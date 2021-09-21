use std::cmp::{max, min};
use std::collections::HashMap;

use crate::frontend::Accelerator;
use crate::pqcache_omega_simulator::PE;
use crate::storage::CsrMatStorage;
use crate::trace_print;

#[derive(Debug, Clone)]
pub struct Block {
    pub is_merge_block: bool,
    pub anchor: [usize; 2],
    pub shape: [usize; 2],
}

impl Block {
    pub fn new(anchor: [usize; 2], shape: [usize; 2], is_merge_block: bool) -> Block {
        Block {
            anchor,
            shape,
            is_merge_block,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Window {
    pub anchor: [usize; 2],
    pub shape: [usize; 2],
    pub alloc: Vec<bool>,
    pub idxs: HashMap<[usize; 2], [usize; 2]>,
    pub token: usize,
}

impl Window {
    pub fn new(
        anchor: [usize; 2],
        shape: [usize; 2],
        idxs: HashMap<[usize; 2], [usize; 2]>,
        token: usize,
    ) -> Window {
        Window {
            anchor,
            shape,
            alloc: vec![false; shape.iter().product()],
            idxs,
            token,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ExecTracker {
    pub touched_fiber_size: usize,
    pub dedup_fiber_size: usize,
    pub output_fiber_size: usize,
    pub miss_size: usize,
    pub psum_rw_size: [usize; 2],

    pub b_cols_done: HashMap<[usize; 2], usize>, // Keep track of current executed b columns in the window.
    pub a_cols_done: Vec<usize>,
    pub psum_row_addr_binding: HashMap<usize, usize>, // Bind the psum of a row to a specific addr.
    pub a_col_remains: Vec<usize>,
    pub block: Block,
    pub window: Window,
    pub window_token_counter: usize,
}

impl ExecTracker {
    pub fn new(block: Block, window: Window, a_col_remains: Vec<usize>) -> ExecTracker {
        ExecTracker {
            b_cols_done: HashMap::new(),
            a_cols_done: vec![usize::MAX; block.shape[1]],
            psum_row_addr_binding: HashMap::new(),
            a_col_remains,
            block,
            window,
            touched_fiber_size: 0,
            dedup_fiber_size: 0,
            output_fiber_size: 0,
            miss_size: 0,
            psum_rw_size: [0, 0],
            window_token_counter: 0,
        }
    }

    pub fn c_reuse(&self) -> f64 {
        self.touched_fiber_size as f64
            / (self.output_fiber_size as f64 * self.window.shape[0] as f64 + 0.00001)
    }

    pub fn b_reuse(&self) -> f64 {
        self.touched_fiber_size as f64
            / (self.dedup_fiber_size as f64 * self.window.shape[1] as f64 + 0.00001)
    }

    pub fn set_window(&mut self, window: Window) {
        self.window = window;
        self.b_cols_done.clear();
    }
}

pub struct BlockTracker {
    pub row_s_list: Vec<usize>,
    pub col_s_list: Vec<Vec<usize>>,
    pub exec_logs: HashMap<[usize; 2], ExecTracker>,
}

impl BlockTracker {
    pub fn new() -> BlockTracker {
        BlockTracker {
            row_s_list: vec![],
            col_s_list: vec![],
            exec_logs: HashMap::new(),
        }
    }

    pub fn find_left(&self, cur_block: &[usize; 2]) -> Option<[usize; 2]> {
        let row_pos = match self.row_s_list.binary_search(&cur_block[1]) {
            Ok(r) | Err(r) => r as i32 - 1,
        };
        if row_pos < 0 {
            return None;
        }
        let row_pos = row_pos as usize;

        let col_pos = match self.col_s_list[row_pos].binary_search(&cur_block[0]) {
            Ok(c) | Err(c) => c as i32 - 1,
        };

        if col_pos < 0 {
            return None;
        } else {
            return Some([self.col_s_list[row_pos][col_pos as usize], cur_block[1]]);
        }
    }

    pub fn find_above(&self, cur_block: &[usize; 2]) -> Option<[usize; 2]> {
        let row_pos = match self.row_s_list.binary_search(&cur_block[1]) {
            Ok(r) | Err(r) => r as i32 - 1,
        };

        if row_pos < 0 || self.col_s_list[row_pos as usize].len() == 0 {
            return None;
        }

        let row_pos = row_pos as usize;

        match self.col_s_list[row_pos].binary_search(&cur_block[0]) {
            Ok(c) => Some([self.col_s_list[row_pos][c], self.row_s_list[row_pos]]),
            Err(c) => {
                let c_l = max(c - 1, 0);
                let c_r = min(c + 1, self.col_s_list[row_pos].len() - 1);
                if (cur_block[0] as i64 - self.col_s_list[row_pos][c_l] as i64).abs()
                    >= (self.col_s_list[row_pos][c_r] as i64 - cur_block[0] as i64).abs()
                {
                    return Some([self.col_s_list[row_pos][c_r], self.row_s_list[row_pos]]);
                } else {
                    return Some([self.col_s_list[row_pos][c_l], self.row_s_list[row_pos]]);
                }
            }
        }
    }

    pub fn exec_tracker(&self, block_idx: &[usize; 2]) -> &ExecTracker {
        self.exec_logs.get(block_idx).unwrap()
    }

    pub fn exec_tracker_mut(&mut self, block_idx: &[usize; 2]) -> &mut ExecTracker {
        self.exec_logs.get_mut(block_idx).unwrap()
    }
}

#[derive(Debug, Clone)]
struct MergeTracker {
    pub asgnd_col_ranges: HashMap<usize, Vec<[usize; 2]>>, // assigned row -> assigned col ranges.
}

impl MergeTracker {
    pub fn new() -> MergeTracker {
        MergeTracker {
            asgnd_col_ranges: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct GroupInfo {
    pub row_range: [usize; 2],
    pub avg_row_len: usize,
    pub cost_num: HashMap<usize, [usize; 2]>,
}

#[derive(Debug, Clone)]
pub struct GroupTracker {
    pub groups: Vec<GroupInfo>,
    pub rgmap: HashMap<usize, usize>,
}

impl GroupTracker {
    pub fn new() -> GroupTracker {
        GroupTracker {
            groups: vec![],
            rgmap: HashMap::new(),
        }
    }

    pub fn add_group(&mut self, gi: GroupInfo) {
        self.groups.push(gi);
        let last_idx = self.groups.len() - 1;
        for rowidx in self.groups[last_idx].row_range[0]..self.groups[last_idx].row_range[1] {
            self.rgmap.insert(rowidx, last_idx);
        }
    }
}

pub fn parse_group(matrix: &CsrMatStorage, var_factor: f32) -> GroupTracker {
    let mut gt = GroupTracker::new();
    let mut prev_row_len = usize::MAX;
    let mut row_s = 0;

    // Parse matrix.
    for idx in 0..matrix.row_num() + 1 {
        if idx == matrix.row_num() {
            // Finish the last group.
            let gi = GroupInfo {
                row_range: [row_s, idx],
                avg_row_len: (matrix.get_ele_num(row_s, idx)) / (idx - row_s),
                cost_num: HashMap::new(),
            };
            gt.add_group(gi);
        } else {
            let row_len = matrix.get_ele_num(idx, idx + 1);
            if row_len == 0 {
                continue;
            } else if prev_row_len == usize::MAX {
                // Init the first group.
                prev_row_len = row_len;
            } else if prev_row_len as f32 * var_factor < row_len as f32
                || prev_row_len as f32 > var_factor * row_len as f32
            {
                // Encounter a new group. Save the current one.
                let gi = GroupInfo {
                    row_range: [row_s, idx],
                    avg_row_len: (matrix.get_ele_num(row_s, idx)) / (idx - row_s),
                    cost_num: HashMap::new(),
                };
                gt.add_group(gi);
                prev_row_len = row_len;
                row_s = idx;
            } else {
                prev_row_len = row_len;
            }
        }
    }

    return gt;
}

pub struct Scheduler {
    a_traversed: bool,
    pe_num: usize,
    lane_num: usize,
    pub block_tracker: BlockTracker,
    merge_queue: Vec<usize>,
    block_shape: [usize; 2],
    output_base_addr: usize,
    output_tracker: HashMap<usize, Vec<usize>>,
    row_s: usize,
    col_s: usize,
    merge_tracker: MergeTracker,
    merge_pe: usize,
    merge_counter: usize,
    merge_period: usize,
    b_sparsity: f32,
    a_group: GroupTracker,
    b_group: GroupTracker,
    row_group: usize,
    sampling_bounds: Vec<usize>,
    set_row_num: usize,
    a_row_num: usize,
    accelerator: Accelerator,
    a_row_lens: Vec<usize>,
    pub b_row_lens: HashMap<usize, usize>,
    merge_block_token: usize,
}

impl Scheduler {
    pub fn new(
        pe_num: usize,
        lane_num: usize,
        block_shape: [usize; 2],
        output_base_addr: usize,
        b_sparsity: f32,
        a_matrix: &CsrMatStorage,
        b_matrix: &CsrMatStorage,
        var_factor: f32,
        a_row_num: usize,
        accelerator: Accelerator,
    ) -> Scheduler {
        Scheduler {
            a_traversed: false,
            pe_num,
            lane_num,
            block_tracker: BlockTracker::new(),
            merge_queue: vec![],
            block_shape,
            output_base_addr,
            output_tracker: HashMap::new(),
            row_s: 0,
            col_s: 0,
            merge_tracker: MergeTracker::new(),
            merge_pe: 0,
            merge_counter: 0,
            merge_period: lane_num / block_shape[1],
            b_sparsity,
            a_group: parse_group(a_matrix, var_factor),
            b_group: parse_group(b_matrix, var_factor),
            row_group: usize::MAX,
            sampling_bounds: vec![],
            set_row_num: usize::MAX,
            a_row_num,
            accelerator,
            a_row_lens: (0..a_matrix.row_num())
                .map(|idx| a_matrix.get_ele_num(idx, idx + 1))
                .collect::<Vec<usize>>(),
            b_row_lens: (0..b_matrix.row_num())
                .map(|idx| (idx, b_matrix.get_ele_num(idx, idx + 1)))
                .collect::<HashMap<usize, usize>>(),
            merge_block_token: 0,
        }
    }

    pub fn assign_jobs(&mut self, pe: &mut PE) -> bool {
        // Check if the workload is finished.
        if self.a_traversed && self.merge_tracker.asgnd_col_ranges.len() == 0 {
            return false;
        }

        // For the PE, check if current assigned block is finished.
        // If finished, assign a new block.
        // If not finished, check the current window.
        if pe.block_anchor == [usize::MAX, usize::MAX] || self.is_block_finished(&pe.block_anchor) {
            // If any merge block is ready, assign the merge block.
            if let Some((block, window)) = self.merge_task() {
                self.update_merge(block, window, pe);
                return true;
            }
            // Otherwise allocate a new block.
            match self.next_block() {
                None => {
                    self.a_traversed = true;
                    self.update_block(None, None, pe);
                    return false;
                }
                Some(block) => {
                    let win = self.next_window(&block.anchor);
                    self.update_block(Some(block), win, pe);
                    return true;
                }
            }
        } else {
            let blk_idx = pe.block_anchor;
            match self.next_window(&blk_idx) {
                None => {
                    self.update_window(None, pe);
                    return false;
                }
                Some(window) => {
                    self.update_window(Some(window), pe);
                    return true;
                }
            }
        }
    }

    pub fn is_block_finished(&self, blk_idx: &[usize; 2]) -> bool {
        let exec_tracker = self.block_tracker.exec_tracker(blk_idx);
        for (c, l) in exec_tracker
            .a_cols_done
            .iter()
            .zip(exec_tracker.a_col_remains.iter())
        {
            if *c + blk_idx[0] < *l {
                return false;
            }
        }

        return true;
    }

    pub fn next_block(&mut self) -> Option<Block> {
        loop {
            // Initial adjust of block.
            if self.row_s == 0 && self.col_s == 0 {
                self.adjust_block([self.col_s, self.row_s]);
            }
            // Return if finished.
            else if self.row_s >= self.a_row_num {
                return None;
            }
            // Prefer to allocate along K dim.
            else if self.is_block_valid(self.row_s, self.block_shape[1], self.col_s) {
                let block = Block {
                    anchor: [self.col_s, self.row_s],
                    shape: self.block_shape.clone(),
                    is_merge_block: false,
                };
                self.col_s += self.block_shape[0];
                return Some(block);
            } else {
                self.row_s += self.block_shape[1];
                self.col_s = 0;
                if self.row_s < self.a_row_num {
                    self.adjust_block([self.col_s, self.row_s]);
                }
            }
        }
    }

    pub fn update_block(&mut self, block: Option<Block>, window: Option<Window>, pe: &mut PE) {
        // Config PE.
        if block.is_none() || window.is_none() {
            pe.stop_stream = true;
            return;
        }

        let block = block.unwrap();
        let window = window.unwrap();

        pe.set_block(&block);
        pe.set_window(&window);
        pe.unbind_win2lane();
        pe.stop_stream = false;

        // Config block_tracker.
        if block.anchor[0] == 0 {
            self.block_tracker.row_s_list.push(block.anchor[1]);
            self.block_tracker.col_s_list.push(vec![]);
        }
        self.block_tracker
            .col_s_list
            .last_mut()
            .unwrap()
            .push(block.anchor[0]);

        // Config merge_tracker.
        for rowid in block.anchor[1]..block.anchor[1] + block.shape[1] {
            self.merge_tracker
                .asgnd_col_ranges
                .entry(rowid)
                .and_modify(
                    |rngs| match rngs.binary_search_by(|p| p[0].cmp(&block.anchor[0])) {
                        Ok(pos) => rngs[pos][1] = block.anchor[0] + block.shape[0],
                        Err(pos) => {
                            rngs.insert(pos, [block.anchor[0], block.anchor[0] + block.shape[0]])
                        }
                    },
                )
                .or_insert(vec![[block.anchor[0], block.anchor[0] + block.shape[0]]]);
        }

        // Config exec tracker.
        assert!(
            !self.block_tracker.exec_logs.contains_key(&block.anchor),
            "Block already added!"
        );
        let a_col_remains = (0..block.shape[1])
            .map(|offset| {
                let ridx = block.anchor[1] + offset;
                let rlen = self.a_row_lens[ridx];
                let btail = block.shape[0];
                min(max(rlen, block.anchor[0]) - block.anchor[0], btail)
            })
            .collect::<Vec<usize>>();
        self.block_tracker
            .exec_logs
            .insert(block.anchor, ExecTracker::new(block, window, a_col_remains));
    }

    pub fn update_window(&mut self, window: Option<Window>, pe: &mut PE) {
        // Config PE.
        if window.is_none() {
            pe.stop_stream = true;
            return;
        }

        let window = window.unwrap();
        pe.set_window(&window);
        pe.stop_stream = false;

        // Config exec tracker.
        let block_idx = pe.block_anchor;
        let exec_tracker = self.block_tracker.exec_tracker_mut(&block_idx);
        exec_tracker.set_window(window);
        for i in 0..exec_tracker.window.shape[1] {
            exec_tracker.psum_row_addr_binding.insert(i, self.output_base_addr);
            self.output_base_addr += 1;
        }
    }

    pub fn update_merge(&mut self, block: Block, window: Window, pe: &mut PE) {
        // Config PE.
        pe.set_block(&block);
        pe.set_window(&window);
        pe.unbind_win2lane();
        pe.stop_stream = false;
        pe.merge_mode = true;

        // Config exec tracker.
        assert!(
            !self.block_tracker.exec_logs.contains_key(&block.anchor),
            "Block already added!"
        );
        let a_col_remains = vec![2; block.shape[1]];
        self.block_tracker
            .exec_logs
            .insert(block.anchor, ExecTracker::new(block, window, a_col_remains));
    }

    pub fn is_block_valid(&self, row_s: usize, row_num: usize, col_s: usize) -> bool {
        for rowid in row_s..row_s + row_num {
            if !self.is_col_s_valid(rowid, col_s) {
                continue;
            } else {
                return true;
            }
        }

        return false;
    }

    pub fn is_col_s_valid(&self, rowid: usize, col_s: usize) -> bool {
        if (rowid >= self.a_row_num) || (self.a_row_lens[rowid] <= col_s) {
            return false;
        } else {
            return true;
        }
    }

    pub fn is_window_finished(&self, blk_idx: &[usize; 2]) -> bool {
        let exec_tracker = self.block_tracker.exec_tracker(blk_idx);
        for r_offset in 0..exec_tracker.window.shape[1] {
            for c_offset in 0..exec_tracker.window.shape[0] {
                match exec_tracker.window.idxs.get(&[c_offset, r_offset]) {
                    None => continue,
                    Some(idx) => {
                        let rlen = self.b_row_lens[&idx[0]];
                        if exec_tracker.b_cols_done[idx] < rlen {
                            return false;
                        }
                    }
                }
            }
        }

        return true;
    }

    pub fn next_window(&mut self, blk_idx: &[usize; 2]) -> Option<Window> {
        let exec_tracker = self.block_tracker.exec_tracker(blk_idx);
        let mut window = exec_tracker.window.clone();
        let block = &exec_tracker.block;
        let b_col_lim = block.anchor[0] + block.shape[0];
        let b_row_lim = block.anchor[1] + block.shape[1];
        loop {
            // Return if finished.
            if window.anchor[1] >= b_row_lim {
                return None;
            }
            // Prefer to allocate along K dim.
            else if window.anchor[0] + window.shape[0] < b_col_lim {
                let new_anchor = [window.anchor[0] + window.shape[0], window.anchor[1]];
                let new_shape = window.shape.clone();
                let new_idxs = HashMap::new();
                let token = self.block_tracker.exec_tracker(blk_idx).window_token_counter;
                self.block_tracker.exec_tracker_mut(blk_idx).window_token_counter += 1;
                return Some(Window::new(new_anchor, new_shape, new_idxs, token));
            }
            // Move to new rows.
            else {
                window.anchor[1] += window.shape[1];
                window.anchor[0] = block.anchor[0];
            }
        }
    }

    pub fn merge_task(&mut self) -> Option<(Block, Window)> {
        let mut pairs = vec!();
        let mut rows = vec!();
        // If `lane_num / 2` pairs of psums are found, the a merge block is ready.
        for (row, psum_addrs) in self.output_tracker.iter_mut() {
            while psum_addrs.len() > 1 {
                pairs.extend(psum_addrs.drain(..2));
                rows.push(*row);
            }
        }

        if rows.len() < self.lane_num / 2 {
            return None;
        } else {
            let token = self.merge_block_token;
            self.merge_block_token += 1;
            let block = Block::new(
                [usize::MAX, token],
                [self.lane_num / 2, 2],
                true);
            let mut window = Window::new(
                [usize::MAX, token],
                [self.lane_num / 2, 2],
                HashMap::new(),
                0
            );
            for r_offset in 0..self.lane_num / 2 {
                for c_offset in 0..2 {
                    window.idxs.insert([c_offset, r_offset], [rows[r_offset], pairs[r_offset*2+c_offset]]);
                }
            }
            return Some((block, window));
        }
    }

    pub fn adjust_block(&mut self, cur_idx: [usize; 2]) {
        match self.accelerator {
            Accelerator::Ip | Accelerator::Omega | Accelerator::Op => {
                // First check if the row group changed.
                if self.a_group.rgmap[&self.row_s] != self.row_group {
                    self.row_group = self.a_group.rgmap[&self.row_s];
                    return;
                }
            }
            Accelerator::NewOmega => {
                let block_adjust_scheme = 8;
                match block_adjust_scheme {
                    8 => {
                        trace_print!("-Adjust block");
                        // Separately treat wide groups and narrow groups.
                        let group_diviser = 128;
                        let sample_num = 4;
                        let mut min_row_num = 1;

                        // First check if the row group changed and prepare for sampling.
                        if self.a_group.rgmap[&self.row_s] != self.row_group {
                            // Start from row_num = 1 to touch the distribution.
                            self.block_shape[1] = 1;
                            self.row_group = self.a_group.rgmap[&self.row_s];
                            let cur_gi = &self.a_group.groups[self.row_group];
                            if cur_gi.row_range[1] - cur_gi.row_range[0] > group_diviser {
                                let mut cur_row = self.row_s + 1;
                                let mut i = 1;
                                self.sampling_bounds.clear();
                                while i <= self.lane_num {
                                    cur_row += sample_num * i;
                                    self.sampling_bounds.push(cur_row);
                                    i *= 2;
                                }
                            }
                            self.set_row_num = usize::MAX;
                            return;
                        }

                        let cur_gi = &self.a_group.groups[self.row_group];
                        if cur_gi.row_range[1] - cur_gi.row_range[0] > group_diviser {
                            // Treat the wide groups.
                            if self.row_s >= *self.sampling_bounds.last().unwrap() {
                                if self.set_row_num == usize::MAX {
                                    // Sampling finished.
                                    // Then adjust based on the cost of different row num.
                                    let mut min_cost = f32::MAX;
                                    let mut cur_row_num = 1;
                                    while cur_row_num <= self.lane_num {
                                        if let Some(cost_num) = self.a_group.groups[self.row_group]
                                            .cost_num
                                            .get_mut(&cur_row_num)
                                        {
                                            let div_cost =
                                                cost_num[0] as f32 / (cost_num[1] as f32 + 0.0001);
                                            if div_cost < min_cost {
                                                min_cost = div_cost;
                                                self.set_row_num = cur_row_num;
                                            }
                                        } else {
                                            self.a_group.groups[self.row_group]
                                                .cost_num
                                                .insert(cur_row_num, [0, 0]);
                                            self.set_row_num = cur_row_num;
                                            break;
                                        }
                                        cur_row_num *= 2;
                                    }
                                    while cur_row_num > 1
                                        && (self.row_s + cur_row_num
                                            >= self.a_group.groups[self.row_group].row_range[1])
                                    {
                                        cur_row_num /= 2;
                                    }
                                }
                                min_row_num = self.set_row_num;
                            } else {
                                // Sampling.
                                trace_print!("---Sampling");
                                min_row_num =
                                    match self.sampling_bounds.binary_search(&(self.row_s)) {
                                        Ok(idx) => 2usize.pow(idx as u32 + 1),
                                        Err(idx) => 2usize.pow(idx as u32),
                                    };
                            }
                            while min_row_num > 1
                                && (self.row_s + min_row_num
                                    >= self.a_group.groups[self.row_group].row_range[1])
                            {
                                min_row_num /= 2;
                            }
                            trace_print!(
                                "group_range {:?} cost num: {:?}",
                                &self.a_group.groups[self.row_group].row_range,
                                self.a_group.groups[self.row_group].cost_num
                            );
                            self.block_shape[1] = min_row_num;
                        } else {
                            // Treat the narrow groups.
                            let n1_block = self.block_tracker.find_above(&cur_idx);
                            if n1_block.is_none() {
                                return;
                            }
                            let n1_block = n1_block.unwrap();
                            let n1_row_num = cur_idx[1] - n1_block[1];
                            let n1_ele_size =
                                (n1_block[1]..cur_idx[1]).fold(0, |s, x| s + self.a_row_lens[x]);

                            let n2_block = self.block_tracker.find_above(&n1_block);
                            if n2_block.is_none() {
                                return;
                            }
                            let n2_block = n2_block.unwrap();
                            let n2_row_num = n1_block[1] - n2_block[1];
                            let n2_ele_size =
                                (n2_block[1]..n1_block[1]).fold(0, |s, x| s + self.a_row_lens[x]);

                            let n1_cost = (self.block_tracker.exec_tracker(&n1_block).miss_size
                                + self.block_tracker.exec_tracker(&n1_block).psum_rw_size[0])
                                * 100
                                + self.block_tracker.exec_tracker(&n1_block).psum_rw_size[1];
                            let n2_cost = (self.block_tracker.exec_tracker(&n2_block).miss_size
                                + self.block_tracker.exec_tracker(&n2_block).psum_rw_size[0])
                                * 100
                                + self.block_tracker.exec_tracker(&n2_block).psum_rw_size[1];

                            trace_print!(
                                "group_range {:?} n1_cost: {}, n1_ele_size: {}, n2_cost: {}, n2_ele_size: {}",
                                &self.a_group.groups[self.row_group].row_range, n1_cost, n1_ele_size, n2_cost, n2_ele_size
                            );

                            if (n1_cost as f32 / n1_ele_size as f32)
                                <= (n2_cost as f32 / n2_ele_size as f32)
                            {
                                if n1_row_num >= n2_row_num {
                                    self.block_shape[1] =
                                        min(self.block_shape[1] * 2, self.lane_num);
                                } else {
                                    self.block_shape[1] = max(self.block_shape[1] / 2, 1);
                                }
                            } else {
                                if n1_row_num >= n2_row_num {
                                    self.block_shape[1] = max(self.block_shape[1] / 2, 1);
                                } else {
                                    self.block_shape[1] =
                                        min(self.block_shape[1] * 2, self.lane_num);
                                }
                            }

                            while self.block_shape[1] > 1
                                && (self.row_s + self.block_shape[1]
                                    >= self.a_group.groups[self.row_group].row_range[1])
                            {
                                self.block_shape[1] /= 2;
                            }
                        }
                    }
                    _ => panic!("Invalid merge scheme: {}", block_adjust_scheme),
                }
            }
        }
    }

    pub fn collect_pending_psums(&mut self, block_anchor: [usize; 2]) {
        let exec_tracker = self.block_tracker.exec_tracker(&block_anchor);
        let window_anchor = exec_tracker.window.anchor;
        for i in 0..exec_tracker.window.shape[1] {
            let row_idx = window_anchor[1] + i;
            let psum_addr = exec_tracker.psum_row_addr_binding[&i];
            self.output_tracker
                .entry(row_idx)
                .and_modify(|ps| ps.push(psum_addr))
                .or_insert(vec!(psum_addr,));
        }
    }
}
