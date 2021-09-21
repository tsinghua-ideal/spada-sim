use std::cmp::{max, min};
use std::collections::HashMap;

use crate::frontend::Accelerator;
use crate::new_pqcache_omega_simulator::PE;
use crate::storage::{Element, CsrMatStorage};
use crate::trace_print;

#[derive(Debug, Clone)]
pub struct Task {
    pub block_token: usize,
    pub window_token: usize,
    pub group_size: usize,
    pub merge_mode: bool,
    pub a_eles: Vec<Option<Element>>,
}

impl Task {
    pub fn new(
        block_token: usize,
        window_token: usize,
        group_size: usize,
        merge_mode: bool,
        a_eles: Vec<Option<Element>>,
    ) -> Task {
        Task {
            block_token,
            window_token,
            group_size,
            merge_mode,
            a_eles,
        }
    }
}

pub struct Token {
    token: usize,
}

impl Token {
    pub fn new() -> Token {
        Token{token: 0}
    }

    pub fn new_from(v: usize) -> Token {
        Token{token: v}
    }

    pub fn tik(&mut self) -> usize {
        let r = self.token;
        self.token += 1;
        return r;
    }
}

pub struct BlockTopoTracker {
    pub row_s_list: Vec<usize>,
    pub col_s_list: Vec<Vec<usize>>,
    pub token_list: Vec<Vec<usize>>,
}

impl BlockTopoTracker {
    pub fn new() -> BlockTopoTracker {
        BlockTopoTracker {
            row_s_list: vec![],
            col_s_list: vec![],
            token_list: vec![],
        }
    }

    pub fn find_left(&self, cur_block: [usize; 2]) -> Option<usize> {
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
            return Some(self.token_list[row_pos][col_pos as usize]);
        }
    }

    pub fn find_above(&self, cur_block: [usize; 2]) -> Option<usize> {
        let row_pos = match self.row_s_list.binary_search(&cur_block[1]) {
            Ok(r) | Err(r) => r as i32 - 1,
        };

        if row_pos < 0 || self.col_s_list[row_pos as usize].len() == 0 {
            return None;
        }

        let row_pos = row_pos as usize;

        match self.col_s_list[row_pos].binary_search(&cur_block[0]) {
            Ok(c) => Some(self.token_list[row_pos][c]),
            Err(c) => {
                let c_l = max(c - 1, 0);
                let c_r = min(c + 1, self.col_s_list[row_pos].len() - 1);
                if (cur_block[0] as i64 - self.col_s_list[row_pos][c_l] as i64).abs()
                    >= (self.col_s_list[row_pos][c_r] as i64 - cur_block[0] as i64).abs()
                {
                    return Some(self.token_list[row_pos][c_r]);
                } else {
                    return Some(self.token_list[row_pos][c_l]);
                }
            }
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

pub struct BlockTracker {
    // Config.
    pub token: usize,
    pub anchor: [usize; 2],
    pub shape: [usize; 2],
    pub is_merge_block: bool,
    // Assign job related.
    pub a_cols_assigned: Vec<usize>,
    pub a_cols_num: Vec<usize>,
    pub window_tokens: Vec<usize>,
    // Adjust block related.
    pub miss_size: usize,
    pub psum_rw_size: [usize; 2],
}

impl BlockTracker {
    pub fn new(token: usize, anchor: [usize; 2], shape: [usize; 2], is_merge_block: bool, a_cols_num: Vec<usize>) -> BlockTracker {
        BlockTracker {
            token,
            anchor,
            shape,
            is_merge_block,
            a_cols_assigned: vec![0; a_cols_num.len()],
            a_cols_num,
            window_tokens: vec![],
            miss_size: 0,
            psum_rw_size: [0, 0],
        }
    }
}

pub struct WindowTracker {
    // Config.
    pub token: usize,
    pub anchor: [usize; 2],
    pub block_token: usize,
    pub shape: [usize; 2],
    // Assign job related.
    pub b_cols_assigned: Vec<usize>,
    // PE execution related.
    pub lane2idx: Vec<Option<[usize; 2]>>, // [lane] -> actual a index.
    pub grp2psum_addr: Vec<usize>, // [group] -> writeback psum addr.
}

impl WindowTracker {
    pub fn new(token: usize, anchor: [usize; 2], block_token: usize, shape: [usize; 2],
        lane2idx: Vec<Option<[usize; 2]>>, grp2psum_addr: Vec<usize>) -> WindowTracker {
        WindowTracker {
            token,
            anchor,
            block_token,
            shape,
            b_cols_assigned: vec![0; shape.iter().product()],
            lane2idx,
            grp2psum_addr,
        }
    }
}

pub struct Scheduler {
    // Config.
    a_traversed: bool,
    pe_num: usize,
    lane_num: usize,
    row_s: usize,
    col_s: usize,
    block_shape: [usize; 2],
    a_row_num: usize,
    accelerator: Accelerator,
    a_row_lens: Vec<usize>,
    pub b_row_lens: HashMap<usize, usize>,
    // Adjust scheme.
    b_sparsity: f32,
    a_group: GroupTracker,
    b_group: GroupTracker,
    row_group: usize,
    sampling_bounds: Vec<usize>,
    set_row_num: usize,
    // Assign job related.
    pub block_tracker: HashMap<usize, BlockTracker>, // block_anchor -> BlockTracker
    pub window_tracker: HashMap<usize, WindowTracker>, // window_token -> WindowTracker
    output_tracker: HashMap<usize, Vec<usize>>, // row idx -> psums
    block_topo_tracker: BlockTopoTracker,
    output_addr_token: Token,
    window_token: Token,
    block_token: Token,
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
        accelerator: Accelerator,
    ) -> Scheduler {
        Scheduler {
            a_traversed: false,
            pe_num,
            lane_num,
            row_s: 0,
            col_s: 0,
            block_shape,
            a_row_num: a_matrix.row_num(),
            accelerator,
            a_row_lens: (0..a_matrix.row_num())
                .map(|idx| a_matrix.get_ele_num(idx, idx + 1))
                .collect::<Vec<usize>>(),
            b_row_lens: (0..b_matrix.row_num())
                .map(|idx| (idx, b_matrix.get_ele_num(idx, idx + 1)))
                .collect::<HashMap<usize, usize>>(),
            b_sparsity: 1.0 - b_matrix.data.len() as f32 / (b_matrix.row_num() * b_matrix.mat_shape[0]) as f32,
            a_group: parse_group(a_matrix, var_factor),
            b_group: parse_group(b_matrix, var_factor),
            row_group: usize::MAX,
            sampling_bounds: vec!(),
            set_row_num: usize::MAX,
            block_tracker: HashMap::new(),
            window_tracker: HashMap::new(),
            output_tracker: HashMap::new(),
            block_topo_tracker: BlockTopoTracker::new(),
            output_addr_token: Token::new_from(output_base_addr),
            window_token: Token::new(),
            block_token: Token::new(),
        }
    }

    pub fn assign_jobs(&mut self, pe: &mut PE, a_matrix: &mut CsrMatStorage) -> Option<Task> {
        if self.a_traversed {
            return None;
        }
        if pe.task.is_none() || self.is_block_finished(pe.task.unwrap().block_token) {
            // If any merge block is ready, assign the merge block.
            if let Some(task) = self.merge_task() {
                return Some(task);
            }
            // Otherwise allocate a new block.
            match self.next_block() {
                None => {
                    self.a_traversed = true;
                    return None;
                }
                Some(blk_token) => {
                    let task = self.next_window(blk_token, a_matrix);
                    return task;
                }
            }
        } else {
            match self.next_window(pe.task.unwrap().block_token, a_matrix) {
                None => {
                    return None;
                }
                Some(task) => {
                    return Some(task);
                }
            }
        }
    }

    pub fn is_block_finished(&self, block_token: usize) -> bool {
        let block_tracker = self.block_tracker.get(&block_token).unwrap();
        for (c, l) in block_tracker
            .a_cols_assigned
            .iter()
            .zip(block_tracker.a_cols_num.iter()) {
            if *c < *l {
                return false;
            }
        }

        return true;
    }

    pub fn next_block(&mut self) -> Option<usize> {
        loop {
            // Initial adjust of block.
            if self.row_s == 0 && self.col_s == 0 {
                self.adjust_block([self.row_s, self.col_s]);
            }
            // Return if finished.
            else if self.row_s >= self.a_row_num {
                return None;
            }
            // Prefer to allocate along K dim.
            else if self.is_block_empty([self.row_s, self.col_s], self.block_shape) {
                let token = self.block_token.tik();
                let a_cols_num = (0..self.block_shape[0])
                    .map(|offset| {
                        let ridx = self.row_s + offset;
                        let rlen = self.a_row_lens[ridx];
                        let btail = self.block_shape[1];
                        min(max(rlen, self.col_s+self.block_shape[1]) - self.col_s, btail)
                    })
                    .collect::<Vec<usize>>();
                // Config block tracker.
                self.block_tracker.insert(token, BlockTracker::new(
                    token,
                    [self.row_s, self.col_s],
                    self.block_shape,
                    false,
                    a_cols_num
                ));
                self.col_s += self.block_shape[1];
                return Some(token);
            } else {
                self.row_s += self.block_shape[0];
                self.col_s = 0;
                if self.row_s < self.a_row_num {
                    self.adjust_block([self.row_s, self.col_s]);
                }
            }
        }
    }

    pub fn merge_task(&mut self) -> Option<Task> {
        let mut pairs = vec!();
        // If `lane_num / 2` pairs of psums are found, the a merge block is ready.
        for (row, psum_addrs) in self.output_tracker.iter_mut() {
            while psum_addrs.len() > 1 {
                for addr in psum_addrs.drain(..2) {
                    pairs.push([*row, addr]);
                }
            }
        }
        if pairs.len() < self.lane_num {
            return None;
        } else {
            let blk_token = self.block_token.tik();
            let win_token = self.window_token.tik();
            let a_cols_num = vec![2; self.lane_num / 2];
            let psum_addrs = vec![self.output_addr_token.tik(); self.lane_num / 2];
            // Create merge task.
            let task = Task::new(
                blk_token,
                win_token,
                2,
                true,
                pairs.iter().map(|p| Some(Element::new(*p, 1.0))).collect(),
            );
            // Config block tracker.
            self.block_tracker.insert(blk_token, BlockTracker::new(
                blk_token,
                [0, 0],
                [self.lane_num / 2, 2],
                true,
                a_cols_num,
            ));
            // Config window tracker.
            self.window_tracker.insert(win_token, WindowTracker::new(
                win_token,
                [0, 0],
                blk_token,
                [self.lane_num / 2, 2],
                pairs.iter().map(|p| Some(*p)).collect::<Vec<Option<[usize; 2]>>>(),
                psum_addrs,
            ));
            return Some(task);
        }
    }

    pub fn next_window(&mut self, block_token: usize, a_matrix: &mut CsrMatStorage) -> Option<Task> {
        let blk_tracker = self.block_tracker.get_mut(&block_token).unwrap();
        let prev_window = blk_tracker.window_tokens.last();
        let mut window_shape: [usize; 2];
        let mut window_token: usize;
        let mut window_anchor: [usize; 2];
        if prev_window.is_none() {
            window_shape = self.adjust_window(block_token);
            window_token = self.window_token.tik();
            window_anchor = blk_tracker.anchor;
        } else {
            let prev_window = prev_window.unwrap();
            let window = self.window_tracker.get(prev_window).unwrap();
            window_anchor = window.anchor;
            window_shape = window.shape;
            let row_lim = blk_tracker.anchor[0] + blk_tracker.shape[0];
            let col_lim = blk_tracker.anchor[1] + blk_tracker.shape[1];
            // Return if finished.
            if window_anchor[0] >= row_lim {
                return None;
            }
            // Prefer to allocate along K dim.
            else if window_anchor[1] + window_shape[1] < col_lim {
                window_anchor[1] += window_shape[1];
            }
            // Move to new rows.
            else {
                while window_anchor[0] < row_lim {
                    window_anchor[1] = blk_tracker.anchor[1];
                    window_anchor[0] += window_shape[0];
                    if !self.is_window_empty(blk_tracker.anchor, blk_tracker.shape, window_anchor, window_shape) {
                        break;
                    }
                }
                if window_anchor[0] >= row_lim {
                    return None;
                }
            }
        }
        let lane2idx = vec!();
        let a_eles = vec!();
        let output_addrs = vec![self.output_addr_token.tik(); window_shape[0]];
        for r_idx in window_anchor[0]..window_anchor[0]+window_shape[0] {
            let num = min(max(self.a_row_lens[r_idx], window_anchor[1]),
                window_anchor[1]+window_shape[1]) - window_anchor[1];
            let element = a_matrix.read_scalars(r_idx, window_anchor[1], num).unwrap();
            for e in element {
                lane2idx.push(Some(e.idx));
                e.idx = [window_token, e.idx[1]];
                a_eles.push(Some(e));
            }
            for _ in element.len()..window_shape[1] {
                lane2idx.push(None);
                a_eles.push(None);
            }
        }
        self.window_tracker.insert(window_token, WindowTracker::new(
            window_token,
            window_anchor,
            block_token,
            window_shape,
            lane2idx,
            output_addrs,
        ));
        blk_tracker.window_tokens.push(window_token);
        let task = Some(Task::new(
            block_token,
            window_token,
            window_shape[1],
            false,
            a_eles,
        ));
        return task;
    }

    pub fn is_block_empty(&self, block_anchor: [usize; 2], block_shape: [usize; 2]) -> bool {
        for rowid in block_anchor[0]..block_anchor[0]+block_shape[0] {
            if rowid >= self.a_row_num || block_anchor[1] >= self.a_row_lens[rowid] {
                continue;
            } else {
                return true;
            }
        }

        return false;
    }

    pub fn is_window_empty(&self, block_anchor: [usize; 2], block_shape: [usize; 2], window_anchor: [usize; 2], window_shape: [usize; 2]) -> bool {
        for rowid in window_anchor[0]..min(window_anchor[0]+window_shape[0], block_anchor[1]+block_shape[1]) {
            if rowid >= self.a_row_num || window_anchor[1] >= self.a_row_lens[rowid] {
                continue;
            } else {
                return true;
            }
        }

        return false;
    }

    pub fn is_window_finished(&self, window_token: usize) -> bool {
        let window_tracker = self.window_tracker.get(&window_token).unwrap();
        for r_offset in 0..window_tracker.shape[0] {
            for c_offset in 0..window_tracker.shape[1] {
                let lane_pos = r_offset * window_tracker.shape[1] + c_offset;
                match window_tracker.lane2idx[lane_pos] {
                    None => continue,
                    Some(idx) => {
                        let rlen = self.b_row_lens[&idx[0]];
                        if window_tracker.b_cols_assigned[lane_pos] < rlen {
                            return false;
                        }
                    }
                }
            }
        }

        return false;
    }

    pub fn adjust_block(&mut self, block_anchor: [usize; 2]) {
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
                            let n1_token = self.block_topo_tracker.find_above(block_anchor);
                            if n1_token.is_none() {
                                return;
                            }
                            let n1_token = n1_token.unwrap();
                            let n1_block = self.block_tracker.get(&n1_token).unwrap().anchor;
                            let n1_row_num = block_anchor[1] - n1_block[1];
                            let n1_ele_size =
                                (n1_block[1]..block_anchor[1]).fold(0, |s, x| s + self.a_row_lens[x]);

                            let n2_token = self.block_topo_tracker.find_above(n1_block);
                            if n2_token.is_none() {
                                return;
                            }
                            let n2_token = n2_token.unwrap();
                            let n2_block = self.block_tracker.get(&n2_token).unwrap().anchor;
                            let n2_row_num = n1_block[1] - n2_block[1];
                            let n2_ele_size =
                                (n2_block[1]..n1_block[1]).fold(0, |s, x| s + self.a_row_lens[x]);

                            let n1_cost = (self.block_tracker[&n1_token].miss_size
                                + self.block_tracker[&n1_token].psum_rw_size[0])
                                * 100
                                + self.block_tracker[&n1_token].psum_rw_size[1];
                            let n2_cost = (self.block_tracker[&n2_token].miss_size
                                + self.block_tracker[&n2_token].psum_rw_size[0])
                                * 100
                                + self.block_tracker[&n2_token].psum_rw_size[1];

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

    pub fn adjust_window(&mut self, block_token: usize) -> [usize; 2] {
        if self.accelerator == Accelerator::NewOmega {
            return [self.lane_num / self.block_shape[1], self.block_shape[1]];
        }

        match self.accelerator {
            Accelerator::Ip | Accelerator::Omega | Accelerator::Op | Accelerator::NewOmega => {
                let mut reduction_window: [usize; 2];
                reduction_window = [self.lane_num / self.block_shape[1], self.block_shape[1]];

                reduction_window
            }
        }
    }

    pub fn collect_pending_psums(&mut self, window_token: usize) {
        let window_tracker = self.window_tracker.get(&window_token).unwrap();
        for i in 0..window_tracker.shape[0] {
            let row_idx = window_tracker.anchor[0] + i;
            let psum_addr = window_tracker.grp2psum_addr[i];
            self.output_tracker
                .entry(row_idx)
                .and_modify(|ps| ps.push(psum_addr))
                .or_insert(vec![psum_addr,]);
        }
    }
}