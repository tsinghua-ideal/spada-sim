use std::cmp::{max, min};
use std::collections::{HashMap, HashSet};
use std::mem;
use std::ops::AddAssign;

use crate::adder_tree::AdderTree;
use crate::block_topo_tracker::BlockTopoTracker;
use crate::colwise_irr_adjust::{ColwiseIrrBlockAdjustTracker, ColwiseIrrBlockInfo};
use crate::colwise_reg_adjust::{ColwiseRegBlockAdjustTracker, ColwiseRegBlockInfo};
use crate::frontend::Accelerator;
use crate::rowwise_adjust::{RowwiseAdjustTracker, RowwiseBlockInfo};
use crate::rowwise_perf_adjust::{RowwiseLatencyAdjustTracker, RowwiseLatencyBlockInfo};
use crate::simulator::PE;
use crate::storage::LatencyPriorityCache;
use crate::storage::{CsrMatStorage, Element};

#[derive(Debug, Clone)]
pub struct Task {
    pub block_token: usize,
    pub window_token: usize,
    pub group_size: usize,
    pub merge_mode: bool,
    pub a_eles: Vec<Option<Element>>,
    pub memory_traffic: usize,
    pub start_cycle: usize,
}

impl Task {
    pub fn new(
        block_token: usize,
        window_token: usize,
        group_size: usize,
        merge_mode: bool,
        a_eles: Vec<Option<Element>>,
        start_cycle: usize,
    ) -> Task {
        Task {
            block_token,
            window_token,
            group_size,
            merge_mode,
            a_eles,
            memory_traffic: 0,
            start_cycle,
        }
    }
}

pub struct Token {
    token: usize,
}

impl Token {
    pub fn new() -> Token {
        Token { token: 0 }
    }

    pub fn new_from(v: usize) -> Token {
        Token { token: v }
    }

    pub fn tik(&mut self) -> usize {
        let r = self.token;
        self.token += 1;
        return r;
    }
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
    // Merge related.
    pub is_tail: Vec<bool>,
}

impl BlockTracker {
    pub fn new(
        token: usize,
        anchor: [usize; 2],
        shape: [usize; 2],
        is_merge_block: bool,
        a_cols_num: Vec<usize>,
        is_tail: Vec<bool>,
    ) -> BlockTracker {
        BlockTracker {
            token,
            anchor,
            shape,
            is_merge_block,
            a_cols_assigned: vec![0; a_cols_num.len()],
            a_cols_num,
            window_tokens: vec![],
            is_tail,
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
    pub arow_addr_pairs: Vec<[usize; 2]>,  // [group] -> writeback psum addr.
}

impl WindowTracker {
    pub fn new(
        token: usize,
        anchor: [usize; 2],
        block_token: usize,
        shape: [usize; 2],
        lane2idx: Vec<Option<[usize; 2]>>,
        arow_addr_pairs: Vec<[usize; 2]>,
    ) -> WindowTracker {
        WindowTracker {
            token,
            anchor,
            block_token,
            shape,
            b_cols_assigned: vec![0; shape.iter().product()],
            lane2idx,
            arow_addr_pairs,
        }
    }
}

pub struct Scheduler {
    // Config.
    pub a_traversed: bool,
    lane_num: usize,
    pub row_s: usize,
    pub col_s: usize,
    block_shape: [usize; 2],
    a_row_num: usize,
    pub accelerator: Accelerator,
    a_row_lens: Vec<usize>,
    pub b_row_lens: HashMap<usize, usize>,
    pub mem_latency: usize,
    pub cache_latency: usize,
    // Adjust scheme.
    adjust_scheme: usize,
    pub rowwise_adjust_tracker: RowwiseAdjustTracker,
    pub rowwise_latency_adjust_tracker: RowwiseLatencyAdjustTracker,
    pub colwise_reg_adjust_tracker: ColwiseRegBlockAdjustTracker,
    pub colwise_irr_adjust_tracker: ColwiseIrrBlockAdjustTracker,
    // Assign job related.
    pub block_tracker: HashMap<usize, BlockTracker>, // block_anchor -> BlockTracker
    pub window_tracker: HashMap<usize, WindowTracker>, // window_token -> WindowTracker
    pub output_tracker: HashMap<usize, Vec<usize>>,  // row idx -> psums
    block_topo_tracker: BlockTopoTracker,
    output_addr_token: Token,
    window_token: Token,
    block_token: Token,
    pub a_tail_produced: HashSet<usize>,
    // pub a_row_finished: HashSet<usize>,
    pub a_row_finished: HashMap<usize, usize>,
    pub a_cols_assigned: Vec<usize>,
    pub a_cols_produced: HashMap<usize, usize>,
    pub row_rgstr_task: HashMap<usize, usize>,
    staged_tasks: Vec<Option<Task>>,
    latest_block_token: usize,
}

impl Scheduler {
    pub fn new(
        pe_num: usize,
        lane_num: usize,
        block_shape: [usize; 2],
        output_base_addr: usize,
        a_matrix: &CsrMatStorage,
        b_matrix: &CsrMatStorage,
        var_factor: f32,
        accelerator: Accelerator,
        mem_latency: usize,
        cache_latency: usize,
    ) -> Scheduler {
        Scheduler {
            a_traversed: false,
            lane_num,
            row_s: usize::MAX,
            col_s: usize::MAX,
            block_shape,
            a_row_num: a_matrix.row_num(),
            accelerator,
            a_row_lens: (0..a_matrix.row_num())
                .map(|idx| a_matrix.get_ele_num(idx, idx + 1))
                .collect::<Vec<usize>>(),
            b_row_lens: (0..b_matrix.row_num())
                .map(|idx| (idx, b_matrix.get_ele_num(idx, idx + 1)))
                .collect::<HashMap<usize, usize>>(),
            adjust_scheme: 3,
            block_tracker: HashMap::new(),
            window_tracker: HashMap::new(),
            output_tracker: HashMap::new(),
            block_topo_tracker: BlockTopoTracker::new(),
            output_addr_token: Token::new_from(output_base_addr),
            window_token: Token::new(),
            block_token: Token::new(),
            a_tail_produced: HashSet::new(),
            // a_row_finished: HashSet::new(),
            a_row_finished: HashMap::new(),
            rowwise_adjust_tracker: RowwiseAdjustTracker::new(
                lane_num, a_matrix, b_matrix, var_factor,
            ),
            rowwise_latency_adjust_tracker: RowwiseLatencyAdjustTracker::new(
                lane_num, a_matrix, b_matrix, var_factor,
            ),
            colwise_reg_adjust_tracker: ColwiseRegBlockAdjustTracker::new(lane_num),
            colwise_irr_adjust_tracker: ColwiseIrrBlockAdjustTracker::new(
                lane_num, lane_num, lane_num,
            ),
            mem_latency,
            cache_latency,
            a_cols_assigned: vec![0; a_matrix.row_num()],
            a_cols_produced: HashMap::new(),
            row_rgstr_task: HashMap::new(),
            staged_tasks: vec![None; pe_num],
            latest_block_token: usize::MAX,
        }
    }

    pub fn assign_task(
        &mut self,
        pe: &mut PE,
        a_matrix: &mut CsrMatStorage,
        cur_cycle: usize,
    ) -> Option<(usize, Task)> {
        // If previous block is finished, try assign the undone latest block, or alloc a new block.
        if pe.task.is_none() || self.is_block_finished(pe.task.as_ref().unwrap().block_token) {
            if self.latest_block_token != usize::MAX
            // && pe.task.as_ref().unwrap().block_token < self.latest_block_token
            && !self.is_block_finished(self.latest_block_token)
            {
                return self.next_window(self.latest_block_token, a_matrix, cur_cycle);
            } else {
                match self.next_block() {
                    None => {
                        self.a_traversed = true;
                        // Check if there are some merge task remained.
                        if let Some(task) = self.merge_task(cur_cycle) {
                            return Some((0, task));
                        } else {
                            return None;
                        }
                    }
                    Some(blk_token) => {
                        let latency_task = self.next_window(blk_token, a_matrix, cur_cycle);
                        self.latest_block_token = blk_token;
                        return latency_task;
                    }
                }
            }
        } else {
            return self.next_window(pe.task.as_ref().unwrap().block_token, a_matrix, cur_cycle);
        }
    }

    pub fn is_block_finished(&mut self, block_token: usize) -> bool {
        let block_tracker = self.block_tracker.get(&block_token).unwrap();
        for (c, l) in block_tracker
            .a_cols_assigned
            .iter()
            .zip(block_tracker.a_cols_num.iter())
        {
            if *c < *l {
                return false;
            }
        }
        return true;
    }

    pub fn label_finished_rows(&mut self, block_token: usize) {
        let block_tracker = self.block_tracker.get(&block_token).unwrap();
        if !block_tracker.is_merge_block {
            for (offset, is_tail) in block_tracker.is_tail.iter().enumerate() {
                let rowidx = offset + block_tracker.anchor[0];
                if *is_tail && !self.a_row_finished.contains_key(&rowidx) {
                    self.a_tail_produced.insert(rowidx);
                }
            }
        }
    }

    pub fn next_block(&mut self) -> Option<usize> {
        loop {
            // Initial adjust of block.
            if self.row_s == usize::MAX && self.col_s == usize::MAX {
                self.row_s = 0;
                self.col_s = 0;
                if let Accelerator::Spada = self.accelerator {
                    self.adjust_block_row([self.row_s, self.col_s]);
                }
                // Get block stats.
                let token = self.block_token.tik();
                let a_cols_num = (0..self.block_shape[0])
                    .map(|offset| {
                        let ridx = self.row_s + offset;
                        let rlen = self.a_row_lens[ridx];
                        max(min(rlen, self.col_s + self.block_shape[1]), self.col_s) - self.col_s
                    })
                    .collect::<Vec<usize>>();
                let is_tail = (0..self.block_shape[0])
                    .map(|offset| {
                        let ridx = self.row_s + offset;
                        self.col_s + self.block_shape[1] >= self.a_row_lens[ridx]
                    })
                    .collect::<Vec<bool>>();
                // Config trackers.
                self.set_block(
                    token,
                    [self.row_s, self.col_s],
                    self.block_shape,
                    false,
                    a_cols_num,
                    is_tail,
                );
                // Move col_s to next position.
                self.col_s += self.block_shape[1];
                return Some(token);
            }
            // Return if finished.
            else if self.row_s >= self.a_row_num {
                return None;
            }
            // Prefer to allocate along K dim.
            else if !self.is_block_valid([self.row_s, self.col_s], self.block_shape) {
                // Adjust block across cols.
                self.adjust_block_col([self.row_s, self.col_s]);
                // Get block stats.
                let token = self.block_token.tik();
                let a_cols_num = (0..self.block_shape[0])
                    .map(|offset| {
                        let ridx = self.row_s + offset;
                        let rlen = self.a_row_lens[ridx];
                        max(min(rlen, self.col_s + self.block_shape[1]), self.col_s) - self.col_s
                    })
                    .collect::<Vec<usize>>();
                let is_tail = (0..self.block_shape[0])
                    .map(|offset| {
                        let ridx = self.row_s + offset;
                        self.col_s + self.block_shape[1] >= self.a_row_lens[ridx]
                    })
                    .collect::<Vec<bool>>();
                // Config trackers.
                self.set_block(
                    token,
                    [self.row_s, self.col_s],
                    self.block_shape,
                    false,
                    a_cols_num,
                    is_tail,
                );
                // Move col_s to next position.
                self.col_s += self.block_shape[1];
                return Some(token);
            } else {
                self.row_s += self.block_shape[0];
                // Adjust block across rows.
                if self.row_s < self.a_row_num {
                    self.col_s = self.a_cols_assigned[self.row_s];
                    self.adjust_block_row([self.row_s, self.col_s]);
                } else {
                    self.col_s = 0;
                }
            }
        }
    }

    pub fn merge_task(&mut self, cur_cycle: usize) -> Option<Task> {
        let mut psums = vec![];
        let mut pnum = 0;

        // If `lane_num / 2` pairs of psums are found, the a merge block is ready.
        for psum_addrs in self.output_tracker.values() {
            if pnum >= self.lane_num / 2 {
                break;
            }
            pnum += psum_addrs.len() / 2;
        }
        if (self.a_traversed && pnum == 0) || (!self.a_traversed && pnum < self.lane_num / 2) {
            return None;
        }

        for (row, psum_addrs) in self.output_tracker.iter_mut() {
            while psum_addrs.len() > 1 {
                if psums.len() == self.lane_num {
                    break;
                }
                for addr in psum_addrs.drain(..2) {
                    psums.push([*row, addr]);
                }
            }
        }

        let blk_token = self.block_token.tik();
        let win_token = self.window_token.tik();
        let a_cols_num = (0..self.lane_num / 2)
            .map(|r_ofst| if r_ofst < psums.len() / 2 { 2 } else { 0 })
            .collect();
        let mut arow_addr_pairs = vec![];
        let mut a_eles = vec![];
        let mut lane2idx = vec![];
        for r_ofst in 0..self.lane_num / 2 {
            if r_ofst < psums.len() / 2 {
                arow_addr_pairs.push([psums[r_ofst * 2][0], self.output_addr_token.tik()]);
                a_eles.extend(vec![
                    Some(Element::new(psums[r_ofst * 2], 1.0)),
                    Some(Element::new(psums[r_ofst * 2 + 1], 1.0)),
                ]);
                lane2idx.extend(vec![Some(psums[r_ofst * 2]), Some(psums[r_ofst * 2 + 1])]);
            } else {
                arow_addr_pairs.push([usize::MAX, self.output_addr_token.tik()]);
                // a_eles.push(None);
                a_eles.extend(vec![None; 2]);
                lane2idx.extend(vec![None; 2]);
            }
        }
        // Create merge task.
        let task = Task::new(blk_token, win_token, 2, true, a_eles, cur_cycle);
        //Register task in each row.
        for arow_addr in arow_addr_pairs.iter() {
            if arow_addr[0] != usize::MAX {
                self.row_rgstr_task
                    .entry(arow_addr[0])
                    .or_default()
                    .add_assign(1);
            }
        }
        // Config block tracker.
        self.block_tracker.insert(
            blk_token,
            BlockTracker::new(
                blk_token,
                [0, 0],
                [self.lane_num / 2, 2],
                true,
                a_cols_num,
                vec![false; self.lane_num / 2],
            ),
        );
        for r_ofst in 0..self.lane_num / 2 {
            if r_ofst < psums.len() / 2 {
                self.block_tracker
                    .get_mut(&blk_token)
                    .unwrap()
                    .a_cols_assigned[r_ofst] += 2;
            }
        }
        self.block_tracker
            .get_mut(&blk_token)
            .unwrap()
            .window_tokens
            .push(win_token);
        // Config window tracker.
        self.window_tracker.insert(
            win_token,
            WindowTracker::new(
                win_token,
                [0, 0],
                blk_token,
                [self.lane_num / 2, 2],
                lane2idx,
                arow_addr_pairs,
            ),
        );

        return Some(task);
    }

    pub fn next_window(
        &mut self,
        block_token: usize,
        a_matrix: &mut CsrMatStorage,
        cur_cycle: usize,
    ) -> Option<(usize, Task)> {
        let prev_window = self.block_tracker[&block_token]
            .window_tokens
            .last()
            .map(|x| *x);
        let window_shape: [usize; 2];
        let window_token: usize;
        let mut window_anchor: [usize; 2];
        let block_anchor: [usize; 2];
        let a_latency: usize;
        if prev_window.is_none() {
            window_shape = self.adjust_window(block_token);
            window_token = self.window_token.tik();
            window_anchor = self.block_tracker.get_mut(&block_token).unwrap().anchor;
            block_anchor = self.block_tracker.get_mut(&block_token).unwrap().anchor;
            a_latency = self.cache_latency;
        } else {
            let prev_window = prev_window.unwrap();
            let blk_tracker = self.block_tracker.get(&block_token).unwrap();
            let window = self.window_tracker.get(&prev_window).unwrap();
            window_token = self.window_token.tik();
            window_anchor = window.anchor;
            window_shape = window.shape;
            block_anchor = blk_tracker.anchor;
            let row_lim = blk_tracker.anchor[0] + blk_tracker.shape[0];
            let col_lim = blk_tracker.anchor[1]
                + min(
                    blk_tracker.shape[1],
                    *blk_tracker.a_cols_num.iter().max().unwrap(),
                );
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
                    if !self.is_window_valid(
                        blk_tracker.anchor,
                        blk_tracker.shape,
                        window_anchor,
                        window_shape,
                    ) {
                        break;
                    }
                }
                if window_anchor[0] >= row_lim {
                    return None;
                }
            }
            a_latency = 0;
        }
        let mut lane2idx = vec![];
        let mut a_eles = vec![];
        // let output_addrs = vec![self.output_addr_token.tik(); window_shape[0]];
        let output_addrs = (0..window_shape[0])
            .map(|r_offset| [window_anchor[0] + r_offset, self.output_addr_token.tik()])
            .collect::<Vec<[usize; 2]>>();
        for r_idx in window_anchor[0]..window_anchor[0] + window_shape[0] {
            let num = min(
                max(self.a_row_lens[r_idx], window_anchor[1]),
                window_anchor[1] + window_shape[1],
            ) - window_anchor[1];
            let element = a_matrix.read_scalars(r_idx, window_anchor[1], num).unwrap();
            let ele_len = element.len();
            // Increase assigned a col elements.
            let block_tracker = self.block_tracker.get_mut(&block_token).unwrap();
            block_tracker.a_cols_assigned[r_idx - block_anchor[0]] += ele_len;
            for mut e in element {
                lane2idx.push(Some(e.idx));
                e.idx = [window_token, e.idx[1]];
                a_eles.push(Some(e));
            }
            for _ in ele_len..window_shape[1] {
                lane2idx.push(None);
                a_eles.push(None);
            }
        }
        // Register task in each row.
        for arow_addr in output_addrs.iter() {
            self.row_rgstr_task
                .entry(arow_addr[0])
                .or_default()
                .add_assign(1);
        }
        // Config window tracker.
        self.window_tracker.insert(
            window_token,
            WindowTracker::new(
                window_token,
                window_anchor,
                block_token,
                window_shape,
                lane2idx,
                output_addrs,
            ),
        );
        // Config block tracker.
        self.block_tracker
            .get_mut(&block_token)
            .unwrap()
            .window_tokens
            .push(window_token);
        // Config task.
        let task = Task::new(
            block_token,
            window_token,
            window_shape[1],
            false,
            a_eles,
            cur_cycle,
        );
        return Some((a_latency, task));
    }

    pub fn is_block_valid(&self, block_anchor: [usize; 2], block_shape: [usize; 2]) -> bool {
        for rowid in block_anchor[0]..block_anchor[0] + block_shape[0] {
            if rowid >= self.a_row_num || block_anchor[1] >= self.a_row_lens[rowid] {
                continue;
            } else if block_anchor[1] < self.a_cols_assigned[rowid] {
                return true;
            } else {
                return false;
            }
        }

        return true;
    }

    pub fn is_window_valid(
        &self,
        block_anchor: [usize; 2],
        block_shape: [usize; 2],
        window_anchor: [usize; 2],
        window_shape: [usize; 2],
    ) -> bool {
        for rowid in window_anchor[0]
            ..min(
                window_anchor[0] + window_shape[0],
                block_anchor[0] + block_shape[0],
            )
        {
            if rowid >= self.a_row_num || window_anchor[1] >= self.a_row_lens[rowid] {
                continue;
            } else {
                return false;
            }
        }

        return true;
    }

    pub fn is_window_finished(&self, window_token: usize) -> bool {
        // trace_println!("**is_window_finished");
        let window_tracker = self.window_tracker.get(&window_token).unwrap();
        for r_offset in 0..window_tracker.shape[0] {
            for c_offset in 0..window_tracker.shape[1] {
                let lane_pos = r_offset * window_tracker.shape[1] + c_offset;
                match window_tracker.lane2idx[lane_pos] {
                    None => continue,
                    Some(idx) => {
                        let rlen = self.b_row_lens[&idx[1]];
                        // trace_println!("idx: {:?} b_col_asgn: {} rlen: {}", idx, window_tracker.b_cols_assigned[lane_pos], rlen);
                        if window_tracker.b_cols_assigned[lane_pos] < rlen {
                            return false;
                        }
                    }
                }
            }
        }

        return true;
    }

    pub fn adjust_block_row(&mut self, block_anchor: [usize; 2]) {
        match self.accelerator {
            Accelerator::Ip | Accelerator::MultiRow | Accelerator::Op => {
                while self.row_s + self.block_shape[0] > self.a_row_num {
                    self.block_shape[0] = max(1, self.block_shape[0] / 2);
                }
            }
            Accelerator::Spada => {
                self.block_shape = match self.adjust_scheme {
                    0 => self.rowwise_adjust_tracker.adjust_block_shape(
                        block_anchor,
                        self.row_s,
                        self.block_shape,
                        &self.block_topo_tracker,
                        &self.a_row_lens,
                    ),
                    1 => self
                        .colwise_reg_adjust_tracker
                        .adjust_block_shape(self.row_s, self.a_row_num),
                    2 => {
                        if block_anchor == [0; 2] {
                            self.colwise_irr_adjust_tracker.adjust_block_shape(
                                block_anchor,
                                self.a_row_num,
                                &&self.block_topo_tracker,
                            )
                        } else {
                            self.block_shape
                        }
                    }
                    3 => self
                        .rowwise_latency_adjust_tracker
                        .adjust_block_shape(self.row_s, self.block_shape),
                    _ => panic!("Invalid merge scheme: {}", self.adjust_scheme),
                }
            }
        }
    }

    pub fn adjust_block_col(&mut self, block_anchor: [usize; 2]) {
        match self.accelerator {
            Accelerator::Ip | Accelerator::MultiRow | Accelerator::Op => {
                while self.row_s + self.block_shape[0] > self.a_row_num {
                    self.block_shape[0] = max(1, self.block_shape[0] / 2);
                }
            }
            Accelerator::Spada => {
                self.block_shape = match self.adjust_scheme {
                    0 => self.block_shape,
                    1 => self.block_shape,
                    2 => self.colwise_irr_adjust_tracker.adjust_block_shape(
                        block_anchor,
                        self.a_row_num,
                        &&self.block_topo_tracker,
                    ),
                    3 => self.block_shape,
                    _ => panic!("Invalid merge scheme: {}", self.adjust_scheme),
                }
            }
        }
    }

    pub fn adjust_window(&mut self, block_token: usize) -> [usize; 2] {
        match self.accelerator {
            Accelerator::Ip | Accelerator::MultiRow | Accelerator::Op => {
                return [self.block_shape[0], self.lane_num / self.block_shape[0]];
            }
            Accelerator::Spada => match self.adjust_scheme {
                0 => self
                    .rowwise_adjust_tracker
                    .adjust_window_shape(self.block_tracker[&block_token].shape),
                1 => self.colwise_reg_adjust_tracker.adjust_window_shape(
                    block_token,
                    self.block_tracker[&block_token].anchor,
                    self.block_tracker[&block_token].shape,
                    &self.block_topo_tracker,
                ),
                2 => self
                    .colwise_irr_adjust_tracker
                    .adjust_window_shape(self.block_tracker[&block_token].shape),
                3 => self
                    .rowwise_latency_adjust_tracker
                    .adjust_window_shape(self.block_tracker[&block_token].shape),
                _ => panic!("Invalid adjust scheme: {}", self.adjust_scheme),
            },
        }
    }

    pub fn set_block(
        &mut self,
        token: usize,
        block_anchor: [usize; 2],
        block_shape: [usize; 2],
        is_merge_block: bool,
        a_cols_num: Vec<usize>,
        is_tail: Vec<bool>,
    ) {
        let a_ele_num = a_cols_num.iter().sum::<usize>();
        // Config scheduler a col assigned.
        for (offset, col_num) in a_cols_num.iter().enumerate() {
            let rowidx = block_anchor[0] + offset;
            self.a_cols_assigned[rowidx] += *col_num;
        }
        // Config block tracker.
        self.block_tracker.insert(
            token,
            BlockTracker::new(
                token,
                block_anchor,
                block_shape,
                is_merge_block,
                a_cols_num,
                is_tail,
            ),
        );
        // Config adjust tracker.
        self.rowwise_adjust_tracker
            .block_info
            .insert(token, RowwiseBlockInfo::new(a_ele_num));
        self.rowwise_latency_adjust_tracker
            .block_info
            .insert(token, RowwiseLatencyBlockInfo::new(a_ele_num));
        self.colwise_reg_adjust_tracker
            .block_info
            .insert(token, ColwiseRegBlockInfo::new(a_ele_num));
        self.colwise_irr_adjust_tracker
            .block_info
            .insert(token, ColwiseIrrBlockInfo::new(a_ele_num));
        self.rowwise_latency_adjust_tracker
            .block_info
            .insert(token, RowwiseLatencyBlockInfo::new(a_ele_num));
        // Config block topo tracker.
        self.block_topo_tracker.add_block(token, block_anchor);
    }

    pub fn assign_in_cache_merge_task(
        &mut self,
        adder_tree: &mut AdderTree,
        fiber_cache: &LatencyPriorityCache,
        cur_cycle: usize,
    ) -> Option<(usize, Task)> {
        // Prior to merge task.
        if let Some(task) = self.in_cache_merge_task(adder_tree.tree_width, fiber_cache, cur_cycle)
        {
            if adder_tree.task.is_some() && !adder_tree.task.as_ref().unwrap().merge_mode {
                self.staged_tasks[adder_tree.pe_idx] = mem::replace(&mut adder_tree.task, None);
            }
            return Some((0, task));
        } else {
            return None;
        }
    }

    pub fn in_cache_merge_task(
        &mut self,
        tree_width: usize,
        fiber_cache: &LatencyPriorityCache,
        cur_cycle: usize,
    ) -> Option<Task> {
        let mut psums = vec![];
        for (row, psum_addrs) in self.output_tracker.iter_mut() {
            if psum_addrs.len() >= tree_width
                && psum_addrs.iter().all(|addr| fiber_cache.contains_row(addr))
            {
                for addr in psum_addrs.drain(..min(tree_width, psum_addrs.len())) {
                    psums.push([*row, addr]);
                }
                break;
            }
        }
        if psums.len() == 0 {
            for frow in self.a_tail_produced.iter() {
                if !self.output_tracker.contains_key(frow) {
                    continue;
                }
                let frow_len = self.output_tracker[frow].len();
                if frow_len > 1
                    && self.output_tracker[frow]
                        .iter()
                        .all(|addr| fiber_cache.contains_row(addr))
                {
                    for addr in self
                        .output_tracker
                        .get_mut(frow)
                        .unwrap()
                        .drain(..min(tree_width, frow_len))
                    {
                        psums.push([*frow, addr]);
                    }
                    break;
                }
            }
        }
        if psums.len() == 0 {
            return None;
        }
        let blk_token = self.block_token.tik();
        let win_token = self.window_token.tik();
        let a_cols_num = vec![psums.len()];
        let arow_addr_pairs = vec![[psums[0][0], self.output_addr_token.tik()]];
        let mut a_eles = psums
            .iter()
            .map(|p_idx| Some(Element::new(*p_idx, 1.0)))
            .collect::<Vec<_>>();
        let mut lane2idx = psums.iter().map(|p_idx| Some(*p_idx)).collect::<Vec<_>>();
        for _ in a_eles.len()..tree_width {
            a_eles.push(None);
            lane2idx.push(None);
        }
        // Create merge task.
        let task = Task::new(blk_token, win_token, tree_width, true, a_eles, cur_cycle);
        // Register task in each row.
        for arow_addr in arow_addr_pairs.iter() {
            if arow_addr[0] != usize::MAX {
                self.row_rgstr_task
                    .entry(arow_addr[0])
                    .or_default()
                    .add_assign(1);
            }
        }
        // Config block tracker.
        self.block_tracker.insert(
            blk_token,
            BlockTracker::new(
                blk_token,
                [0, 0],
                [1, tree_width],
                true,
                a_cols_num,
                vec![false],
            ),
        );
        self.block_tracker
            .get_mut(&blk_token)
            .unwrap()
            .a_cols_assigned[0] += tree_width;
        self.block_tracker
            .get_mut(&blk_token)
            .unwrap()
            .window_tokens
            .push(win_token);
        self.window_tracker.insert(
            win_token,
            WindowTracker::new(
                win_token,
                [0, 0],
                blk_token,
                [1, tree_width],
                lane2idx,
                arow_addr_pairs,
            ),
        );
        return Some(task);
    }
}
