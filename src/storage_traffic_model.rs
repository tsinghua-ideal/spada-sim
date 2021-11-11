use std::{
    cmp::{max, min},
    collections::{HashMap, VecDeque},
    hash::Hash,
    ops::Range,
};

use itertools::{izip, merge, merge_join_by, Itertools, Merge, MergeJoinBy};
use storage::{LRUCache, LRURandomCache, PriorityCache, RandomCache, VectorStorage};

use crate::frontend::Accelerator;
use crate::trace_println;
use crate::util::gen_rands_from_range;
use crate::{
    print_type_of,
    storage::{self, CsrMatStorage, CsrRow, StorageAPI},
};

#[derive(Debug, Clone)]
struct PE {
    reduction_window: [usize; 2], // [width, height]
    cur_block: Block,
    merge_mode: bool,
    row_s: usize,
    col_s: usize,
}

impl PE {
    pub fn assign_block(&mut self, block: Block) {
        self.row_s = block.row_s;
        self.col_s = block.col_s;
        self.cur_block = block;
    }

    pub fn reset_pe(&mut self) {
        self.row_s = 0;
        self.col_s = 0;
        self.cur_block = Block::new(0, 0, 0, 0, false);
        self.reduction_window = [0, 0];
    }
}

#[derive(Debug, Clone)]
struct Block {
    pub width: usize,
    pub height: usize,
    pub row_s: usize,
    pub col_s: usize,
}

impl Block {
    pub fn new(width: usize, height: usize, row_s: usize, col_s: usize, is_tail: bool) -> Block {
        Block {
            width: width,
            height: height,
            row_s: row_s,
            col_s: col_s,
        }
    }

    pub fn get_idx(&self) -> [usize; 2] {
        [self.col_s, self.row_s]
    }

    pub fn get_shape(&self) -> [usize; 2] {
        [self.width, self.height]
    }
}

struct BlockTracker {
    pub row_s_list: Vec<usize>,
    pub col_s_list: Vec<Vec<usize>>,
}

impl BlockTracker {
    pub fn new() -> BlockTracker {
        BlockTracker {
            row_s_list: vec![],
            col_s_list: vec![],
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
}

#[derive(Debug, Clone)]
struct ExecTracker {
    pub block: [usize; 2],
    pub window: [usize; 2],
    pub touched_fiber_size: usize,
    pub dedup_fiber_size: usize,
    pub output_fiber_size: usize,
    pub miss_size: usize,
    pub psum_rw_size: [usize; 2],
}

impl ExecTracker {
    pub fn new(block_shape: [usize; 2], window_shape: [usize; 2]) -> ExecTracker {
        ExecTracker {
            block: block_shape,
            window: window_shape,
            touched_fiber_size: 0,
            dedup_fiber_size: 0,
            output_fiber_size: 0,
            miss_size: 0,
            psum_rw_size: [0, 0],
        }
    }

    pub fn c_reuse(&self) -> f64 {
        self.touched_fiber_size as f64
            / (self.output_fiber_size as f64 * self.window[0] as f64 + 0.00001)
    }

    pub fn b_reuse(&self) -> f64 {
        self.touched_fiber_size as f64
            / (self.dedup_fiber_size as f64 * self.window[1] as f64 + 0.00001)
    }
}

#[derive(Debug, Clone)]
struct MergeTracker {
    pub finished: bool,
    pub blocks: Vec<[usize; 2]>,
}

impl MergeTracker {
    pub fn new() -> MergeTracker {
        MergeTracker {
            finished: false,
            blocks: vec![],
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

pub struct TrafficModel<'a> {
    a_traversed: bool,
    reduction_window: [usize; 2],
    pe_num: usize,
    lane_num: usize,
    fiber_cache: PriorityCache<'a>,
    pes: Vec<PE>,
    a_mem: &'a mut CsrMatStorage,
    merge_queue: Vec<usize>,
    accelerator: Accelerator,
    block_shape: [usize; 2],
    block_topo: BlockTracker,
    /// Track the relative pos of blocks.
    exec_trackers: HashMap<[usize; 2], ExecTracker>,
    /// Track the execution of each block.
    output_base_addr: usize,
    output_trackers: HashMap<usize, Vec<usize>>,
    row_s: usize,
    col_s: usize,
    merge_trackers: HashMap<usize, MergeTracker>,
    exec_round: usize,
    /// Use each PE to do merge job in a round-robin way.
    merge_pe: usize,
    /// Use merge_period to control merge scheme 3.
    merge_counter: usize,
    merge_period: usize,
    b_sparsity: f32,
    a_group: GroupTracker,
    b_group: GroupTracker,
    row_group: usize,

    /// sampling list. Used by wide groups.
    sampling_bounds: Vec<usize>,
    set_row_num: usize,
}

impl<'a> TrafficModel<'a> {
    pub fn new(
        pe_num: usize,
        lane_num: usize,
        cache_size: usize,
        word_byte: usize,
        output_base_addr: usize,
        default_reduction_window: [usize; 2],
        default_block_shape: [usize; 2],
        a_mem: &'a mut CsrMatStorage,
        b_mem: &'a mut CsrMatStorage,
        psum_mem: &'a mut VectorStorage,
        accelerator: Accelerator,
    ) -> TrafficModel<'a> {
        // Preprocessing. Group each matrix's rows by their row lens.
        let var_factor = 1.5;
        let a_group = parse_group(&a_mem, var_factor);
        let b_group = parse_group(&b_mem, var_factor);

        trace_println!("-- a_group:");
        let a_rg = a_group
            .groups
            .iter()
            .map(|gi| gi.row_range)
            .collect::<Vec<[usize; 2]>>();
        trace_println!("{:?}", &a_rg);
        trace_println!("-- b_group:");
        let b_rg = b_group
            .groups
            .iter()
            .map(|gi| gi.row_range)
            .collect::<Vec<[usize; 2]>>();
        trace_println!("{:?}", &b_rg);

        // Init from the inner-product dataflow.
        // Can be changed to be adaptive.
        TrafficModel {
            b_sparsity: 1.0
                - b_mem.data.len() as f32 / (b_mem.row_num() * b_mem.mat_shape[0]) as f32,
            a_traversed: false,
            reduction_window: default_reduction_window.clone(),
            pe_num: pe_num,
            lane_num: lane_num,
            fiber_cache: PriorityCache::new(
                cache_size,
                word_byte,
                output_base_addr,
                b_mem,
                psum_mem,
            ),
            pes: vec![
                PE {
                    reduction_window: default_reduction_window.clone(),
                    cur_block: Block::new(0, 0, 0, 0, false),
                    merge_mode: false,
                    row_s: 0,
                    col_s: 0,
                };
                pe_num
            ],
            a_mem: a_mem,
            merge_queue: vec![],
            accelerator: accelerator,
            block_shape: default_block_shape.clone(),
            block_topo: BlockTracker::new(),
            exec_trackers: HashMap::new(),
            output_base_addr: output_base_addr,
            output_trackers: HashMap::new(),
            row_s: 0,
            col_s: 0,
            merge_trackers: HashMap::new(),
            exec_round: 0,
            merge_pe: 0,
            merge_counter: 0,
            merge_period: lane_num / default_block_shape[1],
            a_group,
            b_group,
            row_group: usize::MAX,
            sampling_bounds: vec![],
            set_row_num: usize::MAX,
        }
    }

    pub fn execute(&mut self) {
        // Reset the execution round counter.
        self.exec_round = 0;
        loop {
            trace_println!("----");
            trace_println!(
                "merge counter: {}, merge period: {}",
                self.merge_counter,
                self.merge_period
            );
            self.exec_round += 1;
            // Assign jobs to PEs. If no jobs can be assigned, end execution.
            if !self.assign_jobs() {
                break;
            }

            let prev_a_mem_read_count = self.a_mem.read_count;
            let prev_b_mem_read_count = self.fiber_cache.b_mem.read_count;
            let prev_psum_mem_read_count = self.fiber_cache.psum_mem.read_count;
            let prev_psum_mem_write_count = self.fiber_cache.psum_mem.write_count;
            let prev_miss_count = self.fiber_cache.miss_count;
            let prev_b_evict_count = self.fiber_cache.b_evict_count;
            let prev_psum_evict_count = self.fiber_cache.psum_evict_count;
            let prev_cache_read_count = self.fiber_cache.read_count;
            let prev_cache_write_count = self.fiber_cache.write_count;

            // Each PE execute a window.
            for i in 0..self.pe_num {
                let tmp_b_r = self.fiber_cache.b_mem.read_count;
                let tmp_psum_mem_rw =
                    self.fiber_cache.psum_mem.read_count + self.fiber_cache.psum_mem.write_count;
                let tmp_cache_rw = self.fiber_cache.read_count + self.fiber_cache.write_count;

                // Find if the pe is uninitialized.
                if !self.pes[i].merge_mode && (self.pes[i].reduction_window[0] == 0) {
                    continue;
                }
                // Fetch data from memory & cache.
                let (rowidxs, scaling_factors, fibers) = self.fetch_window_data(i);
                trace_println!(
                    "PE: {} scaling factors: {:?}",
                    i,
                    scaling_factors
                        .iter()
                        .map(|x| x.iter().map(|y| y.0).collect::<Vec<usize>>())
                        .collect::<Vec<Vec<usize>>>()
                );

                // Compute the window.
                let output_fibers = self.compute_a_window(&rowidxs, &scaling_factors, fibers);
                trace_println!(
                    "Compute: rows: {:?} cols: {}-{} merge_mode: {} output fiber size: {:?}",
                    &rowidxs,
                    self.pes[i].col_s,
                    self.pes[i].col_s + self.pes[i].reduction_window[0],
                    &self.pes[i].merge_mode,
                    output_fibers
                        .iter()
                        .map(|c| c.as_ref().map_or(0, |v| v.size()))
                        .collect::<Vec<usize>>()
                );
                if !self.pes[i].merge_mode {
                    trace_println!(
                        "Reuse: touched fiber size: {} deduped fiber size: {}, output size: {}",
                        self.exec_trackers[&self.pes[i].cur_block.get_idx()].touched_fiber_size,
                        self.exec_trackers[&self.pes[i].cur_block.get_idx()].dedup_fiber_size,
                        self.exec_trackers[&self.pes[i].cur_block.get_idx()].output_fiber_size
                    );
                }

                // Update reuse tracker if it is not in the merge mode.
                if !self.pes[i].merge_mode {
                    self.exec_trackers
                        .get_mut(&self.pes[i].cur_block.get_idx())
                        .unwrap()
                        .output_fiber_size += output_fibers
                        .iter()
                        .fold(0, |acc, x| acc + x.as_ref().map_or(0, |v| v.size()));
                }

                // Update work mode.
                let pe = &self.pes[i];
                if self.pes[i].merge_mode {
                    for row in rowidxs.iter() {
                        self.merge_queue.push(*row);
                    }
                } else if !self.pes[i].merge_mode && self.pes[i].cur_block.height != 0 {
                    // Finish one traverse over current rows.
                    // Add the finished rows into merge queue and turn into merge mode.
                    for (row_pos, row) in rowidxs.iter().enumerate() {
                        trace_println!("row: {}", row);

                        // // Merge scheme 1:
                        // if output_fibers[row_pos].is_some()
                        //     && !self.is_window_valid(
                        //         *row,
                        //         1,
                        //         self.pes[i].col_s + self.pes[i].reduction_window[0],
                        //         self.pes[i].cur_block.col_s,
                        //         self.pes[i].cur_block.width,
                        //     )
                        // {
                        //     let tracker = self.merge_trackers.get_mut(row).unwrap();
                        //     // Unregister current computed block from the merge tracker.
                        //     tracker.blocks.retain(|x| *x != self.pes[i].cur_block.get_idx());
                        //     // If all related blocks are computed, then start to merge all psums of
                        //     // the row.
                        //     if tracker.finished && tracker.blocks.len() == 0 {
                        //         self.merge_queue.push(*row);
                        //     }
                        // }

                        // // Merge scheme 2:
                        // // Every time a tile of a row is finished, start to merge the psums.
                        // if output_fibers[row_pos].is_some()
                        //     && !self.is_window_valid(
                        //         *row,
                        //         1,
                        //         self.pes[i].col_s + self.pes[i].reduction_window[0],
                        //         self.pes[i].cur_block.col_s,
                        //         self.pes[i].cur_block.width,
                        //     )
                        // {
                        //     let tracker = self.merge_trackers.get_mut(row).unwrap();
                        //     // Unregister current computed block from the merge tracker.
                        //     tracker.blocks.retain(|x| *x != self.pes[i].cur_block.get_idx());
                        //     self.merge_queue.push(*row);
                        // }

                        // Merge scheme 3:
                        // When reaches the merge period.
                        if output_fibers[row_pos].is_some()
                            && self.merge_counter == self.merge_period - 1
                        {
                            self.merge_queue.push(*row);
                        }

                        if output_fibers[row_pos].is_some()
                            && !self.is_window_valid(
                                *row,
                                1,
                                self.pes[i].col_s + self.pes[i].reduction_window[0],
                                self.pes[i].cur_block.col_s,
                                self.pes[i].cur_block.width,
                            )
                        {
                            let tracker = self.merge_trackers.get_mut(row).unwrap();
                            let cur_block_idx = self.pes[i].cur_block.get_idx();
                            // Unregister current computed block from the merge tracker.
                            tracker.blocks.retain(|x| *x != cur_block_idx);
                            self.merge_queue.push(*row);
                        }
                    }
                    self.merge_counter = (self.merge_counter + 1) % self.merge_period;
                }

                // Writeback psums.
                self.write_psum(rowidxs, output_fibers);

                // Calc exec info.
                let delta_b_mem = self.fiber_cache.b_mem.read_count - tmp_b_r;
                let delta_c_mem = self.fiber_cache.psum_mem.read_count
                    + self.fiber_cache.psum_mem.write_count
                    - tmp_psum_mem_rw;
                let delta_cache =
                    self.fiber_cache.read_count + self.fiber_cache.write_count - tmp_cache_rw;

                // Update exec_counter.
                let tracker = self
                    .exec_trackers
                    .get_mut(&self.pes[i].cur_block.get_idx())
                    .unwrap();
                tracker.miss_size += delta_b_mem;
                tracker.psum_rw_size[0] += delta_c_mem;
                tracker.psum_rw_size[1] += delta_cache;
            }

            trace_println!(
                "Cache read_count: + {} -> {}, write_count: + {} -> {}",
                self.fiber_cache.read_count - prev_cache_read_count,
                self.fiber_cache.read_count,
                self.fiber_cache.write_count - prev_cache_write_count,
                self.fiber_cache.write_count
            );
            trace_println!(
                "Cache occp: {} in {}, psum_occp: {}, b_occp: {}",
                self.fiber_cache.cur_num,
                self.fiber_cache.capability,
                self.fiber_cache.psum_occp,
                self.fiber_cache.b_occp
            );
            trace_println!("Cache miss_count: + {} -> {}, b_evict_count: + {} -> {}, psum_evict_count: + {} -> {}",
                self.fiber_cache.miss_count - prev_miss_count, self.fiber_cache.miss_count,
                self.fiber_cache.b_evict_count - prev_b_evict_count, self.fiber_cache.b_evict_count,
                self.fiber_cache.psum_evict_count - prev_psum_evict_count, self.fiber_cache.psum_evict_count);
            trace_println!(
                "A mem: read_count: + {} -> {}",
                self.a_mem.read_count - prev_a_mem_read_count,
                self.a_mem.read_count
            );
            trace_println!(
                "B mem: read_count: + {} -> {}",
                self.fiber_cache.b_mem.read_count - prev_b_mem_read_count,
                self.fiber_cache.b_mem.read_count
            );
            trace_println!(
                "C mem: read_count: + {} -> {}, write_count: +{} -> {}",
                self.fiber_cache.psum_mem.read_count - prev_psum_mem_read_count,
                self.fiber_cache.psum_mem.read_count,
                self.fiber_cache.psum_mem.write_count - prev_psum_mem_write_count,
                self.fiber_cache.psum_mem.write_count
            );
        }
    }

    fn assign_jobs(&mut self) -> bool {
        trace_println!("Merge queue: {:?}", &self.merge_queue);

        // Dedup merge queue & writeback merged fiber.
        let mut i = 0;
        let mut psums_num: usize = 0;
        self.merge_queue.sort();
        self.merge_queue.dedup();
        while i != self.merge_queue.len() {
            let rowid = self.merge_queue[i];
            let psum_addrs = self.output_trackers.get(&rowid).unwrap();
            if psum_addrs.len() == 1 {
                if self.merge_trackers[&rowid].finished
                    && self.merge_trackers[&rowid].blocks.len() == 0
                {
                    if self.fiber_cache.rowmap.contains_key(&psum_addrs[0]) {
                        trace_println!(
                            "Assign jobs: swapout addr {} of {} with size {}",
                            psum_addrs[0],
                            self.merge_queue[i],
                            self.fiber_cache.rowmap.get(&psum_addrs[0]).unwrap().size()
                        );
                    }
                    self.fiber_cache.swapout(psum_addrs[0]);
                }
                self.merge_queue.remove(i);
            } else {
                i += 1;
                psums_num += psum_addrs.len();
            }
        }

        trace_println!("Assign jobs: merge queue: {:?}", &self.merge_queue);

        // No job to assign if no multiplication and merge workloads.
        if self.a_traversed && self.pes.iter().all(|x| x.cur_block.height == 0) && psums_num == 0 {
            return false;
        }

        // Calculate the required merge psums number.
        let merge_pe_num = (psums_num + self.lane_num - 1) / self.lane_num;
        let mut alloc_merge_pe = min(merge_pe_num, self.pe_num);
        // Assign jobs to PEs.
        for offset in 0..self.pe_num {
            // Allocate PEs to merge the unmerged psums in prior.
            let pe_no = (offset + self.merge_pe) % self.pe_num;
            if alloc_merge_pe > 0 {
                trace_println!("PE {} turn into merge mode.", pe_no);
                self.pes[pe_no].merge_mode = true;
                alloc_merge_pe -= 1;
            } else {
                trace_println!("PE {}", pe_no);
                trace_println!(
                    "Current reduction window: {:?}",
                    self.pes[pe_no].reduction_window
                );
                self.pes[pe_no].merge_mode = false;
                // Try to shift the window in the block. Otherwise assign new block to PE.
                if !self.slide_window(pe_no) {
                    trace_println!("Failed to shift window.");
                    // Either empty or finished.
                    // Add block exec log to group.
                    if self.pes[pe_no].cur_block.height != 0 {
                        let blk_idx = self.pes[pe_no].cur_block.get_idx();
                        let grp_idx = self.a_group.rgmap[&self.pes[i].cur_block.row_s];
                        let row_num = self.pes[i].cur_block.height;
                        let cost = (self.exec_trackers[&blk_idx].miss_size
                            + self.exec_trackers[&blk_idx].psum_rw_size[0])
                            * 100
                            + self.exec_trackers[&blk_idx].psum_rw_size[1];
                        let ele_size = (blk_idx[1]
                            ..min(blk_idx[1] + row_num, self.a_mem.row_num()))
                            .fold(0, |ref s, ref x| *s + self.a_mem.get_ele_num(*x, *x + 1));
                        self.a_group.groups[grp_idx]
                            .cost_num
                            .entry(row_num)
                            .and_modify(|e| {
                                e[0] += cost;
                                e[1] += ele_size;
                            })
                            .or_insert([cost, ele_size]);
                    }
                    // Try to assign a new block.
                    match self.get_next_block() {
                        Some(block) => {
                            trace_println!("Assign block {:?} to {}", block.get_idx(), pe_no);
                            println!("row_s: {} block_shape: {:?}", self.row_s, self.block_shape);
                            let reduction_window =
                                self.adjust_window(block.get_idx(), block.get_shape());
                            self.pes[pe_no].assign_block(block);
                            self.pes[pe_no].reduction_window = reduction_window;
                            trace_println!(
                                "Adjust reduction window: {:?}",
                                self.pes[pe_no].reduction_window
                            );
                            // Slide window if the initial window is empty.
                            if !self.is_window_valid(
                                self.pes[pe_no].row_s,
                                self.pes[pe_no].reduction_window[1],
                                self.pes[pe_no].col_s,
                                self.pes[pe_no].cur_block.col_s,
                                self.pes[pe_no].cur_block.width,
                            ) {
                                self.slide_window(pe_no);
                            }

                            self.exec_trackers.insert(
                                self.pes[pe_no].cur_block.get_idx(),
                                ExecTracker::new(
                                    self.pes[pe_no].cur_block.get_shape(),
                                    self.pes[pe_no].reduction_window.clone(),
                                ),
                            );
                        }
                        None => {
                            self.pes[pe_no].reset_pe();
                            self.a_traversed = true;
                        }
                    }
                }
            }
        }

        self.merge_pe = (self.merge_pe + merge_pe_num) % self.pe_num;

        return true;
    }

    fn get_next_block(&mut self) -> Option<Block> {
        loop {
            // Return if finished.
            if self.row_s >= self.a_mem.row_num() {
                return None;
            }

            // Initial adjust of block.
            if self.row_group == usize::MAX {
                self.col_s = 0;
                self.adjust_block([self.col_s, self.row_s]);
            }
            // Try to allocate along K dim.
            else if self.is_block_valid(self.row_s, self.block_shape[1], self.col_s) {
                let block = Block {
                    width: self.block_shape[0],
                    height: self.block_shape[1],
                    row_s: self.row_s,
                    col_s: self.col_s,
                };
                if block.col_s == 0 {
                    self.block_topo.row_s_list.push(block.row_s);
                    self.block_topo.col_s_list.push(vec![]);
                }
                self.block_topo
                    .col_s_list
                    .last_mut()
                    .unwrap()
                    .push(block.col_s);
                self.col_s += self.block_shape[0];

                // Append the new block to the merge tracker.
                for rowid in block.row_s..block.row_s + block.height {
                    if self.is_col_s_valid(rowid, block.col_s) {
                        let row_finished = !self.is_col_s_valid(rowid, block.col_s + block.width);
                        let tracker = self
                            .merge_trackers
                            .entry(rowid)
                            .or_insert(MergeTracker::new());
                        // Register the block in the merge tracker.
                        tracker.blocks.push(block.get_idx());
                        // If the allocated block is the final block to be executed, then
                        // mark the row to be finished.
                        tracker.finished = row_finished;
                    }
                }
                // Adjust the merge period according to the block shape.
                self.merge_counter = 0;
                self.merge_period = self.lane_num / block.height;
                return Some(block);
            } else {
                // Block shape adaptation can be added here. For now we only support adjust block
                // when finishing traverse over K dim.
                self.row_s += self.block_shape[1];
                self.col_s = 0;
                if self.row_s < self.a_mem.row_num() {
                    self.adjust_block([self.col_s, self.row_s]);
                }
            }
        }
    }

    fn is_window_valid(
        &self,
        row_s: usize,
        row_num: usize,
        col_s: usize,
        b_col_s: usize,
        b_width: usize,
    ) -> bool {
        for rowid in row_s..row_s + row_num {
            if !self.is_col_s_valid(rowid, col_s)
                || (col_s < b_col_s)
                || (col_s >= b_col_s + b_width)
            {
                continue;
            } else {
                return true;
            }
        }

        return false;
    }

    fn is_block_valid(&self, row_s: usize, row_num: usize, col_s: usize) -> bool {
        for rowid in row_s..row_s + row_num {
            if !self.is_col_s_valid(rowid, col_s) {
                continue;
            } else {
                return true;
            }
        }

        return false;
    }

    fn is_col_s_valid(&self, rowid: usize, col_s: usize) -> bool {
        if (rowid >= self.a_mem.row_num()) || (self.a_mem.get_ele_num(rowid, rowid + 1) <= col_s) {
            return false;
        } else {
            return true;
        }
    }

    fn slide_window(&mut self, pe_no: usize) -> bool {
        // If no block has been assigned.
        if self.pes[pe_no].cur_block.height == 0 {
            return false;
        }

        // If the row_s exceeds the block limitation.
        if self.pes[pe_no].row_s
            >= self.pes[pe_no].cur_block.row_s + self.pes[pe_no].cur_block.height
        {
            return false;
        }
        // Try to allocate along K dim.
        if self.is_window_valid(
            self.pes[pe_no].row_s,
            self.pes[pe_no].reduction_window[1],
            self.pes[pe_no].col_s + self.pes[pe_no].reduction_window[0],
            self.pes[pe_no].cur_block.col_s,
            self.pes[pe_no].cur_block.width,
        ) {
            self.pes[pe_no].col_s += self.pes[pe_no].reduction_window[0];
        } else {
            self.pes[pe_no].col_s = self.pes[pe_no].cur_block.col_s;
            self.pes[pe_no].row_s += self.pes[pe_no].reduction_window[1];
            if self.pes[pe_no].row_s
                >= self.pes[pe_no].cur_block.row_s + self.pes[pe_no].cur_block.height
            {
                return false;
            }
            while !self.is_window_valid(
                self.pes[pe_no].row_s,
                self.pes[pe_no].reduction_window[1],
                self.pes[pe_no].col_s,
                self.pes[pe_no].cur_block.col_s,
                self.pes[pe_no].cur_block.width,
            ) {
                self.pes[pe_no].row_s += self.pes[pe_no].reduction_window[1];
                if self.pes[pe_no].row_s
                    >= self.pes[pe_no].cur_block.row_s + self.pes[pe_no].cur_block.height
                {
                    return false;
                }
            }
        }

        trace_println!(
            "PE {} shift to row_s {} col_s {}, block: row_s {} col_s {} height {} width {}",
            pe_no,
            self.pes[pe_no].row_s,
            self.pes[pe_no].col_s,
            self.pes[pe_no].cur_block.row_s,
            self.pes[pe_no].cur_block.col_s,
            self.pes[pe_no].cur_block.height,
            self.pes[pe_no].cur_block.width
        );
        true
    }

    /// Block shape adaptation can be added here.
    /// For now we only support adjust block when finishing traverse over K dim.
    fn adjust_block(&mut self, cur_idx: [usize; 2]) {
        match self.accelerator {
            Accelerator::Ip | Accelerator::MultiRow | Accelerator::Op => {
                // First check if the row group changed.
                if self.a_group.rgmap[&self.row_s] != self.row_group {
                    self.row_group = self.a_group.rgmap[&self.row_s];
                    return;
                }
            }
            Accelerator::NewOmega => {
                let block_adjust_scheme = 8;
                match block_adjust_scheme {
                    0 => {
                        // Scheme 0: Based on the reuse of the previous exeuction.
                        let neighbor_blocks = self.get_neighbor_blocks(&cur_idx);

                        // If no neighbor blocks, then use the default reduction window shape.
                        if neighbor_blocks.len() == 0 {
                            return;
                        }
                        // We look at the neighbor blocks and find the block with the largest total reuse.
                        let max_reuse_block = neighbor_blocks[neighbor_blocks
                            .iter()
                            .map(|x| {
                                self.exec_trackers[x].c_reuse() + self.exec_trackers[x].b_reuse()
                            })
                            .position_max_by(|a, b| a.partial_cmp(b).unwrap())
                            .unwrap()];

                        let cr = self.exec_trackers[&max_reuse_block].c_reuse();
                        let br = self.exec_trackers[&max_reuse_block].b_reuse();

                        if cr >= br {
                            if self.block_shape[1] > 1 {
                                self.block_shape[1] /= 2;
                            }
                        } else {
                            if self.block_shape[1] * 2 <= self.lane_num {
                                self.block_shape[1] *= 2;
                            }
                        }
                    }
                    1 => {
                        // Scheme 1: Based on the reuse of the above execution.
                        let above_block = self.block_topo.find_above(&cur_idx);

                        // If no neighbor blocks, then the block shape remains unchanged.
                        if above_block.is_none() {
                            return;
                        }
                        let above_block = above_block.unwrap();

                        let cr = self.exec_trackers[&above_block].c_reuse();
                        let br = self.exec_trackers[&above_block].b_reuse();

                        if cr >= br {
                            if self.block_shape[1] > 1 {
                                self.block_shape[1] /= 2;
                            }
                        } else {
                            if self.block_shape[1] * 2 <= self.lane_num {
                                self.block_shape[1] *= 2;
                            }
                        }
                    }
                    2 => {
                        // Scheme 2: Based on the reuse of the last two block execution.
                        let n1_block = self.block_topo.find_above(&cur_idx);
                        if n1_block.is_none() {
                            return;
                        }
                        let n1_block = n1_block.unwrap();
                        let n1_row_num = cur_idx[1] - n1_block[1];

                        let n2_block = self.block_topo.find_above(&n1_block);
                        if n2_block.is_none() {
                            return;
                        }
                        let n2_block = n2_block.unwrap();
                        let n2_row_num = n1_block[1] - n2_block[1];

                        if self.exec_trackers[&n1_block].b_reuse() as f32 / n1_row_num as f32
                            >= self.exec_trackers[&n2_block].b_reuse() as f32 / n2_row_num as f32
                        {
                            if n1_row_num >= n2_row_num {
                                self.block_shape[1] = min(self.block_shape[1] * 2, self.lane_num);
                            } else {
                                self.block_shape[1] = max(self.block_shape[1] / 2, 1);
                            }
                        } else {
                            if n1_row_num >= n2_row_num {
                                self.block_shape[1] = max(self.block_shape[1] / 2, 1);
                            } else {
                                self.block_shape[1] = min(self.block_shape[1] * 2, self.lane_num);
                            }
                        }
                    }
                    3 => {
                        // Scheme 3: Based on the cost of the last two block execution.
                        let n1_block = self.block_topo.find_above(&cur_idx);
                        if n1_block.is_none() {
                            return;
                        }
                        let n1_block = n1_block.unwrap();
                        let n1_row_num = cur_idx[1] - n1_block[1];
                        let n1_ele_size = (n1_block[1]..cur_idx[1])
                            .fold(0, |s, x| s + self.a_mem.get_ele_num(x, x + 1));

                        let n2_block = self.block_topo.find_above(&n1_block);
                        if n2_block.is_none() {
                            return;
                        }
                        let n2_block = n2_block.unwrap();
                        let n2_row_num = n1_block[1] - n2_block[1];
                        let n2_ele_size = (n2_block[1]..n1_block[1])
                            .fold(0, |s, x| s + self.a_mem.get_ele_num(x, x + 1));

                        let n1_cost = (self.exec_trackers[&n1_block].miss_size
                            + self.exec_trackers[&n1_block].psum_rw_size[0])
                            * 100
                            + self.exec_trackers[&n1_block].psum_rw_size[1];
                        let n2_cost = (self.exec_trackers[&n2_block].miss_size
                            + self.exec_trackers[&n2_block].psum_rw_size[0])
                            * 100
                            + self.exec_trackers[&n2_block].psum_rw_size[1];

                        trace_println!(
                            "n1_cost: {}, n1_ele_size: {}, n2_cost: {}, n2_ele_size: {}",
                            n1_cost,
                            n1_ele_size,
                            n2_cost,
                            n2_ele_size
                        );

                        if (n1_cost as f32 / n1_ele_size as f32)
                            <= (n2_cost as f32 / n2_ele_size as f32)
                        {
                            if n1_row_num >= n2_row_num {
                                self.block_shape[1] = min(self.block_shape[1] * 2, self.lane_num);
                            } else {
                                self.block_shape[1] = max(self.block_shape[1] / 2, 1);
                            }
                        } else {
                            if n1_row_num >= n2_row_num {
                                self.block_shape[1] = max(self.block_shape[1] / 2, 1);
                            } else {
                                self.block_shape[1] = min(self.block_shape[1] * 2, self.lane_num);
                            }
                        }
                    }
                    4 => {
                        // Scheme 4: Based on the cost of the last two block execution.
                        let n1_block = self.block_topo.find_above(&cur_idx);
                        if n1_block.is_none() {
                            return;
                        }
                        let n1_block = n1_block.unwrap();
                        let n1_row_num = cur_idx[1] - n1_block[1];
                        let n1_ele_size = (n1_block[1]..cur_idx[1])
                            .fold(0, |s, x| s + self.a_mem.get_ele_num(x, x + 1));

                        let n2_block = self.block_topo.find_above(&n1_block);
                        if n2_block.is_none() {
                            return;
                        }
                        let n2_block = n2_block.unwrap();
                        let n2_row_num = n1_block[1] - n2_block[1];
                        let n2_ele_size = (n2_block[1]..n1_block[1])
                            .fold(0, |s, x| s + self.a_mem.get_ele_num(x, x + 1));

                        let n1_cost = (self.exec_trackers[&n1_block].miss_size
                            + self.exec_trackers[&n1_block].psum_rw_size[0])
                            * 100
                            + self.exec_trackers[&n1_block].psum_rw_size[1];
                        let n2_cost = (self.exec_trackers[&n2_block].miss_size
                            + self.exec_trackers[&n2_block].psum_rw_size[0])
                            * 100
                            + self.exec_trackers[&n2_block].psum_rw_size[1];

                        let mut max_cachable_row = 0;
                        let mut exp_psum_size = 0.0;
                        let mut temp_idx = cur_idx[1];
                        while max_cachable_row <= self.lane_num - 1
                            && temp_idx < self.a_mem.row_num()
                        {
                            let row_num = self.a_mem.get_ele_num(temp_idx, temp_idx + 1);
                            let merged_psum_row = (1.0 - self.b_sparsity.powi(row_num as i32))
                                * self.fiber_cache.b_mem.mat_shape[0] as f32;
                            exp_psum_size += merged_psum_row * 2.0;
                            if exp_psum_size > self.fiber_cache.capability as f32 {
                                break;
                            }
                            max_cachable_row += 1;
                            temp_idx += 1;
                        }

                        max_cachable_row = max(1, max_cachable_row);

                        trace_println!(
                            "n1_cost: {}, n1_ele_size: {}, n2_cost: {}, n2_ele_size: {}, exp_psum_size: {}, max_cachable_row: {}",
                            n1_cost, n1_ele_size, n2_cost, n2_ele_size, exp_psum_size, max_cachable_row
                        );

                        if (n1_cost as f32 / n1_ele_size as f32)
                            <= (n2_cost as f32 / n2_ele_size as f32)
                        {
                            if n1_row_num >= n2_row_num {
                                self.block_shape[1] = min(self.block_shape[1] * 2, self.lane_num);
                            } else {
                                self.block_shape[1] = max(self.block_shape[1] / 2, 1);
                            }
                        } else {
                            if n1_row_num >= n2_row_num {
                                self.block_shape[1] = max(self.block_shape[1] / 2, 1);
                            } else {
                                self.block_shape[1] = min(self.block_shape[1] * 2, self.lane_num);
                            }
                        }

                        while self.block_shape[1] > max_cachable_row {
                            if self.block_shape[1] == 1 {
                                break;
                            }
                            self.block_shape[1] = max(self.block_shape[1] / 2, 1);
                        }
                    }
                    5 => {
                        // Scheme 5: Based on the row group.
                        // First check if the row group changed.
                        if self.a_group.rgmap[&self.row_s] != self.row_group {
                            // Start from row_num = 1.
                            self.block_shape[1] = 1;
                            self.row_group = self.a_group.rgmap[&self.row_s];
                            return;
                        }

                        // Then adjust based on the cost of different row num.
                        let mut min_row_num = 1;
                        let mut min_cost = f32::MAX;
                        let mut cur_row_num = 1;
                        while cur_row_num <= self.lane_num {
                            if let Some(cost_num) = self.a_group.groups[self.row_group]
                                .cost_num
                                .get_mut(&cur_row_num)
                            {
                                let div_cost = cost_num[0] as f32 / (cost_num[1] as f32 + 0.0001);
                                if div_cost < min_cost {
                                    min_cost = div_cost;
                                    min_row_num = cur_row_num;
                                }
                            } else {
                                self.a_group.groups[self.row_group]
                                    .cost_num
                                    .insert(cur_row_num, [0, 0]);
                                min_row_num = cur_row_num;
                                break;
                            }
                            cur_row_num *= 2;
                        }

                        // Avoid unecessary psum writeback by estimate the max cachable row num.
                        let mut max_cachable_row = 0;
                        let mut exp_psum_size = 0.0;
                        let mut temp_idx = cur_idx[1];
                        while max_cachable_row <= self.lane_num - 1
                            && temp_idx < self.a_mem.row_num()
                        {
                            // trace_print!("rgmap: {:?}, rowptr: {}", self.b_group.rgmap.keys(), self.a_mem.rowptr(temp_idx+1));

                            if self.a_mem.get_ele_num(temp_idx, temp_idx + 1) == 0 {
                                max_cachable_row += 1;
                                temp_idx += 1;
                                continue;
                            }
                            let idx_s = self.a_mem.rowptr(temp_idx);
                            // let idx_t = self.a_mem.rowptr(temp_idx+1);
                            let idx_t = idx_s + self.a_mem.get_ele_num(temp_idx, temp_idx + 1);
                            let brow_s = self.a_mem.indices[idx_s];
                            let brow_t = self.a_mem.indices[idx_t - 1];
                            let bg_s = self.b_group.rgmap[&brow_s];
                            let bg_t = self.b_group.rgmap[&brow_t];
                            let b_width = self.fiber_cache.b_mem.mat_shape[0];
                            let idx_per_g = (idx_t - idx_s) / (bg_t - bg_s + 1);
                            let rm_idxs_num = idx_t - idx_s - (idx_per_g * (bg_t - bg_s + 1));
                            let mut group_row_num: Vec<usize> = vec![idx_per_g; bg_t - bg_s + 1];
                            for idx in gen_rands_from_range(0, bg_t - bg_s + 1, rm_idxs_num) {
                                group_row_num[idx] += 1;
                            }

                            let product_sparsity = self.b_group.groups[bg_s..bg_t + 1]
                                .iter()
                                .zip(group_row_num.iter())
                                .fold(1.0, |p, x| {
                                    p * ((b_width - x.0.avg_row_len) as f32 / b_width as f32)
                                        .powi(*x.1 as i32)
                                });
                            let merged_psum_row = (1.0 - product_sparsity) * b_width as f32;
                            exp_psum_size += merged_psum_row * 2.0;
                            trace_println!(
                                "brow_s {} brow_t {} merged_psum_row {}",
                                brow_s,
                                brow_t,
                                merged_psum_row
                            );
                            if exp_psum_size * 2.0 > self.fiber_cache.capability as f32 {
                                break;
                            }
                            max_cachable_row += 1;
                            temp_idx += 1;
                        }

                        max_cachable_row = max(1, max_cachable_row);

                        trace_println!(
                            "cost num: {:?}, min_row_num: {}, exp_psum_size: {}, max_cachable_row: {}",
                            self.a_group.groups[self.row_group].cost_num,
                            min_row_num,
                            exp_psum_size,
                            max_cachable_row
                        );

                        while min_row_num > max_cachable_row {
                            if min_row_num == 1 {
                                break;
                            }
                            min_row_num = max(min_row_num / 2, 1);
                        }

                        self.block_shape[1] = min_row_num;
                    }
                    6 => {
                        // First check if the row group changed.
                        if self.a_group.rgmap[&self.row_s] != self.row_group {
                            // Start from row_num = 1.
                            self.block_shape[1] = 1;
                            self.row_group = self.a_group.rgmap[&self.row_s];
                            return;
                        }

                        let n1_block = self.block_topo.find_above(&cur_idx);
                        if n1_block.is_none() {
                            return;
                        }
                        let n1_block = n1_block.unwrap();
                        let n1_row_num = cur_idx[1] - n1_block[1];
                        let n1_ele_size = (n1_block[1]..cur_idx[1])
                            .fold(0, |s, x| s + self.a_mem.get_ele_num(x, x + 1));

                        let n2_block = self.block_topo.find_above(&n1_block);
                        if n2_block.is_none() {
                            return;
                        }
                        let n2_block = n2_block.unwrap();
                        let n2_row_num = n1_block[1] - n2_block[1];
                        let n2_ele_size = (n2_block[1]..n1_block[1])
                            .fold(0, |s, x| s + self.a_mem.get_ele_num(x, x + 1));

                        let n1_cost = (self.exec_trackers[&n1_block].miss_size
                            + self.exec_trackers[&n1_block].psum_rw_size[0])
                            * 100
                            + self.exec_trackers[&n1_block].psum_rw_size[1];
                        let n2_cost = (self.exec_trackers[&n2_block].miss_size
                            + self.exec_trackers[&n2_block].psum_rw_size[0])
                            * 100
                            + self.exec_trackers[&n2_block].psum_rw_size[1];

                        // Avoid unecessary psum writeback by estimate the max cachable row num.
                        let mut max_cachable_row = 0;
                        let mut exp_psum_size = 0.0;
                        let mut temp_idx = cur_idx[1];
                        while max_cachable_row <= self.lane_num - 1
                            && temp_idx < self.a_mem.row_num()
                        {
                            // trace_print!("rgmap: {:?}, rowptr: {}", self.b_group.rgmap.keys(), self.a_mem.rowptr(temp_idx+1));
                            let idx_s = self.a_mem.rowptr(temp_idx);
                            // let idx_t = self.a_mem.rowptr(temp_idx+1);
                            let idx_t = idx_s + self.a_mem.get_ele_num(temp_idx, temp_idx + 1);
                            let brow_s = self.a_mem.indices[idx_s];
                            let brow_t = self.a_mem.indices[idx_t - 1];
                            let bg_s = self.b_group.rgmap[&brow_s];
                            let bg_t = self.b_group.rgmap[&brow_t];
                            let b_width = self.fiber_cache.b_mem.mat_shape[0];
                            let product_sparsity = self.b_group.groups[bg_s..bg_t + 1]
                                .iter()
                                .fold(1.0, |p, x| p * x.avg_row_len as f32 / b_width as f32);
                            let merged_psum_row = (1.0 - product_sparsity) * b_width as f32;
                            // Include index & value.
                            exp_psum_size += merged_psum_row * 2.0;
                            // if exp_psum_size > self.fiber_cache.capability as f32 {

                            // Consider two psum on chip may need to merge.
                            if exp_psum_size * 2.0 > self.fiber_cache.capability as f32 {
                                break;
                            }
                            max_cachable_row += 1;
                            temp_idx += 1;
                        }

                        max_cachable_row = max(1, max_cachable_row);

                        trace_println!(
                            "n1_cost: {}, n1_ele_size: {}, n2_cost: {}, n2_ele_size: {}, exp_psum_size: {}, max_cachable_row: {}",
                            n1_cost, n1_ele_size, n2_cost, n2_ele_size, exp_psum_size, max_cachable_row
                        );

                        if (n1_cost as f32 / n1_ele_size as f32)
                            <= (n2_cost as f32 / n2_ele_size as f32)
                        {
                            if n1_row_num >= n2_row_num {
                                self.block_shape[1] = min(self.block_shape[1] * 2, self.lane_num);
                            } else {
                                self.block_shape[1] = max(self.block_shape[1] / 2, 1);
                            }
                        } else {
                            if n1_row_num >= n2_row_num {
                                self.block_shape[1] = max(self.block_shape[1] / 2, 1);
                            } else {
                                self.block_shape[1] = min(self.block_shape[1] * 2, self.lane_num);
                            }
                        }

                        while self.block_shape[1] > max_cachable_row {
                            if self.block_shape[1] == 1 {
                                break;
                            }
                            self.block_shape[1] = max(self.block_shape[1] / 2, 1);
                        }
                    }
                    7 => {
                        // First check if the row group changed.
                        if self.a_group.rgmap[&self.row_s] != self.row_group {
                            // Start from row_num = 1 to touch the distribution.
                            self.block_shape[1] = 1;
                            self.row_group = self.a_group.rgmap[&self.row_s];
                            return;
                        }

                        // After touching the distribution, clear the exec log to get rid of the
                        // initial thrashing.
                        if self.row_s == self.a_group.groups[self.row_group].row_range[0] + 1 {
                            self.a_group.groups[self.row_group].cost_num.remove(&1);
                        }

                        // Then adjust based on the cost of different row num.
                        let mut min_row_num = 1;
                        let mut min_cost = f32::MAX;
                        let mut cur_row_num = 1;

                        trace_println!(
                            "cost num: {:?}",
                            self.a_group.groups[self.row_group].cost_num
                        );

                        while cur_row_num <= self.lane_num {
                            if let Some(cost_num) = self.a_group.groups[self.row_group]
                                .cost_num
                                .get_mut(&cur_row_num)
                            {
                                let div_cost = cost_num[0] as f32 / (cost_num[1] as f32 + 0.0001);
                                if div_cost < min_cost {
                                    min_cost = div_cost;
                                    min_row_num = cur_row_num;
                                }
                            } else {
                                self.a_group.groups[self.row_group]
                                    .cost_num
                                    .insert(cur_row_num, [0, 0]);
                                min_row_num = cur_row_num;
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

                        self.block_shape[1] = min_row_num;
                    }
                    8 => {
                        trace_println!("-Adjust block");
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
                                trace_println!("---Sampling");
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
                            trace_println!(
                                "group_range {:?} cost num: {:?}",
                                &self.a_group.groups[self.row_group].row_range,
                                self.a_group.groups[self.row_group].cost_num
                            );
                            self.block_shape[1] = min_row_num;
                        } else {
                            // Treat the narrow groups.
                            let n1_block = self.block_topo.find_above(&cur_idx);
                            if n1_block.is_none() {
                                return;
                            }
                            let n1_block = n1_block.unwrap();
                            let n1_row_num = cur_idx[1] - n1_block[1];
                            let n1_ele_size = (n1_block[1]..cur_idx[1])
                                .fold(0, |s, x| s + self.a_mem.get_ele_num(x, x + 1));

                            let n2_block = self.block_topo.find_above(&n1_block);
                            if n2_block.is_none() {
                                return;
                            }
                            let n2_block = n2_block.unwrap();
                            let n2_row_num = n1_block[1] - n2_block[1];
                            let n2_ele_size = (n2_block[1]..n1_block[1])
                                .fold(0, |s, x| s + self.a_mem.get_ele_num(x, x + 1));

                            let n1_cost = (self.exec_trackers[&n1_block].miss_size
                                + self.exec_trackers[&n1_block].psum_rw_size[0])
                                * 100
                                + self.exec_trackers[&n1_block].psum_rw_size[1];
                            let n2_cost = (self.exec_trackers[&n2_block].miss_size
                                + self.exec_trackers[&n2_block].psum_rw_size[0])
                                * 100
                                + self.exec_trackers[&n2_block].psum_rw_size[1];

                            trace_println!(
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
                    _ => panic!("Invalid merge scheme:{}", block_adjust_scheme),
                }
            }
        }
    }

    /// Adjust the reduction window for the current block.
    fn adjust_window(&mut self, cur_idx: [usize; 2], block_shape: [usize; 2]) -> [usize; 2] {
        // The new omega scheme.
        if self.accelerator == Accelerator::NewOmega {
            return [self.lane_num / self.block_shape[1], self.block_shape[1]];
        }

        match self.accelerator {
            Accelerator::NewOmega => [self.lane_num / self.block_shape[1], self.block_shape[1]],
            Accelerator::Ip | Accelerator::MultiRow | Accelerator::Op => {
                let mut reduction_window: [usize; 2];
                // // window adjust scheme 0 that reduction window is decoupled from block.
                // let neighbor_blocks = self.get_neighbor_blocks(&cur_idx);

                // // If no neighbor blocks, then use the default reduction window shape.
                // if neighbor_blocks.len() == 0 {
                //     return [self.lane_num, 1];
                // }
                // // We look at the neighbor blocks and find the block with the largest total reuse.
                // let max_reuse_block = neighbor_blocks[neighbor_blocks
                //     .iter()
                //     .map(|x| self.exec_trackers[x].c_reuse() + self.exec_trackers[x].b_reuse())
                //     .position_max_by(|a, b| a.partial_cmp(b).unwrap())
                //     .unwrap()];

                // let cr = self.exec_trackers[&max_reuse_block].c_reuse();
                // let br = self.exec_trackers[&max_reuse_block].b_reuse();
                // reduction_window = self.exec_trackers[&max_reuse_block].window;

                // if cr >= br {
                //     if reduction_window[1] > 1 && reduction_window[0] * 2 <= block_shape[0] {
                //         reduction_window[1] /= 2;
                //         reduction_window[0] *= 2;
                //     }
                // } else {
                //     if reduction_window[0] > 1 && reduction_window[1] * 2 <= block_shape[1] {
                //         reduction_window[0] /= 2;
                //         reduction_window[1] *= 2;
                //     }
                // }

                // // window adjust scheme 1 that pin the window.
                // reduction_window = [8, 1];

                // window adjust scheme 2 that coupled with block.
                reduction_window = [self.lane_num / self.block_shape[1], self.block_shape[1]];

                reduction_window
            }
        }
    }

    /// The neighbor blocks can be defined here.
    /// Currently we use the left & above block as neighbor blocks, if possible.
    fn get_neighbor_blocks(&mut self, cur_idx: &[usize; 2]) -> Vec<[usize; 2]> {
        let mut blocks = vec![];
        if let Some(left) = self.block_topo.find_left(cur_idx) {
            blocks.push(left);
        }
        if let Some(above) = self.block_topo.find_above(cur_idx) {
            blocks.push(above);
        }

        blocks
    }

    /// Fetch data in the window from the cache & memory.
    fn fetch_window_data(
        &mut self,
        pe_no: usize,
    ) -> (Vec<usize>, Vec<Vec<(usize, f64)>>, Vec<Vec<CsrRow>>) {
        let pe = &self.pes[pe_no];
        let mut scaling_factors = vec![];
        let mut fibers = vec![];
        let mut rowidxs = vec![];

        if pe.merge_mode {
            let mut unused_lane_num = self.lane_num;
            while unused_lane_num > 0 && self.merge_queue.len() > 0 {
                let rowidx = self.merge_queue.first().unwrap();
                let psums = self.output_trackers.get_mut(rowidx).unwrap();
                let used_num = min(psums.len(), unused_lane_num);
                let mut fbs = vec![];
                let mut sfs = vec![];
                for colid in psums.drain(0..used_num) {
                    let csrrow = self.fiber_cache.consume(colid).unwrap();
                    fbs.push(csrrow);
                    sfs.push((colid, 1f64));
                }
                scaling_factors.push(sfs);
                fibers.push(fbs);
                rowidxs.push(*rowidx);

                if psums.len() == 0 {
                    self.merge_queue.remove(0);
                }
                unused_lane_num -= used_num;
            }
        } else {
            rowidxs = (pe.row_s..min(pe.row_s + pe.reduction_window[1], self.a_mem.row_num()))
                .filter(|x| self.a_mem.get_ele_num(*x, *x + 1) as i32 >= 0)
                .collect();
            let mut broadcast_cache: HashMap<usize, CsrRow> = HashMap::new();
            for rowidx in rowidxs.iter() {
                let mut r_sfs = CsrRow::new(*rowidx);
                if self.a_mem.get_ele_num(*rowidx, *rowidx + 1) > pe.col_s {
                    let ele_num = min(
                        pe.reduction_window[0],
                        self.a_mem.get_ele_num(*rowidx, *rowidx + 1) - pe.col_s,
                    );
                    r_sfs = self.a_mem.read(*rowidx, pe.col_s, ele_num).unwrap();
                }
                let mut fbs = vec![];
                let mut sfs = vec![];
                for (colid, value) in r_sfs.enumerate() {
                    if broadcast_cache.contains_key(colid) {
                        let csrrow = broadcast_cache[colid].clone();
                        fbs.push(csrrow);
                        sfs.push((*colid, *value));
                    } else {
                        let missed = !self.fiber_cache.rowmap.contains_key(colid);
                        match self.fiber_cache.read([*rowidx, *colid]) {
                            Some(csrrow) => {
                                broadcast_cache.insert(*colid, csrrow.clone());
                                fbs.push(csrrow);
                                sfs.push((*colid, *value));
                            }
                            None => (),
                        }
                    }
                }
                scaling_factors.push(sfs);
                fibers.push(fbs);
            }
            // Update reuse tracker data.
            // trace_print!("Fetch row data: previous touched: {}, dedup: {}", self.reuse_trackers[pe_no].touched_fiber_size, self.reuse_trackers[pe_no].dedup_fiber_size);
            self.exec_trackers
                .get_mut(&pe.cur_block.get_idx())
                .unwrap()
                .touched_fiber_size += fibers.iter().flatten().fold(0, |acc, x| acc + x.size());
            self.exec_trackers
                .get_mut(&pe.cur_block.get_idx())
                .unwrap()
                .dedup_fiber_size += fibers
                .iter()
                .flatten()
                .sorted_by(|a, b| Ord::cmp(&a.rowptr, &b.rowptr))
                .dedup_by(|x, y| x.rowptr == y.rowptr)
                .fold(0, |acc, x| acc + x.size());
            // trace_print!("Fetch row data: current touched: {}, dedup: {}", self.reuse_trackers[pe_no].touched_fiber_size, self.reuse_trackers[pe_no].dedup_fiber_size)
        }

        return (rowidxs, scaling_factors, fibers);
    }

    fn compute_a_window(
        &self,
        rowidxs: &Vec<usize>,
        scaling_factors: &Vec<Vec<(usize, f64)>>,
        fibers: Vec<Vec<CsrRow>>,
    ) -> Vec<Option<CsrRow>> {
        let mut psums = vec![];
        for (rowidx, sfs, fbs) in izip!(rowidxs, scaling_factors, fibers) {
            // Compute psum.
            if sfs.len() == 0 {
                psums.push(None);
                continue;
            }
            let mut psum = CsrRow::new(*rowidx);
            for (sf, fb) in izip!(sfs, fbs) {
                for (colid, value) in izip!(fb.indptr, fb.data) {
                    match psum.indptr.binary_search(&colid) {
                        Ok(pos) => psum.data[pos] += sf.1 * value,
                        Err(pos) => {
                            psum.data.insert(pos, sf.1 * value);
                            psum.indptr.insert(pos, colid);
                        }
                    }
                }
            }
            psums.push(Some(psum));
        }

        psums
    }

    fn write_psum(&mut self, rowidxs: Vec<usize>, output_fibers: Vec<Option<CsrRow>>) {
        for (rowidx, output_fiber) in rowidxs
            .into_iter()
            .zip(output_fibers.into_iter())
            .filter(|(_, y)| y.is_some())
        {
            self.output_trackers
                .entry(rowidx)
                .or_default()
                .push(self.output_base_addr);
            trace_println!("write_psum: {:?}", self.output_trackers[&rowidx]);
            let mut output_fiber = output_fiber.unwrap();
            output_fiber.rowptr = self.output_base_addr;
            self.fiber_cache
                .write(output_fiber, [self.output_base_addr, self.output_base_addr]);
            self.output_base_addr += 1;
        }
    }

    pub fn get_exec_result(&mut self) -> Vec<CsrRow> {
        let mut c = vec![];
        for rowid in 0..self.a_mem.row_num() {
            let mut csrrow = CsrRow::new(rowid);
            // if self.a_mem.rowptr(rowid+1) - self.a_mem.rowptr(rowid) > 0 {
            if self.a_mem.get_ele_num(rowid, rowid + 1) > 0 {
                let raw_rowid = if self.a_mem.remapped {
                    self.a_mem.row_remap[&rowid]
                } else {
                    rowid
                };
                // let raw_rowid = self.a_mem.row_remap[&rowid];
                let addrs = self.output_trackers.get(&rowid).unwrap();
                trace_println!(
                    "Get result: row: {} row len: {}",
                    raw_rowid,
                    self.fiber_cache.psum_mem.data[&addrs[0]].size() / 2
                );
                assert!(
                    addrs.len() == 1,
                    "Partially merged psums! {:?} of row {}",
                    &addrs,
                    raw_rowid
                );
                let addr = addrs[0];
                csrrow = match self.fiber_cache.psum_mem.data.get(&addr) {
                    Some(row) => row.clone(),
                    None => self.fiber_cache.rowmap.get(&addr).unwrap().clone(),
                };
                csrrow.rowptr = raw_rowid;
            }
            c.push(csrrow);
        }
        c.sort_by(|a, b| a.rowptr.cmp(&b.rowptr));
        return c;
    }

    pub fn get_a_mat_stat(&self) -> (usize, usize) {
        (self.a_mem.read_count, self.a_mem.write_count)
    }

    pub fn get_b_mat_stat(&self) -> (usize, usize) {
        (
            self.fiber_cache.b_mem.read_count,
            self.fiber_cache.b_mem.write_count,
        )
    }

    pub fn get_c_mat_stat(&self) -> (usize, usize) {
        (
            self.fiber_cache.psum_mem.read_count,
            self.fiber_cache.psum_mem.write_count,
        )
    }

    pub fn get_exec_round(&self) -> usize {
        self.exec_round
    }

    pub fn get_cache_stat(&self) -> (usize, usize) {
        (self.fiber_cache.read_count, self.fiber_cache.write_count)
    }
}
