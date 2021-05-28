use std::{cmp::{max, min}, collections::{HashMap, VecDeque}, hash::Hash, ops::Range};

use itertools::{Itertools, Merge, izip, merge, merge_join_by};
use storage::{LRUCache, VectorStorage};

use crate::{print_type_of, storage::{self, CsrMatStorage, CsrRow, StorageAPI, StorageError}};
use crate::frontend::Accelerator;

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
        self.reduction_window= [0, 0];
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
        let row_pos = self.row_s_list.binary_search(&cur_block[1]).unwrap();
        let col_pos = match self.col_s_list[row_pos].binary_search(&cur_block[0]) {
            Ok(c) | Err(c) => c as i32 - 1
        };

        if col_pos < 0 {
            return None;
        } else {
            return Some([self.col_s_list[row_pos][col_pos as usize], cur_block[1]]);
        }
    }

    pub fn find_above(&self, cur_block: &[usize; 2]) -> Option<[usize; 2]> {
        let row_pos = match self.row_s_list.binary_search(&cur_block[1]) {
            Ok(r) | Err(r) => r as i32 - 1
        };

        if row_pos < 0 || self.col_s_list[row_pos as usize].len() == 0{
            return None;
        }

        let row_pos = row_pos as usize;

        match self.col_s_list[row_pos].binary_search(&cur_block[0]) {
            Ok(c) => Some([self.col_s_list[row_pos][c], self.row_s_list[row_pos]]),
            Err(c) => {
                let c_l = max(c - 1, 0);
                let c_r = min(c+1, self.col_s_list[row_pos].len()-1);
                if (cur_block[0] as i64 - self.col_s_list[row_pos][c_l] as i64).abs() >=
                    (self.col_s_list[row_pos][c_r] as i64 - cur_block[0] as i64).abs() {
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
}

impl ExecTracker {
    pub fn new(block_shape: [usize; 2], window_shape: [usize; 2]) -> ExecTracker {
        ExecTracker {
            block: block_shape,
            window: window_shape,
            touched_fiber_size: 0,
            dedup_fiber_size: 0,
            output_fiber_size: 0,
        }
    }

    pub fn c_reuse(&self) -> f64 {
        self.touched_fiber_size as f64 / (self.output_fiber_size as f64 *
            self.window[0] as f64 + 0.00001)
    }

    pub fn b_reuse(&self) -> f64 {
        self.touched_fiber_size as f64 / (self.dedup_fiber_size as f64 *
            self.window[1] as f64 + 0.00001)
    }
}

pub struct TrafficModel<'a> {
    a_traversed: bool,
    reduction_window: [usize; 2],
    pe_num: usize,
    lane_num: usize,
    fiber_cache: LRUCache<'a>,
    pes: Vec<PE>,
    a_mem: &'a mut CsrMatStorage,
    merge_queue: Vec<usize>,
    accelerator: Accelerator,
    block_shape: [usize; 2],
    block_topo: BlockTracker, /// Track the relative pos of blocks.
    exec_trackers: HashMap<[usize; 2], ExecTracker>, /// Track the execution of each block.
    output_base_addr: usize,
    output_tracker: HashMap<usize, Vec<usize>>,
    row_s: usize,
    col_s: usize,
}

impl<'a> TrafficModel<'a> {
    pub fn new(
        pe_num: usize,
        lane_num: usize,
        cache_size: usize,
        word_byte: usize,
        output_base_addr: usize,
        a_mem: &'a mut CsrMatStorage,
        b_mem: &'a mut CsrMatStorage,
        psum_mem: &'a mut VectorStorage,
        accelerator: Accelerator,
    ) -> TrafficModel<'a> {
        // Init from the inner-product dataflow.
        // Can be changed to be adaptive.
        let reduction_window = match accelerator {
            Accelerator::Ip | Accelerator::Omega => [lane_num, 1],
            Accelerator::Op => [1, lane_num],
        };

        let block_shape = match accelerator {
            Accelerator::Ip | Accelerator::Omega => [a_mem.indices.len() / (a_mem.indptr.len() - 1) / 2, 1],
            Accelerator::Op => [1, usize::MAX],
        };

        TrafficModel {
            a_traversed: false,
            reduction_window: reduction_window,
            pe_num: pe_num,
            lane_num: lane_num,
            fiber_cache: LRUCache::new(cache_size, word_byte, output_base_addr, b_mem, psum_mem),
            pes: vec![
                PE {
                    reduction_window: [lane_num, 1],
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
            block_shape: block_shape,
            block_topo: BlockTracker::new(),
            exec_trackers: HashMap::new(),
            output_base_addr: output_base_addr,
            output_tracker: HashMap::new(),
            row_s: 0,
            col_s: 0,
        }
    }

    pub fn execute(&mut self) {
        loop {
            println!("----");

            // Assign jobs to PEs. If no jobs can be assigned, end execution.
            if !self.assign_jobs() { break; }

            let prev_a_mem_read_count = self.a_mem.read_count;
            let prev_b_mem_read_count = self.fiber_cache.b_mem.read_count;
            let prev_psum_mem_read_count = self.fiber_cache.psum_mem.read_count;
            let prev_psum_mem_write_count = self.fiber_cache.psum_mem.write_count;
            let prev_miss_count = self.fiber_cache.miss_count;
            let prev_b_evict_count = self.fiber_cache.b_evict_count;
            let prev_psum_evict_count = self.fiber_cache.psum_evict_count;

            // Each PE execute a window.
            for i in 0..self.pe_num {
                // Fetch data from memory & cache.
                let (rowidxs, scaling_factors, fibers) = self.fetch_window_data(i);
                println!("PE: {} scaling factors: {:?}", i,
                    scaling_factors.iter().map(|x| x.iter().map(|y| y.0).collect::<Vec<usize>>()).collect::<Vec<Vec<usize>>>());

                // Compute the window.
                let output_fibers = self.compute_a_window(&rowidxs, scaling_factors, fibers);
                println!("Compute: rows: {:?} cols: {}-{} merge_mode: {} output fiber size: {:?}",
                    &rowidxs, self.pes[i].col_s, self.pes[i].col_s+self.pes[i].reduction_window[0],
                    &self.pes[i].merge_mode, output_fibers.iter().map(|c| c.len()).collect::<Vec<usize>>());
                println!("Reuse: touched fiber size: {} deduped fiber size: {}, output size: {}",
                    self.exec_trackers[&self.pes[i].cur_block.get_idx()].touched_fiber_size,
                    self.exec_trackers[&self.pes[i].cur_block.get_idx()].dedup_fiber_size,
                    self.exec_trackers[&self.pes[i].cur_block.get_idx()].output_fiber_size);

                    // Update reuse tracker if it is not in the merge mode.
                if !self.pes[i].merge_mode {
                    self.exec_trackers.get_mut(&self.pes[i].cur_block.get_idx())
                        .unwrap().output_fiber_size += 
                        output_fibers.iter().fold(0, |acc, x| acc + x.size());
                }

                // Update work mode.
                let pe = &self.pes[i];
                if pe.merge_mode {
                    for row in rowidxs.iter() {
                        self.merge_queue.push(*row);
                    }
                } else if !pe.merge_mode && pe.cur_block.height != 0 {
                    // Finish one traverse over current rows.
                    // Add the finished rows into merge queue and turn into merge mode.
                    for row in rowidxs.iter() {
                        if self.a_mem.get_rowptr(*row+1) - self.a_mem.get_rowptr(*row) <=
                            pe.col_s + pe.reduction_window[0] {
                            self.merge_queue.push(*row);
                        }
                    }
                }

                // Writeback psums.
                self.write_psum(rowidxs, output_fibers);
            }

            println!("Cache occp: {} in {}, miss_count: + {} -> {}, b_evict_count: + {} -> {}, psum_evict_count: + {} -> {}", self.fiber_cache.cur_num, self.fiber_cache.capability,
                self.fiber_cache.miss_count - prev_miss_count, self.fiber_cache.miss_count,
                self.fiber_cache.b_evict_count - prev_b_evict_count, self.fiber_cache.b_evict_count,
                self.fiber_cache.psum_evict_count - prev_psum_evict_count, self.fiber_cache.psum_evict_count);
            println!("A mem: read_count: + {} -> {}", self.a_mem.read_count - prev_a_mem_read_count, self.a_mem.read_count);
            println!("B mem: read_count: + {} -> {}", self.fiber_cache.b_mem.read_count - prev_b_mem_read_count, self.fiber_cache.b_mem.read_count);
            println!("C mem: read_count: + {} -> {}, write_count: +{} -> {}",
                self.fiber_cache.psum_mem.read_count - prev_psum_mem_read_count, self.fiber_cache.psum_mem.read_count,
                self.fiber_cache.psum_mem.write_count - prev_psum_mem_write_count, self.fiber_cache.psum_mem.write_count);
        }
    }

    fn assign_jobs(&mut self) -> bool {
        println!("Merge queue: {:?}", &self.merge_queue);

        // Dedup merge queue & writeback merged fiber.
        let mut i = 0;
        let mut psums_num: usize = 0;
        self.merge_queue.sort();
        self.merge_queue.dedup();
        while i != self.merge_queue.len() {
            let psum_addrs = self.output_tracker.get(&self.merge_queue[i]).unwrap();
            if psum_addrs.len() == 1 {
                println!("Assign jobs: swapout addr {} of {}", psum_addrs[0], self.merge_queue[i]);
                self.merge_queue.remove(i);
                self.fiber_cache.swapout(psum_addrs[0]);
            } else {
                i += 1;
                psums_num += psum_addrs.len();
            }
        }

        println!("Assign jobs: merge queue: {:?}", &self.merge_queue);

        // No job to assign if no multiplication and merge workloads.
        if self.a_traversed && self.pes.iter().all(|x| x.cur_block.height == 0) && psums_num == 0 {
            return false;
        }

        // Calculate the required merge psums number.
        let mut merge_pe_num = (psums_num + self.lane_num - 1) / self.lane_num;

        // Assign jobs to PEs.
        for pe_no in 0..self.pe_num {
            // Allocate PEs to merge the unmerged psums in prior.
            println!("PE no: {}", pe_no);
            if merge_pe_num > 0 {
                self.pes[pe_no].merge_mode = true;
                merge_pe_num -= 1;
            } else {
                println!("No merge to do.");
                println!("Reduction window: {:?}", self.reduction_window);
                self.pes[pe_no].merge_mode = false;
                // Try to shift the window in the block. Otherwise assign new block to PE.
                if !self.slide_window(pe_no) {
                    println!("Failed to shift window.");
                    // Either empty or finished.
                    match self.get_next_block() {
                        Some(block) => {
                            println!("Assign block {:?} to {}", block.get_idx(), pe_no);
                            let reduction_window = self.adjust_window(block.get_idx(), block.get_shape());
                            self.pes[pe_no].assign_block(block);
                            self.pes[pe_no].reduction_window = reduction_window;
                            if !self.is_col_valid(self.pes[pe_no].row_s,
                                self.pes[pe_no].reduction_window[1],
                                self.pes[pe_no].col_s+self.pes[pe_no].reduction_window[0],
                                self.pes[pe_no].cur_block.col_s,
                                self.pes[pe_no].cur_block.width) {
                            }
                            self.exec_trackers.insert(
                                self.pes[pe_no].cur_block.get_idx(),
                                ExecTracker::new(self.pes[pe_no].cur_block.get_shape(),
                                self.pes[pe_no].reduction_window.clone()));
                        },
                        None => {
                            self.pes[pe_no].reset_pe();
                            self.a_traversed = true;
                        }
                    }
                }
            }
        }

        return true;
    }

    fn get_next_block(&mut self) -> Option<Block> {
        loop {
            if self.row_s >= self.a_mem.get_row_len() { return None; }

            // Try to allocate along K dim.
            if self.is_col_valid(self.row_s, self.block_shape[1], self.col_s, self.col_s, self.block_shape[0]) {
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
                self.block_topo.col_s_list.last_mut().unwrap().push(block.col_s);
                self.col_s += self.block_shape[0];
                return Some(block);
            } else {
                // Block shape adaptation can be added here. For now we only support adjust block
                // when finishing traverse over K dim.
                self.adjust_block();
                self.row_s += self.block_shape[1];
                self.col_s = 0;
            }
        }
    }

    fn is_col_valid(&self, row_s: usize, row_num: usize, col_s: usize, b_col_s: usize, b_width: usize) -> bool {
        for rowid in row_s..row_s+row_num {
            if (rowid >= self.a_mem.get_row_len()) ||
                (self.a_mem.get_rowptr(rowid+1) - self.a_mem.get_rowptr(rowid) <= col_s) ||
                (col_s < b_col_s) ||
                (col_s >= b_col_s + b_width) {
                    continue;
            } else {
                return true;
            }
        }

        return false;
    }

    fn slide_window(&mut self, pe_no: usize) -> bool {
        // If no block has been assigned.
        if self.pes[pe_no].cur_block.height == 0 { return false; }

        // If the row_s exceeds the block limitation.
        if self.pes[pe_no].row_s >= self.pes[pe_no].cur_block.row_s + self.pes[pe_no].cur_block.height { return false; }
        // Try to allocate along K dim.
        if self.is_col_valid(self.pes[pe_no].row_s, self.pes[pe_no].reduction_window[1],
                self.pes[pe_no].col_s+self.pes[pe_no].reduction_window[0],
                self.pes[pe_no].cur_block.col_s,
                self.pes[pe_no].cur_block.width) {
            self.pes[pe_no].col_s += self.pes[pe_no].reduction_window[0];
        } else {
            self.pes[pe_no].col_s = self.pes[pe_no].cur_block.col_s;
            self.pes[pe_no].row_s += self.pes[pe_no].reduction_window[1];
            if self.pes[pe_no].row_s >= self.pes[pe_no].cur_block.row_s + self.pes[pe_no].cur_block.height { return false; }
            while !self.is_col_valid(self.pes[pe_no].row_s, self.pes[pe_no].reduction_window[1], self.pes[pe_no].col_s,
                    self.pes[pe_no].cur_block.col_s, self.pes[pe_no].cur_block.width) {
                self.pes[pe_no].row_s += self.pes[pe_no].reduction_window[1];
                if self.pes[pe_no].row_s >= self.pes[pe_no].cur_block.row_s + self.pes[pe_no].cur_block.height { return false; }
            }
        }

        // Check if the current window is at the tail.
        if !self.is_col_valid(self.pes[pe_no].row_s, self.pes[pe_no].reduction_window[1],
                self.pes[pe_no].col_s+self.pes[pe_no].reduction_window[0],
                self.pes[pe_no].cur_block.col_s, self.pes[pe_no].cur_block.width) {
        }

        println!("{} shift to row_s {} col_s {}, block: row_s {} col_s {} height {} width {}",
            pe_no, self.pes[pe_no].row_s, self.pes[pe_no].col_s, self.pes[pe_no].cur_block.row_s,
            self.pes[pe_no].cur_block.col_s, self.pes[pe_no].cur_block.height, self.pes[pe_no].cur_block.width);
        true
    }

    /// Block shape adaptation can be added here.
    /// For now we only support adjust block when finishing traverse over K dim.
    fn adjust_block(&mut self) {
    }

    /// Adjust the reduction window for the current block.
    fn adjust_window(&mut self, cur_idx: [usize; 2], block_shape: [usize; 2]) -> [usize; 2] {
        let neighbor_blocks = self.get_neighbor_blocks(&cur_idx);

        // If no neighbor blocks, then use the default reduction window shape.
        if neighbor_blocks.len() == 0 {
            return [self.lane_num, 1];
        }
        // We look at the neighbor blocks and find the block with the largest total reuse.
        let max_reuse_block = neighbor_blocks[neighbor_blocks
            .iter()
            .map(|x| self.exec_trackers[x].c_reuse()+self.exec_trackers[x].b_reuse())
            .position_max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()];

        let cr = self.exec_trackers[&max_reuse_block].c_reuse();
        let br = self.exec_trackers[&max_reuse_block].b_reuse();
        let mut reduction_window = self.exec_trackers[&max_reuse_block].window;

        if cr >= br {
            if reduction_window[1] > 1 && reduction_window[0] * 2 <= block_shape[0] {
                reduction_window[1] /= 2;
                reduction_window[0] *= 2;
            }
        } else {
            if reduction_window[0] > 1 && reduction_window[1] * 2 <= block_shape[1]{
                reduction_window[0] /= 2;
                reduction_window[1] *= 2;
            }
        }

        reduction_window
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
    fn fetch_window_data(&mut self, pe_no: usize) -> (Vec<usize>, Vec<Vec<(usize, f64)>>, Vec<Vec<CsrRow>>) {
        let pe = &self.pes[pe_no];
        let mut scaling_factors = vec![];
        let mut fibers = vec![];
        let mut rowidxs = vec![];

        if pe.merge_mode {
            let mut unused_lane_num = self.lane_num;
            while unused_lane_num > 0 && self.merge_queue.len() > 0 {
                let rowidx = self.merge_queue.first().unwrap();
                let psums = self.output_tracker.get_mut(rowidx).unwrap();
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
            rowidxs = (pe.row_s..pe.row_s+pe.reduction_window[1])
                .filter(|x|self.a_mem.get_rowptr(*x+1) as i32 - self.a_mem.get_rowptr(*x) as i32 >= 0)
                .collect();
            let mut broadcast_cache: HashMap<usize, CsrRow> = HashMap::new();
            for rowidx in rowidxs.iter() {
                let mut r_sfs = CsrRow::new(*rowidx);
                if self.a_mem.get_rowptr(*rowidx+1) > self.a_mem.get_rowptr(*rowidx) + pe.col_s {
                    let ele_num = min(pe.reduction_window[0], self.a_mem.get_rowptr(*rowidx+1) - self.a_mem.get_rowptr(*rowidx) - pe.col_s);
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
                        match self.fiber_cache.read(*colid) {
                            Some(csrrow) => {
                                broadcast_cache.insert(*colid, csrrow.clone());
                                fbs.push(csrrow);
                                sfs.push((*colid, *value));
                            },
                            None => (),
                        }
                    }
                }
                scaling_factors.push(sfs);
                fibers.push(fbs);
            }
            // Update reuse tracker data.
            // println!("Fetch row data: previous touched: {}, dedup: {}", self.reuse_trackers[pe_no].touched_fiber_size, self.reuse_trackers[pe_no].dedup_fiber_size);
            self.exec_trackers.get_mut(&pe.cur_block.get_idx()).unwrap().touched_fiber_size +=
                fibers.iter().flatten().fold(0, |acc, x| acc + x.size());
            self.exec_trackers.get_mut(&pe.cur_block.get_idx()).unwrap().dedup_fiber_size +=
                fibers.iter().flatten().sorted_by(|a, b| Ord::cmp(&a.rowptr, &b.rowptr)).dedup_by(|x, y| x.rowptr == y.rowptr).fold(0, |acc, x| acc + x.size());
            // println!("Fetch row data: current touched: {}, dedup: {}", self.reuse_trackers[pe_no].touched_fiber_size, self.reuse_trackers[pe_no].dedup_fiber_size)
        }

        return (rowidxs, scaling_factors, fibers);
    }

    fn compute_a_window(
        &self,
        rowidxs: &Vec<usize>,
        scaling_factors: Vec<Vec<(usize, f64)>>,
        fibers: Vec<Vec<CsrRow>>)
        -> Vec<CsrRow> {
            let mut psums = vec![];
            for (rowidx, sfs, fbs) in izip!(rowidxs, scaling_factors, fibers) {
                // Compute psum.
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
                psums.push(psum);
            }

            return psums;
        }

    fn write_psum(&mut self, rowidxs: Vec<usize>, output_fibers: Vec<CsrRow>) {
        for (rowidx, mut output_fiber) in rowidxs.into_iter().zip(output_fibers.into_iter()) {
            self.output_tracker
                .entry(rowidx)
                .or_default()
                .push(self.output_base_addr);
            println!("write_psum: {:?}", self.output_tracker[&rowidx]);
            output_fiber.rowptr = self.output_base_addr;
            self.output_base_addr += 1;
            self.fiber_cache.write(output_fiber);
        }
    }

    pub fn get_result(&mut self) -> Vec<CsrRow> {
        let mut c = vec![];
        // for rowid in 0..self.a_mem.indptr.len() - 1 {
        for rowid in 0..self.a_mem.get_row_len() {
            let mut csrrow = CsrRow::new(rowid);
            // if self.a_mem.indptr[rowid+1] - self.a_mem.indptr[rowid] > 0 {
            if self.a_mem.get_rowptr(rowid+1) - self.a_mem.get_rowptr(rowid) > 0 {
                let raw_rowid = if self.a_mem.remapped {
                    self.a_mem.row_remap[&rowid]
                } else {rowid};
                // let raw_rowid = self.a_mem.row_remap[&rowid];
                let addrs = self.output_tracker.get(&rowid).unwrap();
                println!("Get result: row: {} addrs: {:?}", raw_rowid, &addrs);
                assert!(addrs.len() == 1, "Partially merged psums! {:?} of row {}", &addrs, raw_rowid);
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
        (self.fiber_cache.b_mem.read_count, self.fiber_cache.b_mem.write_count)
    }

    pub fn get_c_mat_stat(&self) -> (usize, usize) {
        (self.fiber_cache.psum_mem.read_count, self.fiber_cache.psum_mem.write_count)
    }

}