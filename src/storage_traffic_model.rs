use std::{cmp::{max, min}, collections::{HashMap, VecDeque}, hash::Hash};

use itertools::{Itertools, Merge, izip, merge, merge_join_by};
use pyo3::ffi::PyExc_NotImplementedError;
use sprs::CsMat;
use storage::{LRUCache, VectorStorage};

use crate::storage::{self, CsrMatStorage, CsrRow, StorageAPI, StorageError};
use crate::frontend::Accelerator;

#[derive(Debug, Clone)]
struct PE {
    reduction_window: [usize; 2], // [width, height]
    cur_rows: Vec<usize>,
    cur_col: usize,
    merge_mode: bool,
}

struct Block {
    pub width: usize, // spatial range
    pub height: usize, // compressed range
}

#[derive(Debug, Clone)]
struct ReuseTracker {
    pub touched_fiber_size: usize,
    pub dedup_fiber_size: usize,
    pub output_fiber_size: usize,
}

impl ReuseTracker {
    pub fn new() -> ReuseTracker {
        ReuseTracker {
            touched_fiber_size: 0,
            dedup_fiber_size: 0,
            output_fiber_size: 0,
        }
    }

    pub fn c_reuse(&self, reduction_window: &[usize; 2]) -> f64 {
        self.touched_fiber_size as f64 / self.output_fiber_size as f64 / reduction_window[0] as f64
    }

    pub fn b_reuse(&self, reduction_window: &[usize; 2]) -> f64 {
        self.touched_fiber_size as f64 / self.dedup_fiber_size as f64 / reduction_window[1] as f64
    }
}

pub struct TrafficModel<'a> {
    no_new_product: bool,
    reduction_window: [usize; 2],
    reuse_trackers: Vec<ReuseTracker>,
    local_pe: usize,
    pe_num: usize,
    lane_num: usize,
    unalloc_row: usize,
    fiber_cache: LRUCache<'a>,
    pes: Vec<PE>,
    a_mem: &'a mut CsrMatStorage,
    output_base_addr: usize,
    output_tracker: HashMap<usize, Vec<usize>>,
    merge_queue: Vec<usize>,
    accelerator: Accelerator,
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

        TrafficModel {
            no_new_product: false,
            reduction_window: reduction_window,
            reuse_trackers: vec![ReuseTracker::new(); pe_num],
            local_pe: pe_num,
            pe_num: pe_num,
            lane_num: lane_num,
            unalloc_row: 0,
            fiber_cache: LRUCache::new(cache_size, word_byte, output_base_addr, b_mem, psum_mem),
            pes: vec![
                PE {
                    reduction_window: [lane_num, 1],
                    cur_rows: vec![],
                    cur_col: 0,
                    merge_mode: false,
                };
                pe_num
            ],
            output_base_addr: output_base_addr,
            output_tracker: HashMap::new(),
            a_mem,
            merge_queue: vec![],
            accelerator: accelerator,
        }
    }

    fn get_valid_rowids(&mut self, row_num: usize) -> Vec<usize> {
        let mut rowids = vec![];
        let mut rowid = self.unalloc_row;
        for _ in 0..row_num {
            loop {
                // if rowid >= (self.a_mem.indptr.len() - 1) {
                if rowid >= self.a_mem.get_row_len() {
                    self.unalloc_row = rowid;
                    return rowids;
                // } else if self.a_mem.indptr[rowid + 1] - self.a_mem.indptr[rowid] == 0 {
                } else if self.a_mem.get_rowptr(rowid+1) - self.a_mem.get_rowptr(rowid) == 0 {
                    rowid += 1;
                } else {
                    rowids.push(rowid);
                    rowid += 1;
                    break;
                }
            }
        }

        self.unalloc_row = rowid;

        return rowids;
    }

    fn is_col_start_valid(&self, rowids: &Vec<usize>, cur_col_s: usize) -> bool {
        for rowid in rowids.iter() {
            // if (*rowid >= self.a_mem.indptr.len() - 1)
            // || (self.a_mem.indptr[*rowid + 1] - self.a_mem.indptr[*rowid] <= cur_col_s)
            if (*rowid >= self.a_mem.get_row_len())
            || (self.a_mem.get_rowptr(*rowid+1) - self.a_mem.get_rowptr(*rowid) <= cur_col_s)
            {
                continue;
            } else {
                return true;
            }
        }

        return false;
    }

    fn get_read_element_num(&self, rowid: usize, cur_col_s: usize, window_width: usize) -> usize {
        // min(window_width, self.a_mem.indptr[rowid+1] - self.a_mem.indptr[rowid] - cur_col_s)
        min(window_width, self.a_mem.get_rowptr(rowid+1) - self.a_mem.get_rowptr(rowid) - cur_col_s)
    }

    fn adjust_window(&mut self) {
        if self.local_pe >= self.pe_num {
            return;
        }
        let cr = self.reuse_trackers[self.local_pe].c_reuse(&self.pes[self.local_pe].reduction_window);
        let br = self.reuse_trackers[self.local_pe].b_reuse(&self.pes[self.local_pe].reduction_window);
        if cr >= br {
            if self.reduction_window[1] > 1 {
                self.reduction_window[1] /= 2;
                self.reduction_window[0] *= 2;
            }
        } else {
            if self.reduction_window[0] > 1 {
                self.reduction_window[0] /= 2;
                self.reduction_window[1] *= 2;
            }
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
        if self.no_new_product && self.pes.iter().all(|x| x.cur_rows.len() == 0) && psums_num == 0 {
            return true;
        }

        // Calculate the required merge psums number.
        let mut merge_pe_num = (psums_num + self.lane_num - 1) / self.lane_num;

        // Assign jobs to PEs.
        let mut adjusted = false;
        for pe_no in 0..self.pe_num {
            // Allocate PEs to merge the unmerged psums in prior.
            if merge_pe_num > 0 {
                self.pes[pe_no].merge_mode = true;
                merge_pe_num -= 1;
            } else {
                self.pes[pe_no].merge_mode = false;
                if self.pes[pe_no].cur_rows.len() == 0 {
                    if !adjusted {
                        adjusted = true;
                        // Adjust window according to previous execution.
                        println!("Assign jobs: from {:?}", &self.reduction_window);
                        if self.accelerator == Accelerator::Omega {
                            self.adjust_window();
                        }
                        println!("Assign jobs: adjust reduction window to: {:?}", &self.reduction_window);
                    }
                    let rowids = self.get_valid_rowids(self.reduction_window[1]);
                    if rowids.len() > 0 {
                        self.pes[pe_no].cur_rows = rowids;
                        self.pes[pe_no].cur_col = 0;
                        self.pes[pe_no].reduction_window = self.reduction_window.clone();
                        self.local_pe = pe_no;
                    } else {
                        self.no_new_product = true;
                    }
                } else {
                    // Try to allocate along K dim.
                    assert!(self.is_col_start_valid(&self.pes[pe_no].cur_rows,
                        self.pes[pe_no].cur_col+self.pes[pe_no].reduction_window[0]));
                    self.pes[pe_no].cur_col += self.pes[pe_no].reduction_window[0];
                }
            }
        }

        return false;
    }

    fn fetch_window_data(&mut self, pe_no: usize) -> (Vec<usize>, Vec<Vec<(usize, f64)>>, Vec<Vec<CsrRow>>) {
        let pe = &mut self.pes[pe_no];
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
                // pe.cur_rows = rowidxs.clone();
            }
        } else {
            rowidxs = pe.cur_rows.clone();
            let mut broadcast_cache: HashMap<usize, CsrRow> = HashMap::new();
            for rowidx in rowidxs.iter() {
                let mut r_sfs = CsrRow::new(*rowidx);
                // if self.a_mem.indptr[*rowidx+1] > self.a_mem.indptr[*rowidx] + pe.cur_col {
                    // let ele_num = min(pe.reduction_window[0], self.a_mem.indptr[*rowidx+1] - self.a_mem.indptr[*rowidx] - pe.cur_col);
                if self.a_mem.get_rowptr(*rowidx+1) > self.a_mem.get_rowptr(*rowidx) + pe.cur_col {
                    let ele_num = min(pe.reduction_window[0], self.a_mem.get_rowptr(*rowidx+1) - self.a_mem.get_rowptr(*rowidx) - pe.cur_col);
                    r_sfs = self.a_mem.read(*rowidx, pe.cur_col, ele_num).unwrap();
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
            self.reuse_trackers[pe_no].touched_fiber_size += fibers.iter().flatten().fold(0, |acc, x| acc + x.size());
            self.reuse_trackers[pe_no].dedup_fiber_size += fibers.iter().flatten().sorted_by(|a, b| Ord::cmp(&a.rowptr, &b.rowptr)).dedup_by(|x, y| x.rowptr == y.rowptr).fold(0, |acc, x| acc + x.size());
            // println!("Fetch row data: current touched: {}, dedup: {}", self.reuse_trackers[pe_no].touched_fiber_size, self.reuse_trackers[pe_no].dedup_fiber_size)
        }

        return (rowidxs, scaling_factors, fibers);
    }

    fn compute_a_window(
        &self,
        rowidxs: &Vec<usize>,
        scaling_factors: Vec<Vec<(usize, f64)>>,
        fibers: Vec<Vec<CsrRow>>,
    ) -> Vec<CsrRow> {
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

    pub fn execute(&mut self) {
        loop {
            println!("----");

            // Assign jobs to PEs.
            let done = self.assign_jobs();

            // If no jobs can be assigned, end execution.
            if done {
                break;
            }

            let prev_a_mem_read_count = self.a_mem.read_count;
            let prev_b_mem_read_count = self.fiber_cache.b_mem.read_count;
            let prev_psum_mem_read_count = self.fiber_cache.psum_mem.read_count;
            let prev_psum_mem_write_count = self.fiber_cache.psum_mem.write_count;
            let prev_miss_count = self.fiber_cache.miss_count;
            let prev_b_evict_count = self.fiber_cache.b_evict_count;
            let prev_psum_evict_count = self.fiber_cache.psum_evict_count;

            // Each PE executes a window.
            for i in 0..self.pe_num {
                let (rowidxs, scaling_factors, fibers) = self.fetch_window_data(i);
                println!("PE: {}, Fetch data: scaling_factors: {:?}", i,
                    scaling_factors.iter().map(|x| x.iter().map(|y| y.0).collect::<Vec<usize>>()).collect::<Vec<Vec<usize>>>());
                let output_fibers = self.compute_a_window(&rowidxs, scaling_factors, fibers);
                println!("Compute: rowidxs: {:?} cur_col: {} merge_mode: {} output fiber size: {:?}", &rowidxs, &self.pes[i].cur_col, &self.pes[i].merge_mode, output_fibers.iter().map(|c| c.len()).collect::<Vec<usize>>());
                println!("Reuse: touched fiber size: {} deduped fiber size: {}, output size: {}", self.reuse_trackers[i].touched_fiber_size, self.reuse_trackers[i].dedup_fiber_size, self.reuse_trackers[i].output_fiber_size);
                // Update reuse tracker if it is not in the merge mode.
                if !self.pes[i].merge_mode {
                    self.reuse_trackers[i].output_fiber_size += output_fibers.iter().fold(0, |acc, x| acc + x.size());
                }

                // Update work mode.
                let pe = &self.pes[i];
                if pe.merge_mode {
                    for row in rowidxs.iter() {
                        self.merge_queue.push(*row);
                    }
                } else if !pe.merge_mode && pe.cur_rows.len() != 0 && !self.is_col_start_valid(&pe.cur_rows, pe.cur_col+pe.reduction_window[0]) {
                    // Finish one traverse over current rows.
                    // Add the finished rows into merge queue and turn into merge mode.
                    for row in pe.cur_rows.iter() {
                        self.merge_queue.push(*row);
                    }
                    // Clear the current jobs.
                    self.pes[i].cur_rows.clear();
                    self.pes[i].cur_col = 0;
                    self.reuse_trackers[i].touched_fiber_size = 0;
                    self.reuse_trackers[i].dedup_fiber_size = 0;
                    self.reuse_trackers[i].output_fiber_size = 0;
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
