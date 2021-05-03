use std::collections::HashMap;

use itertools::{izip, merge, Merge};
use sprs::CsMat;
use storage::{LRUCache, VectorStorage};

use crate::storage::{self, CsrMatStorage, CsrRow, StorageAPI, StorageError};

#[derive(Debug, Clone)]
struct PE {
    reduction_window: [usize; 2],
    cur_rows: Vec<usize>,
    cur_col: usize,
    merge_mode: bool,
}

pub struct OmegaTraffic<'a> {
    done: bool,
    reduction_window: [usize; 2],
    reuse_hist: Vec<[usize; 2]>,
    pe_num: usize,
    lane_num: usize,
    unalloc_row: usize,
    unalloc_col: usize,
    fiber_cache: LRUCache<'a>,
    pes: Vec<PE>,
    A_mem: &'a mut CsrMatStorage,
    output_base_addr: usize,
    output_tracker: HashMap<usize, Vec<usize>>,
}

impl<'a> OmegaTraffic<'a> {
    pub fn new(
        pe_num: usize,
        lane_num: usize,
        cache_size: usize,
        word_byte: usize,
        output_base_addr: usize,
        A_mem: &'a mut CsrMatStorage,
        B_mem: &'a mut CsrMatStorage,
        psum_mem: &'a mut VectorStorage,
    ) -> OmegaTraffic<'a> {
        // Init from the inner-product dataflow.
        // Can be changed to be adaptive.
        OmegaTraffic {
            done: false,
            reduction_window: [1, lane_num],
            reuse_hist: vec![[0, 0]; lane_num],
            pe_num: pe_num,
            lane_num: lane_num,
            unalloc_row: 0,
            unalloc_col: 0,
            fiber_cache: LRUCache::new(cache_size, word_byte, output_base_addr, B_mem, psum_mem),
            pes: vec![
                PE {
                    reduction_window: [1, lane_num],
                    cur_rows: vec![],
                    cur_col: 0,
                    merge_mode: false,
                };
                pe_num
            ],
            output_base_addr: output_base_addr,
            output_tracker: HashMap::new(),
            A_mem: A_mem,
        }
    }

    fn get_valid_rowids(&mut self, row_num: usize) -> Option<Vec<usize>> {
        let mut rowids = vec![];
        let mut rowid = self.unalloc_row;
        for _ in 0..row_num {
            loop {
                if rowid >= (self.A_mem.indptr.len() - 1) {
                    return None;
                } else if self.A_mem.indptr[rowid + 1] - self.A_mem.indptr[rowid] == 0 {
                    rowid += 1;
                } else {
                    rowids.push(rowid);
                    break;
                }
            }
        }

        self.unalloc_row = rowid + 1;

        return Some(rowids);
    }

    fn is_col_start_valid(&self, rowids: &Vec<usize>, cur_col_s: usize) -> bool {
        for rowid in rowids.iter() {
            if (*rowid >= self.A_mem.indptr.len() - 1)
                || (self.A_mem.indptr[*rowid + 1] - self.A_mem.indptr[*rowid] < cur_col_s)
            {
                continue;
            } else {
                return true;
            }
        }

        return false;
    }

    fn adjust_window(&mut self) {}

    fn assign_jobs(&mut self, pe_no: usize) {
        let pe = &self.pes[pe_no];
        if pe.cur_rows.len() == 0 {
            // Assign initial rows to PEs.
            match self.get_valid_rowids(self.reduction_window[0]) {
                Some(rowids) => {
                    // self.pes[pe_no].cur_rows.extend(rowids);
                    self.pes[pe_no].cur_rows = rowids;
                    self.pes[pe_no].cur_col = 0;
                    self.pes[pe_no].reduction_window = self.reduction_window.clone();
                }
                None => {
                    self.done = true;
                }
            }
        } else if self.is_col_start_valid(&pe.cur_rows, pe.cur_col + pe.reduction_window[1]) {
            // Try to allocate along K dim.
            self.pes[pe_no].cur_col += self.pes[pe_no].reduction_window[1];
        } else {
            // Finish one tranverse over current rows. Turn PE into
        }
    }

    fn fetch_window_data(
        &mut self,
        pe_no: usize,
    ) -> (Vec<usize>, Vec<Vec<(usize, f64)>>, Vec<Vec<CsrRow>>) {
        let pe = &mut self.pes[pe_no];
        let rowidxs = pe.cur_rows.clone();
        let mut scaling_factors = vec![];
        let mut fibers = vec![];

        let mut broadcast_cache: HashMap<usize, CsrRow> = HashMap::new();
        for rowidx in rowidxs.iter() {
            let r_sfs = match self.A_mem.read(*rowidx, pe.cur_col, pe.reduction_window[1]) {
                Ok(csrrow) => csrrow,
                Err(err) => CsrRow::new(*rowidx),
            };
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

        return (rowidxs, scaling_factors, fibers);
    }

    fn compute_a_window(
        &self,
        rowidxs: Vec<usize>,
        scaling_factors: Vec<Vec<(usize, f64)>>,
        fibers: Vec<Vec<CsrRow>>,
    ) -> Vec<CsrRow> {
        let mut psums = vec![];
        for (rowidx, sfs, fbs) in izip!(rowidxs, scaling_factors, fibers) {
            // Compute psum.
            let mut psum = CsrRow::new(rowidx);
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

    fn write_psum(&mut self, output_fibers: Vec<CsrRow>) {
        for mut output_fiber in output_fibers {
            self.output_tracker
                .entry(output_fiber.rowptr)
                .or_default()
                .push(self.output_base_addr);
            output_fiber.rowptr = self.output_base_addr;
            self.output_base_addr += 1;
            self.fiber_cache.write(output_fiber);
        }
    }

    pub fn execute(&mut self) {
        while !self.done {
            // Assign rows.
            self.adjust_window();
            for i in 0..self.pe_num {
                self.assign_jobs(i);
                println!("{} reduction window {:?}", i, &self.pes[i].reduction_window);
                println!(
                    "{} assign row {:?} col {} - {}",
                    i,
                    &self.pes[i].cur_rows,
                    self.pes[i].cur_col,
                    self.pes[i].cur_col + self.pes[i].reduction_window[1]
                );
            }

            // Each PE execute a window.
            for i in 0..self.pe_num {
                let (rowidxs, scaling_factors, fibers) = self.fetch_window_data(i);
                let output_fibers = self.compute_a_window(rowidxs, scaling_factors, fibers);
                // println!("Compute psum of: {:#?}", output_fibers.iter().map(|c| c.rowptr).collect::<Vec<usize>>());
                // println!("Output fiber size: {:?}", output_fibers.iter().map(|c| c.len()).collect::<Vec<usize>>());
                self.write_psum(output_fibers);
            }
        }
    }
}
