use std::collections::HashMap;

use itertools::{izip, merge};
use sprs::CsMat;
use storage::LRUCache;

use crate::storage::{self, CsrRow, Storage, StorageAPI, StorageError};

#[derive(Debug, Clone)]
struct PE {
    reduction_window: [usize; 2],
    cur_rows: Vec<usize>,
    cur_col: usize,
}

struct OmegaTraffic<'a> {
    done: bool,
    reduction_window: [usize; 2],
    reuse_hist: Vec<[usize; 2]>,
    pe_num: usize,
    lane_num: usize,
    unalloc_row: usize,
    unalloc_col: usize,
    cache: LRUCache,
    pes: Vec<PE>,
    A_mem: &'a mut Storage,
    B_mem: &'a mut Storage,
    output_base_addr: usize,
}

impl<'a> OmegaTraffic<'a> {
    fn new(
        pe_num: usize,
        lane_num: usize,
        cache_size: usize,
        word_byte: usize,
        A_mem: &'a mut Storage,
        B_mem: &'a mut Storage,
    ) -> OmegaTraffic<'a> {
        // Init from the inner-product dataflow.
        // Can be changed to be adaptive.
        OmegaTraffic {
            done: false,
            reduction_window: [lane_num, 1],
            reuse_hist: vec![[0, 0]; lane_num],
            pe_num: pe_num,
            lane_num: lane_num,
            unalloc_row: 0,
            unalloc_col: 0,
            cache: LRUCache::new(cache_size, word_byte),
            pes: vec![
                PE {
                    reduction_window: [lane_num, 1],
                    cur_rows: vec![],
                    cur_col: 0,
                };
                pe_num
            ],
            output_base_addr: B_mem.indptr.len(),
            A_mem: A_mem,
            B_mem: B_mem,
        }
    }

    fn get_valid_rowids(&self, row_num: usize) -> Option<Vec<usize>> {
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
        // Try to allocate along K dim.
        let pe = &self.pes[pe_no];
        if pe.cur_rows.len() > 0
            && self.is_col_start_valid(&pe.cur_rows, pe.cur_col + pe.reduction_window[1])
        {
            self.pes[pe_no].cur_col += self.pes[pe_no].reduction_window[1];
        } else {
            // Allocate new rows to PE.
            match self.get_valid_rowids(self.reduction_window[0]) {
                Some(rowids) => {
                    self.pes[pe_no].cur_rows.extend(rowids);
                    self.pes[pe_no].cur_col = 0;
                    self.pes[pe_no].reduction_window = self.reduction_window.clone();
                }
                None => {
                    self.done = true;
                }
            }
        }
    }

    fn fetch_window_data(
        &mut self,
        pe_no: usize,
    ) -> (Vec<usize>, Vec<Vec<(usize, f64)>>, Vec<Vec<CsrRow>>) {
        let mut pe = &mut self.pes[pe_no];
        let rowidxs = pe.cur_rows.clone();
        let mut scaling_factors = vec![];
        let mut fibers = vec![];

        let mut broadcast_cache: HashMap<usize, CsrRow> = HashMap::new();
        for rowidx in rowidxs.iter() {
            let mut r_sfs = self
                .A_mem
                .read(*rowidx, pe.cur_col, pe.reduction_window[1])
                .unwrap();
            let mut fbs = vec![];
            let mut sfs = vec![];
            for (colid, value) in r_sfs.enumerate() {
                if broadcast_cache.contains_key(colid) {
                    let csrrow = broadcast_cache[colid].clone();
                    fbs.push(csrrow);
                    sfs.push((*colid, *value));
                } else {
                    match self.B_mem.read_row(*colid) {
                        Ok(csrrow) => {
                            broadcast_cache.insert(*colid, csrrow.clone());
                            fbs.push(csrrow);
                            sfs.push((*colid, *value));
                        }
                        Err(_) => (),
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

    fn write(&mut self, output_fibers: Vec<CsrRow>) {
        for mut output_fiber in output_fibers {
            output_fiber.rowptr += self.output_base_addr;
            self.cache.write(output_fiber);
        }
    }

    fn execute(&mut self) {
        while !self.done {
            // Assign rows.
            for i in 0..self.pe_num {
                self.adjust_window();
                if self.pes[i].cur_rows.len() == 0 {
                    self.assign_jobs(i);
                }
            }

            // Each PE execute a window.
            for i in 0..self.pe_num {
                let (rowidxs, scaling_factors, fibers) = self.fetch_window_data(i);
                let output_fibers = self.compute_a_window(rowidxs, scaling_factors, fibers);
                self.write(output_fibers);
            }
        }
    }
}
