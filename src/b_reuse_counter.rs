use std::{cmp::{min, max}, collections::{HashMap, VecDeque}, hash::Hash};

use itertools::any;
use sprs::vec;

use crate::storage::CsrMatStorage;

struct LRUCacheSimu {
    pub cache_size: usize,
    pub word_byte: usize,
    pub capability: usize,
    pub cur_num: usize,
    pub rowmap: HashMap<usize, usize>,
    pub lru_queue: VecDeque<usize>,
}

impl LRUCacheSimu {
    pub fn new(cache_size: usize, word_byte: usize) -> LRUCacheSimu {
        LRUCacheSimu {
            cache_size: cache_size,
            word_byte: word_byte,
            capability: cache_size / word_byte,
            cur_num: 0,
            rowmap: HashMap::new(),
            lru_queue: VecDeque::new(),
        }
    }

    pub fn write(&mut self, rowidx: usize, size: usize) {
        if self.cur_num + size <= self.capability {
            self.cur_num += size;
            self.lru_queue.push_back(rowidx);
            self.rowmap.insert(rowidx, size);
        } else {
            if let Err(err) = self.freeup_space(size) {
                panic!("{}", err);
            }
            self.cur_num += size;
            self.lru_queue.push_back(rowidx);
            self.rowmap.insert(rowidx, size);
        }
    }

    pub fn read(&mut self, rowidx: usize) {
        if let Some(pos) = self.lru_queue.iter().position(|&x| x == rowidx) {
            self.lru_queue.remove(pos);
        }
        self.lru_queue.push_back(rowidx);
    }

    pub fn freeup_space(&mut self, size: usize) -> Result<(), String> {
        while self.lru_queue.len() > 0 && (self.cur_num + size > self.capability) {
            let mut popid: usize;
            loop {
                popid = self.lru_queue.pop_front().unwrap();
                if self.rowmap.contains_key(&popid) {
                    break;
                }
            }
            let evict_size = self.rowmap.remove(&popid).unwrap();
            self.cur_num -= evict_size;
        }
        if self.cur_num + size > self.capability {
            return Err(format!(
                "freeup_space: Not enough space for {}",
                size
            ));
        } else {
            return Ok(());
        }
    }
}

pub struct BReuseCounter<'a> {
    pub a_mem: &'a mut CsrMatStorage,
    pub b_mem: &'a mut CsrMatStorage,
    cache_size: usize,
    word_byte: usize,
}

impl<'a> BReuseCounter<'a> {
    pub fn new(
        a_mem: &'a mut CsrMatStorage,
        b_mem: &'a mut CsrMatStorage,
        cache_size: usize,
        word_byte: usize,
    ) -> BReuseCounter<'a> {
        BReuseCounter {
            a_mem: a_mem,
            b_mem: b_mem,
            cache_size: cache_size,
            word_byte: word_byte,
        }
    }

    pub fn oracle_fetch(&mut self) -> HashMap<usize, usize> {
        // Assume an unlimited cache so that every unique fiber requires only one access.
        println!("--oracle_fetch");
        let mut collect = HashMap::new();
        for i in 0..self.a_mem.get_row_len() {
            let s = self.a_mem.indptr[i];
            let t = self.a_mem.indptr[i+1];
            for colptr in self.a_mem.indices[s..t].iter() {
                *collect.entry(*colptr).or_insert(0) += 1;
            }
        }
        println!("");

        return collect;
    }

    pub fn cached_fetch(&mut self) -> HashMap<usize, usize> {
        // Assume a GAMMA like access pattern.
        println!("--cached_fetch");
        let mut collect = HashMap::new();
        let mut cache = LRUCacheSimu::new(self.cache_size, 8);
        for i in 0..self.a_mem.get_row_len() {
            print!("{} ", i);
            let s = self.a_mem.indptr[i];
            let t = self.a_mem.indptr[i+1];
            for colidx in self.a_mem.indices[s..t].iter() {
                let size = 2 * (self.b_mem.indptr[*colidx+1] - self.b_mem.indptr[*colidx]);
                if cache.rowmap.contains_key(colidx) {
                    cache.read(*colidx);
                    // print!("h{} ", collect.values().sum::<usize>());
                    print!("h ");
                } else {
                    cache.write(*colidx, size);
                    // Scheme 1: Only stat on A column index number.
                    // *collect.entry(*colidx).or_insert(0) += 1;

                    // Scheme 2: Stat B row size.
                    *collect.entry(*colidx).or_insert(0) += size;
                    // print!("m{} ", collect.values().sum::<usize>());
                    print!("m ");
                }
            }
            println!("");
        }

        return collect;
    }

    pub fn blocked_fetch(&mut self, row_num: usize) -> HashMap<usize, usize> {
        // Multi-row access pattern.
        println!("--blocked_fetch");
        let mut collect = HashMap::new();
        let mut cache = LRUCacheSimu::new(self.cache_size, 8);
        for i in (0..self.a_mem.get_row_len()).step_by(row_num) {
            let mut ss = vec![];
            let mut st = vec![];
            for rowidx in i..min(i+row_num, self.a_mem.get_row_len()) {
                ss.push(self.a_mem.indptr[rowidx]);
                st.push(self.a_mem.indptr[rowidx+1]);
            }
            let mut coloffset = 0;
            loop {
                let mut finished = true;
                for rowidx in i..min(i+row_num, self.a_mem.get_row_len()) {
                    print!("{} ", rowidx);
                    let colidx = ss[rowidx - i] + coloffset;
                    if colidx >= st[rowidx - i] { continue; }
                    let colptr = self.a_mem.indices[colidx];
                    finished = false;
                    let size = 2 * (self.b_mem.indptr[colptr+1] - self.b_mem.indptr[colptr]);
                    if cache.rowmap.contains_key(&colptr) {
                        cache.read(colptr);
                        // print!("h{} ", collect.values().sum::<usize>());
                        print!("h ");
                    } else {
                        cache.write(colptr, size);
                        // Scheme 1: Only stat on A column index number.
                        // *collect.entry(colptr).or_insert(0) += 1;
                        // Scheme 2: Stat B row size.
                        *collect.entry(colptr).or_insert(0) += size;
                        // print!("m{} ", collect.values().sum::<usize>());
                        print!("m ");
                    }
                }
                println!("");
                if finished { break; }
                coloffset += 1;
            }
        }

        return collect;
    }

    pub fn reuse_row_distance(&mut self) -> HashMap<usize, [f32; 2]> {
        let mut collect: HashMap<usize, [f32; 2]> = HashMap::new();
        let mut prev_pos: HashMap<usize, [usize; 2]> = HashMap::new();
        let mut occr_counter = HashMap::new();
        let mut ele_counter: usize = 0;
        for i in 0..self.a_mem.get_row_len() {
            let s = self.a_mem.indptr[i];
            let t = self.a_mem.indptr[i+1];
            for colptr in self.a_mem.indices[s..t].iter() {
                *occr_counter.entry(*colptr).or_insert(0) += 1;
                if prev_pos.contains_key(colptr) {
                    let row_dist = (i - prev_pos[colptr][0]) as f32;
                    let ele_dist = (ele_counter - prev_pos[colptr][1]) as f32;
                    collect
                    .entry(*colptr)
                    .and_modify(|e|
                        *e = [(e[0] * (occr_counter[colptr] - 2) as f32 + row_dist) / (occr_counter[colptr] - 1) as f32,
                                (e[1] * (occr_counter[colptr] - 2) as f32 + ele_dist) / (occr_counter[colptr] - 1) as f32])
                    .or_insert([row_dist, ele_dist]);
                }
                prev_pos.insert(*colptr, [i, ele_counter]);
                ele_counter += 1;
            }
        }

        return collect;
    }

    pub fn collect_row_length(&mut self) -> HashMap<usize, usize> {
        let mut collect: HashMap<usize, usize> = HashMap::new();
        for i in 0..self.b_mem.get_row_len() {
            collect.insert(i, self.b_mem.indptr[i+1] - self.b_mem.indptr[i]);
        }

        return collect;
    }

    pub fn neighbor_row_affinity(&mut self) -> HashMap<usize, usize> {
        println!("neighbor_row_affinity");
        let neighbor_num = self.cache_size / self.word_byte / 2;
        let neighbor_row = neighbor_num / (self.a_mem.get_nonzero() / self.a_mem.get_row_len());
        let mut collect: HashMap<usize, usize> = HashMap::new();
        for i in 0..self.a_mem.get_row_len() {
            let s = self.a_mem.indptr[i];
            let t = self.a_mem.indptr[i+1];
            for col in self.a_mem.indices[s..t].iter() {
                let mut founded = false;
                for j in max(0, i as i32 - neighbor_row as i32) as usize ..i {
                    let s = self.a_mem.indptr[j];
                    let t = self.a_mem.indptr[j+1];
                    for colptr in self.a_mem.indices[s..t].iter() {
                        if *colptr == *col {
                            *collect.entry(i).or_insert(0) += 1;
                            founded = true;
                            break;
                        }
                    }
                    if founded { break; }
                }
            }
        }

        return collect;
    }
}