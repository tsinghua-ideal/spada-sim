use std::{cmp::{Reverse, max, min}, collections::{BinaryHeap, HashMap, VecDeque}, hash::Hash};

use itertools::any;
use sprs::vec;

use crate::storage::{CsrMatStorage, CsrRow, PriorityCache, Snapshotable, LogItem};

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


#[derive(Debug, Clone)]
struct PriorityCacheSimuSnapshot {
    pub cur_num: usize,
    pub priority_queue: BinaryHeap<Reverse<[usize; 2]>>,
    pub rowmap_inc: HashMap<usize, Option<usize>>,
    pub old_pq_row_track: HashMap<usize, LogItem>,
}

#[derive(Debug, Clone)]
pub struct PriorityCacheSimu {
    pub cache_size: usize,
    pub word_byte: usize,
    pub capability: usize,
    pub cur_num: usize,
    pub rowmap: HashMap<usize, usize>,
    pub priority_queue: BinaryHeap<Reverse<[usize; 2]>>,
    pub valid_pq_row_dict: HashMap<usize, usize>,
    snapshot: Option<PriorityCacheSimuSnapshot>,
}

impl Snapshotable for PriorityCacheSimu {
    fn take_snapshot(&mut self) {
        self.snapshot = Some(PriorityCacheSimuSnapshot {
            cur_num: self.cur_num,
            priority_queue: self.priority_queue.clone(),
            rowmap_inc: HashMap::new(),
            old_pq_row_track: HashMap::new(),
        });
    }

    fn drop_snapshot(&mut self) {
        self.snapshot = None;
    }

    fn restore_from_snapshot(&mut self) {
        match self.snapshot {
            Some(ref mut snp) => {
                // Restore rowmap from execution log.
                for (rowid, size) in snp.rowmap_inc.drain() {
                    if let Some(s) = size {
                        self.rowmap.insert(rowid, s);
                    } else {
                        self.rowmap.remove(&rowid);
                    }
                }

                // Restore valid_pq_row_dict from execution log.
                for (colptr, logitem) in snp.old_pq_row_track.iter() {
                    match logitem {
                        LogItem::Update(x) => self.valid_pq_row_dict.insert(*colptr, *x),
                        LogItem::Insert => self.valid_pq_row_dict.remove(colptr),
                    };
                }
                self.cur_num = snp.cur_num;
                self.priority_queue = snp.priority_queue.clone();
            },
            None => {
                panic!("No snapshot to be restored!");
            }
        }
    }
}

impl PriorityCacheSimu {
    pub fn new(cache_size: usize, word_byte: usize) -> PriorityCacheSimu {
        PriorityCacheSimu {
            cache_size: cache_size,
            word_byte: word_byte,
            capability: cache_size / word_byte,
            cur_num: 0,
            rowmap: HashMap::new(),
            priority_queue: BinaryHeap::new(),
            valid_pq_row_dict: HashMap::new(),
            snapshot: None,
        }
    }

    fn rowmap_insert(&mut self, rowptr: usize, size: usize) {
        if let Some(ref mut snp) = self.snapshot {
            if !snp.rowmap_inc.contains_key(&rowptr) {
                snp.rowmap_inc.insert(rowptr, None);
            }
        }
        self.rowmap.insert(rowptr, size);
    }

    fn rowmap_remove(&mut self, rowptr: &usize) -> Option<usize> {
        let size = self.rowmap.remove(rowptr);
        if let Some(ref mut snp) = self.snapshot {
            if !snp.rowmap_inc.contains_key(&rowptr) {
                if let Some(ref c) = size {
                    snp.rowmap_inc.insert(*rowptr, Some(*c));
                }
            }
        }
        size
    }

    fn priority_queue_push(&mut self, a_loc: [usize; 2]) {
        self.priority_queue.push(Reverse(a_loc));
    }

    fn priority_queue_pop(&mut self) -> Option<[usize; 2]> {
        self.priority_queue.pop().map(|s|s.0)
    }

    pub fn write(&mut self, a_loc: [usize; 2], size: usize) {
        // Freeup space first if necessary.
        if self.cur_num + size <= self.capability {
            self.cur_num += size;
        } else {
            if let Err(err) = self.freeup_space(size) {
                panic!("{}", err);
            }
            self.cur_num += size;
        }

        // Track snapshot.
        if let Some(ref mut snp) = self.snapshot {
            if !snp.old_pq_row_track.contains_key(&a_loc[1]) {
                if self.valid_pq_row_dict.contains_key(&a_loc[1]) {
                    snp.old_pq_row_track.insert(a_loc[1], LogItem::Update(self.valid_pq_row_dict[&a_loc[1]]));
                } else {
                    snp.old_pq_row_track.insert(a_loc[1], LogItem::Insert);
                }
            }
        }

        // Update priority status.
        self.valid_pq_row_dict.entry(a_loc[1])
            .and_modify(|x| *x = max(*x, a_loc[0]))
            .or_insert(a_loc[0]);
        self.priority_queue_push([self.valid_pq_row_dict[&a_loc[1]], a_loc[1]]);

        self.rowmap_insert(a_loc[1], size);

    }

    pub fn freeup_space(&mut self, size: usize) -> Result<(), String> {
        while self.priority_queue.len() > 0 && (self.cur_num + size > self.capability) {
            let mut popid: [usize; 2];
            loop {
                popid = self.priority_queue_pop().unwrap();
                // println!("freeup_space: popid: {:?}", popid);
                if self.valid_pq_row_dict[&popid[1]] == popid[0] && self.rowmap.contains_key(&popid[1]) {
                    break;
                }
            }
            let evict_size = self.rowmap_remove(&popid[1]).unwrap();
            // println!("*freeup {} with {}", popid[1], evict_size);
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

    pub fn read(&mut self, a_loc: [usize; 2]) -> Option<usize> {
        if self.rowmap.contains_key(&a_loc[1]) {
            // Track snapshot.
            if let Some(ref mut snp) = self.snapshot {
                if !snp.old_pq_row_track.contains_key(&a_loc[1]) {
                    if self.valid_pq_row_dict.contains_key(&a_loc[1]) {
                        snp.old_pq_row_track.insert(a_loc[1], LogItem::Update(self.valid_pq_row_dict[&a_loc[1]]));
                    } else {
                        snp.old_pq_row_track.insert(a_loc[1], LogItem::Insert);
                    }
                }
            }

            self.valid_pq_row_dict.entry(a_loc[1])
            .and_modify(|x| *x = max(*x, a_loc[0]))
            .or_insert(a_loc[0]);
            self.priority_queue_push([self.valid_pq_row_dict[&a_loc[1]], a_loc[1]]);
            let size = self.rowmap.get(&a_loc[1]).unwrap().clone();
            return Some(size);
        } else {
            return None;
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
        let mut cache = PriorityCacheSimu::new(self.cache_size, 8);
        for i in 0..self.a_mem.get_row_len() {
            print!("{} ", i);
            let s = self.a_mem.indptr[i];
            let t = self.a_mem.indptr[i+1];
            for colidx in self.a_mem.indices[s..t].iter() {
                let size = 2 * (self.b_mem.indptr[*colidx+1] - self.b_mem.indptr[*colidx]);
                if cache.rowmap.contains_key(colidx) {
                    cache.read([i, *colidx]);
                    // print!("h{} ", collect.values().sum::<usize>());
                    print!("h ");
                } else {
                    cache.write([i, *colidx], size);
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
        let mut cache = PriorityCacheSimu::new(self.cache_size, 8);
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
                    // print!("{} ", rowidx);
                    let colidx = ss[rowidx - i] + coloffset;
                    if colidx >= st[rowidx - i] { continue; }
                    let colptr = self.a_mem.indices[colidx];
                    finished = false;
                    let size = 2 * (self.b_mem.indptr[colptr+1] - self.b_mem.indptr[colptr]);
                    if cache.rowmap.contains_key(&colptr) {
                        cache.read([rowidx, colptr]);
                        // print!("h{} ", collect.values().sum::<usize>());
                        // print!("h ");
                    } else {
                        cache.write([rowidx, colptr], size);
                        // Scheme 1: Only stat on A column index number.
                        // *collect.entry(colptr).or_insert(0) += 1;
                        // Scheme 2: Stat B row size.
                        *collect.entry(colptr).or_insert(0) += size;
                        // print!("m{} ", collect.values().sum::<usize>());
                        // print!("m ");
                    }
                }
                // println!("");
                if finished { break; }
                coloffset += 1;
            }
            println!("last row: {} b fetch: {}", min(i+row_num, self.a_mem.get_row_len()), collect.values().sum::<usize>());
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

    pub fn collect_ip_reuse_distance(&mut self) -> HashMap<usize, Vec<usize>> {
        let mut collect: HashMap<usize, Vec<usize>> = HashMap::new();
        let mut prev_pos: HashMap<usize, usize> = HashMap::new();
        let mut ele_counter: usize = 0;
        for i in 0..self.a_mem.get_row_len() {
            let s = self.a_mem.indptr[i];
            let t = self.a_mem.indptr[i+1];
            for colptr in self.a_mem.indices[s..t].iter() {
                if prev_pos.contains_key(colptr) {
                    let ele_dist = ele_counter - prev_pos[colptr];
                    collect.entry(*colptr)
                        .and_modify(|x| x.push(ele_dist))
                        .or_insert(vec![ele_dist,]);
                }
                ele_counter += self.b_mem.indptr[*colptr+1] - self.b_mem.indptr[*colptr];
                prev_pos.insert(*colptr, ele_counter);
            }
        }

        return collect;
    }

    pub fn collect_omega_reuse_distance(&mut self, row_num: usize) -> HashMap<usize, Vec<usize>> {
        let mut collect: HashMap<usize, Vec<usize>> = HashMap::new();
        let mut prev_pos: HashMap<usize, usize> = HashMap::new();
        let mut ele_counter: usize = 0;
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
                    let colidx = ss[rowidx - i] + coloffset;
                    if colidx >= st[rowidx - i] { continue; }
                    let colptr = self.a_mem.indices[colidx];
                    finished = false;

                    if prev_pos.contains_key(&colptr) {
                        let ele_dist = ele_counter - prev_pos[&colptr];
                        collect.entry(colptr)
                            .and_modify(|x| x.push(ele_dist))
                            .or_insert(vec![ele_dist,]);
                    }
                    ele_counter += self.b_mem.indptr[colptr+1] - self.b_mem.indptr[colptr];
                    prev_pos.insert(colptr, ele_counter);
                }
                if finished { break; }
                coloffset += 1;
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

    pub fn improved_reuse(&mut self, row_num: usize) -> (usize, usize, f32) {
        let mut improved_counter = 0;
        let mut total_reuse_counter = 0;
        let ip_dist = self.collect_ip_reuse_distance();
        let omega_dist = self.collect_omega_reuse_distance(row_num);
        for k in ip_dist.keys() {
            for (ip_dist, o_dist) in ip_dist[k].iter().zip(omega_dist[k].iter()) {
                total_reuse_counter += 1;
                if *ip_dist >= self.cache_size && self.cache_size >= *o_dist {
                    improved_counter += 1;
                }
            }
        }

        return (total_reuse_counter, improved_counter, improved_counter as f32 / total_reuse_counter as f32);
    }

    pub fn oracle_blocked_fetch(&mut self) -> HashMap<usize, usize> {
        println!("--oracle blocked fetch");
        let collect: HashMap<usize,usize> = HashMap::new();
        let mut cache = PriorityCacheSimu::new(self.cache_size, 8);

        let mut result_track: HashMap<i32, (HashMap<usize, usize>, PriorityCacheSimu)> = HashMap::new();
        let prev_result_offset = [1, 2, 4, 8];

        // Execute for row 0.
        // self._exec(&mut collect, &mut cache, 0, 1);
        // cache.take_snapshot();
        // result_track.insert(0, (collect, cache));

        // Set the ordinary case for -1.
        cache.take_snapshot();
        result_track.insert(-1, (collect, cache));

        for row_end in 0..self.a_mem.get_row_len() {
            let mut min_b_fetch = usize::MAX;
            let mut min_collect = None;
            let mut min_cache = None;
            let mut min_offset = 0;
            println!("--Row: {}", row_end);

            for offset in prev_result_offset.iter() {
                // Prepare the base execution environment.
                println!("offset {} ", offset);
                if row_end + 1 < *offset { continue; }
                let prev_result_idx = row_end as i32 - *offset as i32;
                let (ref mut base_collect, ref mut base_cache) = result_track.get_mut(&(prev_result_idx as i32)).unwrap();
                let row_start = row_end + 1 - *offset;

                // Execute.
                print!("Before fetch {}, ", base_collect.values().sum::<usize>());
                let rev_collect_log = self._exec(base_collect, base_cache, row_start, row_end+1);
                // print!("collect log: {:?} ", &rev_collect_log);

                // Update the dp tape if the scheme is better.
                let cur_b_fetch = base_collect.values().sum::<usize>();
                print!("After fetch {}\n", cur_b_fetch);
                if cur_b_fetch < min_b_fetch {
                    min_collect = Some(base_collect.clone());
                    min_cache = Some(base_cache.clone());
                    min_b_fetch = cur_b_fetch;
                    min_offset = *offset;
                }

                // Restore the base condition.
                base_cache.restore_from_snapshot();
                for (colptr, logitem) in rev_collect_log.iter() {
                    match logitem {
                        LogItem::Update(x) => base_collect.insert(*colptr, *x),
                        LogItem::Insert => base_collect.remove(colptr),
                    };
                }
            }

            if min_cache.is_some() && min_collect.is_some() {
                result_track.insert(row_end as i32, (
                    min_collect.unwrap(),
                    min_cache.map(|mut x|{x.take_snapshot(); x}).unwrap()
                ));
            }

            println!("oracle offset: {} with fetch: {}", min_offset, min_b_fetch);

            result_track.retain(|k, _| *k + 8 >= max(row_end as i32, 7));
        }

        return result_track.get(&(self.a_mem.get_row_len() as i32 - 1)).unwrap().0.clone();
    }

    pub fn _exec(&mut self, base_collect: &mut HashMap<usize, usize>, base_cache: &mut PriorityCacheSimu, row_s: usize, row_t: usize) -> HashMap<usize, LogItem> {
        let mut old_collect_track: HashMap<usize, LogItem> = HashMap::new();
        let mut ss = vec![];
        let mut st = vec![];
        for i in row_s..row_t {
            ss.push(self.a_mem.indptr[i]);
            st.push(self.a_mem.indptr[i+1]);
        }

        let mut coloffset = 0;
        loop {
            let mut finished = true;
            for rowidx in row_s..row_t {
                // print!("{} " , rowidx);
                let colidx = ss[rowidx - row_s] + coloffset;
                if colidx >= st[rowidx - row_s] { continue; }
                let colptr = self.a_mem.indices[colidx];
                finished = false;
                let size = 2 * (self.b_mem.indptr[colptr+1] - self.b_mem.indptr[colptr]);
                if base_cache.rowmap.contains_key(&colptr) {
                    // println!("read: rowidx: {} colptr: {}", rowidx, colptr);
                    base_cache.read([rowidx, colptr]);
                } else {
                    // println!("write: rowidx: {} colptr: {} size: {} cur_num: {}", rowidx, colptr, size, base_cache.cur_num);
                    base_cache.write([rowidx, colptr], size);
                    // Track old value to restore later.
                    if !old_collect_track.contains_key(&colptr) {
                        if base_collect.contains_key(&colptr) {
                            old_collect_track.insert(colptr, LogItem::Update(base_collect[&colptr]));
                        } else {
                            old_collect_track.insert(colptr, LogItem::Insert);
                        }
                    }
                    *base_collect.entry(colptr).or_insert(0) += size;
                }
            }
            if finished { break; }
            coloffset += 1;
        }

        return old_collect_track;
    }
}