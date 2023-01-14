use crate::gemm::GEMM;
use crate::trace_println;
use itertools::izip;
use std::{
    cmp::{max, min, Reverse},
    collections::{BinaryHeap, HashMap},
    fmt, usize,
};

#[derive(Debug, Clone)]
pub enum StorageError {
    ReadEmptyRowError(String),
    ReadOverBoundError(String),
}

impl fmt::Display for StorageError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", &self)
    }
}

#[derive(Debug, Clone)]
pub struct Element {
    pub idx: [usize; 2],
    pub value: f64,
}

impl Element {
    pub fn new(idx: [usize; 2], value: f64) -> Element {
        Element { idx, value }
    }
}

#[derive(Debug, Clone)]
pub struct CsrRow {
    pub rowptr: usize,
    pub data: Vec<f64>,
    pub indptr: Vec<usize>,
    consumed: usize,
}

impl CsrRow {
    pub fn new(rowptr: usize) -> CsrRow {
        CsrRow {
            rowptr: rowptr,
            data: vec![],
            indptr: vec![],
            consumed: 0,
        }
    }

    pub fn new_from_data(rowptr: usize, data: Vec<f64>, indptr: Vec<usize>) -> CsrRow {
        CsrRow {
            rowptr,
            data,
            indptr,
            consumed: 0,
        }
    }

    pub fn as_element_vec(self) -> Vec<Element> {
        let mut result = vec![];
        for (d, col_idx) in izip!(self.data, self.indptr) {
            result.push(Element {
                idx: [self.rowptr, col_idx],
                value: d,
            });
        }

        return result;
    }

    pub fn size(&self) -> usize {
        return self.data.len() + self.indptr.len() - 2 * self.consumed;
    }

    pub fn len(&self) -> usize {
        return self.indptr.len();
    }

    pub fn append(&mut self, csrrow: CsrRow) {
        assert!(
            self.rowptr == csrrow.rowptr,
            "Not the same row ({}, {}), cannot be combined!",
            self.rowptr,
            csrrow.rowptr
        );
        self.data.extend(csrrow.data.iter());
        self.indptr.extend(csrrow.indptr.iter());
    }

    pub fn append_element(&mut self, element: Element) {
        assert!(
            self.rowptr == element.idx[0],
            "Not the same row ({}, {}), cannot be combined!",
            self.rowptr,
            element.idx[0]
        );
        self.data.push(element.value);
        self.indptr.push(element.idx[1]);
    }

    pub fn consume_front(&mut self, num: usize) -> Result<(), &str> {
        for _ in 0..num {
            if self.size() > 0 {
                self.consumed += 1;
            } else {
                return Err("Empty csrrow!");
            }
        }
        Ok(())
    }
}

impl fmt::Display for CsrRow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let display_len = min(self.data.len(), 5);
        write!(
            f,
            "rowptr: {} indptr: {:?} data: {:?}",
            self.rowptr,
            &self.indptr[0..display_len],
            &self.data[0..display_len]
        )
    }
}

pub fn sorted_element_vec_to_csr_row(srt_ele_vec: Vec<Element>) -> CsrRow {
    let rowptr = srt_ele_vec[0].idx[0];
    let data = srt_ele_vec.iter().map(|e| e.value).collect::<Vec<f64>>();
    let indptr = srt_ele_vec.iter().map(|e| e.idx[1]).collect::<Vec<usize>>();
    return CsrRow {
        rowptr,
        data,
        indptr,
        consumed: 0,
    };
}

pub trait StorageAPI {
    fn read(
        &mut self,
        row_ptr: usize,
        col_s: usize,
        ele_num: usize,
    ) -> Result<CsrRow, StorageError>;
    fn write(&mut self, rows: &mut Vec<CsrRow>) -> Result<Vec<usize>, StorageError>;
}

pub struct CsrMatStorage {
    pub data: Vec<f64>,
    pub indptr: Vec<usize>,
    pub indices: Vec<usize>,
    pub read_count: usize,
    pub write_count: usize,
    pub remapped: bool,
    pub row_remap: HashMap<usize, usize>,
    pub track_count: bool,
    pub mat_shape: [usize; 2],
}

impl StorageAPI for CsrMatStorage {
    fn read(&mut self, rawp: usize, col_s: usize, ele_num: usize) -> Result<CsrRow, StorageError> {
        if rawp >= self.indptr.len() {
            return Err(StorageError::ReadOverBoundError(format!(
                "Invalid row_ptr: {}",
                rawp
            )));
        }

        let row_ptr = if self.remapped {
            self.row_remap[&rawp]
        } else {
            rawp
        };

        let cur_row_pos = self.indptr[row_ptr];
        let end_row_pos = self.indptr[row_ptr + 1];
        let s = cur_row_pos + col_s;
        let t = s + ele_num;
        if (s <= t) && (t <= end_row_pos) {
            let csrrow =
                CsrRow::new_from_data(rawp, self.data[s..t].to_vec(), self.indices[s..t].to_vec());
            if self.track_count {
                self.read_count += csrrow.size();
            }
            return Ok(csrrow);
        } else {
            return Err(StorageError::ReadEmptyRowError(format!(
                "Invalid col_pos: {}..{} with end_row_pos {} for row {}.",
                s, t, end_row_pos, rawp
            )));
        }
    }

    fn write(&mut self, rows: &mut Vec<CsrRow>) -> Result<Vec<usize>, StorageError> {
        let mut indptrs = vec![];
        for row in rows.iter_mut() {
            let indptr = self.data.len();
            indptrs.push(indptr);
            if self.track_count {
                self.write_count += 2 * row.data.len() + 1;
            }
            self.data.extend(row.data.iter());
            self.indices.extend(row.indptr.iter());
            self.indptr.insert(self.indptr.len() - 1, indptr);
            *self.indptr.last_mut().unwrap() = self.data.len();
        }
        Ok(indptrs)
    }
}

impl CsrMatStorage {
    pub fn init_with_gemm(gemm: GEMM) -> (CsrMatStorage, CsrMatStorage) {
        (
            CsrMatStorage {
                data: gemm.a.data().to_vec(),
                indptr: gemm.a.indptr().as_slice().unwrap().to_vec(),
                indices: gemm.a.indices().to_vec(),
                read_count: 0,
                write_count: 0,
                remapped: false,
                row_remap: HashMap::new(),
                track_count: true,
                mat_shape: [gemm.a.shape().1, gemm.a.shape().0],
            },
            CsrMatStorage {
                data: gemm.b.data().to_vec(),
                indptr: gemm.b.indptr().as_slice().unwrap().to_vec(),
                indices: gemm.b.indices().to_vec(),
                read_count: 0,
                write_count: 0,
                remapped: false,
                row_remap: HashMap::new(),
                track_count: true,
                mat_shape: [gemm.b.shape().1, gemm.b.shape().0],
            },
        )
    }

    pub fn read_row(&mut self, row_ptr: usize) -> Result<CsrRow, StorageError> {
        if row_ptr >= self.indptr.len() {
            return Err(StorageError::ReadEmptyRowError(format!(
                "Invalid row_ptr: {}",
                row_ptr
            )));
        }
        let row_len = self.indptr[row_ptr + 1] - self.indptr[row_ptr];
        return self.read(row_ptr, 0, row_len);
    }

    pub fn reorder_row(&mut self, rowmap: HashMap<usize, usize>) {
        self.remapped = true;
        self.row_remap = rowmap;
    }

    pub fn rowptr(&self, rowid: usize) -> usize {
        if self.remapped {
            return self.indptr[self.row_remap[&rowid]];
        } else {
            return self.indptr[rowid];
        }
    }

    pub fn row_num(&self) -> usize {
        self.indptr.len() - 1
    }

    pub fn get_ele_num(&self, row_s: usize, row_t: usize) -> usize {
        let mut ele_num = 0;
        for i in row_s..row_t {
            let rawidx = if self.remapped { self.row_remap[&i] } else { i };
            ele_num += self.indptr[rawidx + 1] - self.indptr[rawidx];
        }

        return ele_num;
    }

    pub fn read_scalars(
        &mut self,
        row_idx: usize,
        col_idx: usize,
        num: usize,
    ) -> Result<Vec<Element>, StorageError> {
        // trace_println!(
        //     "***storage read_scalars: row_idx {} col_idx {} num {}",
        //     row_idx,
        //     col_idx,
        //     num
        // );
        if row_idx >= self.indptr.len() {
            return Err(StorageError::ReadOverBoundError(format!(
                "Invalid row_ptr: {}",
                row_idx
            )));
        } else if num == 0 {
            return Ok(vec![]);
        }

        let row_idx = if self.remapped {
            self.row_remap[&row_idx]
        } else {
            row_idx
        };

        let cur_row_pos = self.indptr[row_idx];
        let end_row_pos = self.indptr[row_idx + 1];
        let s = cur_row_pos + col_idx;
        if s < end_row_pos {
            let elements = (s..min(s + num, end_row_pos))
                .map(|idx| Element::new([row_idx, self.indices[idx]], self.data[idx]))
                .collect::<Vec<Element>>();
            if self.track_count {
                self.read_count += elements.len() * 2;
            }
            return Ok(elements);
        } else {
            return Err(StorageError::ReadEmptyRowError(format!(
                "Invalid col_pos: {}",
                s
            )));
        }
    }
}

pub struct VectorStorage {
    pub data: HashMap<usize, CsrRow>,
    pub read_count: usize,
    pub write_count: usize,
    pub track_count: bool,
}

impl StorageAPI for VectorStorage {
    fn read(
        &mut self,
        row_ptr: usize,
        col_s: usize,
        ele_num: usize,
    ) -> Result<CsrRow, StorageError> {
        match self.data.get(&row_ptr) {
            Some(csrrow) => {
                if col_s + ele_num <= csrrow.data.len() {
                    if self.track_count {
                        self.read_count += csrrow.size();
                    }
                    return Ok(CsrRow::new_from_data(
                        csrrow.rowptr,
                        csrrow.data[col_s..col_s + ele_num].to_vec(),
                        csrrow.indptr[col_s..col_s + ele_num].to_vec(),
                    ));
                } else {
                    return Err(StorageError::ReadEmptyRowError(format!(
                        "Invalid col_pos: {}..{} in row {}",
                        col_s,
                        col_s + ele_num,
                        csrrow.rowptr
                    )));
                }
            }
            None => {
                return Err(StorageError::ReadOverBoundError(format!(
                    "Invalid rowptr: {}",
                    row_ptr
                )))
            }
        }
    }

    fn write(&mut self, rows: &mut Vec<CsrRow>) -> Result<Vec<usize>, StorageError> {
        let mut indptrs = vec![];
        for row in rows.iter_mut() {
            let indptr = row.rowptr;
            indptrs.push(indptr);
            // If already exists, append to the previous, otherwise insert it.
            self.data
                .entry(indptr)
                .and_modify(|p| p.append(row.to_owned()))
                .or_insert(row.to_owned());
            if self.track_count {
                self.write_count += row.size();
            }
        }

        return Ok(indptrs);
    }
}

impl VectorStorage {
    pub fn new() -> VectorStorage {
        VectorStorage {
            data: HashMap::new(),
            read_count: 0,
            write_count: 0,
            track_count: true,
        }
    }

    pub fn read_row(&mut self, row_ptr: usize) -> Result<CsrRow, StorageError> {
        match self.data.get(&row_ptr) {
            Some(csrrow) => {
                if self.track_count {
                    self.read_count += csrrow.size();
                }
                return Ok(csrrow.clone());
            }
            None => {
                return Err(StorageError::ReadOverBoundError(format!(
                    "Invalid rowptr: {}",
                    row_ptr
                )))
            }
        }
    }

    pub fn consume_scalars(
        &mut self,
        row_idx: usize,
        col_idx: usize,
        num: usize,
    ) -> Result<Vec<Element>, StorageError> {
        match self.data.get(&row_idx) {
            Some(_cr) => {
                let elements = self.data.get(&row_idx).unwrap().clone().as_element_vec();
                let col_t = min(col_idx + num, elements.len());
                let ele_size = (col_t - col_idx) * 2;
                if self.track_count {
                    self.read_count += ele_size;
                }
                if col_t == elements.len() {
                    self.data.remove(&row_idx);
                }
                return Ok(elements[col_idx..col_t].to_vec());
            }
            None => {
                return Err(StorageError::ReadOverBoundError(format!(
                    "Invalid rowptr: {}",
                    row_idx
                )))
            }
        }
    }

    pub fn contains_row(&self, row_idx: &usize) -> bool {
        self.data.contains_key(row_idx)
    }

    pub fn write_element(&mut self, element: Element) -> Result<usize, StorageError> {
        let indptr = element.idx[0];
        self.data
            .entry(indptr)
            .or_insert(CsrRow::new(indptr))
            .append_element(element);
        if self.track_count {
            self.write_count += 2;
        }
        return Ok(indptr);
    }
}

pub struct LatencyPriorityCache<'a> {
    pub cache_size: usize,
    pub word_byte: usize,
    pub capability: usize,
    pub cur_num: usize,
    pub read_count: usize,
    pub write_count: usize,
    pub rowmap: HashMap<usize, CsrRow>,
    pub priority_queue: BinaryHeap<Reverse<[usize; 2]>>,
    pub valid_pq_row_dict: HashMap<usize, usize>,
    pub output_base_addr: usize,
    pub b_mem: &'a mut CsrMatStorage,
    pub psum_mem: &'a mut VectorStorage,
    pub miss_count: usize,
    pub b_evict_count: usize,
    pub psum_evict_count: usize,
    pub b_occp: usize,
    pub psum_occp: usize,
    pub track_count: bool,
    // Latency related.
    pub mem_latency: usize,
    pub cache_latency: usize,
    pub pending_request: HashMap<[usize; 2], usize>, // addr -> finish cycle
}

impl<'a> LatencyPriorityCache<'a> {
    pub fn new(
        cache_size: usize,
        word_byte: usize,
        output_base_addr: usize,
        b_mem: &'a mut CsrMatStorage,
        psum_mem: &'a mut VectorStorage,
        mem_latency: usize,
        cache_latency: usize,
    ) -> LatencyPriorityCache<'a> {
        LatencyPriorityCache {
            cache_size: cache_size,
            word_byte: word_byte,
            capability: cache_size / word_byte,
            cur_num: 0,
            read_count: 0,
            write_count: 0,
            rowmap: HashMap::new(),
            priority_queue: BinaryHeap::new(),
            valid_pq_row_dict: HashMap::new(),
            output_base_addr: output_base_addr,
            b_mem: b_mem,
            psum_mem: psum_mem,
            miss_count: 0,
            b_evict_count: 0,
            psum_evict_count: 0,
            b_occp: 0,
            psum_occp: 0,
            track_count: true,
            mem_latency,
            cache_latency,
            pending_request: HashMap::new(),
        }
    }

    fn rowmap_insert(&mut self, rowptr: usize, csrrow: CsrRow) {
        self.rowmap.insert(rowptr, csrrow);
    }

    fn rowmap_remove(&mut self, rowptr: &usize) -> Option<CsrRow> {
        let csrrow = self.rowmap.remove(rowptr);
        csrrow
    }

    fn rowmap_consume(&mut self, rowptr: &usize, num: usize) {
        let mut consumed = false;
        self.rowmap.entry(*rowptr).and_modify(|r| {
            r.consume_front(num).unwrap();
            if r.size() == 0 {
                consumed = true;
            }
        });
        if consumed {
            trace_println!("*{} released after consume.", rowptr);
            self.rowmap.remove(rowptr);
        }
    }

    fn priority_queue_push(&mut self, a_loc: [usize; 2]) {
        self.priority_queue.push(Reverse(a_loc));
    }

    fn priority_queue_pop(&mut self, pinned_addrs: Vec<usize>) -> Option<[usize; 2]> {
        let mut popped = vec![];
        loop {
            let p = self.priority_queue.pop().map(|s| s.0);
            popped.push(p);
            if p.is_none() || !pinned_addrs.contains(&p.unwrap()[1]) {
                break;
            }
        }
        let result = popped.pop().unwrap();
        for p in popped {
            if p.is_some() {
                self.priority_queue_push(p.unwrap());
            }
        }
        return result;
    }

    pub fn write(&mut self, csrrow: CsrRow, a_loc: [usize; 2]) {
        let row_size = csrrow.size();
        // Freeup space first if necessary.
        if self.cur_num + row_size <= self.capability {
            self.cur_num += row_size;
        } else {
            if let Err(err) = self.freeup_space(csrrow.rowptr, row_size) {
                panic!("{}", err);
            }
            self.cur_num += row_size;
        }

        // println!("*cache write invoked with count {} row {}", self.write_count, row_size);
        if self.is_psum_row(csrrow.rowptr) {
            self.psum_occp += row_size;
            trace_println!("*psum write: + {} -> {}", row_size, self.psum_occp);
        } else {
            self.b_occp += row_size;
        }

        // Update priority status.
        self.valid_pq_row_dict
            .entry(a_loc[1])
            .and_modify(|x| *x = max(*x, a_loc[0]))
            .or_insert(a_loc[0]);
        self.priority_queue_push([self.valid_pq_row_dict[&a_loc[1]], a_loc[1]]);

        if self.track_count {
            self.write_count += row_size;
        }

        self.rowmap_insert(a_loc[1], csrrow);
    }

    pub fn freeup_space(&mut self, addr: usize, space_required: usize) -> Result<(), String> {
        while self.priority_queue.len() > 0 && (self.cur_num + space_required > self.capability) {
            // trace_println!(
            //     "freeup_space: space_required: {} by {}",
            //     space_required,
            //     addr
            // );
            let poprow: usize;
            if self.b_occp < space_required {
                poprow = *self
                    .rowmap
                    .keys()
                    .filter(|&&rowid| self.is_psum_row(rowid) && rowid != addr)
                    .next()
                    .unwrap();
            } else {
                loop {
                    let popid = self.priority_queue_pop(vec![addr]).unwrap();
                    // trace_println!("freeup_space: popid: {:?}", popid);
                    if self.valid_pq_row_dict[&popid[1]] == popid[0]
                        && self.rowmap.contains_key(&popid[1])
                    {
                        poprow = popid[1];
                        break;
                    }
                }
            }
            if self.is_psum_row(poprow) {
                let popped_csrrow = self.rowmap_remove(&poprow).unwrap();
                // trace_println!("*freerow {:?} and get {}", poprow, popped_csrrow.size());
                self.cur_num -= popped_csrrow.size();
                if self.track_count {
                    self.psum_evict_count += popped_csrrow.size();
                }
                self.psum_occp -= popped_csrrow.size();
                trace_println!(
                    "*psum freeup: - {} -> {}",
                    popped_csrrow.size(),
                    self.psum_occp
                );
                self.psum_mem.write(&mut vec![popped_csrrow]).unwrap();
            } else {
                let evict_size = self.rowmap_remove(&poprow).unwrap().size();
                // trace_println!("*freerow {:?} and get {}", poprow, evict_size);
                self.cur_num -= evict_size;
                self.b_occp -= evict_size;
                if self.track_count {
                    self.b_evict_count += evict_size;
                }
            }
        }
        if self.cur_num + space_required > self.capability {
            return Err(format!(
                "freeup_space: Not enough space for {}",
                space_required
            ));
        } else {
            return Ok(());
        }
    }

    pub fn swapout(&mut self, rowid: usize) {
        if self.rowmap.contains_key(&rowid) {
            let popped_csrrow = self.rowmap_remove(&rowid).unwrap();
            trace_println!("swapout {} with size {}", rowid, popped_csrrow.size());
            self.cur_num -= popped_csrrow.size();
            if self.is_psum_row(rowid) {
                self.psum_occp -= popped_csrrow.size();
                trace_println!(
                    "*psum swapout: - {} -> {}",
                    popped_csrrow.size(),
                    self.psum_occp
                );
            } else {
                self.b_occp -= popped_csrrow.size();
            }
            self.psum_mem.write(&mut vec![popped_csrrow]).unwrap();
        } else {
            panic!("Swapout non-exist row: {}", rowid);
        }
    }

    pub fn is_psum_row(&self, rowid: usize) -> bool {
        return rowid >= self.output_base_addr;
    }

    pub fn append_psum_to(&mut self, addr: usize, csrrow: CsrRow) {
        let row_size = csrrow.size();

        // If the same addr psum is in the cache, append to current one.
        if self.rowmap.contains_key(&addr) {
            // Update occp.
            self.freeup_space(addr, row_size).unwrap();
            self.cur_num += row_size;
            if self.is_psum_row(addr) {
                self.psum_occp += row_size;
                trace_println!("*psum append: + {} -> {}", row_size, self.psum_occp);
            } else {
                self.b_occp += row_size;
            }
            // Update write count.
            if self.track_count {
                self.write_count += row_size;
            }
            // Update data.
            let psum = self.rowmap.get_mut(&addr).unwrap();
            psum.append(csrrow);

        // If swapped out, direct write the partial psum into psum memory.
        } else if self.psum_mem.contains_row(&addr) {
            self.psum_mem.write(&mut vec![csrrow]).unwrap();

        // Otherwise, alloc in cache.
        } else {
            // Update priority status.
            self.valid_pq_row_dict
                .entry(addr)
                .and_modify(|x| *x = max(*x, addr))
                .or_insert(addr);
            self.priority_queue_push([self.valid_pq_row_dict[&addr], addr]);
            // Update occp.
            self.freeup_space(addr, row_size).unwrap();
            self.cur_num += row_size;
            if self.is_psum_row(addr) {
                self.psum_occp += row_size;
                trace_println!("*psum append: + {} -> {}", row_size, self.psum_occp);
            } else {
                self.b_occp += row_size;
            }
            // Update write_count.
            if self.track_count {
                self.write_count += row_size;
            }
            // Update data.
            self.rowmap_insert(addr, csrrow);
        }
    }

    pub fn request_read_scalars(
        &mut self,
        a_loc: [usize; 2],
        col_s: usize,
        num: usize,
        cur_cycle: usize,
        no_delay: bool,
    ) -> Option<Vec<Element>> {
        // Read 0 elements should not mean the end of the row, thus return None.
        if num == 0 {
            return None;
        }
        // Pending the request.
        if !self.pending_request.contains_key(&a_loc) {
            if self.rowmap.contains_key(&a_loc[1]) {
                let ele_row = self.rowmap.get(&a_loc[1]).unwrap().len();
                let mut b_latency = if col_s == 0 && ele_row > 0 {
                    self.cache_latency
                } else {
                    0
                };
                if no_delay {
                    b_latency = 0;
                }
                self.pending_request.insert(a_loc, cur_cycle + b_latency);
            } else {
                let mut b_latency = self.mem_latency + self.cache_latency;
                if no_delay {
                    b_latency = 0;
                }
                self.pending_request.insert(a_loc, cur_cycle + b_latency);
            }
        }
        // Process the pending request.
        if cur_cycle < self.pending_request[&a_loc] {
            return None;
        }
        self.pending_request.remove(&a_loc);

        if self.rowmap.contains_key(&a_loc[1]) {
            // Only update when col_s is 0.
            if col_s == 0 {
                self.valid_pq_row_dict
                    .entry(a_loc[1])
                    .and_modify(|x| *x = max(*x, a_loc[0]))
                    .or_insert(a_loc[0]);
                self.priority_queue_push([self.valid_pq_row_dict[&a_loc[1]], a_loc[1]]);
            }
            let elements = self.rowmap.get(&a_loc[1]).unwrap().clone().as_element_vec();
            let col_t = min(col_s + num, elements.len());
            let ele_size = (col_t - col_s) * 2;
            if self.track_count {
                self.read_count += ele_size;
            }
            return Some(elements[col_s..col_t].to_vec());
        } else {
            if self.is_psum_row(a_loc[1]) {
                match self.psum_mem.read_row(a_loc[1]) {
                    Ok(csrrow) => {
                        if self.track_count {
                            self.miss_count += csrrow.size();
                        }
                        self.write(csrrow.clone(), a_loc);
                        let elements = csrrow.as_element_vec();
                        return Some(elements[col_s..min(col_s + num, elements.len())].to_vec());
                    }
                    Err(_) => panic!("{} should be in psum mem but not found!", a_loc[1]),
                }
            } else {
                match self.b_mem.read_row(a_loc[1]) {
                    Ok(csrrow) => {
                        if self.track_count {
                            self.miss_count += csrrow.size();
                        }
                        self.write(csrrow.clone(), a_loc);
                        let elements = csrrow.as_element_vec();
                        return Some(elements[col_s..min(col_s + num, elements.len())].to_vec());
                    }
                    Err(_) => panic!("{} should be in b mem but not found!", a_loc[1]),
                }
            }
        }
    }

    pub fn request_consume_scalars(
        &mut self,
        a_loc: [usize; 2],
        col_s: usize,
        num: usize,
        cur_cycle: usize,
        no_delay: bool,
    ) -> Option<Vec<Element>> {
        // Read 0 elements should not mean the end of the row, thus return None.
        if num == 0 {
            return None;
        }
        // Pending the request.
        if !self.pending_request.contains_key(&a_loc) {
            if self.rowmap.contains_key(&a_loc[1]) {
                let ele_row = self.rowmap.get(&a_loc[1]).unwrap().len();
                let mut b_latency = if col_s == 0 && ele_row > 0 {
                    self.cache_latency
                } else {
                    0
                };
                if no_delay {
                    b_latency = 0;
                }
                self.pending_request.insert(a_loc, cur_cycle + b_latency);
            } else {
                let mut b_latency = self.mem_latency + self.cache_latency;
                if no_delay {
                    b_latency = 0;
                }
                self.pending_request.insert(a_loc, cur_cycle + b_latency);
            }
        }
        // Process the pending request.
        if self.pending_request[&a_loc] < cur_cycle {
            return None;
        }
        self.pending_request.remove(&a_loc);

        if self.rowmap.contains_key(&a_loc[1]) {
            // Convert the csrrow to element vector.
            let elements = self.rowmap.get(&a_loc[1]).unwrap().clone().as_element_vec();
            // Track the tail of the readout.
            let col_t = min(col_s + num, elements.len());
            let ele_size = (col_t - col_s) * 2;
            let eles = elements[col_s..col_t].to_vec();
            // Update the counter.
            if self.track_count {
                self.read_count += ele_size;
            }
            // Update the occupation.
            self.cur_num -= ele_size;
            if self.is_psum_row(a_loc[1]) {
                self.psum_occp -= ele_size;
                trace_println!(
                    "*psum {} consume: - {} -> {}",
                    a_loc[1],
                    ele_size,
                    self.psum_occp
                );
            } else {
                self.b_occp -= ele_size;
            }
            // Release the consumed row after traversing it.
            // if col_t == elements.len() {
            //     trace_println!("*{} released after consume.", &a_loc[1]);
            //     self.rowmap_remove(&a_loc[1]).unwrap();
            // }
            self.rowmap_consume(&a_loc[1], col_t - col_s);
            return Some(eles);
        } else {
            if self.is_psum_row(a_loc[1]) {
                match self.psum_mem.consume_scalars(a_loc[1], col_s, num) {
                    Ok(eles) => {
                        if self.track_count {
                            self.read_count += eles.len() * 2;
                            self.miss_count += eles.len() * 2;
                        }
                        Some(eles)
                    }
                    Err(_) => Some(vec![]),
                }
            } else {
                match self.b_mem.read_scalars(a_loc[1], col_s, num) {
                    Ok(eles) => {
                        if self.track_count {
                            self.read_count += eles.len() * 2;
                            self.miss_count += eles.len() * 2;
                        }
                        Some(eles)
                    }
                    Err(_) => Some(vec![]),
                }
            }
        }
    }

    pub fn contains_row(&self, row_idx: &usize) -> bool {
        self.rowmap.contains_key(row_idx)
    }

    pub fn append_element_to(&mut self, addr: usize, element: Element) {
        let element_size = 2;
        // If the same addr psum is in the cache, append to current one.
        if self.rowmap.contains_key(&addr) {
            // Update occp.
            self.freeup_space(addr, element_size).unwrap();
            self.cur_num += element_size;
            if self.is_psum_row(addr) {
                self.psum_occp += element_size;
                trace_println!(
                    "*psum append element: + {} -> {}",
                    element_size,
                    self.psum_occp
                );
            } else {
                self.b_occp += element_size;
            }
            // Update write count.
            if self.track_count {
                self.write_count += element_size;
            }
            // Update data.
            let psum = self.rowmap.get_mut(&addr).unwrap();
            psum.append_element(element);
        // If swapped out, direct write the partial psum into psum memory.
        } else if self.psum_mem.contains_row(&addr) {
            self.psum_mem.write_element(element).unwrap();
        // Otherwise, alloc in cache.
        } else {
            // Update priority status.
            self.valid_pq_row_dict
                .entry(addr)
                .and_modify(|x| *x = max(*x, addr))
                .or_insert(addr);
            self.priority_queue_push([self.valid_pq_row_dict[&addr], addr]);
            // Update occp.
            self.freeup_space(addr, element_size).unwrap();
            self.cur_num += element_size;
            if self.is_psum_row(addr) {
                self.psum_occp += element_size;
                trace_println!(
                    "*psum append element: + {} -> {}",
                    element_size,
                    self.psum_occp
                );
            } else {
                self.b_occp += element_size;
            }
            // Update write_count.
            if self.track_count {
                self.write_count += element_size;
            }
            // Update data.
            let mut csrrow = CsrRow::new(element.idx[0]);
            csrrow.append_element(element);
            self.rowmap_insert(addr, csrrow);
        }

        if self.rowmap.contains_key(&addr) {
            trace_println!(
                "append psum {} -> {}",
                &addr,
                self.rowmap.get(&addr).unwrap().size()
            );
        }
    }

    pub fn print_psums(&self) {
        trace_println!("Total psum occp: {}", self.psum_occp);
        let mut psum_sum = 0;
        for (id, csrrow) in self.rowmap.iter() {
            if self.is_psum_row(*id) {
                trace_println!("psum {}: {}", id, csrrow.size());
                psum_sum += csrrow.size();
            }
        }
        trace_println!("");
        trace_println!("Psum sum: {}", psum_sum);
        assert!(
            self.psum_occp == psum_sum,
            "Unequal psum occp {}, {}",
            self.psum_occp,
            psum_sum
        );
    }
}
