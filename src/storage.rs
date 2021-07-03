use fmt::write;
use itertools::{Itertools, izip};
use pyo3::ffi::PyCodec_StrictErrors;
use std::{cmp::{max, min}, collections::{HashMap, VecDeque}, fmt, hash::Hash, mem, ops::Index, usize};
use crate::gemm::GEMM;

#[derive(Debug, Clone)]
pub enum StorageError {
    WriteError(String),
    ReadEmptyRowError(String),
    ReadOverBoundError(String),
}

impl fmt::Display for StorageError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", &self)
    }
}

pub struct Element {
    pub row_idx: usize,
    pub value: f64,
    pub col_idx: usize,
}

#[derive(Debug, Clone)]
pub struct CsrRow {
    pub rowptr: usize,
    pub data: Vec<f64>,
    pub indptr: Vec<usize>,
}

impl CsrRow {
    pub fn new(rowptr: usize) -> CsrRow {
        CsrRow {
            rowptr: rowptr,
            data: vec![],
            indptr: vec![],
        }
    }

    pub fn as_element_vec(self) -> Vec<Element> {
        let mut result = vec![];
        for (d, col_idx) in izip!(self.data, self.indptr) {
            result.push(Element {
                row_idx: self.rowptr,
                value: d,
                col_idx: col_idx,
            });
        }

        return result;
    }

    pub fn size(&self) -> usize {
        return self.data.len() + self.indptr.len();
    }

    pub fn enumerate(&self) -> impl Iterator<Item = (&usize, &f64)> {
        return self.indptr.iter().zip(self.data.iter());
    }

    pub fn len(&self) -> usize {
        return self.indptr.len();
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
    let rowptr = srt_ele_vec[0].row_idx;
    let data = srt_ele_vec.iter().map(|e| e.value).collect::<Vec<f64>>();
    let indptr = srt_ele_vec
        .iter()
        .map(|e| e.col_idx)
        .collect::<Vec<usize>>();
    return CsrRow {
        rowptr,
        data,
        indptr,
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

pub trait Snapshotable {
    fn take_snapshot(&mut self);
    fn drop_snapshot(&mut self);
    fn restore_from_snapshot(&mut self);
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
    snapshot: Option<(usize, usize)>
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
            let csrrow = CsrRow {
                rowptr: rawp,
                data: self.data[s..t].to_vec(),
                indptr: self.indices[s..t].to_vec(),
            };
            if self.track_count { self.read_count += csrrow.size(); }
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
            if self.track_count { self.write_count += 2 * row.data.len() + 1; }
            self.data.extend(row.data.iter());
            self.indices.extend(row.indptr.iter());
            self.indptr.insert(self.indptr.len() - 1, indptr);
            *self.indptr.last_mut().unwrap() = self.data.len();
        }
        Ok(indptrs)
    }
}

impl Snapshotable for CsrMatStorage {
    fn take_snapshot(&mut self) {
        // Since currently the A & B matrices are read-only, we can simply dump the count_metrics.
        self.snapshot = Some((self.read_count, self.write_count));
    }

    fn drop_snapshot(&mut self) {
        // To drop the snapshot we simply turn it into None.
        self.snapshot = None;
    }

    fn restore_from_snapshot(&mut self) {
        match self.snapshot {
            Some(ref snp) => {
                self.read_count = snp.0;
                self.write_count = snp.1;
            },
            None => {
                panic!("No snapshot to be restored!");
            }
        }
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
                snapshot: None,
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
                snapshot: None,
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

    pub fn get_rowptr(&self, rowid: usize) -> usize {
        if self.remapped {
            return self.indptr[self.row_remap[&rowid]];
        } else {
            return self.indptr[rowid];
        }
    }

    pub fn get_row_len(&self) -> usize {
        self.indptr.len() - 1
    }

    pub fn get_nonzero(&self) -> usize {
        self.indices.len()
    }
}

pub struct VectorStorage {
    pub data: HashMap<usize, CsrRow>,
    pub read_count: usize,
    pub write_count: usize,
    pub track_count: bool,
    snapshot: Option<(Vec<usize>, usize, usize)>
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
                let cur_row_pos = csrrow.indptr[row_ptr];
                let end_row_pos = csrrow.indptr[row_ptr + 1];
                if col_s + ele_num <= csrrow.data.len() {
                    if self.track_count { self.read_count += csrrow.size(); }
                    return Ok(CsrRow {
                        rowptr: csrrow.rowptr,
                        data: csrrow.data[col_s..col_s + ele_num].to_vec(),
                        indptr: csrrow.indptr[col_s..col_s + ele_num].to_vec(),
                    });
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
            self.data.insert(indptr, row.clone());
            if self.track_count { self.write_count += row.size(); }
        }

        return Ok(indptrs);
    }
}

impl Snapshotable for VectorStorage {
    fn take_snapshot(&mut self) {
        // Since C matrix may be written, we need to track the item added to the Hashmap.
        self.snapshot = Some((
            self.data.keys().map(|x| *x).collect_vec(),
            self.read_count,
            self.write_count,
        ));
    }

    fn drop_snapshot(&mut self) {
        // To drop the snapshot we simply turn it into None.
        self.snapshot = None;
    }

    fn restore_from_snapshot(&mut self) {
        match self.snapshot {
            Some(ref snp) => {
                self.data.retain(|k, _| snp.0.contains(k));
                self.read_count = snp.1;
                self.write_count = snp.2;
            }
            None => {
                panic!("No snapshot to be restored!");
            }
        }
    }
}

impl VectorStorage {
    pub fn new() -> VectorStorage {
        VectorStorage {
            data: HashMap::new(),
            read_count: 0,
            write_count: 0,
            track_count: true,
            snapshot: None,
        }
    }

    pub fn read_row(&mut self, row_ptr: usize) -> Result<CsrRow, StorageError> {
        match self.data.get(&row_ptr) {
            Some(csrrow) => {
                if self.track_count { self.read_count += csrrow.size(); }
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
}

pub struct LRUCache<'a> {
    pub cache_size: usize,
    pub word_byte: usize,
    pub capability: usize,
    pub cur_num: usize,
    pub read_count: usize,
    pub write_count: usize,
    pub rowmap: HashMap<usize, CsrRow>,
    pub lru_queue: VecDeque<usize>,
    pub output_base_addr: usize,
    pub b_mem: &'a mut CsrMatStorage,
    pub psum_mem: &'a mut VectorStorage,
    pub miss_count: usize,
    pub b_evict_count: usize,
    pub psum_evict_count: usize,
    pub b_occp: usize,
    pub psum_occp: usize,
    pub track_count: bool,
    snapshot: Option<CacheSnapshot>,
}

impl<'a> Snapshotable for LRUCache<'a> {
    fn take_snapshot(&mut self) {
        // First dump cache's info, then each mem dump its own info.
        self.snapshot = Some(CacheSnapshot {
            cur_num: self.cur_num,
            read_count: self.read_count,
            write_count: self.write_count,
            rowmap: self.rowmap.clone(),
            lru_queue: self.lru_queue.clone(),
            output_base_addr: self.output_base_addr,
            miss_count: self.miss_count,
            b_evict_count: self.b_evict_count,
            psum_evict_count: self.psum_evict_count,
            b_occp: self.b_occp,
            psum_occp: self.psum_occp,
            rowmap_inc: vec![],
        });
        self.b_mem.take_snapshot();
        self.psum_mem.take_snapshot();
    }

    fn drop_snapshot(&mut self) {
        self.snapshot = None;
        self.b_mem.drop_snapshot();
        self.psum_mem.drop_snapshot();
    }

    fn restore_from_snapshot(&mut self) {
        match self.snapshot {
            Some(ref snp) => {
                self.cur_num = snp.cur_num;
                self.read_count = snp.read_count;
                self.write_count = snp.write_count;
                self.rowmap = snp.rowmap.clone();
                self.lru_queue = snp.lru_queue.clone();
                self.output_base_addr = snp.output_base_addr;
                self.miss_count = snp.miss_count;
                self.b_evict_count = snp.b_evict_count;
                self.psum_evict_count = snp.psum_evict_count;
                self.b_occp = snp.b_occp;
                self.psum_occp = snp.psum_occp;
            },
            None => {
                panic!("No snapshot to be restored!");
            }
        }
        self.b_mem.restore_from_snapshot();
        self.psum_mem.restore_from_snapshot();
    }
}

impl<'a> LRUCache<'a> {
    pub fn new(
        cache_size: usize,
        word_byte: usize,
        output_base_addr: usize,
        b_mem: &'a mut CsrMatStorage,
        psum_mem: &'a mut VectorStorage,
    ) -> LRUCache<'a> {
        LRUCache {
            cache_size: cache_size,
            word_byte: word_byte,
            capability: cache_size / word_byte,
            cur_num: 0,
            read_count: 0,
            write_count: 0,
            rowmap: HashMap::new(),
            lru_queue: VecDeque::new(),
            output_base_addr: output_base_addr,
            b_mem: b_mem,
            psum_mem: psum_mem,
            miss_count: 0,
            b_evict_count: 0,
            psum_evict_count: 0,
            b_occp: 0,
            psum_occp: 0,
            track_count: true,
            snapshot: None,
        }
    }

    fn rowmap_insert(&mut self, rowptr: usize, csrrow: CsrRow) {
        if let Some(ref mut snp) = self.snapshot {
            snp.rowmap_inc.push((rowptr, None));
        }
        self.rowmap.insert(rowptr, csrrow);
    }

    fn rowmap_remove(&mut self, rowid: &usize) -> Option<CsrRow> {
        let csrrow = self.rowmap.remove(rowid);
        if let Some(ref mut snp) = self.snapshot {
            if let Some(ref c) = csrrow {
                snp.rowmap_inc.push((*rowid, Some(c.clone())));
            }
        }
        return csrrow;
    }

    pub fn write(&mut self, csrrow: CsrRow) {
        let row_size = csrrow.size();
        // println!("*cache write invoked with count {} row {}", self.write_count, row_size);
        if self.is_psum_row(csrrow.rowptr) {
            self.psum_occp += row_size;
        } else {
            self.b_occp += row_size;
        }

        if self.cur_num + row_size <= self.capability {
            self.cur_num += row_size;
            self.lru_queue.push_back(csrrow.rowptr);
            if self.track_count { self.write_count += row_size; }
            self.rowmap_insert(csrrow.rowptr, csrrow);
        } else {
            if let Err(err) = self.freeup_space(row_size) {
                panic!("{}", err);
            }
            self.cur_num += row_size;
            self.lru_queue.push_back(csrrow.rowptr);
            if self.track_count { self.write_count += row_size; }
            self.rowmap_insert(csrrow.rowptr, csrrow);
        }
    }

    pub fn freeup_space(&mut self, space_required: usize) -> Result<(), String> {
        while self.lru_queue.len() > 0 && (self.cur_num + space_required > self.capability) {
            let mut popid: usize;
            loop {
                popid = self.lru_queue.pop_front().unwrap();
                if self.rowmap.contains_key(&popid) {
                    break;
                }
            }
            if self.is_psum_row(popid) {
                let popped_csrrow = self.rowmap_remove(&popid).unwrap();
                println!("*freerow {} and get {}", popid, popped_csrrow.size());
                self.cur_num -= popped_csrrow.size();
                if self.track_count { self.psum_evict_count += popped_csrrow.size(); }
                self.psum_occp -= popped_csrrow.size();
                self.psum_mem.write(&mut vec![popped_csrrow]).unwrap();
            } else {
                let evict_size = self.rowmap_remove(&popid).unwrap().size();
                println!("*freerow {} and get {}", popid, evict_size);
                self.cur_num -= evict_size;
                self.b_occp -= evict_size;
                if self.track_count { self.b_evict_count += evict_size; }
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

    pub fn freeup_row(&mut self, rowid: usize) -> Result<CsrRow, String> {
        if self.rowmap.contains_key(&rowid) {
            let removed_row = self.rowmap_remove(&rowid).unwrap();
            self.cur_num -= removed_row.size();
            if self.is_psum_row(rowid) {
                self.psum_occp -= removed_row.size();
            } else {
                self.b_occp -= removed_row.size();
            }
            return Ok(removed_row);
        } else {
            return Err(format!("freeup_row: row {} not found", rowid));
        }
    }

    pub fn read_cache(&mut self, rowid: usize) -> Option<CsrRow> {
        if self.rowmap.contains_key(&rowid) {
            // self.lru_queue
            //     .remove(self.lru_queue.iter().position(|&x| x == rowid).unwrap());
            if let Some(pos) = self.lru_queue.iter().position(|&x| x == rowid) {
                self.lru_queue.remove(pos);
            }
            self.lru_queue.push_back(rowid);
            let csrrow = self.rowmap.get(&rowid).unwrap().clone();
            if self.track_count { self.read_count += csrrow.size(); }
            return Some(csrrow);
        } else {
            return None;
        }
    }

    pub fn consume(&mut self, rowid: usize) -> Option<CsrRow> {
        match self.freeup_row(rowid) {
            Ok(csrrow) => {
                self.read_count += csrrow.size();
                Some(csrrow)
            }
            Err(_) => {
                if self.is_psum_row(rowid) {
                    match self.psum_mem.read_row(rowid) {
                        Ok(csrrow) => {
                            if self.track_count {
                                self.read_count += csrrow.size();
                                self.miss_count += csrrow.size();
                            }
                            Some(csrrow)
                        }
                        Err(_) => None,
                    }
                } else {
                    match self.b_mem.read_row(rowid) {
                        Ok(csrrow) => {
                            if self.track_count {
                                self.read_count += csrrow.size();
                                self.miss_count += csrrow.size();
                            }
                            Some(csrrow)
                        }
                        Err(_) => None,
                    }
                }
            }
        }
    }

    pub fn read(&mut self, rowid: usize) -> Option<CsrRow> {
        match self.read_cache(rowid) {
            Some(csrrow) => {
                // if self.track_count { self.read_count += csrrow.size(); }
                Some(csrrow)
            }
            None => {
                if self.is_psum_row(rowid) {
                    match self.psum_mem.read_row(rowid) {
                        Ok(csrrow) => {
                            if self.track_count {
                                self.read_count += csrrow.size();
                                self.miss_count += csrrow.size();
                            }
                            self.write(csrrow.clone());
                            Some(csrrow)
                        }
                        Err(_) => None,
                    }
                } else {
                    match self.b_mem.read_row(rowid) {
                        Ok(csrrow) => {
                            if self.track_count {
                                self.read_count += csrrow.size();
                                self.miss_count += csrrow.size();
                            }
                            self.write(csrrow.clone());
                            Some(csrrow)
                        }
                        Err(_) => None,
                    }
                }
            }
        }
    }

    pub fn swapout(&mut self, rowid: usize) {
        if self.rowmap.contains_key(&rowid) {
            let popped_csrrow = self.rowmap_remove(&rowid).unwrap();
            self.cur_num -= popped_csrrow.size();
            if self.is_psum_row(rowid) {
                self.psum_occp -= popped_csrrow.size();
            } else {
                self.b_occp -= popped_csrrow.size();
            }
            self.psum_mem.write(&mut vec![popped_csrrow]).unwrap();
        }
    }

    pub fn is_psum_row(&self, rowid: usize) -> bool {
        return rowid >= self.output_base_addr;
    }

    /// Partially restore the statistic data without recovering the queue.
    pub fn partial_restore_from_snapshot(&mut self) {
        match self.snapshot {
            Some(ref mut snp) => {
                for (rowid, csrrow) in snp.rowmap_inc.drain(..) {
                    // If csrrow is None then it is an add operation; Otherwise it is a deletion.
                    if let Some(c) = csrrow {
                        self.rowmap.insert(rowid, c);
                    } else {
                        self.rowmap.remove(&rowid);
                    }
                }
                self.cur_num = snp.cur_num;
                self.read_count = snp.read_count;
                self.write_count = snp.write_count;
                self.output_base_addr = snp.output_base_addr;
                self.miss_count = snp.miss_count;
                self.b_evict_count = snp.b_evict_count;
                self.psum_evict_count = snp.psum_evict_count;
                self.b_occp = snp.b_occp;
                self.psum_occp = snp.psum_occp;
            },
            None => {
                panic!("No snapshot to be restored!");
            }
        }
        self.b_mem.restore_from_snapshot();
        self.psum_mem.restore_from_snapshot();
    }

    /// Restore data directly from using the snapshot data without cloning.
    pub fn restore_and_drop_snapshot(&mut self) {
        match self.snapshot {
            Some(ref mut snp) => {
                self.cur_num = snp.cur_num;
                self.read_count = snp.read_count;
                self.write_count = snp.write_count;
                // self.rowmap = snp.rowmap.clone();
                self.rowmap = mem::replace(&mut snp.rowmap, HashMap::new());
                // self.lru_queue = snp.lru_queue.clone();
                self.lru_queue = mem::replace(&mut snp.lru_queue, VecDeque::new());
                self.output_base_addr = snp.output_base_addr;
                self.miss_count = snp.miss_count;
                self.b_evict_count = snp.b_evict_count;
                self.psum_evict_count = snp.psum_evict_count;
                self.b_occp = snp.b_occp;
                self.psum_occp = snp.psum_occp;
            },
            None => {
                panic!("No snapshot to be restored!");
            }
        }
        self.b_mem.restore_from_snapshot();
        self.psum_mem.restore_from_snapshot();
    }
}

struct CacheSnapshot {
    pub cur_num: usize,
    pub read_count: usize,
    pub write_count: usize,
    pub rowmap: HashMap<usize, CsrRow>,
    pub lru_queue: VecDeque<usize>,
    pub output_base_addr: usize,
    pub miss_count: usize,
    pub b_evict_count: usize,
    pub psum_evict_count: usize,
    pub b_occp: usize,
    pub psum_occp: usize,
    pub rowmap_inc: Vec<(usize, Option<CsrRow>)>,
}
