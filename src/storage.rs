use fmt::write;
use itertools::izip;
use std::{cmp::{max, min}, collections::{HashMap, VecDeque}, fmt, ops::Index};

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
        write!(f, "rowptr: {} indptr: {:?} data: {:?}", self.rowptr, &self.indptr[0..display_len], &self.data[0..display_len])
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

pub struct CsrMatStorage {
    pub data: Vec<f64>,
    pub indptr: Vec<usize>,
    pub indices: Vec<usize>,
    pub read_count: usize,
    pub write_count: usize,
}

impl StorageAPI for CsrMatStorage {
    fn read(
        &mut self,
        row_ptr: usize,
        col_s: usize,
        ele_num: usize,
    ) -> Result<CsrRow, StorageError> {
        if row_ptr >= self.indptr.len() {
            return Err(StorageError::ReadOverBoundError(format!(
                "Invalid row_ptr: {}",
                row_ptr
            )));
        }
        let cur_row_pos = self.indptr[row_ptr];
        let end_row_pos = self.indptr[row_ptr + 1];
        let s = cur_row_pos + col_s;
        let t = s + ele_num;
        if (s <= t) && (t <= end_row_pos) {
            self.read_count += 2 * (t - s) + 1;
            return Ok(CsrRow {
                rowptr: row_ptr,
                data: self.data[s..t].to_vec(),
                indptr: self.indices[s..t].to_vec(),
            });
        } else {
            return Err(StorageError::ReadEmptyRowError(format!(
                "Invalid col_pos: {}..{} with end_row_pos {} for row {}.",
                s, t, end_row_pos, row_ptr
            )));
        }
    }

    fn write(&mut self, rows: &mut Vec<CsrRow>) -> Result<Vec<usize>, StorageError> {
        let mut indptrs = vec![];
        for row in rows.iter_mut() {
            let indptr = self.data.len();
            indptrs.push(indptr);
            self.write_count += 2 * row.data.len() + 1;
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
            },
            CsrMatStorage {
                data: gemm.b.data().to_vec(),
                indptr: gemm.b.indptr().as_slice().unwrap().to_vec(),
                indices: gemm.b.indices().to_vec(),
                read_count: 0,
                write_count: 0,
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
}


pub struct VectorStorage {
    pub data: HashMap<usize, CsrRow>,
    pub read_count: usize,
    pub write_count: usize,
}

impl StorageAPI for VectorStorage {
    fn read(&mut self, row_ptr: usize, col_s: usize, ele_num: usize) -> Result<CsrRow, StorageError> {
        match self.data.get(&row_ptr) {
            Some(csrrow) => {
                let cur_row_pos = csrrow.indptr[row_ptr];
                let end_row_pos = csrrow.indptr[row_ptr + 1];
                if col_s + ele_num <= csrrow.data.len() {
                    self.read_count += csrrow.size();
                    return Ok(CsrRow {
                        rowptr: csrrow.rowptr,
                        data: csrrow.data[col_s..col_s+ele_num].to_vec(),
                        indptr: csrrow.indptr[col_s..col_s+ele_num].to_vec(),
                    });
                } else {
                    return Err(StorageError::ReadEmptyRowError(format!(
                        "Invalid col_pos: {}..{} in row {}",
                        col_s, col_s + ele_num, csrrow.rowptr
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
            self.write_count += row.size();
        }

        return Ok(indptrs);
    }
}

impl VectorStorage {
    pub fn new() -> VectorStorage {
        VectorStorage{
            data: HashMap::new(),
            read_count: 0,
            write_count: 0,
        }
    }

    pub fn read_row(&mut self, row_ptr: usize) -> Result<CsrRow, StorageError> {
        match self.data.get(&row_ptr) {
            Some(csrrow) => {
                self.read_count += csrrow.size();
                return Ok(csrrow.clone());
            },
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
    pub rowmap: HashMap<usize, CsrRow>,
    pub lru_queue: VecDeque<usize>,
    pub output_base_addr: usize,
    pub b_mem: &'a mut CsrMatStorage,
    pub psum_mem: &'a mut VectorStorage,
}

impl<'a> LRUCache<'a> {
    pub fn new(cache_size: usize, word_byte: usize, output_base_addr: usize, B_mem: &'a mut CsrMatStorage, psum_mem: &'a mut VectorStorage) -> LRUCache<'a> {
        LRUCache {
            cache_size: cache_size,
            word_byte: word_byte,
            capability: cache_size / word_byte,
            cur_num: 0,
            rowmap: HashMap::new(),
            lru_queue: VecDeque::new(),
            output_base_addr: output_base_addr,
            b_mem: B_mem,
            psum_mem: psum_mem,
        }
    }

    pub fn write(&mut self, csrrow: CsrRow) {
        let num = csrrow.size();
        if self.cur_num + num <= self.capability {
            self.cur_num += num;
            self.lru_queue.push_back(csrrow.rowptr);
            self.rowmap.insert(csrrow.rowptr, csrrow);
        } else {
            if let Err(err) = self.freeup_space(self.cur_num + num - self.capability) {
                panic!("{}", err);
            }
            self.cur_num += num;
            self.lru_queue.push_back(csrrow.rowptr);
            self.rowmap.insert(csrrow.rowptr, csrrow);
        }
    }

    pub fn debug_write(&mut self, csrrow: CsrRow) {
        let num = csrrow.size();
        if self.cur_num + num <= self.capability {
            self.cur_num += num;
            self.lru_queue.push_back(csrrow.rowptr);
            println!("Insert {} into cache.", csrrow.rowptr);
            self.rowmap.insert(csrrow.rowptr.clone(), csrrow);
        } else {
            if let Err(err) = self.freeup_space(self.cur_num + num - self.capability) {
                panic!("{}", err);
            }
            self.cur_num += num;
            self.lru_queue.push_back(csrrow.rowptr);
            println!("Insert {} into cache.", csrrow.rowptr);
            self.rowmap.insert(csrrow.rowptr, csrrow);
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
                let popped_csrrow = self.rowmap.remove(&popid).unwrap();
                self.cur_num -= popped_csrrow.size();
                self.psum_mem.write(&mut vec![popped_csrrow,]).unwrap();
            } else {
                self.cur_num -= self.rowmap.remove(&popid).unwrap().size();
            }
        }
        if self.capability - self.cur_num < space_required {
            return Err(format!("Not enough space for {}", space_required));
        } else {
            return Ok(());
        }
    }

    pub fn read_cache(&mut self, rowid: usize) -> Option<CsrRow> {
        if self.rowmap.contains_key(&rowid) {
            self.lru_queue
                .remove(self.lru_queue.iter().position(|&x| x == rowid).unwrap());
            self.lru_queue.push_back(rowid);
            let csrrow = self.rowmap.get(&rowid).unwrap().clone();
            return Some(csrrow);
        } else {
            return None;
        }
    }

    pub fn consume(&mut self, rowid: usize) -> Option<CsrRow> {
        if self.rowmap.contains_key(&rowid) {
            let csrrow = self.rowmap.remove(&rowid).unwrap();
            self.cur_num -= csrrow.size();
            return Some(csrrow);
        } else {
            return None;
        }
    }

    pub fn read(&mut self, rowid: usize) -> Option<CsrRow> {
        match self.read_cache(rowid) {
            Some(csrrow) => Some(csrrow),
            None => {if self.is_psum_row(rowid) { match self.psum_mem.read_row(rowid) {
                    Ok(csrrow) => {
                        self.write(csrrow.clone());
                        Some(csrrow)
                    },
                    Err(_) => None,
                }}else { match self.b_mem.read_row(rowid) {
                    Ok(csrrow) => {
                        self.write(csrrow.clone());
                        Some(csrrow)
                    },
                    Err(_) => None,
            }}}
        }
    }

    pub fn debug_read(&mut self, rowid: usize) -> Option<CsrRow> {
        match self.read_cache(rowid) {
            Some(csrrow) => {
                println!("Get {} from cache", rowid);
                Some(csrrow)
            },
            None => {if self.is_psum_row(rowid) {
                println!("Read {} from psum memory.", rowid);
                match self.psum_mem.read_row(rowid) {
                    Ok(csrrow) => {
                        self.write(csrrow.clone());
                        Some(csrrow)
                    },
                    Err(_) => None,
                }}else {
                println!("Read {} from fiber memory.", rowid);
                match self.b_mem.read_row(rowid) {
                    Ok(csrrow) => {
                        self.write(csrrow.clone());
                        Some(csrrow)
                    },
                    Err(_) => None,
            }}}
        }
    }

    pub fn swapout(&mut self, rowid: usize) {
        if self.rowmap.contains_key(&rowid) {
            let popped_csrrow = self.rowmap.remove(&rowid).unwrap();
            self.cur_num -= popped_csrrow.size();
            self.psum_mem.write(&mut vec![popped_csrrow,]).unwrap();
        }
    }

    pub fn is_psum_row(&self, rowid: usize) -> bool {
        return rowid >= self.output_base_addr;
    }
}
