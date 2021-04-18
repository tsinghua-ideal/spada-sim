use std::{fmt,};
use itertools::izip;


#[derive(Debug, Clone)]
pub struct StorageReadError(String);
impl fmt::Display for StorageReadError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "StorageReadError: {}", &self)
    }
}


#[derive(Debug, Clone)]
pub struct StorageWriteError(String);
impl fmt::Display for StorageWriteError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "StorageWriteError: {}", &self)
    }
}


pub struct Element {
    pub row_idx: usize,
    pub value: f64,
    pub col_idx: usize,
}


pub struct CsrRow {
    rowptr: usize,
    data: Vec<f64>,
    indptr: Vec<usize>,
}

impl CsrRow {
    pub fn as_element_vec(self) -> Vec<Element> {
        let mut result = vec![];
        for (d, col_idx) in izip!(self.data, self.indptr) {
            result.push(Element{
                row_idx: self.rowptr,
                value: d,
                col_idx: col_idx,
            });
        }
    
        return result;
    }
}


pub fn sorted_element_vec_to_csr_row(srt_ele_vec: Vec<Element>) -> CsrRow {
    let rowptr = srt_ele_vec[0].row_idx;
    let data = srt_ele_vec.iter().map(|e| e.value).collect::<Vec<f64>>();
    let indptr = srt_ele_vec.iter().map(|e| e.col_idx).collect::<Vec<usize>>();
    return CsrRow {rowptr, data, indptr};

}


pub trait StorageAPI {
    fn read(&mut self, row_ptr: usize, col_s: usize, ele_num: usize) -> Result<CsrRow, StorageReadError>;
    fn write(&mut self, rows: &mut Vec<CsrRow>) -> Result<Vec<usize>, StorageWriteError>;
}


pub struct Storage {
    data: Vec<f64>,
    indptr: Vec<usize>,
    indices: Vec<usize>,
    read_count: usize,
    write_count: usize,
}

impl StorageAPI for Storage {
    fn read(&mut self, row_ptr: usize, col_s: usize, ele_num: usize) 
            -> Result<CsrRow, StorageReadError> {
        if row_ptr >= self.indptr.len() {
            return Err(StorageReadError(format!("Invalid row_ptr: {}", row_ptr)));
        }
        let cur_row_pos = self.indptr[row_ptr];
        let end_row_pos = self.indptr[row_ptr+1];
        let s = cur_row_pos + col_s;
        let t = s + ele_num;
        if (s <= t) && (t < end_row_pos) {
            self.read_count += 2 * (t - s) + 1;
            return Ok(CsrRow {
                rowptr: row_ptr,
                data: self.data[s..t].to_vec(),
                indptr: self.indptr[s..t].to_vec(),
            })
        } else {
            return Err(StorageReadError(format!("Invalid col_pos: {}..{} with end_row_pos {}",
                s, t, end_row_pos)));
        }
    }

    fn write(&mut self, rows: &mut Vec<CsrRow>) -> Result<Vec<usize>, StorageWriteError> {
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
