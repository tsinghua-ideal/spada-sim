use std::{fmt,};
use std::rc::Rc;

#[derive(Debug, Clone)]
struct StorageReadError(String);
impl fmt::Display for StorageReadError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "StorageReadError: {}", &self)
    }
}


#[derive(Debug, Clone)]
struct StorageWriteError(String);
impl fmt::Display for StorageWriteError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "StorageWriteError: {}", &self)
    }
}


struct CsrRow {
    rowptr: usize,
    data: Vec<f64>,
    indptr: Vec<usize>,
}


trait StorageAPI {
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
            self.data.append(&mut row.data);
            self.indices.append(&mut row.indptr);
            self.indptr.insert(self.indptr.len() - 1, indptr);
            *self.indptr.last_mut().unwrap() = self.data.len();
        }
        Ok(indptrs)
    }
}
