use std::fmt;
use std::cmp::{min, max};
use numpy::PyArray1;
use pyo3::{buffer::ReadOnlyCell, prelude::{PyResult, Python}};
use pyo3::{prelude::*, types::{IntoPyDict, PyModule}};
use sprs::{CsMat};

fn main() {
    let gemm = load_pickled_gemms().unwrap();
    println!("Get GEMM {}", gemm.name);
    println!("{}", &gemm);
}


pub struct GEMM {
    name: String,
    a: CsMat<f64>,
    b: CsMat<f64>,
}

impl GEMM {
    fn new (gn: & str, csrt: CsrTuple) -> GEMM {
        GEMM {
            name: gn.to_owned(),
            a: CsMat::new(csrt.0, csrt.1, csrt.2, csrt.3),
            b: CsMat::new(csrt.4, csrt.5, csrt.6, csrt.7),
        }
    }
}

impl fmt::Display for GEMM {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "---- {} ----\n", self.name)?;
        write!(f, "--A: {:?}\n", self.a.shape())?;
        write!(f, "data: {:?} .. \n", &self.a.data()[..min(self.a.data().len(), 5)])?;
        write!(f, "indices: {:?} ...\n", &self.a.indices()[..min(self.a.indices().len(), 5)])?;
        write!(f, "indptr: {:?} ...\n", &self.a.indptr().as_slice()
            .unwrap()[..min(self.a.indptr().len(), 5)])?;
        write!(f, "--B: {:?}\n", self.b.shape())?;
        write!(f, "data: {:?} ...\n", &self.a.data()[..min(self.b.data().len(), 5)])?;
        write!(f, "indices: {:?} ...\n", &self.a.indices()[..min(self.b.indices().len(), 5)])?;
        write!(f, "indptr: {:?} ...\n", &self.b.indptr().as_slice()
            .unwrap()[..min(self.b.indptr().len(), 5)])
    }
}

#[derive(FromPyObject, Debug)]
struct CsrTuple((usize, usize), Vec<usize>, Vec<usize>, Vec<f64>,
                (usize, usize), Vec<usize>, Vec<usize>, Vec<f64>);

fn load_pickled_gemms() -> PyResult<GEMM> {
    let code = r#"
def retrieve_pickled_csr(pickle_gemm_fp, pickle_gemm_name):
    print('--- Python Interface ---')
    import pickle
    import numpy as np
    from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
    print(f'% Load {pickle_gemm_name} from', pickle_gemm_fp)
    with open(pickle_gemm_fp, 'rb') as f:
        gemms = pickle.load(f)

        A, B = gemms[pickle_gemm_name]
        if isinstance(A, (csc_matrix, coo_matrix)):
            A = A.tocsr()
        elif isinstance(A, (np.ndarray)):
            A = csr_matrix(A)
        else:
            raise TypeError('Unsupported matrix type: {}'.format(type(A)))

        if isinstance(B, (csc_matrix, coo_matrix)):
            B = B.tocsr()
        elif isinstance(B, (np.ndarray)):
            B = csr_matrix(B)
        else:
            raise TypeError('Unsupported matrix type: {}'.format(type(B)))

        shape_A, shape_B = A.shape, B.shape
        data_A, data_B = A.data, B.data
        indices_A, indices_B = A.indices, B.indices
        indptr_A, indptr_B = A.indptr, B.indptr

        print(f'% -- A --')
        print(f'% shape: {shape_A} data: {data_A[:5]}... indices: {indices_A[:5]}... indptr: {indptr_A[:5]}...')
        print(f'% shape: {shape_B} data: {data_B[:5]}... indices: {indices_B[:5]}... indptr: {indptr_B[:5]}...')
    print('--- Return from Python Interface ---\n')
    return (shape_A, indptr_A, indices_A, data_A, shape_B, indptr_B, indices_B, data_B)
    "#;

    let file_name = "retrieve_pickled_csr.py";
    let module_name = "retrieve_pickled_csr";

    let pickle_gemm_fp = "../adaptive_sparse_accelerator/workload/pickled_gemms/SS_gemms_01_27_20_37.pkl";
    let pickle_gemm_name = "2cubes_sphere";

    Python::with_gil(|py| {
        let load_gemm_from_path = PyModule::from_code(
            py, code, file_name, module_name).unwrap();
        let csr_tuple: CsrTuple = 
            load_gemm_from_path.getattr("retrieve_pickled_csr").unwrap()
            .call1((pickle_gemm_fp, pickle_gemm_name)).unwrap()
            .extract().unwrap();
        // println!("csr_tuple:\n {:#?}", csr_tuple);
        let gemm = GEMM::new(pickle_gemm_name, csr_tuple);
        Ok(gemm)
    })

}