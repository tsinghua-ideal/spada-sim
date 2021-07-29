use pyo3::prelude::*;
use sprs::CsMat;
use std::cmp::min;
use std::fmt;

#[derive(FromPyObject, Debug)]
pub struct GEMMRawTuple(
    pub (usize, usize),
    pub Vec<usize>,
    pub Vec<usize>,
    pub Vec<f64>,
    pub (usize, usize),
    pub Vec<usize>,
    pub Vec<usize>,
    pub Vec<f64>,
);

#[derive(FromPyObject, Debug)]
pub struct CsrTuple(
    pub (usize, usize),
    pub Vec<usize>,
    pub Vec<usize>,
    pub Vec<f64>,
);

pub struct GEMM {
    pub name: String,
    pub a: CsMat<f64>,
    pub b: CsMat<f64>,
}

impl GEMM {
    pub fn new(gn: &str, grt: GEMMRawTuple) -> GEMM {
        GEMM {
            name: gn.to_owned(),
            a: CsMat::new(grt.0, grt.1, grt.2, grt.3),
            b: CsMat::new(grt.4, grt.5, grt.6, grt.7),
        }
    }

    pub fn from_mat(mn: &str, mat: CsMat<f64>) -> GEMM {
        // If the matrix is square, use A * A, otherwise A * AT.
        let b_mat = if mat.shape().0 == mat.shape().1 {
            mat.clone()
        } else {
            mat.clone().transpose_into().to_csr()
        };
        GEMM {
            name: mn.to_owned(),
            a: mat,
            b: b_mat,
        }
    }
}

impl fmt::Display for GEMM {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "---- {} ----\n", self.name)?;
        write!(f, "--A: {:?}\n", self.a.shape())?;
        write!(
            f,
            "data: {:?} .. \n",
            &self.a.data()[..min(self.a.data().len(), 5)]
        )?;
        write!(
            f,
            "indices: {:?} ...\n",
            &self.a.indices()[..min(self.a.indices().len(), 5)]
        )?;
        write!(
            f,
            "indptr: {:?} ...\n",
            &self.a.indptr().as_slice().unwrap()[..min(self.a.indptr().len(), 5)]
        )?;
        write!(f, "--B: {:?}\n", self.b.shape())?;
        write!(
            f,
            "data: {:?} ...\n",
            &self.a.data()[..min(self.b.data().len(), 5)]
        )?;
        write!(
            f,
            "indices: {:?} ...\n",
            &self.a.indices()[..min(self.b.indices().len(), 5)]
        )?;
        write!(
            f,
            "indptr: {:?} ...\n",
            &self.b.indptr().as_slice().unwrap()[..min(self.b.indptr().len(), 5)]
        )
    }
}
