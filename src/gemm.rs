use pyo3::prelude::*;
use sprs::CsMat;
use std::cmp::min;
use std::fmt;

#[derive(FromPyObject, Debug)]
pub struct CsrTuple(
    (usize, usize),
    Vec<usize>,
    Vec<usize>,
    Vec<f64>,
    (usize, usize),
    Vec<usize>,
    Vec<usize>,
    Vec<f64>,
);

pub struct GEMM {
    pub name: String,
    pub a: CsMat<f64>,
    pub b: CsMat<f64>,
}

impl GEMM {
    pub fn new(gn: &str, csrt: CsrTuple) -> GEMM {
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
