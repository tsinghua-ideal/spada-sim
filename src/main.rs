mod gemm;
mod py2rust;
mod storage;
mod pipeline_simu;
mod components;

use crate::py2rust::load_pickled_gemms;
use crate::storage::Storage;
use crate::pipeline_simu::PipelineSimulator;
use crate::components::{StreamBuffer};

fn main() {
    let gemm_filepath = "../adaptive_sparse_accelerator/workload/
        pickled_gemms/SS_gemms_01_27_20_37.pkl";
    let gemm_name = "2cubes_sphere";
    let gemm = load_pickled_gemms(gemm_filepath, gemm_name).unwrap();

    // TODO: Write the blocking mechanism.
    // TODO: Initialize the storage.
    // TODO: Initialize the StreamBuffer component.

    let mut omega = PipelineSimulator::new();
    // TODO: Add StreamBuffer to omega.
    println!("Get GEMM {}", gemm.name);
    println!("{}", &gemm);
}

struct OmegaSimu {
    memory: Storage,
    cache: Storage,
}
