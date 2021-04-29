mod components;
mod gemm;
mod pipeline_simu;
mod py2rust;
mod storage;
mod storage_traffic_model;

use gemm::GEMM;

use crate::components::StreamBuffer;
use crate::pipeline_simu::PipelineSimulator;
use crate::py2rust::load_pickled_gemms;
use crate::storage::Storage;

fn main() {
    let gemm_filepath = "../adaptive_sparse_accelerator/workload/
        pickled_gemms/SS_gemms_01_27_20_37.pkl";
    let gemm_name = "2cubes_sphere";
    let gemm = load_pickled_gemms(gemm_filepath, gemm_name).unwrap();
    println!("Get GEMM {}", gemm.name);
    println!("{}", &gemm);
    let (mut dram_a, mut dram_b) = Storage::init_with_gemm(gemm);

    // Storage-traffic simulator.

    // Cycle-accurate simulator.
    // TODO: Write the blocking mechanism.
    // TODO: Initialize the StreamBuffer component.
    // let mut omega = PipelineSimulator::new();
    // TODO: Add StreamBuffer to omega.
}
