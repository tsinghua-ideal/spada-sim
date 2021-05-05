mod components;
mod gemm;
mod pipeline_simu;
mod py2rust;
mod storage;
mod storage_traffic_model;

use std::cmp::min;

use gemm::GEMM;
use storage::VectorStorage;
use storage_traffic_model::OmegaTraffic;

use crate::components::StreamBuffer;
use crate::pipeline_simu::PipelineSimulator;
use crate::py2rust::load_pickled_gemms;
use crate::storage::CsrMatStorage;

fn main() {
    let gemm_filepath = "../adaptive_sparse_accelerator/workload/pickled_gemms/SS_gemms_01_27_20_37.pkl";
    let gemm_name = "2cubes_sphere";
    let gemm = load_pickled_gemms(gemm_filepath, gemm_name).unwrap();
    println!("Get GEMM {}", gemm.name);
    println!("{}", &gemm);

    let validating_product_mat = (&gemm.a * &gemm.b).to_csr();

    let (mut dram_a, mut dram_b) = CsrMatStorage::init_with_gemm(gemm);
    let mut dram_psum = VectorStorage::new();
    
    let output_base_addr = dram_b.indptr.len();
    let mut omega_traffic_simu = OmegaTraffic::new(
        4,
        8,
        3 * 1024 * 1024,
        8,
        output_base_addr,
        &mut dram_a,
        &mut dram_b,
        &mut dram_psum,
    );
    omega_traffic_simu.execute();
    let result = omega_traffic_simu.get_result();
    let a_count = omega_traffic_simu.get_a_mat_stat();
    let b_count = omega_traffic_simu.get_b_mat_stat();
    let c_count = omega_traffic_simu.get_c_mat_stat();

    println!("-----Result-----");
    println!("-----Access count");
    println!("A matrix count: read {} write {}", a_count.0, a_count.1);
    println!("B matrix count: read {} write {}", b_count.0, b_count.1);
    println!("C matrix count: read {} write {}", c_count.0, c_count.1);


    println!("-----Output product matrix");
    for idx in 0..min(result.len(), 5) {
        println!("{}", &result[idx]);
    }

    println!("----Validating output product matrix");
    let v_indptr = validating_product_mat.indptr().as_slice().unwrap().to_vec();
    let v_data = validating_product_mat.data().to_vec();
    let v_indices = validating_product_mat.indices().to_vec();
    
    for idx in 0..min(v_indptr.len()-1, 5) {
        let sliced_len = min(v_indptr[idx+1] - v_indptr[idx], 5);
        let sliced_indptr = &v_indices[v_indptr[idx]..v_indptr[idx]+sliced_len];
        let sliced_data = &v_data[v_indptr[idx]..v_indptr[idx]+sliced_len];
        println!("rowptr: {} indptr: {:?} data: {:?}", &idx, sliced_indptr, sliced_data);
    }

    // Storage-traffic simulator.
    
    // Cycle-accurate simulator.
    // TODO: Write the blocking mechanism.
    // TODO: Initialize the StreamBuffer component.
    // let mut omega = PipelineSimulator::new();
    // TODO: Add StreamBuffer to omega.
}
