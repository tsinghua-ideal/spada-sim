mod components;
mod gemm;
mod pipeline_simu;
mod py2rust;
mod storage;
mod storage_traffic_model;
mod frontend;

use std::cmp::min;

use gemm::GEMM;
use storage::VectorStorage;
use storage_traffic_model::TrafficModel;

use crate::components::StreamBuffer;
use crate::pipeline_simu::PipelineSimulator;
use crate::py2rust::load_pickled_gemms;
use crate::storage::CsrMatStorage;
use structopt::StructOpt;
use crate::frontend::{parse_config, Cli, Simulator, Accelerator, WorkloadCate};

fn main() {
    let omega_config = parse_config("omega_config.json").unwrap();
    let cli: Cli = Cli::from_args();

    match cli.simulator {
        Simulator::TrafficModel => {
            let gemm_fp = match cli.category {
                WorkloadCate::NN => omega_config.nn_filepath,
                WorkloadCate::SS => omega_config.ss_filepath,
            };
            let gemm = load_pickled_gemms(&gemm_fp, &cli.workload).unwrap();
            println!("Get GEMM {}", gemm.name);
            println!("{}", &gemm);
            println!("Avg row len of A: {}, Avg row len of B: {}", gemm.a.nnz() / gemm.a.rows(), gemm.b.nnz() / gemm.b.rows());

            let validating_product_mat = (&gemm.a * &gemm.b).to_csr();

            let (mut dram_a, mut dram_b) = CsrMatStorage::init_with_gemm(gemm);
            let mut dram_psum = VectorStorage::new();

            let output_base_addr = dram_b.indptr.len();
            let mut traffic_model = TrafficModel::new(
                4,
                8,
                3 * 1024 * 1024,
                8,
                output_base_addr,
                &mut dram_a,
                &mut dram_b,
                &mut dram_psum,
                cli.accelerator.clone(),
            );

            traffic_model.execute();

            let result = traffic_model.get_result();
            let a_count = traffic_model.get_a_mat_stat();
            let b_count = traffic_model.get_b_mat_stat();
            let c_count = traffic_model.get_c_mat_stat();

            println!("-----Result-----");
            println!("-----Access count");
            println!("A matrix count: read {} write {}", a_count.0, a_count.1);
            println!("B matrix count: read {} write {}", b_count.0, b_count.1);
            println!("C matrix count: read {} write {}", c_count.0, c_count.1);

            println!("-----Output product matrix");
            for idx in 0..min(result.len(), 10) {
                println!("{}", &result[idx]);
            }

            println!("----Validating output product matrix");
            let v_indptr = validating_product_mat.indptr().as_slice().unwrap().to_vec();
            let v_data = validating_product_mat.data().to_vec();
            let v_indices = validating_product_mat.indices().to_vec();

            for idx in 0..min(v_indptr.len()-1, 10) {
                let sliced_len = min(v_indptr[idx+1] - v_indptr[idx], 5);
                let sliced_indptr = &v_indices[v_indptr[idx]..v_indptr[idx]+sliced_len];
                let sliced_data = &v_data[v_indptr[idx]..v_indptr[idx]+sliced_len];
                println!("rowptr: {} indptr: {:?} data: {:?}", &idx, sliced_indptr, sliced_data);
            }
        },

        Simulator::AccurateSimu => {
            // Cycle-accurate simulator.
            // TODO: Write the blocking mechanism.
            // TODO: Initialize the StreamBuffer component.
            // let mut omega = PipelineSimulator::new();
            // TODO: Add StreamBuffer to omega.
        }
    }
}


pub fn print_type_of<T>(_: &T) {
    println!("{}", std::any::type_name::<T>())
}
