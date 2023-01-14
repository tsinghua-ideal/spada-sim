#![feature(drain_filter)]
#![feature(hash_drain_filter)]

mod adder_tree;
mod block_topo_tracker;
mod colwise_irr_adjust;
mod colwise_reg_adjust;
mod frontend;
mod gemm;
mod preprocessing;
mod py2rust;
mod rowwise_adjust;
mod rowwise_perf_adjust;
mod scheduler;
mod simulator;
mod storage;
mod util;

use std::cmp::min;

use gemm::GEMM;

use crate::frontend::{parse_config, Accelerator, Cli, Mode, WorkloadCate};
use crate::preprocessing::sort_by_length;
use crate::py2rust::{load_mm_mat, load_pickled_gemms};
use crate::simulator::Simulator;
use crate::storage::{CsrMatStorage, VectorStorage};
use structopt::StructOpt;

fn main() {
    let cli: Cli = Cli::from_args();
    let spada_config = parse_config(&cli.configuration).unwrap();
    let gemm: GEMM;
    match cli.category {
        WorkloadCate::NN => {
            gemm = load_pickled_gemms(&spada_config.nn_filepath, &cli.workload).unwrap();
        }
        WorkloadCate::SS => {
            let mat = load_mm_mat(&spada_config.ss_filepath, &cli.workload).unwrap();
            gemm = GEMM::from_mat(&cli.workload, mat);
        }
    };

    let a_avg_row_len = gemm.a.nnz() / gemm.a.rows();
    let b_avg_row_len = gemm.b.nnz() / gemm.b.rows();
    println!("Get GEMM {}", gemm.name);
    println!("{}", &gemm);
    println!(
        "Avg row len of A: {}, Avg row len of B: {}",
        a_avg_row_len, b_avg_row_len
    );

    match cli.simulator {
        Mode::AccurateSimu => {
            // Cycle-accurate simulator.
            let (mut dram_a, mut dram_b) = CsrMatStorage::init_with_gemm(gemm);
            let mut dram_psum = VectorStorage::new();

            // Preprocessing.
            if cli.preprocess {
                let rowmap = sort_by_length(&mut dram_a);
                dram_a.reorder_row(rowmap);
            }

            let output_base_addr = dram_b.indptr.len();
            // Determine the default window & block shape.
            let default_block_shape = match cli.accelerator {
                Accelerator::Ip => spada_config.block_shape,
                Accelerator::MultiRow => [spada_config.block_shape[0], spada_config.block_shape[1]],
                Accelerator::Op => [spada_config.lane_num, 1],
                Accelerator::Spada => spada_config.block_shape,
            };

            let mut cycle_simu = Simulator::new(
                spada_config.pe_num,
                spada_config.at_num,
                spada_config.lane_num,
                spada_config.cache_size,
                spada_config.word_byte,
                output_base_addr,
                default_block_shape,
                &mut dram_a,
                &mut dram_b,
                &mut dram_psum,
                cli.accelerator.clone(),
                spada_config.mem_latency,
                spada_config.cache_latency,
                spada_config.freq,
                spada_config.channel,
                spada_config.bandwidth_per_channel,
            );

            cycle_simu.execute();

            let result = cycle_simu.get_exec_result();
            let a_count = cycle_simu.get_a_mat_stat();
            let b_count = cycle_simu.get_b_mat_stat();
            let c_count = cycle_simu.get_c_mat_stat();
            let exec_count = cycle_simu.get_exec_cycle();
            let cache_count = cycle_simu.get_cache_stat();

            println!("-----Result-----");
            println!("-----Access count");
            println!("Execution count: {}", exec_count);
            println!("A matrix count: read {} write {}", a_count[0], a_count[1]);
            println!("B matrix count: read {} write {}", b_count[0], b_count[1]);
            println!("C matrix count: read {} write {}", c_count[0], c_count[1]);
            println!(
                "Cache count: read {} write {}",
                cache_count[0], cache_count[1]
            );

            println!("-----Output product matrix");
            for idx in 0..min(result.len(), 10) {
                println!("{}", &result[idx]);
            }
        }

        _ => panic!("Unimplemented simulator {}", cli.simulator),
    }
}
