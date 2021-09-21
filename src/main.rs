#![feature(drain_filter)]

mod b_reuse_counter;
mod frontend;
mod gemm;
mod oracle_storage_traffic_model;
mod pipeline_simu;
mod pqcache_omega_simulator;
mod pqcache_storage_traffic_model;
mod preprocessing;
mod py2rust;
mod scheduler;
mod storage;
mod storage_traffic_model;
mod util;
mod new_scheduler;
mod new_pqcache_omega_simulator;

use std::cmp::min;

use gemm::GEMM;
use storage::VectorStorage;
use storage_traffic_model::TrafficModel;

use crate::frontend::{parse_config, Accelerator, Cli, Simulator, WorkloadCate};
use crate::pipeline_simu::PipelineSimulator;
use crate::pqcache_omega_simulator::CycleAccurateSimulator;
use crate::preprocessing::{affinity_based_row_reordering, sort_by_length};
use crate::py2rust::{load_mm_mat, load_pickled_gemms};
use crate::storage::CsrMatStorage;
use b_reuse_counter::BReuseCounter;
use structopt::StructOpt;

// Workload included:
// ss: ['2cubes_sphere', 'amazon0312', 'ca-CondMat', 'cage12', 'cit-Patents',
// 'cop20k_A', 'email-Enron', 'filter3D', 'm133-b3', 'mario002', 'offshore', 'p2p-Gnutella31',
// 'patents_main', 'poisson3Da', 'roadNet-CA', 'scircuit', 'web-Google', 'webbase-1M', 'wiki-Vote',
// 'degme', 'EternityII_Etilde', 'Ge87H76', 'Ge99H100', 'gupta2', 'm_t1', 'Maragal_7', 'msc10848',
// 'nemsemm1', 'NotreDame_actors', 'opt1', 'raefsky3', 'ramage02', 'relat8', 'ship_001', 'sme3Db',
// 'vsp_bcsstk30_500sep_10in_1Kout', 'x104']
// nn: ['alexnetconv0', 'alexnetconv1', 'alexnetconv2', 'alexnetconv3', 'alexnetconv4',
// 'alexnetfc0', 'alexnetfc1', 'alexnetfc2', 'resnet50conv0', 'resnet50layer1_conv1',
// 'resnet50layer1_conv2', 'resnet50layer1_conv3', 'resnet50layer2_conv1', 'resnet50layer2_conv2',
// 'resnet50layer2_conv3', 'resnet50layer3_conv1', 'resnet50layer3_conv2', 'resnet50layer3_conv3',
// 'resnet50layer4_conv1', 'resnet50layer4_conv2', 'resnet50layer4_conv3', 'resnet50fc']

fn main() {
    // let omega_config = parse_config("omega_config_3mb.json").unwrap();
    let cli: Cli = Cli::from_args();
    // let omega_config = parse_config("omega_config_1mb.json").unwrap();
    let omega_config = parse_config(&cli.configuration).unwrap();

    let gemm: GEMM;
    match cli.category {
        WorkloadCate::NN => {
            gemm = load_pickled_gemms(&omega_config.nn_filepath, &cli.workload).unwrap();
        }
        WorkloadCate::SS => {
            let mat = load_mm_mat(&omega_config.ss_filepath, &cli.workload).unwrap();
            gemm = GEMM::from_mat(&cli.workload, mat);
            // gemm = load_pickled_gemms(&omega_config.ss_filepath, &cli.workload).unwrap();
        }
        WorkloadCate::Desired => {
            gemm = load_pickled_gemms(&omega_config.desired_filepath, &cli.workload).unwrap();
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

    // let validating_product_mat = (&gemm.a * &gemm.b).to_csr();

    match cli.simulator {
        Simulator::TrafficModel => {
            let (mut dram_a, mut dram_b) = CsrMatStorage::init_with_gemm(gemm);
            let mut dram_psum = VectorStorage::new();

            // Preprocessing.
            if cli.preprocess {
                // if let Some(rowmap) = affinity_based_row_reordering(
                //     &mut dram_a,
                //     omega_config.cache_size,
                //     a_avg_row_len,
                //     b_avg_row_len,
                // ) {
                //     dram_a.reorder_row(rowmap);
                // }
                let rowmap = sort_by_length(&mut dram_a);
                dram_a.reorder_row(rowmap);
            }

            let output_base_addr = dram_b.indptr.len();
            // Determine the default window & block shape.
            let default_block_shape = match cli.accelerator {
                Accelerator::Ip => [omega_config.lane_num, 1],
                Accelerator::Omega => [omega_config.block_shape[0], omega_config.block_shape[1]],
                Accelerator::Op => [1, usize::MAX],
                Accelerator::NewOmega => [omega_config.block_shape[0], omega_config.block_shape[1]],
            };

            let default_reduction_window = match cli.accelerator {
                Accelerator::Ip | Accelerator::Omega | Accelerator::NewOmega => [
                    omega_config.lane_num / omega_config.block_shape[1],
                    omega_config.block_shape[1],
                ],
                Accelerator::Op => [1, omega_config.lane_num],
            };

            // Oracle execution: to use the optimal reduction window shape.
            let oracle_exec = true;

            // let mut traffic_model = storage_traffic_model::TrafficModel::new(
            //     omega_config.pe_num,
            //     omega_config.lane_num,
            //     omega_config.cache_size,
            //     omega_config.word_byte,
            //     output_base_addr,
            //     default_reduction_window,
            //     default_block_shape,
            //     &mut dram_a,
            //     &mut dram_b,
            //     &mut dram_psum,
            //     cli.accelerator.clone(),
            // );

            // let mut traffic_model = oracle_storage_traffic_model::TrafficModel::new(
            //         omega_config.pe_num,
            //         omega_config.lane_num,
            //         omega_config.cache_size,
            //         omega_config.word_byte,
            //         output_base_addr,
            //         default_reduction_window,
            //         default_block_shape,
            //         &mut dram_a,
            //         &mut dram_b,
            //         &mut dram_psum,
            //         cli.accelerator.clone(),
            //         oracle_exec,
            // );

            let mut traffic_model = pqcache_storage_traffic_model::TrafficModel::new(
                omega_config.pe_num,
                omega_config.lane_num,
                omega_config.cache_size,
                omega_config.word_byte,
                output_base_addr,
                default_reduction_window,
                default_block_shape,
                &mut dram_a,
                &mut dram_b,
                &mut dram_psum,
                cli.accelerator.clone(),
            );

            traffic_model.execute();

            let result = traffic_model.get_exec_result();
            let a_count = traffic_model.get_a_mat_stat();
            let b_count = traffic_model.get_b_mat_stat();
            let c_count = traffic_model.get_c_mat_stat();
            let exec_count = traffic_model.get_exec_round();
            let cache_count = traffic_model.get_cache_stat();

            println!("-----Result-----");
            println!("-----Access count");
            println!("Execution count: {}", exec_count);
            println!("A matrix count: read {} write {}", a_count.0, a_count.1);
            println!("B matrix count: read {} write {}", b_count.0, b_count.1);
            println!("C matrix count: read {} write {}", c_count.0, c_count.1);
            println!(
                "Cache count: read {} write {}",
                cache_count.0, cache_count.1
            );

            println!("-----Output product matrix");
            for idx in 0..min(result.len(), 10) {
                println!("{}", &result[idx]);
            }

            // println!("----Validating output product matrix");
            // let v_indptr = validating_product_mat.indptr().as_slice().unwrap().to_vec();
            // let v_data = validating_product_mat.data().to_vec();
            // let v_indices = validating_product_mat.indices().to_vec();

            // for idx in 0..min(v_indptr.len() - 1, 10) {
            //     let sliced_len = min(v_indptr[idx + 1] - v_indptr[idx], 5);
            //     let sliced_indptr = &v_indices[v_indptr[idx]..v_indptr[idx] + sliced_len];
            //     let sliced_data = &v_data[v_indptr[idx]..v_indptr[idx] + sliced_len];
            //     println!(
            //         "rowptr: {} indptr: {:?} data: {:?}",
            //         &idx, sliced_indptr, sliced_data
            //     );
            // }
        }

        Simulator::BReuseCounter => {
            let (mut dram_a, mut dram_b) = CsrMatStorage::init_with_gemm(gemm);
            let mut b_reuse_counter = BReuseCounter::new(
                &mut dram_a,
                &mut dram_b,
                omega_config.cache_size,
                omega_config.word_byte,
            );
            let block_num = 8;
            let oracle_fetch = b_reuse_counter.oracle_fetch();
            // let b_row_len = b_reuse_counter.collect_row_length();
            // let avg_reuse_distance = b_reuse_counter.reuse_row_distance();
            let oracle_blocked_fetch = b_reuse_counter.oracle_blocked_fetch();
            // let cache_restricted_collect = b_reuse_counter.cached_fetch();
            // let blocked_fetch = b_reuse_counter.blocked_fetch(block_num);
            // let reuse_dist_guided_fetch = b_reuse_counter.reuse_dist_guided_blocked_fetch(block_num, 4);
            // let affinity_collect = b_reuse_counter.neighbor_row_affinity();
            // let improved_reuse = b_reuse_counter.improved_reuse(block_num);

            // let mut scanned_b_fetch = vec![];
            // for row_num in vec![1, 2, 4, 8, 16, 32, 64] {
            //     let b_fetch = b_reuse_counter.blocked_fetch(row_num);
            //     scanned_b_fetch.push(b_fetch.values().sum::<usize>());
            // }

            println!("-----Result-----");
            // println!("Row length dist: entries: {} >=256: {} >=180: {} >=128: {} >=90: {}",
            //     b_row_len.len(),
            //     b_row_len.values().filter(|&x| *x >= 256).count(),
            //     b_row_len.values().filter(|&x| *x >= 180).count(),
            //     b_row_len.values().filter(|&x| *x >= 128).count(),
            //     b_row_len.values().filter(|&x| *x >= 90).count());
            // println!("Row reuse distance {} | x occr {} | x occr & len {}",
            //     avg_reuse_distance.values().fold(0.0, |c, v| c + v[0]) / avg_reuse_distance.len() as f32,
            //     avg_reuse_distance.iter().map(|(k, v)| oracle_fetch[k] as f32 * v[0]).sum::<f32>()
            //         / oracle_fetch.values().sum::<usize>() as f32,
            //     avg_reuse_distance.iter().fold(0.0,
            //         |c, (k, v)| c + (oracle_fetch[k] * b_row_len[k]) as f32 * v[0])
            //         / oracle_fetch.iter().fold(0.0, |c, (k, _)| c + (oracle_fetch[k] * b_row_len[k]) as f32)
            //     );
            // println!("Ele reuse distance {} | x occr {} | x occr & len {}",
            //     avg_reuse_distance.values().fold(0.0, |c, v| c + v[1]) / avg_reuse_distance.len() as f32,
            //     avg_reuse_distance.iter().map(|(k, v)| oracle_fetch[k] as f32 * v[1]).sum::<f32>()
            //         / oracle_fetch.values().sum::<usize>() as f32,
            //     avg_reuse_distance.iter().fold(0.0,
            //         |c, (k, v)| c + (oracle_fetch[k] * b_row_len[k]) as f32 * v[1])
            //         / oracle_fetch.iter().fold(0.0, |c, (k, _)| c + (oracle_fetch[k] * b_row_len[k]) as f32)
            //     );
            // // println!("Row distance dist: <= 4: {:.5} <= 8: {:.5} <= 16: {:.5} <= 32: {:.5} <= 64: {:.5}",
            // //     reuse_distance.values().filter(|x| x[0] <= 4.0).count() as f32 / reuse_distance.len() as f32,
            // //     reuse_distance.values().filter(|x| x[0] <= 8.0).count() as f32 / reuse_distance.len() as f32,
            // //     reuse_distance.values().filter(|x| x[0] <= 16.0).count() as f32 / reuse_distance.len() as f32,
            // //     reuse_distance.values().filter(|x| x[0] <= 32.0).count() as f32 / reuse_distance.len() as f32,
            // //     reuse_distance.values().filter(|x| x[0] <= 64.0).count() as f32 / reuse_distance.len() as f32);
            // // println!("Ele distance dist: <= 256: {:.5} <= 1024: {:.5} <= 4096: {:.5} <= 16384: {:.5} <= 65536: {:.5}",
            // //     reuse_distance.values().filter(|x| x[1] <= 256.0).count() as f32 / reuse_distance.len() as f32,
            // //     reuse_distance.values().filter(|x| x[1] <= 1024.0).count() as f32 / reuse_distance.len() as f32,
            // //     reuse_distance.values().filter(|x| x[1] <= 4096.0).count() as f32 / reuse_distance.len() as f32,
            // //     reuse_distance.values().filter(|x| x[1] <= 16384.0).count() as f32 / reuse_distance.len() as f32,
            // //     reuse_distance.values().filter(|x| x[1] <= 65536.0).count() as f32 / reuse_distance.len() as f32);
            // println!("Row distance dist: entries: {} <=4: {} <=8: {} <=16: {} <=32: {} \
            //         <=64: {} <=128: {} <=256: {} <=512: {} <=1024: {}",
            //     avg_reuse_distance.len(),
            //     avg_reuse_distance.values().filter(|x| x[0] <= 4.0).count(),
            //     avg_reuse_distance.values().filter(|x| x[0] <= 8.0).count(),
            //     avg_reuse_distance.values().filter(|x| x[0] <= 16.0).count(),
            //     avg_reuse_distance.values().filter(|x| x[0] <= 32.0).count(),
            //     avg_reuse_distance.values().filter(|x| x[0] <= 64.0).count(),
            //     avg_reuse_distance.values().filter(|x| x[0] <= 128.0).count(),
            //     avg_reuse_distance.values().filter(|x| x[0] <= 256.0).count(),
            //     avg_reuse_distance.values().filter(|x| x[0] <= 512.0).count(),
            //     avg_reuse_distance.values().filter(|x| x[0] <= 1024.0).count());
            // println!("Ele distance dist: entries: {} <=16: {} <=32: {} <=64: {} <=128: {} \
            //         <=256: {} <=512: {} <=1024: {} <=4096: {}",
            //     avg_reuse_distance.len(),
            //     avg_reuse_distance.values().filter(|x| x[1] <= 16.0).count(),
            //     avg_reuse_distance.values().filter(|x| x[1] <= 32.0).count(),
            //     avg_reuse_distance.values().filter(|x| x[1] <= 64.0).count(),
            //     avg_reuse_distance.values().filter(|x| x[1] <= 128.0).count(),
            //     avg_reuse_distance.values().filter(|x| x[1] <= 256.0).count(),
            //     avg_reuse_distance.values().filter(|x| x[1] <= 512.0).count(),
            //     avg_reuse_distance.values().filter(|x| x[1] <= 1024.0).count(),
            //     avg_reuse_distance.values().filter(|x| x[1] <= 4096.0).count());
            // // println!("Affinity dist: entries: {} >=128: {} >=64: {} >=16: {} >=4: {}",
            // //     affinity_collect.len(),
            // //     affinity_collect.values().filter(|&x| *x >= 128).count(),
            // //     affinity_collect.values().filter(|&x| *x >= 64).count(),
            // //     affinity_collect.values().filter(|&x| *x >= 16).count(),
            // //     affinity_collect.values().filter(|&x| *x >= 4).count());
            // println!("Nonzero entries: {}", b_reuse_counter.b_mem.get_nonzero());
            println!("Oracle fetch: {}", oracle_fetch.len());
            println!(
                "Oracle blocked fetch: {}",
                oracle_blocked_fetch.values().sum::<usize>()
            );
            // println!("Cache restricted fetch: {}", cache_restricted_collect.values().sum::<usize>());
            // println!("{} blocked fetch: {}", block_num, blocked_fetch.values().sum::<usize>());
            // println!("Total reuse: {} improved reuse: {}, improved ratio: {:.2}", improved_reuse.0, improved_reuse.1, improved_reuse.2);
            // println!("Reuse dist guided fetch: {}", reuse_dist_guided_fetch.values().sum::<usize>());
            // println!("{:?}", scanned_b_fetch);
        }

        Simulator::AccurateSimu => {
            // Cycle-accurate simulator.
            // TODO: Write the blocking mechanism.
            // TODO: Initialize the StreamBuffer component.
            // let mut omega = PipelineSimulator::new();
            // TODO: Add StreamBuffer to omega.
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
                Accelerator::Ip => [omega_config.lane_num, 1],
                Accelerator::Omega => [omega_config.block_shape[0], omega_config.block_shape[1]],
                Accelerator::Op => [1, usize::MAX],
                Accelerator::NewOmega => [omega_config.block_shape[0], omega_config.block_shape[1]],
            };

            let default_reduction_window = match cli.accelerator {
                Accelerator::Ip | Accelerator::Omega | Accelerator::NewOmega => [
                    omega_config.lane_num / omega_config.block_shape[1],
                    omega_config.block_shape[1],
                ],
                Accelerator::Op => [1, omega_config.lane_num],
            };

            let mut cycle_simu = CycleAccurateSimulator::new(
                omega_config.pe_num,
                omega_config.lane_num,
                omega_config.cache_size,
                omega_config.word_byte,
                output_base_addr,
                default_reduction_window,
                default_block_shape,
                &mut dram_a,
                &mut dram_b,
                &mut dram_psum,
                cli.accelerator.clone(),
            );

            cycle_simu.execute();
        }
    }
}

pub fn print_type_of<T>(_: &T) {
    println!("{}", std::any::type_name::<T>())
}
