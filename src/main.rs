mod gemm;
mod py2rust;
mod storage;

use crate::py2rust::load_pickled_gemms;
use crate::storage::Storage;

fn main() {
    let gemm_filepath = "../adaptive_sparse_accelerator/workload/
        pickled_gemms/SS_gemms_01_27_20_37.pkl";
    let gemm_name = "2cubes_sphere";
    let gemm = load_pickled_gemms(gemm_filepath, gemm_name).unwrap();
    println!("Get GEMM {}", gemm.name);
    println!("{}", &gemm);
}

struct OmegaSimu {
    memory: Storage,
    cache: Storage,
}