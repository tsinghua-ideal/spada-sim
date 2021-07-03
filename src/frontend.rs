use serde::Deserialize;
use std::error::Error;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use structopt::{clap::arg_enum, StructOpt};

#[derive(Debug, Deserialize)]
pub struct OmegaConfig {
    pub ss_filepath: String,
    pub nn_filepath: String,
    pub desired_filepath: String,
    pub pe_num: usize,
    pub lane_num: usize,
    pub cache_size: usize,
    pub word_byte: usize,
    pub block_shape: [usize; 2],
}

arg_enum! {
    #[derive(Debug)]
    pub enum Simulator {
        AccurateSimu,
        TrafficModel,
        BReuseCounter,
    }
}

arg_enum! {
    #[derive(Debug, Clone, PartialEq)]
    pub enum Accelerator {
        Ip,
        Op,
        Omega,
        NewOmega,
    }
}

arg_enum! {
    #[derive(Debug)]
    pub enum WorkloadCate {
        SS,
        NN,
        Desired,
    }
}

#[derive(Debug, StructOpt)]
pub struct Cli {
    /// The simulator to use.
    #[structopt(possible_values=&Simulator::variants(), case_insensitive=true)]
    pub simulator: Simulator,

    /// The accelerator to simulate.
    #[structopt(possible_values=&Accelerator::variants(), case_insensitive=true)]
    pub accelerator: Accelerator,

    /// The workload category to search for the workload.
    #[structopt(possible_values=&WorkloadCate::variants(), case_insensitive=true)]
    pub category: WorkloadCate,

    /// The workload name.
    pub workload: String,

    /// Preprocessing.
    #[structopt(short, long)]
    pub preprocess: bool,
}

pub fn parse_config(config_fp: &str) -> Result<OmegaConfig, Box<dyn Error>> {
    let config_fp = Path::new(config_fp);
    let file = File::open(config_fp)?;
    let reader = BufReader::new(file);

    let omega_config = serde_json::from_reader(reader)?;
    Ok(omega_config)
}
