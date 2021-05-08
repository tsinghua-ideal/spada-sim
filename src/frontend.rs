use serde::Deserialize;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::error::Error;
use structopt::{StructOpt, clap::arg_enum};

#[derive(Debug, Deserialize)]
pub struct OmegaConfig {
    pub ss_filepath: String,
    pub nn_filepath: String,
}

arg_enum! {
    #[derive(Debug)]
    pub enum Simulator {
        AccurateSimu,
        TrafficModel,
    }
}

arg_enum! {
    #[derive(Debug, Clone, PartialEq)]
    pub enum Accelerator {
        Gamma,
        Omega,
    }
}

arg_enum! {
    #[derive(Debug)]
    pub enum WorkloadCate {
        SS,
        NN,
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
}

pub fn parse_config(config_fp: &str) -> Result<OmegaConfig, Box<dyn Error>> {
    let config_fp = Path::new(config_fp);
    let file = File::open(config_fp)?;
    let reader = BufReader::new(file);

    let omega_config = serde_json::from_reader(reader)?;
    Ok(omega_config)
}