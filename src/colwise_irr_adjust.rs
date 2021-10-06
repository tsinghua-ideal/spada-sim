use std::cmp::{max, min};
use std::collections::{HashMap};
use crate::storage::{CsrMatStorage};
use crate::{trace_println};
use crate::block_topo_tracker::BlockTopoTracker;
use crate::scheduler::BlockTracker;

#[derive(Debug, Clone)]
pub struct ColwiseIrrBlockInfo {
    pub a_ele_num: usize,
    pub miss_size: usize,
    pub psum_rw_size: [usize; 2],
}

impl ColwiseIrrBlockInfo {
    pub fn new(a_ele_num: usize) -> ColwiseRegBlockInfo {
        ColwiseRegBlockInfo {
            a_ele_num,
            miss_size: 0,
            psum_rw_size: [0; 2],
        }
    }
}