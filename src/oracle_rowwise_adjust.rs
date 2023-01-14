use std::cmp::Reverse;
use std::cmp::{max, min};
use std::collections::{BinaryHeap, HashMap};

use crate::block_topo_tracker::BlockTopoTracker;
use crate::scheduler::BlockTracker;
use crate::storage::CsrMatStorage;
use crate::trace_println;

pub struct SimplePriorityCache {
    pub rowmap: HashMap<usize, usize>, // b row index -> b row size
    pub priority_queue: BinaryHeap<Reverse<[usize; 2]>>, // a index
    pub occp: usize,
    pub capacity: usize,
}

pub struct OracleRowwiseBlockInfo {
    pub a_ele_num: usize,
    pub in_window_matching: HashMap<usize, usize>,
    pub in_block_matching: HashMap<usize, usize>,
}

pub struct OracleRowwiseAdjustTracker {
    pub block_info: HashMap<usize, OracleRowwiseBlockInfo>, // block_token -> oracle rowwise block info
    pub lane_num: usize,
}
