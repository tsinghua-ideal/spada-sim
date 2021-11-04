use crate::block_topo_tracker::BlockTopoTracker;
use crate::scheduler::BlockTracker;
use crate::storage::CsrMatStorage;
use crate::trace_println;
use std::cmp::{max, min};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct GroupInfo {
    pub row_range: [usize; 2],
    pub avg_row_len: usize,
    pub latency_num: HashMap<usize, [usize; 2]>,
}

#[derive(Debug, Clone)]
pub struct GroupTracker {
    pub groups: Vec<GroupInfo>,
    pub rgmap: HashMap<usize, usize>,
}

impl GroupTracker {
    pub fn new() -> GroupTracker {
        GroupTracker {
            groups: vec![],
            rgmap: HashMap::new(),
        }
    }

    pub fn parse_group(matrix: &CsrMatStorage, var_factor: f32) -> GroupTracker {
        let mut gt = GroupTracker::new();
        let mut prev_row_len = usize::MAX;
        let mut row_s = 0;
    
        // Parse matrix.
        for idx in 0..matrix.row_num() + 1 {
            if idx == matrix.row_num() {
                // Finish the last group.
                let gi = GroupInfo {
                    row_range: [row_s, idx],
                    avg_row_len: (matrix.get_ele_num(row_s, idx)) / (idx - row_s),
                    latency_num: HashMap::new(),
                };
                gt.add_group(gi);
            } else {
                let row_len = matrix.get_ele_num(idx, idx + 1);
                if row_len == 0 {
                    continue;
                } else if prev_row_len == usize::MAX {
                    // Init the first group.
                    prev_row_len = row_len;
                } else if prev_row_len as f32 * var_factor < row_len as f32
                    || prev_row_len as f32 > var_factor * row_len as f32
                {
                    // Encounter a new group. Save the current one.
                    let gi = GroupInfo {
                        row_range: [row_s, idx],
                        avg_row_len: (matrix.get_ele_num(row_s, idx)) / (idx - row_s),
                        latency_num: HashMap::new(),
                    };
                    gt.add_group(gi);
                    prev_row_len = row_len;
                    row_s = idx;
                } else {
                    prev_row_len = row_len;
                }
            }
        }

        return gt;
    }
}

pub struct RowwiseBlockInfo {
    pub a_ele_num: usize,
    pub latency: usize,
}

impl RowwiseBlockInfo {

}