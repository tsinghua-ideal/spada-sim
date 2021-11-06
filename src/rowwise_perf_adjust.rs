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

    pub fn add_group(&mut self, gi: GroupInfo) {
        self.groups.push(gi);
        let last_idx = self.groups.len() - 1;
        for rowidx in self.groups[last_idx].row_range[0]..self.groups[last_idx].row_range[1] {
            self.rgmap.insert(rowidx, last_idx);
        }
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

pub struct RowwiseLatencyBlockInfo {
    pub a_ele_num: usize,
    pub latency: usize,
}

impl RowwiseLatencyBlockInfo {
    pub fn new(a_ele_num: usize) -> RowwiseLatencyBlockInfo {
        RowwiseLatencyBlockInfo {
            a_ele_num,
            latency: 0,
        }
    }
}

pub struct RowwiseLatencyAdjustTracker {
    pub block_info: HashMap<usize, RowwiseLatencyBlockInfo>, // block_token -> rowwise block info
    pub a_group: GroupTracker,
    pub b_group: GroupTracker,
    pub row_group: usize,
    pub sampling_bounds: Vec<usize>,
    pub set_row_num: usize,
    pub lane_num: usize,
}

impl RowwiseLatencyAdjustTracker {
    pub fn new(
        lane_num: usize,
        a_matrix: &CsrMatStorage,
        b_matrix: &CsrMatStorage,
        var_factor: f32,
    ) -> RowwiseLatencyAdjustTracker {
        RowwiseLatencyAdjustTracker {
            block_info: HashMap::new(),
            a_group: parse_group(a_matrix, var_factor),
            b_group: parse_group(b_matrix, var_factor),
            row_group: usize::MAX,
            sampling_bounds: vec![],
            set_row_num: usize::MAX,
            lane_num,
        }
    }

    pub fn adjust_block_shape(
        &mut self,
        prev_blk_anchor: [usize; 2],
        row_s: usize,
        block_shape: [usize; 2],
        block_topo: &BlockTopoTracker,
        a_row_lens: &Vec<usize>,
    ) -> [usize; 2] {
        let mut block_shape = block_shape;
        trace_println!("-Rowwise adjust");
        // Separately treat wide groups and narrow groups.
        let group_diviser = 128;
        let sample_num = 4;
        let mut min_row_num = 1;

        trace_println!(
            "rgmap: {} cur_group: {}",
            self.a_group.rgmap[&row_s],
            self.row_group
        );

        // First check if the row group changed and prepare for sampling.
        if self.a_group.rgmap[&row_s] != self.row_group {
            // Start from row_num = 1 to touch the distribution.
            block_shape[0] = 1;
            self.row_group = self.a_group.rgmap[&row_s];
            let cur_gi = &self.a_group.groups[self.row_group];
            if cur_gi.row_range[1] - cur_gi.row_range[0] > group_diviser {
                let mut cur_row = row_s + 1;
                let mut i = 1;
                self.sampling_bounds.clear();
                while i <= self.lane_num {
                    cur_row += sample_num * i;
                    self.sampling_bounds.push(cur_row);
                    i *= 2;
                }
            }
            self.set_row_num = usize::MAX;
            return block_shape;
        }

        let cur_gi = &self.a_group.groups[self.row_group];
        trace_println!("cur_gi: {:?}", &cur_gi);
        if cur_gi.row_range[1] - cur_gi.row_range[0] > group_diviser {
            println!("latency_num: {:?}", &self.a_group.groups[self.row_group].latency_num);
            // Treat the wide groups.
            if row_s >= *self.sampling_bounds.last().unwrap() {
                if self.set_row_num == usize::MAX {
                    // Sampling finished.
                    // Then adjust based on the cost of different row num.
                    let mut min_latency = f32::MAX;
                    let mut cur_row_num = 1;
                    while cur_row_num <= self.lane_num {
                        if let Some(latency_num) = self.a_group.groups[self.row_group]
                            .latency_num
                            .get_mut(&cur_row_num)
                        {
                            let div_latency = latency_num[0] as f32 / (latency_num[1] as f32 + 0.0001);
                            if div_latency < min_latency {
                                min_latency = div_latency;
                                self.set_row_num = cur_row_num;
                            }
                        } else {
                            self.a_group.groups[self.row_group]
                                .latency_num
                                .insert(cur_row_num, [0, 0]);
                            self.set_row_num = cur_row_num;
                            break;
                        }
                        cur_row_num *= 2;
                    }
                    while cur_row_num > 1
                        && (row_s + cur_row_num >= self.a_group.groups[self.row_group].row_range[1])
                    {
                        cur_row_num /= 2;
                    }
                }
                min_row_num = self.set_row_num;
            } else {
                // Sampling.
                trace_println!("---Sampling");
                min_row_num = match self.sampling_bounds.binary_search(&(row_s)) {
                    Ok(idx) => 2usize.pow(idx as u32 + 1),
                    Err(idx) => 2usize.pow(idx as u32),
                };
            }
            while min_row_num > 1
                && (row_s + min_row_num >= self.a_group.groups[self.row_group].row_range[1])
            {
                min_row_num /= 2;
            }
            trace_println!(
                "group_range {:?} cost num: {:?}",
                &self.a_group.groups[self.row_group].row_range,
                self.a_group.groups[self.row_group].latency_num
            );
            block_shape[0] = min_row_num;
        } else {
            // Treat the narrow groups.
            let n1_tk_acr = block_topo.find_above(prev_blk_anchor);
            if n1_tk_acr.is_none() {
                return block_shape;
            }
            let (n1_token, n1_block) = n1_tk_acr.unwrap();
            let n1_row_num = prev_blk_anchor[1] - n1_block[1];
            let n1_ele_size = (n1_block[1]..prev_blk_anchor[1]).fold(0, |s, x| s + a_row_lens[x]);

            let n2_tk_acr = block_topo.find_above(n1_block);
            if n2_tk_acr.is_none() {
                return block_shape;
            }
            let (n2_token, n2_block) = n2_tk_acr.unwrap();
            let n2_row_num = n1_block[1] - n2_block[1];
            let n2_ele_size = (n2_block[1]..n1_block[1]).fold(0, |s, x| s + a_row_lens[x]);

            let n1_latency = self.block_info[&n1_token].latency;
            let n2_latency = self.block_info[&n2_token].latency;

            println!(
                "group_range {:?} n1_latency: {}, n1_ele_size: {}, n2_latency: {}, n2_ele_size: {}",
                &self.a_group.groups[self.row_group].row_range,
                n1_latency,
                n1_ele_size,
                n2_latency,
                n2_ele_size
            );

            if (n1_latency as f32 / n1_ele_size as f32) <= (n2_latency as f32 / n2_ele_size as f32) {
                if n1_row_num >= n2_row_num {
                    block_shape[0] = min(block_shape[0] * 2, self.lane_num);
                } else {
                    block_shape[0] = max(block_shape[0] / 2, 1);
                }
            } else {
                if n1_row_num >= n2_row_num {
                    block_shape[0] = max(block_shape[0] / 2, 1);
                } else {
                    block_shape[0] = min(block_shape[0] * 2, self.lane_num);
                }
            }

            while block_shape[0] > 1
                && (row_s + block_shape[0] >= self.a_group.groups[self.row_group].row_range[1])
            {
                block_shape[0] /= 2;
            }
        }

        return block_shape;
    }

    pub fn update_group_cost(&mut self, block_tracker: &BlockTracker) {
        let blk_row_s = block_tracker.anchor[0];
        let row_num = block_tracker.shape[0];
        let grp_idx = self.a_group.rgmap[&blk_row_s];
        let block_info = self.block_info.get(&block_tracker.token).unwrap();
        let latency = block_info.latency;
        let ele_size = block_tracker.a_cols_num.iter().sum();
        self.a_group.groups[grp_idx]
            .latency_num
            .entry(row_num)
            .and_modify(|e| {
                e[0] += latency;
                e[1] += ele_size;
            })
            .or_insert([latency, ele_size]);
    }

    pub fn adjust_window_shape(&mut self, block_shape: [usize; 2]) -> [usize; 2] {
        return [block_shape[0], self.lane_num / block_shape[0]];
    }
}