use crate::block_topo_tracker::BlockTopoTracker;
use std::cmp::{max, min};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct ColwiseRegBlockInfo {
    pub a_ele_num: usize,
    pub miss_size: usize,
    pub psum_rw_size: [usize; 2],
}

impl ColwiseRegBlockInfo {
    pub fn new(a_ele_num: usize) -> ColwiseRegBlockInfo {
        ColwiseRegBlockInfo {
            a_ele_num,
            miss_size: 0,
            psum_rw_size: [0; 2],
        }
    }
}

pub struct ColwiseRegBlockAdjustTracker {
    pub block_info: HashMap<usize, ColwiseRegBlockInfo>, // block token -> block info
    pub lane_num: usize,
    pub window_shape: HashMap<usize, [usize; 2]>, // block token -> window shape
}

impl ColwiseRegBlockAdjustTracker {
    pub fn new(lane_num: usize) -> ColwiseRegBlockAdjustTracker {
        ColwiseRegBlockAdjustTracker {
            block_info: HashMap::new(),
            lane_num,
            window_shape: HashMap::new(),
        }
    }

    pub fn adjust_block_shape(&self, row_s: usize, a_row_num: usize) -> [usize; 2] {
        // Regular colwise block adjust scheme now sets a regular block size.
        let block_width: usize = 8;
        let mut block_shape = [block_width; 2];
        while row_s + block_shape[0] > a_row_num {
            block_shape[0] = min(1, block_shape[0] / 2);
        }
        return block_shape;
    }

    pub fn adjust_window_shape(
        &mut self,
        block_token: usize,
        block_anchor: [usize; 2],
        block_shape: [usize; 2],
        block_topo: &BlockTopoTracker,
    ) -> [usize; 2] {
        // Instead, we adjust the window shape.
        let n1_tk_acr = block_topo.find_left(block_anchor);
        if n1_tk_acr.is_none() {
            let mut win_h = self.lane_num;
            while win_h > block_shape[0] {
                win_h = max(1, win_h / 2);
            }
            let window_shape = [win_h, self.lane_num / win_h];
            self.window_shape.insert(block_token, window_shape);
            return window_shape;
        }
        let (n1_token, n1_block) = n1_tk_acr.unwrap();
        let n2_tk_acr = block_topo.find_left(n1_block);
        if n2_tk_acr.is_none() {
            let mut win_h = self.lane_num / 2;
            while win_h > block_shape[0] {
                win_h = max(1, win_h / 2);
            }
            let window_shape = [win_h, self.lane_num / win_h];
            self.window_shape.insert(block_token, window_shape);
            return window_shape;
        }
        let (n2_token, _n2_block) = n2_tk_acr.unwrap();
        let n1_block_info = self.block_info.get(&n1_token).unwrap();
        let n1_ele_size = n1_block_info.a_ele_num;
        let n1_cost = (n1_block_info.miss_size + n1_block_info.psum_rw_size[0]) * 100
            + n1_block_info.psum_rw_size[1];
        let n1_win_h = self.window_shape[&n1_token][0];
        let n2_block_info = self.block_info.get(&n2_token).unwrap();
        let n2_ele_size = n2_block_info.a_ele_num;
        let n2_cost = (n2_block_info.miss_size + n2_block_info.psum_rw_size[0]) * 100
            + n2_block_info.psum_rw_size[1];
        let n2_win_h = self.window_shape[&n2_token][0];

        let mut win_h =
            if (n1_cost as f32 / n1_ele_size as f32) <= (n2_cost as f32 / n2_ele_size as f32) {
                if n1_win_h >= n2_win_h {
                    min(self.lane_num, n1_win_h * 2)
                } else {
                    max(1, n1_win_h / 2)
                }
            } else {
                if n1_win_h >= n2_win_h {
                    max(1, n1_win_h / 2)
                } else {
                    min(self.lane_num, n1_win_h * 2)
                }
            };
        while win_h > block_shape[0] {
            win_h = max(1, win_h / 2);
        }
        let window_shape = [win_h, self.lane_num / win_h];
        self.window_shape.insert(block_token, window_shape);

        return window_shape;
    }
}
