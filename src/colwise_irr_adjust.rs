use crate::block_topo_tracker::BlockTopoTracker;
use std::cmp::max;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct ColwiseIrrBlockInfo {
    pub a_ele_num: usize,
    pub miss_size: usize,
    pub psum_rw_size: [usize; 2],
}

impl ColwiseIrrBlockInfo {
    pub fn new(a_ele_num: usize) -> ColwiseIrrBlockInfo {
        ColwiseIrrBlockInfo {
            a_ele_num,
            miss_size: 0,
            psum_rw_size: [0; 2],
        }
    }
}

pub struct ColwiseIrrBlockAdjustTracker {
    pub block_info: HashMap<usize, ColwiseIrrBlockInfo>, // block token -> block info
    pub lane_num: usize,
    pub block_shape: HashMap<[usize; 2], [usize; 2]>, // block anchor -> block shape
    pub group_size: usize,
    pub block_width: usize,
    pub group_shape: HashMap<usize, [usize; 2]>, // group no -> group block shape
}

impl ColwiseIrrBlockAdjustTracker {
    pub fn new(
        lane_num: usize,
        group_size: usize,
        block_width: usize,
    ) -> ColwiseIrrBlockAdjustTracker {
        ColwiseIrrBlockAdjustTracker {
            block_info: HashMap::new(),
            lane_num,
            block_shape: HashMap::new(),
            group_size,
            block_width,
            group_shape: HashMap::new(),
        }
    }

    pub fn adjust_block_shape(
        &mut self,
        block_anchor: [usize; 2],
        a_row_num: usize,
        block_topo: &BlockTopoTracker,
    ) -> [usize; 2] {
        // Irregular colwise block adjust scheme allows irregular block size.
        // Irregular colwise block adjust only adjust on the top blocks shape
        // and only in a degraded way.
        if block_anchor[0] % self.group_size == 0 {
            let n1_tk_acr = block_topo.find_left(block_anchor);
            if n1_tk_acr.is_none() {
                let mut blk_h = self.lane_num;
                while block_anchor[0] + blk_h > a_row_num {
                    blk_h = max(1, blk_h / 2);
                }
                let block_shape = [blk_h, self.block_width];
                self.block_shape.insert(block_anchor, block_shape);
                self.group_shape
                    .insert(block_anchor[0] / self.group_size, block_shape);
                return block_shape;
            }
            let (n1_token, n1_block) = n1_tk_acr.unwrap();
            let n2_tk_acr = block_topo.find_left(n1_block);
            if n2_tk_acr.is_none() {
                let mut blk_h = self.lane_num / 2;
                while block_anchor[0] + blk_h > a_row_num {
                    blk_h = max(1, blk_h / 2);
                }
                let block_shape = [blk_h, self.block_width];
                self.block_shape.insert(block_anchor, block_shape);
                self.group_shape
                    .insert(block_anchor[0] / self.group_size, block_shape);
                return block_shape;
            }
            let (n2_token, _) = n2_tk_acr.unwrap();
            let n1_block_info = self.block_info.get(&n1_token).unwrap();
            let n1_ele_size = n1_block_info.a_ele_num;
            let n1_cost = (n1_block_info.miss_size + n1_block_info.psum_rw_size[0]) * 100
                + n1_block_info.psum_rw_size[1];
            let n1_blk_h = self.block_shape[&n1_block][0];
            let n2_block_info = self.block_info.get(&n2_token).unwrap();
            let n2_ele_size = n2_block_info.a_ele_num;
            let n2_cost = (n2_block_info.miss_size + n2_block_info.psum_rw_size[0]) * 100
                + n2_block_info.psum_rw_size[1];

            let mut blk_h =
                if (n1_cost as f32 / n1_ele_size as f32) < (n2_cost as f32 / n2_ele_size as f32) {
                    max(1, n1_blk_h / 2)
                } else {
                    n1_blk_h
                };
            while block_anchor[0] + blk_h > a_row_num {
                blk_h = max(1, blk_h / 2);
            }
            let block_shape = [blk_h, self.block_width];
            self.block_shape.insert(block_anchor, block_shape);
            self.group_shape
                .insert(block_anchor[0] / self.group_size, block_shape);
            return block_shape;
        } else {
            return self.group_shape[&(block_anchor[0] / self.group_size)];
        }
    }

    pub fn adjust_window_shape(&mut self, block_shape: [usize; 2]) -> [usize; 2] {
        return [block_shape[0], self.lane_num / block_shape[0]];
    }
}
