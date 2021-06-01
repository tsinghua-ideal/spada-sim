use crate::pipeline_simu::Tickable;
use crate::storage::{CsrMatStorage, Element, StorageAPI};
use std::cmp::{max, min};
use std::collections::VecDeque;

pub const STREAMBUFFER_CAP: usize = 2;

struct SBLane {
    pub lane_data: VecDeque<Element>,
    pub row_idx: usize,
    pub col_idx: usize,
}

pub struct StreamBuffer<'a> {
    dram: &'a mut CsrMatStorage,
    lanes: Vec<SBLane>,
}

impl<'a> Tickable for StreamBuffer<'a> {}

impl<'a> StreamBuffer<'a> {
    fn tick(&mut self) -> Vec<Element> {
        let mut poped_eles = vec![];

        self.fill_lanes();

        poped_eles.push(self.lanes[0].lane_data.pop_front().unwrap());

        for lidx in 1..self.lanes.len() - 1 {
            let first_ele = self.lanes[lidx].lane_data.front().unwrap();
            let neighbor_ele = self.lanes[lidx - 1].lane_data.front().unwrap();
            let pop_ele = if first_ele.col_idx <= neighbor_ele.col_idx {
                self.lanes[lidx].lane_data.pop_front().unwrap()
            } else {
                self.lanes[lidx - 1].lane_data.pop_front().unwrap()
            };
            poped_eles.push(pop_ele);
        }

        return poped_eles;
    }

    fn fill_lanes(&mut self) {
        for lane in self.lanes.iter_mut() {
            let ds = self
                .dram
                .read(
                    lane.row_idx,
                    lane.col_idx,
                    max(STREAMBUFFER_CAP - lane.lane_data.len(), 0),
                )
                .unwrap()
                .as_element_vec();
            lane.lane_data.extend(ds);
        }
    }
}
