use std::{
    cmp::{max, min},
    collections::{HashMap, VecDeque},
    hash::Hash,
    ops::Range,
};

use itertools::{izip, merge, merge_join_by, Itertools, Merge, MergeJoinBy};
use storage::{Element, LRUCache, LRURandomCache, PriorityCache, RandomCache, VectorStorage};

use crate::frontend::Accelerator;
use crate::scheduler::{Block, Scheduler, Window};
use crate::trace_print;
use crate::util::gen_rands_from_range;
use crate::{
    print_type_of,
    storage::{self, CsrMatStorage, CsrRow, StorageAPI},
};

#[derive(Debug, Clone)]
pub struct Multiplier {
    a: Option<Element>,
    b: Option<Element>,
    c: Option<Element>,
}

impl Multiplier {
    pub fn new() -> Multiplier {
        Multiplier {
            a: None,
            b: None,
            c: None,
        }
    }

    pub fn set_a(&mut self, a: Option<Element>) {
        self.a = a;
    }

    pub fn set_b(&mut self, b: Option<Element>) {
        self.b = b;
    }

    pub fn retrieve_c(&mut self) -> Option<Element> {
        return self.c.clone();
    }

    pub fn multiply(&mut self) {
        if self.a.is_none() || self.b.is_none() {
            self.c = None;
        } else {
            let a = self.a.as_ref().unwrap();
            let b = self.b.as_ref().unwrap();
            if a.idx[0] == b.idx[1] {
                self.c = Some(Element::new([b.idx[0], a.idx[1]], a.value * b.value));
            } else {
                self.c = None;
            }
        }
    }

    pub fn a_idx(&self) -> [usize; 2] {
        return self.a.as_ref().unwrap().idx;
    }

    pub fn b_idx(&self) -> [usize; 2] {
        return self.b.as_ref().unwrap().idx;
    }

    pub fn reset(&mut self) {
        self.a = None;
        self.b = None;
        self.c = None;
    }
}

#[derive(Debug, Clone)]
pub struct SortingNetwork {
    // For now we simply assume a one-cycle sorting-network.
    elements: Vec<Vec<Option<Element>>>,
    latency_counter: Vec<usize>,
    group_lane_num: usize,
    ele_per_lane: usize,
    latency: usize,
}

impl SortingNetwork {
    pub fn new(
        lane_num: usize,
        group_lane_num: usize,
        ele_per_lane: usize,
        latency: usize,
    ) -> SortingNetwork {
        SortingNetwork {
            elements: vec![],
            latency_counter: vec![],
            group_lane_num,
            ele_per_lane,
            latency,
        }
    }

    pub fn push_elements(&mut self, elements: Vec<Option<Element>>) {
        self.elements.push(elements);
        self.latency_counter.push(self.latency);
    }

    pub fn pop_elements(&mut self) -> Vec<Vec<Element>> {
        let mut sorted_results = vec![];
        for idx in 0..self.elements.len() {
            if self.latency_counter[idx] > 0 {
                self.latency_counter[idx] -= 1;
                continue;
            }

            let mut elements = self.elements.remove(idx);
            self.latency_counter.remove(idx);

            while !elements.is_empty() {
                let mut g = elements
                    .drain(..self.group_lane_num * self.ele_per_lane)
                    .collect::<Vec<Option<Element>>>()
                    .drain_filter(|x| x.is_some())
                    .map(|x| x.unwrap())
                    .collect::<Vec<Element>>();
                g.sort_by(|a, b| a.idx[0].cmp(&b.idx[0]));
                sorted_results.push(g);
            }
        }
        return sorted_results;
    }

    pub fn reset(&mut self) {
        self.elements.clear();
        self.latency_counter.clear();
    }
}

#[derive(Debug, Clone)]
pub struct MergeTree {
    // For now wee simply assume a single-cycle sorting-network.
    elements: Vec<Vec<Vec<Element>>>,
    latency_counter: Vec<usize>,
    latency: usize,
}

impl MergeTree {
    pub fn new(latency: usize) -> MergeTree {
        MergeTree {
            elements: vec![],
            latency_counter: vec![],
            latency,
        }
    }

    pub fn push_elements(&mut self, elements: Vec<Vec<Element>>) {
        self.elements.push(elements);
        self.latency_counter.push(self.latency);
    }

    pub fn pop_elements(&mut self) -> Vec<Vec<Element>> {
        let mut merged_results = vec![];
        for idx in 0..self.elements.len() {
            if self.latency_counter[idx] > 0 {
                self.latency_counter[idx] -= 1;
                continue;
            }

            let elements = self.elements.remove(idx);
            self.latency_counter.remove(idx);

            for es in elements {
                let mut prev_idx = usize::MAX;
                let mut m = vec![];
                for e in es {
                    if e.idx[0] != prev_idx {
                        prev_idx = e.idx[0];
                        m.push(e);
                    } else {
                        m.last_mut().unwrap().value += e.value;
                    }
                }
                merged_results.push(m);
            }
        }
        return merged_results;
    }

    pub fn reset(&mut self) {
        self.elements.clear();
        self.latency_counter.clear();
    }
}

#[derive(Debug, Clone)]
pub struct PE {
    pub block_anchor: [usize; 2],
    pub window_anchor: [usize; 2],
    pub window_shape: [usize; 2],
    pub wl_binding: HashMap<usize, [usize; 2]>,
    pub stream_buffers: Vec<VecDeque<Element>>,
    pub multipliers: Vec<Multiplier>,
    pub psum_buffers: Vec<VecDeque<Element>>,
    pub sorting_network: SortingNetwork,
    pub merge_tree: MergeTree,
    pub stream_buffer_size: usize,
    pub psum_buffer_size: usize,

    pub lane_num: usize,
    pub stop_stream: bool,
    pub look_aside: bool,
    pub tail_flags: Vec<usize>,
    pub block_token: usize,
}

impl PE {
    pub fn new(sb_size: usize, pb_size: usize, lane_num: usize, pop_num_per_lane: usize) -> PE {
        PE {
            block_anchor: [usize::MAX; 2],
            window_anchor: [usize::MAX; 2],
            window_shape: [lane_num, 1],
            wl_binding: HashMap::new(),
            stream_buffers: vec![VecDeque::new(); lane_num],
            multipliers: vec![Multiplier::new(); lane_num],
            psum_buffers: vec![VecDeque::new(); lane_num],
            sorting_network: SortingNetwork::new(lane_num, lane_num, pop_num_per_lane, 4),
            merge_tree: MergeTree::new(lane_num),
            stream_buffer_size: sb_size,
            psum_buffer_size: pb_size,
            lane_num,
            stop_stream: false,
            look_aside: true,
            tail_flags: vec![0; lane_num],
            block_token: usize::MAX,
        }
    }

    pub fn reset_pe(&mut self) {
        self.block_anchor = [usize::MAX; 2];
        self.window_anchor = [usize::MAX; 2];
        self.window_shape = [1, self.lane_num];
        self.wl_binding.clear();
        for i in 0..self.lane_num {
            self.stream_buffers[i].clear();
            self.multipliers[i].reset();
            self.psum_buffers[i].clear();
        }
        self.sorting_network.reset();
        self.merge_tree.reset();
    }

    pub fn write_psum(&mut self, psums: Vec<([usize; 2], f64)>) {
        unimplemented!()
    }

    /// Bind lane to a relative position of the window.
    pub fn bind_lane(&mut self, lane_idx: usize) {
        let r_offset = lane_idx / self.window_shape[0];
        let c_offset = lane_idx % self.window_shape[0];
        self.wl_binding.insert(lane_idx, [r_offset, c_offset]);
    }

    pub fn set_block(&mut self, block: &Block) {
        self.block_anchor = block.anchor;
    }

    pub fn set_window(&mut self, window: &Window) {
        self.window_anchor = window.anchor;
        self.window_shape = window.shape;
    }

    pub fn unbind_win2lane(&mut self) {
        self.wl_binding.clear();
    }

    pub fn push_stream_buffer(&mut self, values: Vec<Element>, lane_idx: usize) {
        for value in values {
            self.stream_buffers[lane_idx].push_back(value);
        }
    }

    pub fn pop_stream_buffer(&mut self, lane_idx: usize) -> Option<Element> {
        if lane_idx == 0 {
            self.stream_buffers[lane_idx].pop_front()
        } else if self.look_aside
            && (lane_idx - 1) / self.window_shape[0] == lane_idx / self.window_shape[0]
        {
            let (ab_sb, sb) = self.stream_buffers.split_at_mut(lane_idx);
            let ab_sb = ab_sb.last_mut().unwrap();
            let sb = sb.get_mut(0).unwrap();

            if ab_sb.len() <= 1 {
                sb.pop_front()
            } else if sb.len() == 0 {
                ab_sb.pop_front()
            } else {
                if ab_sb[1].idx[0] < sb[0].idx[0] {
                    ab_sb.pop_front()
                } else {
                    sb.pop_front()
                }
            }
        } else {
            self.stream_buffers[lane_idx].pop_front()
        }
    }

    pub fn update_tail_flags(&mut self) {
        let tail_flags = usize::MAX;
        for s in (0..self.lane_num).step_by(self.window_shape[0]) {
            let tail_flag = self.multipliers[s..s + self.window_shape[0]]
                .iter()
                .fold(usize::MAX, |tf, x| min(tf, x.b_idx()[0]));
            self.tail_flags[s..s + self.window_shape[0]].fill(tail_flag);
        }
    }

    pub fn push_psum_buffer(&mut self, lane_idx: usize, prod: Element) {
        self.psum_buffers[lane_idx].push_back(prod);
    }

    pub fn pop_psum_buffer(&mut self, lane_idx: usize, pop_num: usize) -> Vec<Option<Element>> {
        let mut psums = vec![];
        let pb = &mut self.psum_buffers[lane_idx];
        let tf = self.tail_flags[lane_idx];
        for _ in 0..pop_num {
            match pb.front().map(|ref e| e.idx[0] < tf) {
                Some(false) | None => psums.push(None),
                Some(true) => psums.push(pb.pop_front()),
            }
        }

        return psums;
    }
}

pub struct CycleAccurateSimulator<'a> {
    pe_num: usize,
    lane_num: usize,
    fiber_cache: PriorityCache<'a>,
    pes: Vec<PE>,
    a_mem: &'a mut CsrMatStorage,
    exec_cycle: usize,
    scheduler: Scheduler,
}

impl<'a> CycleAccurateSimulator<'a> {
    pub fn new(
        pe_num: usize,
        lane_num: usize,
        cache_size: usize,
        word_byte: usize,
        output_base_addr: usize,
        default_reduction_window: [usize; 2],
        default_block_shape: [usize; 2],
        a_mem: &'a mut CsrMatStorage,
        b_mem: &'a mut CsrMatStorage,
        psum_mem: &'a mut VectorStorage,
        accelerator: Accelerator,
    ) -> CycleAccurateSimulator<'a> {
        let var_factor = 1.5;
        CycleAccurateSimulator {
            scheduler: Scheduler::new(
                pe_num,
                lane_num,
                default_block_shape,
                output_base_addr,
                1.0 - b_mem.data.len() as f32 / (b_mem.row_num() * b_mem.mat_shape[0]) as f32,
                &a_mem,
                &b_mem,
                var_factor,
                a_mem.row_num(),
                accelerator,
            ),
            pe_num: pe_num,
            lane_num: lane_num,
            fiber_cache: PriorityCache::new(
                cache_size,
                word_byte,
                output_base_addr,
                b_mem,
                psum_mem,
            ),
            pes: vec![PE::new(4, 8, lane_num, 2); pe_num],
            a_mem: a_mem,
            exec_cycle: 0,
        }
    }

    pub fn execute(&mut self) {
        // Reset the execution round counter.
        self.exec_cycle = 0;
        loop {
            trace_print!("---- cycle {}", self.exec_cycle);
            // Fetch data stage.
            for pe_idx in 0..self.pe_num {
                if self
                    .scheduler
                    .is_window_finished(&self.pes[pe_idx].block_anchor)
                {
                    if !self.scheduler.assign_jobs(&mut self.pes[pe_idx]) {
                        continue;
                    }
                }

                for lane_idx in 0..self.lane_num {
                    // Bind lane to an index in current window.
                    // Fetch the corresponding scalar to the multiplier unit.
                    if !self.pes[pe_idx].wl_binding.contains_key(&lane_idx) {
                        self.pes[pe_idx].bind_lane(lane_idx);
                    }
                    self.fetch_a_scalar(pe_idx, lane_idx);

                    // Update current stream buffer fetch data.
                    let rb_num = self.pes[pe_idx].stream_buffer_size
                        - self.pes[pe_idx].stream_buffers[lane_idx].len();
                    let bs = self.stream_b_row(pe_idx, lane_idx, rb_num);
                    self.pes[pe_idx].push_stream_buffer(bs, lane_idx);
                }
            }

            // Production phase.
            for pe_idx in 0..self.pe_num {
                for lane_idx in 0..self.lane_num {
                    // Pop from stream buffer.
                    let b = self.pes[pe_idx].pop_stream_buffer(lane_idx);

                    // Set b element to multiplier.
                    let prod = self.pes[pe_idx].multipliers[lane_idx].retrieve_c();
                    self.pes[pe_idx].multipliers[lane_idx].set_b(b);
                    self.pes[pe_idx].multipliers[lane_idx].multiply();
                    // Push prod to the psum buffer.
                    if prod.is_some() {
                        self.pes[pe_idx].push_psum_buffer(lane_idx, prod.unwrap());
                    }
                }
            }

            // Collect psum phase.
            for pe_idx in 0..self.pe_num {
                // Update the tail flag.
                self.pes[pe_idx].update_tail_flags();
                // Collect psum phase.
                let mut collected_psums = vec![];
                for lane_idx in 0..self.lane_num {
                    let pop_num = self.pes[pe_idx].sorting_network.ele_per_lane;
                    collected_psums
                        .append(&mut self.pes[pe_idx].pop_psum_buffer(lane_idx, pop_num));
                }
                assert!(
                    collected_psums.len()
                        == self.lane_num * self.pes[pe_idx].sorting_network.ele_per_lane,
                    "Invalid collected psums num!"
                );
                self.pes[pe_idx]
                    .sorting_network
                    .push_elements(collected_psums);
            }

            // Sort & merge phase.
            for pe_idx in 0..self.pe_num {
                let sorted_elements = self.pes[pe_idx].sorting_network.pop_elements();
                self.pes[pe_idx].merge_tree.push_elements(sorted_elements);
                let merged_psums = self.pes[pe_idx].merge_tree.pop_elements();
                // self.pes[pe_idx].write_psums(merged_psums);
            }
        }
    }

    /// Fetch scalar from A matrix and write to the lane's multiplier.
    pub fn fetch_a_scalar(&mut self, pe_no: usize, lane_no: usize) {
        let pe = &self.pes[pe_no];
        let rltv_pos = pe.wl_binding[&lane_no];
        let scalar = self
            .a_mem
            .read_a_scalar(
                pe.window_anchor[0] + rltv_pos[0],
                pe.window_anchor[1] + rltv_pos[1],
            )
            .unwrap();

        self.scheduler
            .block_tracker
            .exec_tracker_mut(&pe.block_anchor)
            .window
            .idxs
            .insert(rltv_pos, scalar.idx);
        self.pes[pe_no].multipliers[lane_no].set_a(Some(scalar));
    }

    /// Stream b row to the lane's stream buffer.
    pub fn stream_b_row(&mut self, pe_idx: usize, lane_idx: usize, num: usize) -> Vec<Element> {
        // let b_row_idx = self.pes[pe_idx].multipliers[lane_idx].b_row_idx();
        let a_idx = self.pes[pe_idx].multipliers[lane_idx].a_idx();
        let b_col_idx = self
            .scheduler
            .block_tracker
            .exec_tracker_mut(&self.pes[pe_idx].block_anchor)
            .b_cols_done
            .entry(a_idx)
            .or_insert(0);

        return self
            .fiber_cache
            .read_scalars(a_idx, *b_col_idx, num)
            .unwrap();
    }
}
