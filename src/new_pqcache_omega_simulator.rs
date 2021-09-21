use std::{
    cmp::{max, min},
    collections::{VecDeque},
};

use itertools::{merge, merge_join_by, Itertools, Merge, MergeJoinBy};
use storage::{Element, LRUCache, LRURandomCache, PriorityCache, RandomCache, VectorStorage};

use crate::frontend::Accelerator;
use crate::new_scheduler::{Scheduler, Task};
use crate::trace_print;
use crate::util::gen_rands_from_range;
use crate::{
    print_type_of,
    storage::{self, CsrMatStorage, CsrRow, StorageAPI, sorted_element_vec_to_csr_row},
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
            if a.idx[1] == b.idx[0] {
                self.c = Some(Element::new([a.idx[0], b.idx[1]], a.value * b.value));
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

    pub fn is_empty(&self) -> bool {
        return self.a.is_none() || self.b.is_none();
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
                g.sort_by(|a, b| a.idx[1].cmp(&b.idx[1]));
                sorted_results.push(g);
            }
        }
        return sorted_results;
    }

    pub fn reset(&mut self) {
        self.elements.clear();
        self.latency_counter.clear();
    }

    pub fn is_empty(&self) -> bool {
        return self.elements.len() == 0;
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
                        prev_idx = e.idx[1];
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

    pub fn is_empty(&self) -> bool {
        return self.elements.len() == 0;
    }
}

#[derive(Debug, Clone)]
pub struct PE {
    // HW components.
    pub stream_buffers: Vec<VecDeque<Element>>,
    pub multipliers: Vec<Multiplier>,
    pub psum_buffers: Vec<VecDeque<Element>>,
    pub sorting_network: SortingNetwork,
    pub merge_tree: MergeTree,
    pub stream_buffer_size: usize,
    pub psum_buffer_size: usize,
    // Config.
    pub lane_num: usize,
    pub look_aside: bool,
    pub tail_flags: Vec<usize>,
    pub task: Option<Task>,
}

impl PE {
    pub fn new(
        sb_size: usize,
        pb_size: usize,
        lane_num: usize,
        pop_num_per_lane: usize,
        sn_latency: usize,
        mt_latency: usize) -> PE {
        PE {
            stream_buffers: vec![VecDeque::new(); lane_num],
            multipliers: vec![Multiplier::new(); lane_num],
            psum_buffers: vec![VecDeque::new(); lane_num],
            sorting_network: SortingNetwork::new(
                lane_num,
                lane_num,
                pop_num_per_lane,
                sn_latency),
            merge_tree: MergeTree::new(mt_latency),
            stream_buffer_size: sb_size,
            psum_buffer_size: pb_size,
            lane_num,
            look_aside: true,
            tail_flags: vec![0; lane_num],
            task: None,
        }
    }

    pub fn idle(&self) -> bool {
        let is_idle = self.stream_buffers.iter().fold(true,
                |p, fd| p&&fd.is_empty())
            && self.multipliers.iter().fold(true,
                |p, fd| p&&fd.is_empty())
            && self.psum_buffers.iter().fold(true,
                |p, pb| p&&pb.is_empty())
            && self.sorting_network.is_empty()
            && self.merge_tree.is_empty();
        return is_idle;
    }

    pub fn update_tail_flags(&mut self) {
        let group_size = self.task.as_ref().unwrap().group_size;
        for s in (0..self.lane_num).step_by(group_size) {
            let tail_flag = self.multipliers[s..s+self.task.as_ref().unwrap().group_size]
                .iter()
                .fold(usize::MAX, |tf, x| min(tf, x.b_idx()[1]));
            self.tail_flags[s..s+self.task.as_ref().unwrap().group_size].fill(tail_flag);
        }
    }

    pub fn push_stream_buffer(&mut self, lane_idx: usize, elements: Vec<Element>) {
        for e in elements {
            self.stream_buffers[lane_idx].push_back(e);
        }
    }

    pub fn pop_stream_buffer(&mut self, lane_idx: usize) -> Option<Element> {
        if self.task.is_none() {
            return None;
        }

        let group_size = self.task.as_ref().unwrap().group_size;
        if lane_idx == 0 {
            self.stream_buffers[lane_idx].pop_front()
        } else if self.look_aside
            && (lane_idx - 1) / group_size == lane_idx / group_size {
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

    pub fn push_psum_buffer(&mut self, lane_idx: usize, element: Element) {
        self.psum_buffers[lane_idx].push_back(element);
    }

    pub fn pop_psum_buffer(&mut self, lane_idx: usize, pop_num: usize) -> Vec<Option<Element>> {
        let mut psums= vec![];
        let pb = &mut self.psum_buffers[lane_idx];
        let tf = self.tail_flags[lane_idx];
        for _ in 0..pop_num {
            match pb.front().map(|e| e.idx[0] < tf) {
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
    a_matrix: &'a mut CsrMatStorage,
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
        default_block_shape: [usize; 2],
        a_matrix: &'a mut CsrMatStorage,
        b_matrix: &'a mut CsrMatStorage,
        psum_matrix: &'a mut VectorStorage,
        accelerator: Accelerator,
    ) -> CycleAccurateSimulator<'a> {
        let var_factor = 1.5;
        let sb_size = 4;
        let pb_size = 8;
        let pop_num_per_lane = 2;
        let sn_latency = 4;
        let mt_latency = 4;
        CycleAccurateSimulator {
            pe_num,
            lane_num,
            fiber_cache: PriorityCache::new(
                cache_size,
                word_byte,
                output_base_addr,
                b_matrix,
                psum_matrix,
            ),
            pes: vec![PE::new(4, 8, lane_num, 2, 4, 4); pe_num],
            a_matrix,
            exec_cycle: 0,
            scheduler: Scheduler::new(
                pe_num,
                lane_num,
                default_block_shape,
                output_base_addr,
                1.0 - b_matrix.data.len() as f32 / (b_matrix.row_num() * b_matrix.mat_shape[0]) as f32,
                a_matrix,
                b_matrix,
                var_factor,
                accelerator,
            ),
        }
    }

    pub fn execute(&mut self) {
        // Reset the execution round counter.
        self.exec_cycle = 0;
        loop {
            trace_print!("---- cycle {}", self.exec_cycle);
            // Fetch data stage.
            for pe_idx in 0..self.pe_num {
                // When a window is finished, collect merge jobs.
                if self.pes[pe_idx].task.is_some()
                && self.scheduler.is_window_finished(self.pes[pe_idx].task.as_ref().unwrap().window_token) {
                    self.scheduler.collect_pending_psums(self.pes[pe_idx].task.as_ref().unwrap().window_token);
                }

                // Assign new jobs if finished or init.
                if (self.pes[pe_idx].task.is_none()
                || self.scheduler.is_window_finished(self.pes[pe_idx].task.as_ref().unwrap().window_token))
                && self.pes[pe_idx].idle() {
                    self.pes[pe_idx].task = self.scheduler.assign_jobs(&mut self.pes[pe_idx], &mut self.a_matrix);
                }

                // Stream buffer fetch data.
                for lane_idx in 0..self.lane_num {
                    let rb_num = self.pes[pe_idx].stream_buffer_size -
                        self.pes[pe_idx].stream_buffers[lane_idx].len();
                    let bs = self.stream_b_row(pe_idx, lane_idx, rb_num);
                    self.pes[pe_idx].push_stream_buffer(lane_idx, bs);
                }

                // Production phase.
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

                // Collect psum phase.
                self.pes[pe_idx].update_tail_flags();
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

                // Sort & merge phase.
                let sorted_elements = self.pes[pe_idx].sorting_network.pop_elements();
                self.pes[pe_idx].merge_tree.push_elements(sorted_elements);
                let merged_psums = self.pes[pe_idx].merge_tree.pop_elements();
                self.write_psums(pe_idx, merged_psums);
            }
        }
    }

    pub fn stream_b_row(&mut self, pe_idx: usize, lane_idx: usize, rb_num: usize) -> Vec<Element> {
        if self.pes[pe_idx].task.is_none() {
            return vec!();
        }
        let task = self.pes[pe_idx].task.as_ref().unwrap();

        let window_tracker = self.scheduler.window_tracker.get(&task.window_token).unwrap();
        let scalar_idx = window_tracker.lane2idx[lane_idx];
        if scalar_idx.is_none() {
            return vec!();
        }

        let scalar_idx = scalar_idx.unwrap();
        let b_col_idx = window_tracker.b_cols_assigned[lane_idx];
        let elements = self.fiber_cache.read_scalars(scalar_idx, b_col_idx, rb_num).unwrap();

        return elements;
    }

    pub fn write_psums(&mut self, pe_idx: usize, psums: Vec<Vec<Element>>) {
        if self.pes[pe_idx].task.is_none() {
            return;
        }
        let task = self.pes[pe_idx].task.as_ref().unwrap();

        for (gidx, ps) in psums.into_iter().enumerate() {
            let mut csrrow = sorted_element_vec_to_csr_row(ps);
            let addr = self.scheduler.window_tracker[&task.window_token].grp2psum_addr[gidx];
            let a_row_idx = self.scheduler.window_tracker[&task.window_token].lane2idx[gidx*task.group_size].unwrap();
            csrrow.rowptr = a_row_idx[0];
            self.scheduler.b_row_lens
                .entry(addr)
                .and_modify(|l| *l += csrrow.len())
                .or_insert(csrrow.len());
            self.fiber_cache.append_psum_to(addr, csrrow);
        }
    }
}