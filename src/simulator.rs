use itertools::Itertools;
use std::ops::AddAssign;

use crate::adder_tree::AdderTree;
use crate::frontend::Accelerator;
use crate::scheduler::{Scheduler, Task};
use crate::storage::{
    sorted_element_vec_to_csr_row, CsrMatStorage, CsrRow, Element, LatencyPriorityCache,
    VectorStorage,
};
use crate::{trace_print, trace_println};
use std::{
    cmp::{max, min},
    collections::VecDeque,
};

pub fn merge_idx(a: &VecDeque<Element>, b: &VecDeque<Element>, merge_num: usize) -> [usize; 2] {
    let mut a_num = 0;
    let mut b_num = 0;
    let mut merge_num = merge_num;
    while merge_num > 0 && (a_num < a.len() || b_num < b.len()) {
        if a_num == a.len() {
            b_num += 1;
            merge_num -= 1;
        } else if b_num == b.len() {
            a_num += 1;
            merge_num -= 1;
        } else if a[a_num].idx[1] <= b[b_num].idx[1] {
            a_num += 1;
            merge_num -= 1;
        } else {
            b_num += 1;
            merge_num -= 1;
        }
    }

    return [a_num, b_num];
}

#[derive(Debug, Clone)]
pub struct MultiplierArray {
    multiplier_num: usize,
    pub a_eles: Vec<Option<Element>>,
    pub b_eles: Vec<Option<Element>>,
    pub c_eles: Vec<Option<Element>>,
    row_drained: Vec<bool>,
}

impl MultiplierArray {
    pub fn new(multiplier_num: usize) -> MultiplierArray {
        MultiplierArray {
            multiplier_num,
            a_eles: (0..multiplier_num).map(|_| None).collect_vec(),
            b_eles: (0..multiplier_num).map(|_| None).collect_vec(),
            c_eles: (0..multiplier_num).map(|_| None).collect_vec(),
            row_drained: vec![false; multiplier_num],
        }
    }

    pub fn set_as(&mut self, a_eles: Vec<Option<Element>>) {
        for (idx, a) in a_eles.into_iter().enumerate() {
            if a.is_some() {
                self.row_drained[idx] = false;
            } else {
                self.row_drained[idx] = true;
            }
            self.a_eles[idx] = a;
        }
    }

    pub fn set_bs(&mut self, b_eles: Vec<Option<Element>>) {
        for (idx, b) in b_eles.into_iter().enumerate() {
            if b.is_some() && b.as_ref().unwrap().idx == [usize::MAX; 2] {
                self.row_drained[idx] = true;
                self.b_eles[idx] = None;
            } else {
                self.b_eles[idx] = b;
            }
        }
    }

    pub fn retrieve_cs(&mut self) -> Vec<Option<Element>> {
        return self.c_eles.clone();
    }

    pub fn multiply(&mut self, group_size: usize) {
        for idx in 0..self.multiplier_num {
            if self.b_eles[idx].is_none() {
                self.c_eles[idx] = None;
            } else {
                let b = self.b_eles[idx].as_ref().unwrap();
                let group_idx = idx / group_size;
                let mut matched = false;
                for a_idx in group_idx * group_size..(group_idx + 1) * group_size {
                    if self.a_eles[a_idx].is_none() {
                        continue;
                    }
                    let a = self.a_eles[a_idx].as_ref().unwrap();
                    if a.idx[1] == b.idx[0] {
                        self.c_eles[idx] =
                            Some(Element::new([a.idx[0], b.idx[1]], a.value * b.value));
                        matched = true;
                        break;
                    }
                }
                if !matched {
                    panic!("No matching a index: for b {}", b.idx[0]);
                }
            }
        }
    }
    pub fn is_empty(&self, mult_idx: usize) -> bool {
        self.a_eles[mult_idx].is_none() || self.row_drained[mult_idx]
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
    pub fn new(group_lane_num: usize, ele_per_lane: usize, latency: usize) -> SortingNetwork {
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
            break;
        }

        for idx in 0..self.elements.len() {
            self.latency_counter[idx] -= 1;
        }

        return sorted_results;
    }

    pub fn is_empty(&self) -> bool {
        return self.elements.len() == 0;
    }
}

#[derive(Debug, Clone)]
pub struct MergeTree {
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
                continue;
            }

            let elements = self.elements.remove(idx);
            self.latency_counter.remove(idx);

            for es in elements {
                let mut prev_idx = usize::MAX;
                let mut m = vec![];
                for e in es {
                    if e.idx[1] != prev_idx {
                        prev_idx = e.idx[1];
                        m.push(e);
                    } else {
                        m.last_mut().unwrap().value += e.value;
                    }
                }
                merged_results.push(m);
            }
            break;
        }

        for idx in 0..self.elements.len() {
            self.latency_counter[idx] -= 1;
        }

        return merged_results;
    }

    pub fn is_empty(&self) -> bool {
        return self.elements.len() == 0;
    }
}

#[derive(Debug, Clone)]
pub struct PE {
    // HW components.
    pub stream_buffers: Vec<VecDeque<Element>>,
    pub multiplier_array: MultiplierArray,
    pub psum_buffers: Vec<VecDeque<Element>>,
    pub sorting_network: SortingNetwork,
    pub merge_tree: MergeTree,
    pub stream_buffer_size: usize,
    pub psum_buffer_size: usize,
    // Config.
    pub pe_idx: usize,
    pub lane_num: usize,
    pub look_aside: bool,
    pub task: Option<Task>,
    // Control.
    pub tail_flags: Vec<usize>,
    pub sb_drained: Vec<bool>,
    pub full_flags: Vec<bool>,
    pub mem_finish_cycle: Option<usize>,
    // Drain latency.
    pub drain_cycle: Option<usize>,
    pub config_unchanged: bool,
}

impl PE {
    pub fn new(
        pe_idx: usize,
        sb_size: usize,
        pb_size: usize,
        lane_num: usize,
        pop_num_per_lane: usize,
        sn_latency: usize,
        mt_latency: usize,
    ) -> PE {
        PE {
            stream_buffers: vec![VecDeque::new(); lane_num],
            multiplier_array: MultiplierArray::new(lane_num),
            psum_buffers: vec![VecDeque::new(); lane_num],
            sorting_network: SortingNetwork::new(lane_num, pop_num_per_lane, sn_latency),
            merge_tree: MergeTree::new(mt_latency),
            stream_buffer_size: sb_size,
            psum_buffer_size: pb_size,
            pe_idx,
            lane_num,
            look_aside: false,
            tail_flags: vec![0; lane_num],
            task: None,
            sb_drained: vec![true; lane_num],
            full_flags: vec![false; lane_num],
            mem_finish_cycle: None,
            drain_cycle: None,
            config_unchanged: false,
        }
    }

    pub fn idle(&self) -> bool {
        let is_idle = self
            .stream_buffers
            .iter()
            .fold(true, |p, fd| p && fd.is_empty())
            && (0..self.lane_num).fold(true, |p, l| p && self.multiplier_array.is_empty(l))
            && self
                .psum_buffers
                .iter()
                .fold(true, |p, pb| p && pb.is_empty())
            && self.sorting_network.is_empty()
            && self.merge_tree.is_empty();
        return is_idle;
    }

    pub fn update_tail_flags(&mut self) {
        let group_size = self.task.as_ref().unwrap().group_size;
        for s in (0..self.lane_num).step_by(group_size) {
            let mut tail_flag = usize::MAX;
            for lane_idx in s..min(s + group_size, self.lane_num) {
                if self.psum_buffers[lane_idx].len() >= 3 {
                    tail_flag = min(tail_flag, self.psum_buffers[lane_idx][2].idx[1]);
                } else if !self.multiplier_array.is_empty(lane_idx) {
                    tail_flag = min(
                        tail_flag,
                        self.multiplier_array.b_eles[lane_idx].as_ref().unwrap().idx[1],
                    );
                }
                // Else the row is all emitted, the tail flag can be set to MAX.
            }
            self.tail_flags[s..min(s + group_size, self.lane_num)].fill(tail_flag);
        }
    }

    pub fn push_stream_buffer(&mut self, lane_idx: usize, elements: Option<Vec<Element>>) {
        if let Some(es) = elements {
            for e in es {
                self.stream_buffers[lane_idx].push_back(e);
            }
        } else {
            if !self.sb_drained[lane_idx] {
                self.stream_buffers[lane_idx].push_back(Element::new([usize::MAX; 2], 0.0));
                self.sb_drained[lane_idx] = true;
            }
        }
    }

    pub fn pop_stream_buffer(&mut self, lane_idx: usize) -> Option<Element> {
        if self.task.is_none() || self.full_flags[lane_idx] {
            return None;
        }

        let group_size = self.task.as_ref().unwrap().group_size;
        let left_lane = (lane_idx / 2) * 2;
        let right_lane = left_lane + 1;
        if group_size < 2 {
            self.stream_buffers[lane_idx].pop_front()
        } else {
            let drain_num = merge_idx(
                &self.stream_buffers[left_lane],
                &self.stream_buffers[right_lane],
                1,
            );
            if drain_num[0] > 0 && self.stream_buffers[left_lane].len() > 1 {
                self.stream_buffers[left_lane].pop_front()
            } else if drain_num[1] > 0 && self.stream_buffers[right_lane].len() > 1 {
                self.stream_buffers[right_lane].pop_front()
            } else {
                self.stream_buffers[lane_idx].pop_front()
            }
        }
    }

    pub fn push_psum_buffer(&mut self, lane_idx: usize, element: Element) {
        self.psum_buffers[lane_idx].push_back(element);
    }

    pub fn pop_psum_buffer(&mut self, lane_idx: usize, pop_num: usize) -> Vec<Option<Element>> {
        let mut psums = vec![];
        let pb = &mut self.psum_buffers[lane_idx];
        let tf = self.tail_flags[lane_idx];
        for _ in 0..pop_num {
            match pb.front().map(|e| e.idx[1] < tf) {
                Some(false) | None => psums.push(None),
                Some(true) => psums.push(pb.pop_front()),
            }
        }

        return psums;
    }

    pub fn set_task(&mut self, task: Option<(usize, Task)>) -> usize {
        if task.is_none() {
            self.multiplier_array.set_as(vec![None; self.lane_num]);
            // Pb, sn, mt remain the previous configuration.
            self.mem_finish_cycle = None;
            self.task = None;
            self.drain_cycle = None;
            self.config_unchanged = false;
            return 0;
        } else {
            let (a_latency, task) = task.unwrap();
            self.config_unchanged =
                self.task.is_some() && (self.task.as_ref().unwrap().group_size == task.group_size);
            self.task = Some(task.clone());
            self.sorting_network.group_lane_num = task.group_size;
            for (lane_idx, e) in task.a_eles.iter().enumerate() {
                self.sb_drained[lane_idx] = e.is_none();
            }
            self.multiplier_array.set_as(task.a_eles.clone());
            self.mem_finish_cycle = None;
            self.drain_cycle = None;
            return a_latency;
        }
    }
}

pub struct Simulator<'a> {
    pe_num: usize,
    adder_tree_num: usize,
    lane_num: usize,
    fiber_cache: LatencyPriorityCache<'a>,
    pes: Vec<PE>,
    a_matrix: &'a mut CsrMatStorage,
    exec_cycle: usize,
    scheduler: Scheduler,
    adder_trees: Vec<AdderTree>,
    // Storage access latency related.
    pub a_pending_cycle: Vec<usize>,
    pub channel: usize,
    pub word_cycle_chan_bw: f32,
    // Debug info.
    pub drain_cycles: Vec<usize>,
    pub mult_util: Vec<f32>,
    pub active_cycle: Vec<usize>,
}

impl<'a> Simulator<'a> {
    pub fn new(
        pe_num: usize,
        adder_tree_num: usize,
        lane_num: usize,
        cache_size: usize,
        word_byte: usize,
        output_base_addr: usize,
        default_block_shape: [usize; 2],
        a_matrix: &'a mut CsrMatStorage,
        b_matrix: &'a mut CsrMatStorage,
        psum_matrix: &'a mut VectorStorage,
        accelerator: Accelerator,
        mem_latency: usize,
        cache_latency: usize,
        freq: f32,
        channel: usize,
        bandwidth_per_channel: f32,
    ) -> Simulator<'a> {
        let var_factor = 1.5;
        let sb_size = 4;
        let pb_size = 8;
        let pop_num_per_lane = 2;
        let sn_latency = 4;
        let mt_latency = 4;
        let tree_width = 8;
        let word_cycle_chan_bw = bandwidth_per_channel / freq / word_byte as f32;
        Simulator {
            scheduler: Scheduler::new(
                pe_num,
                lane_num,
                default_block_shape,
                output_base_addr,
                a_matrix,
                b_matrix,
                var_factor,
                accelerator,
                mem_latency,
                cache_latency,
            ),
            pe_num,
            adder_tree_num,
            lane_num,
            fiber_cache: LatencyPriorityCache::new(
                cache_size,
                word_byte,
                output_base_addr,
                b_matrix,
                psum_matrix,
                mem_latency,
                cache_latency,
            ),
            pes: (0..pe_num)
                .map(|pe_idx| {
                    PE::new(
                        pe_idx,
                        sb_size,
                        pb_size,
                        lane_num,
                        pop_num_per_lane,
                        sn_latency,
                        mt_latency,
                    )
                })
                .collect::<Vec<PE>>(),
            adder_trees: (0..adder_tree_num)
                .map(|idx| AdderTree::new(idx, tree_width))
                .collect_vec(),
            a_matrix,
            exec_cycle: 0,
            a_pending_cycle: vec![0; pe_num],
            channel,
            word_cycle_chan_bw,
            drain_cycles: vec![0; pe_num],
            mult_util: vec![0.0; pe_num],
            active_cycle: vec![0; pe_num],
        }
    }

    pub fn execute(&mut self) {
        // Reset the execution round counter.
        self.exec_cycle = 0;
        loop {
            trace_println!("\n---- cycle {}", self.exec_cycle);

            let mut prev_a_rs = vec![0; self.pe_num];
            let mut prev_b_rs = vec![0; self.pe_num];
            let mut prev_psum_rs = vec![0; self.pe_num];
            let mut prev_psum_ws = vec![0; self.pe_num];
            let prev_cache_miss = self.fiber_cache.miss_count;
            let prev_b_evict = self.fiber_cache.b_evict_count;
            let prev_psum_evict = self.fiber_cache.psum_evict_count;
            let mut prev_cache_rs = vec![0; self.pe_num];
            let mut prev_cache_ws = vec![0; self.pe_num];

            trace_println!("Psum in memory:");
            self.fiber_cache.print_psums();

            // Fetch data stage.
            for pe_idx in 0..self.pe_num {
                trace_println!("\n---pe {}", pe_idx);
                // Pending when access a or b matrix.
                if self.a_pending_cycle[pe_idx] > 0 {
                    self.a_pending_cycle[pe_idx] -= 1;
                    continue;
                }
                // Collect prev exec stats.
                prev_a_rs[pe_idx] = self.a_matrix.read_count;
                prev_b_rs[pe_idx] = self.fiber_cache.b_mem.read_count;
                prev_psum_rs[pe_idx] = self.fiber_cache.psum_mem.read_count;
                prev_psum_ws[pe_idx] = self.fiber_cache.psum_mem.write_count;
                prev_cache_rs[pe_idx] = self.fiber_cache.read_count;
                prev_cache_ws[pe_idx] = self.fiber_cache.write_count;
                trace_println!("idle: {}", self.pes[pe_idx].idle());
                // Track drain cycle.
                if self.pes[pe_idx].task.is_some()
                    && self.pes[pe_idx].drain_cycle.is_none()
                    && self
                        .scheduler
                        .is_window_finished(self.pes[pe_idx].task.as_ref().unwrap().window_token)
                // && self.pes[pe_idx].stream_buffers.iter().fold(true, |p, fd| p && fd.is_empty())
                {
                    self.pes[pe_idx].drain_cycle = Some(self.exec_cycle);
                }
                // Assign new jobs if finished or init.
                if (self.pes[pe_idx].task.is_none()
                    || self
                        .scheduler
                        .is_window_finished(self.pes[pe_idx].task.as_ref().unwrap().window_token))
                    && self.pes[pe_idx].idle()
                {
                    // Stat the memory transfer cycle and calc the overlapped latency.
                    if self.pes[pe_idx].mem_finish_cycle.is_none() {
                        if self.pes[pe_idx].task.is_some() {
                            let task = self.pes[pe_idx].task.as_ref().unwrap();
                            print!("pe: {} cur_cycle: {} ", pe_idx, self.exec_cycle);
                            if task.merge_mode {
                                println!("merge");
                            } else {
                                println!(
                                    "anchor: {:?}, shape: {:?}",
                                    self.scheduler.window_tracker[&task.window_token].anchor,
                                    self.scheduler.window_tracker[&task.window_token].shape
                                );
                            }
                            trace_println!(
                                "cache occp: {} in {}, psum_occp: {}, b_occp: {}",
                                self.fiber_cache.cur_num,
                                self.fiber_cache.capability,
                                self.fiber_cache.psum_occp,
                                self.fiber_cache.b_occp
                            );
                            trace_println!(
                                "active cycle: {:?} mult_utils: {:?} avg_mult_util: {}",
                                self.active_cycle,
                                self.mult_util,
                                self.mult_util.iter().sum::<f32>() / self.mult_util.len() as f32
                            );
                            if !task.merge_mode {
                                let latency = max(
                                    self.exec_cycle - task.start_cycle,
                                    (task.memory_traffic as f32
                                        / (self.word_cycle_chan_bw * self.channel as f32
                                            / self.pe_num as f32))
                                        as usize,
                                );
                                self.scheduler
                                    .rowwise_latency_adjust_tracker
                                    .block_info
                                    .get_mut(&task.block_token)
                                    .unwrap()
                                    .latency
                                    .add_assign(latency);
                            }
                            self.pes[pe_idx].mem_finish_cycle = Some(
                                task.start_cycle
                                    + (task.memory_traffic as f32
                                        / (self.word_cycle_chan_bw * self.channel as f32
                                            / self.pe_num as f32))
                                        as usize,
                            );
                        } else {
                            self.pes[pe_idx].mem_finish_cycle = None;
                        }

                        if self.pes[pe_idx].mem_finish_cycle.is_some() {
                            // Add to discount drain cycle.
                            let drain_cycle = if self.pes[pe_idx].drain_cycle.is_some() {
                                self.exec_cycle - *self.pes[pe_idx].drain_cycle.as_ref().unwrap()
                            } else {
                                0
                            };
                            let mem_exec_cycle =
                                *self.pes[pe_idx].mem_finish_cycle.as_ref().unwrap();
                            let discounted_exec_cycle = self.exec_cycle - drain_cycle;
                            if self.exec_cycle > mem_exec_cycle && self.pes[pe_idx].config_unchanged
                            {
                                self.drain_cycles[pe_idx] +=
                                    self.exec_cycle - max(discounted_exec_cycle, mem_exec_cycle);
                            }
                            trace_println!("drain cycle: {:?}", self.drain_cycles[pe_idx]);
                        }
                    }
                    // Wait to catch the pending cycle.
                    if self.pes[pe_idx].mem_finish_cycle.is_some()
                        && *self.pes[pe_idx].mem_finish_cycle.as_ref().unwrap() > self.exec_cycle
                    {
                        continue;
                    }
                    // Collect output psums.
                    if self.pes[pe_idx].task.is_some() {
                        let prev_win_token = self.pes[pe_idx].task.as_ref().unwrap().window_token;
                        for arow_addr in self.scheduler.window_tracker[&prev_win_token]
                            .arow_addr_pairs
                            .iter()
                        {
                            self.scheduler
                                .row_rgstr_task
                                .entry(arow_addr[0])
                                .and_modify(|e| *e -= 1);
                            if self.scheduler.b_row_lens.contains_key(&arow_addr[1]) {
                                self.scheduler
                                    .output_tracker
                                    .entry(arow_addr[0])
                                    .and_modify(|ps| {
                                        if !ps.contains(&arow_addr[1]) {
                                            ps.push(arow_addr[1]);
                                        }
                                    })
                                    .or_insert(vec![arow_addr[1]]);
                            }
                        }
                    }
                    // Collect stats of the prev finished task.
                    if self.pes[pe_idx].task.is_some()
                        && !self.pes[pe_idx].task.as_ref().unwrap().merge_mode
                    {
                        let prev_blk_tk = self.pes[pe_idx].task.as_ref().unwrap().block_token;
                        if self.scheduler.is_block_finished(prev_blk_tk) {
                            // Label finished rows.
                            self.scheduler.label_finished_rows(prev_blk_tk);
                            match self.scheduler.accelerator {
                                Accelerator::Spada => {
                                    // Update the rowwise adjust tracker.
                                    let block_tracker = &self.scheduler.block_tracker[&prev_blk_tk];
                                    self.scheduler
                                        .rowwise_adjust_tracker
                                        .update_group_cost(block_tracker);
                                    self.scheduler
                                        .rowwise_latency_adjust_tracker
                                        .update_group_cost(block_tracker);
                                }
                                _ => {}
                            }
                        }
                    }
                    // Swapout those finished rows.
                    self.swapout_finished_psums();
                    // Assign new tasks.
                    let task = self.scheduler.assign_task(
                        &mut self.pes[pe_idx],
                        &mut self.a_matrix,
                        self.exec_cycle,
                    );
                    let latency = self.pes[pe_idx].set_task(task);
                    self.a_pending_cycle[pe_idx] += latency;
                }
                if self.pes[pe_idx].task.is_some() {
                    let block_token = self.pes[pe_idx].task.as_ref().unwrap().block_token;
                    let block_tracker = &self.scheduler.block_tracker[&block_token];
                    let window_token = self.pes[pe_idx].task.as_ref().unwrap().window_token;
                    let window_tracker = &self.scheduler.window_tracker[&window_token];
                    trace_println!("-block anchor: {:?} block shape {:?} window anchor: {:?} window shape: {:?}",
                    block_tracker.anchor, block_tracker.shape, window_tracker.anchor, window_tracker.shape);
                    for r_offset in 0..window_tracker.shape[0] {
                        for c_offset in 0..window_tracker.shape[1] {
                            let lane_pos = r_offset * window_tracker.shape[1] + c_offset;
                            match window_tracker.lane2idx[lane_pos] {
                                None => {
                                    trace_print!("{:?} None  ", lane_pos);
                                }
                                Some(idx) => {
                                    let rlen = self.scheduler.b_row_lens[&idx[1]];
                                    trace_print!(
                                        "{:?} asgn:{} len:{}  ",
                                        idx,
                                        window_tracker.b_cols_assigned[lane_pos],
                                        rlen
                                    );
                                }
                            }
                        }
                    }
                } else {
                    continue;
                }

                // Stream buffer fetch data.
                for lane_idx in 0..self.lane_num {
                    let sb_len = self.pes[pe_idx].stream_buffers[lane_idx]
                        .iter()
                        .filter(|e| e.idx[0] != usize::MAX)
                        .count();
                    let rb_num = self.pes[pe_idx].stream_buffer_size - sb_len;
                    let bs = self.stream_b_row(pe_idx, lane_idx, rb_num, self.exec_cycle);
                    self.pes[pe_idx].push_stream_buffer(lane_idx, bs);
                }

                // Production phase.
                let group_size = self.pes[pe_idx].task.as_ref().unwrap().group_size;
                let mut bs = vec![];
                for lane_idx in 0..self.lane_num {
                    // Update full flag.
                    if self.pes[pe_idx].psum_buffers[lane_idx].len()
                        >= self.pes[pe_idx].psum_buffer_size - 1
                    {
                        self.pes[pe_idx].full_flags[lane_idx] = true;
                    } else {
                        self.pes[pe_idx].full_flags[lane_idx] = false;
                    }
                    // Pop from stream buffer.
                    let b = if self.pes[pe_idx].multiplier_array.a_eles[lane_idx].is_some() {
                        self.pes[pe_idx].pop_stream_buffer(lane_idx)
                    } else {
                        None
                    };
                    bs.push(b);
                }
                // Set bs to multiplier array.
                let prods = self.pes[pe_idx].multiplier_array.retrieve_cs();
                self.pes[pe_idx].multiplier_array.set_bs(bs);
                self.pes[pe_idx].multiplier_array.multiply(group_size);
                let mut mult_in_use = 0;
                for (lane_idx, prod) in prods.into_iter().enumerate() {
                    if prod.is_some() {
                        mult_in_use += 1;
                        self.pes[pe_idx].push_psum_buffer(lane_idx, prod.unwrap());
                    }
                }
                let mult_util = mult_in_use as f32 / self.pes[pe_idx].lane_num as f32;
                if !self.pes[pe_idx].idle() && !self.pes[pe_idx].task.as_ref().unwrap().merge_mode {
                    self.mult_util[pe_idx] =
                        (self.mult_util[pe_idx] * self.active_cycle[pe_idx] as f32 + mult_util)
                            / (self.active_cycle[pe_idx] + 1) as f32;
                    self.active_cycle[pe_idx] += 1;
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
                if collected_psums.iter().any(|p| p.is_some()) {
                    self.pes[pe_idx]
                        .sorting_network
                        .push_elements(collected_psums);
                }

                // Sort & merge phase.
                let sorted_elements = self.pes[pe_idx].sorting_network.pop_elements();
                if sorted_elements.len() > 0 {
                    self.pes[pe_idx].merge_tree.push_elements(sorted_elements);
                }
                let merged_psums = self.pes[pe_idx].merge_tree.pop_elements();
                self.write_psums(pe_idx, merged_psums);

                if self.pes[pe_idx].task.is_some() {
                    let blk_token = self.pes[pe_idx].task.as_ref().unwrap().block_token;
                    // Collect post exec info.
                    let delta_b = self.fiber_cache.b_mem.read_count - prev_b_rs[pe_idx];
                    let delta_psum = self.fiber_cache.psum_mem.read_count
                        + self.fiber_cache.psum_mem.write_count
                        - prev_psum_rs[pe_idx]
                        - prev_psum_ws[pe_idx];
                    let delta_cache = self.fiber_cache.read_count + self.fiber_cache.write_count
                        - prev_cache_rs[pe_idx]
                        - prev_cache_ws[pe_idx];
                    // Update block tracker.
                    if !self.pes[pe_idx].task.as_ref().unwrap().merge_mode {
                        self.update_energy_adjust_tracker(
                            blk_token,
                            delta_b,
                            delta_psum,
                            delta_cache,
                        );
                    }
                }

                let memory_traffic = self.a_matrix.read_count - prev_a_rs[pe_idx]
                    + self.fiber_cache.b_mem.read_count
                    - prev_b_rs[pe_idx]
                    + self.fiber_cache.psum_mem.read_count
                    - prev_psum_rs[pe_idx]
                    + self.fiber_cache.psum_mem.write_count
                    - prev_psum_ws[pe_idx];
                if self.pes[pe_idx].task.is_some() {
                    self.pes[pe_idx].task.as_mut().unwrap().memory_traffic += memory_traffic;
                }
            }

            for idx in 0..self.adder_tree_num {
                self.adder_tree_exec(idx);
            }

            if self.scheduler.a_traversed
                && self.pes.iter().all(|p| p.idle() && p.task.is_none())
                && self
                    .adder_trees
                    .iter()
                    .all(|a| a.idle() && a.task.is_none())
            {
                break;
            }

            trace_println!(
                "Cache read_count: + {} -> {}, write_count: + {} -> {}",
                self.fiber_cache.read_count - prev_cache_rs[0],
                self.fiber_cache.read_count,
                self.fiber_cache.write_count - prev_cache_ws[0],
                self.fiber_cache.write_count
            );
            trace_println!(
                "Cache occp: {} in {}, psum_occp: {}, b_occp: {}",
                self.fiber_cache.cur_num,
                self.fiber_cache.capability,
                self.fiber_cache.psum_occp,
                self.fiber_cache.b_occp
            );
            trace_println!("Cache miss_count: + {} -> {}, b_evict_count: + {} -> {}, psum_evict_count: + {} -> {}",
                self.fiber_cache.miss_count - prev_cache_miss, self.fiber_cache.miss_count,
                self.fiber_cache.b_evict_count - prev_b_evict, self.fiber_cache.b_evict_count,
                self.fiber_cache.psum_evict_count - prev_psum_evict, self.fiber_cache.psum_evict_count);
            trace_println!(
                "A mem: read_count: + {} -> {}",
                self.a_matrix.read_count - prev_a_rs[0],
                self.a_matrix.read_count
            );
            trace_println!(
                "B mem: read_count: + {} -> {}",
                self.fiber_cache.b_mem.read_count - prev_b_rs[0],
                self.fiber_cache.b_mem.read_count
            );
            trace_println!(
                "C mem: read_count: + {} -> {}, write_count: +{} -> {}",
                self.fiber_cache.psum_mem.read_count - prev_psum_rs[0],
                self.fiber_cache.psum_mem.read_count,
                self.fiber_cache.psum_mem.write_count - prev_psum_ws[0],
                self.fiber_cache.psum_mem.write_count
            );

            self.exec_cycle += 1;
        }
    }

    pub fn stream_b_row(
        &mut self,
        pe_idx: usize,
        lane_idx: usize,
        rb_num: usize,
        cur_cycle: usize,
    ) -> Option<Vec<Element>> {
        if self.pes[pe_idx].task.is_none() {
            return None;
        }
        let task = self.pes[pe_idx].task.as_mut().unwrap();

        let window_tracker = self
            .scheduler
            .window_tracker
            .get_mut(&task.window_token)
            .unwrap();
        let scalar_idx = window_tracker.lane2idx[lane_idx];
        if scalar_idx.is_none() {
            return None;
        }

        let scalar_idx = scalar_idx.unwrap();
        let b_col_idx = window_tracker.b_cols_assigned[lane_idx];
        if !self.fiber_cache.contains_row(&scalar_idx[1]) && b_col_idx == 0 {
            task.memory_traffic +=
                (self.fiber_cache.mem_latency as f32 * self.word_cycle_chan_bw) as usize
        }
        let elements = if self.pes[pe_idx].task.as_ref().unwrap().merge_mode {
            match self
                .fiber_cache
                .request_consume_scalars(scalar_idx, b_col_idx, rb_num, cur_cycle, true)
            {
                Some(es) => {
                    if es.len() == 0 {
                        None
                    } else {
                        window_tracker.b_cols_assigned[lane_idx] += es.len();
                        Some(es)
                    }
                }
                None => Some(vec![]), // Pending cycle, not drained.
            }
        } else {
            match self
                .fiber_cache
                .request_read_scalars(scalar_idx, b_col_idx, rb_num, cur_cycle, true)
            {
                Some(es) => {
                    if es.len() == 0 {
                        None
                    } else {
                        window_tracker.b_cols_assigned[lane_idx] += es.len();
                        Some(es)
                    }
                }
                None => Some(vec![]), // Pending cycle, not drained.
            }
        };

        return elements;
    }

    pub fn write_psums(&mut self, pe_idx: usize, psums: Vec<Vec<Element>>) {
        if self.pes[pe_idx].task.is_none() {
            return;
        }
        let task = self.pes[pe_idx].task.as_ref().unwrap();

        // Write psums to cache.
        for (gidx, ps) in psums.into_iter().enumerate() {
            if ps.len() == 0 {
                continue;
            }
            let mut csrrow = sorted_element_vec_to_csr_row(ps);
            let window_tracker = self
                .scheduler
                .window_tracker
                .get(&task.window_token)
                .unwrap();
            let arow_addr = window_tracker.arow_addr_pairs[gidx];
            // Assign the output address.
            csrrow.rowptr = arow_addr[1];
            trace_println!("-write_psum: {:?}", &csrrow);
            self.scheduler
                .b_row_lens
                .entry(arow_addr[1])
                .and_modify(|l| *l += csrrow.len())
                .or_insert(csrrow.len());
            self.fiber_cache.append_psum_to(arow_addr[1], csrrow);
        }
    }

    pub fn swapout_finished_psums(&mut self) {
        let output_tracker = &mut self.scheduler.output_tracker;
        let row_rgstr_task = &self.scheduler.row_rgstr_task;
        let swapable_rows = self
            .scheduler
            .a_tail_produced
            .drain_filter(|row| {
                row_rgstr_task.get(row).map_or(true, |r| *r == 0)
                    && output_tracker.get(row).map_or(true, |ps| ps.len() == 1)
            })
            .collect::<Vec<usize>>();
        for row in swapable_rows {
            if output_tracker.contains_key(&row) {
                let addr = output_tracker[&row][0];
                self.scheduler.a_row_finished.insert(row, addr);
                output_tracker.remove(&row);
                if self.fiber_cache.rowmap.contains_key(&addr) {
                    self.fiber_cache.swapout(addr);
                }
            }
        }
    }

    pub fn get_a_mat_stat(&self) -> [usize; 2] {
        [self.a_matrix.read_count, self.a_matrix.write_count]
    }

    pub fn get_b_mat_stat(&self) -> [usize; 2] {
        [
            self.fiber_cache.b_mem.read_count,
            self.fiber_cache.b_mem.write_count,
        ]
    }

    pub fn get_c_mat_stat(&self) -> [usize; 2] {
        [
            self.fiber_cache.psum_mem.read_count,
            self.fiber_cache.psum_mem.write_count,
        ]
    }

    pub fn get_exec_cycle(&self) -> usize {
        self.exec_cycle - self.drain_cycles.iter().min().unwrap()
    }

    pub fn get_cache_stat(&self) -> [usize; 2] {
        [self.fiber_cache.read_count, self.fiber_cache.write_count]
    }

    pub fn get_exec_result(&mut self) -> Vec<CsrRow> {
        let mut c = vec![];
        for rowid in 0..self.a_matrix.row_num() {
            let mut csrrow = CsrRow::new(rowid);
            if self.a_matrix.get_ele_num(rowid, rowid + 1) > 0 {
                let raw_rowid = if self.a_matrix.remapped {
                    self.a_matrix.row_remap[&rowid]
                } else {
                    rowid
                };
                if self.scheduler.a_row_finished.contains_key(&rowid) {
                    let addr = self.scheduler.a_row_finished.get(&rowid).unwrap();
                    trace_println!(
                        "Get result: row: {} row len: {}",
                        raw_rowid,
                        self.fiber_cache.psum_mem.data[&addr].size() / 2
                    );
                    csrrow = match self.fiber_cache.psum_mem.data.get(&addr) {
                        Some(row) => row.clone(),
                        None => self.fiber_cache.rowmap.get(&addr).unwrap().clone(),
                    };
                    csrrow.rowptr = raw_rowid;
                }
            }
            c.push(csrrow);
        }
        c.sort_by(|a, b| a.rowptr.cmp(&b.rowptr));
        return c;
    }

    pub fn update_energy_adjust_tracker(
        &mut self,
        block_token: usize,
        delta_b: usize,
        delta_psum: usize,
        delta_cache: usize,
    ) {
        // Rowwise adjust tracker.
        let rowwise_tracker = self
            .scheduler
            .rowwise_adjust_tracker
            .block_info
            .get_mut(&block_token)
            .unwrap();
        rowwise_tracker.miss_size += delta_b;
        rowwise_tracker.psum_rw_size[0] += delta_psum;
        rowwise_tracker.psum_rw_size[1] += delta_cache;
        // Colwise reg adjust tracker.
        let colwise_reg_tracker = self
            .scheduler
            .colwise_reg_adjust_tracker
            .block_info
            .get_mut(&block_token)
            .unwrap();
        colwise_reg_tracker.miss_size += delta_b;
        colwise_reg_tracker.psum_rw_size[0] += delta_psum;
        colwise_reg_tracker.psum_rw_size[1] += delta_cache;
    }

    pub fn adder_tree_exec(&mut self, idx: usize) {
        trace_println!("\n--adder_tree {}", idx);
        if (self.adder_trees[idx].task.is_none()
            || self
                .scheduler
                .is_window_finished(self.adder_trees[idx].task.as_ref().unwrap().window_token))
            && self.adder_trees[idx].idle()
        {
            // Collect output psums.
            if self.adder_trees[idx].task.is_some() {
                let prev_win_token = self.adder_trees[idx].task.as_ref().unwrap().window_token;
                for arow_addr in self.scheduler.window_tracker[&prev_win_token]
                    .arow_addr_pairs
                    .iter()
                {
                    self.scheduler
                        .row_rgstr_task
                        .entry(arow_addr[0])
                        .and_modify(|e| *e -= 1);
                    if self.scheduler.b_row_lens.contains_key(&arow_addr[1]) {
                        self.scheduler
                            .output_tracker
                            .entry(arow_addr[0])
                            .and_modify(|ps| {
                                if !ps.contains(&arow_addr[1]) {
                                    ps.push(arow_addr[1]);
                                }
                            })
                            .or_insert(vec![arow_addr[1]]);
                    }
                }
            }
            // Collect stats of the prev finished task.
            if self.adder_trees[idx].task.is_some()
                && !self.adder_trees[idx].task.as_ref().unwrap().merge_mode
            {
                let prev_blk_tk = self.adder_trees[idx].task.as_ref().unwrap().block_token;
                if self.scheduler.is_block_finished(prev_blk_tk) {
                    // Label finished rows.
                    self.scheduler.label_finished_rows(prev_blk_tk);
                }
            }
            // Swapout those finished rows.
            self.swapout_finished_psums();
            // Assign new tasks.
            let task = self.scheduler.assign_in_cache_merge_task(
                &mut self.adder_trees[idx],
                &self.fiber_cache,
                self.exec_cycle,
            );
            self.adder_trees[idx].set_task(task);
            // trace_println!("new task: {:?}", &self.adder_trees[idx].task);
        }
        if self.adder_trees[idx].task.is_some() {
            let block_token = self.adder_trees[idx].task.as_ref().unwrap().block_token;
            let block_tracker = &self.scheduler.block_tracker[&block_token];
            let window_token = self.adder_trees[idx].task.as_ref().unwrap().window_token;
            let window_tracker = &self.scheduler.window_tracker[&window_token];
            trace_println!(
                "-block anchor: {:?} block shape {:?} window anchor: {:?} window shape: {:?}",
                block_tracker.anchor,
                block_tracker.shape,
                window_tracker.anchor,
                window_tracker.shape
            );
        } else {
            return;
        }
        // Stream data from each b row.
        for lane_idx in 0..self.adder_trees[idx].tree_width {
            if self.adder_trees[idx]
                .merge_tree
                .is_leaf_node_empty(lane_idx)
            {
                let b = self.stream_b_element(idx, lane_idx, self.exec_cycle);
                self.adder_trees[idx].merge_tree.push_element(lane_idx, b);
            }
        }
        // Update merge tree & pop an sorted element.
        let sb = self.adder_trees[idx].merge_tree.update();
        // Set b element to multiplier.
        let prod = self.adder_trees[idx].multiplier.retrieve_c();
        self.adder_trees[idx].multiplier.set_b(sb);
        self.adder_trees[idx].multiplier.multiply();
        // Push prod to the adder.
        let psum = self.adder_trees[idx].adder.add(prod);
        // Write back.
        self.adder_tree_write_psum(idx, psum);
    }

    pub fn adder_tree_write_psum(&mut self, idx: usize, mut element: Option<Element>) {
        if self.adder_trees[idx].task.is_none() || element.is_none() {
            return;
        }
        let task = self.adder_trees[idx].task.as_ref().unwrap();
        let window_tracker = self
            .scheduler
            .window_tracker
            .get(&task.window_token)
            .unwrap();
        let arow_addr = window_tracker.arow_addr_pairs[0];
        element.as_mut().unwrap().idx[0] = arow_addr[1];
        self.scheduler
            .b_row_lens
            .entry(arow_addr[1])
            .or_default()
            .add_assign(1);
        self.fiber_cache
            .append_element_to(arow_addr[1], element.unwrap());
    }

    pub fn stream_b_element(
        &mut self,
        idx: usize,
        lane_idx: usize,
        cur_cycle: usize,
    ) -> Option<Element> {
        if self.adder_trees[idx].task.is_none() {
            return Some(Element::new([usize::MAX; 2], 0.0));
        }
        let task = self.adder_trees[idx].task.as_mut().unwrap();
        let window_tracker = self
            .scheduler
            .window_tracker
            .get_mut(&task.window_token)
            .unwrap();
        let scalar_idx = window_tracker.lane2idx[lane_idx];
        if scalar_idx.is_none() {
            return Some(Element::new([usize::MAX; 2], 0.0));
        }
        let scalar_idx = scalar_idx.unwrap();
        let b_col_idx = window_tracker.b_cols_assigned[lane_idx];
        let element = if task.merge_mode {
            self.fiber_cache
                .request_consume_scalars(scalar_idx, b_col_idx, 1, cur_cycle, true)
                .map(|mut es| {
                    if es.len() == 0 {
                        Element::new([usize::MAX; 2], 0.0)
                    } else {
                        window_tracker.b_cols_assigned[lane_idx] += 1;
                        es.pop().unwrap()
                    }
                })
        } else {
            self.fiber_cache
                .request_read_scalars(scalar_idx, b_col_idx, 1, cur_cycle, true)
                .map(|mut es| {
                    if es.len() == 0 {
                        Element::new([usize::MAX; 2], 0.0)
                    } else {
                        window_tracker.b_cols_assigned[lane_idx] += 1;
                        es.pop().unwrap()
                    }
                })
        };

        return element;
    }
}
