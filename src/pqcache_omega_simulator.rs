use std::{cmp::min, collections::VecDeque};

use storage::{Element, PriorityCache, VectorStorage};

use crate::frontend::Accelerator;
use crate::scheduler::{Scheduler, Task};
use crate::storage::{self, sorted_element_vec_to_csr_row, CsrMatStorage, CsrRow};
use crate::{trace_print, trace_println};

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
                panic!("Mistach index: a {} b {}", a.idx[1], b.idx[0]);
            }
        }
    }

    pub fn a_idx(&self) -> [usize; 2] {
        if self.a.is_none() {
            return [usize::MAX; 2];
        } else {
            return self.a.as_ref().unwrap().idx;
        }
    }

    pub fn b_idx(&self) -> [usize; 2] {
        if self.b.is_none() {
            return [usize::MAX; 2];
        } else {
            return self.b.as_ref().unwrap().idx;
        }
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
        mt_latency: usize,
    ) -> PE {
        PE {
            stream_buffers: vec![VecDeque::new(); lane_num],
            multipliers: vec![Multiplier::new(); lane_num],
            psum_buffers: vec![VecDeque::new(); lane_num],
            sorting_network: SortingNetwork::new(lane_num, lane_num, pop_num_per_lane, sn_latency),
            merge_tree: MergeTree::new(mt_latency),
            stream_buffer_size: sb_size,
            psum_buffer_size: pb_size,
            lane_num,
            look_aside: false,
            tail_flags: vec![0; lane_num],
            task: None,
        }
    }

    pub fn idle(&self) -> bool {
        let is_idle = self
            .stream_buffers
            .iter()
            .fold(true, |p, fd| p && fd.is_empty())
            && self
                .multipliers
                .iter()
                .fold(true, |p, fd| p && fd.is_empty())
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
            for lane_idx in 0..self.lane_num {
                // trace_println!("tail flag {} {:?} {:?}", tail_flag, self.psum_buffers[lane_idx], self.multipliers);
                if self.psum_buffers[lane_idx].len() >= 3 {
                    tail_flag = min(tail_flag, self.psum_buffers[lane_idx][2].idx[1]);
                // Accumulate for enough elements.
                } else if !self.multipliers[lane_idx].is_empty() {
                    tail_flag = 0;
                }
                // Else the row is all emitted, the tail flag can be set to MAX.
            }
            self.tail_flags[s..s + group_size].fill(tail_flag);
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
        } else if self.look_aside && (lane_idx - 1) / group_size == lane_idx / group_size {
            let (ab_sb, sb) = self.stream_buffers.split_at_mut(lane_idx);
            let ab_sb = ab_sb.last_mut().unwrap();
            let sb = sb.get_mut(0).unwrap();

            if ab_sb.len() <= 1 {
                sb.pop_front()
            } else if sb.len() == 0 {
                ab_sb.pop_front()
            } else {
                if ab_sb[1].idx[1] < sb[0].idx[1] {
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

    pub fn set_task(&mut self, task: Option<Task>) {
        self.task = task.clone();
        if task.is_none() {
            for lane_idx in 0..self.lane_num {
                self.multipliers[lane_idx].set_a(None);
            }
            // Pb, sn, mt remain the previous configuration.
        } else {
            self.sorting_network.group_lane_num = task.as_ref().unwrap().group_size;
            for (lane_idx, e) in task.unwrap().a_eles.into_iter().enumerate() {
                self.multipliers[lane_idx].set_a(e);
            }
        }
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
            scheduler: Scheduler::new(
                pe_num,
                lane_num,
                default_block_shape,
                output_base_addr,
                1.0 - b_matrix.data.len() as f32
                    / (b_matrix.row_num() * b_matrix.mat_shape[0]) as f32,
                a_matrix,
                b_matrix,
                var_factor,
                accelerator,
            ),
            pe_num,
            lane_num,
            fiber_cache: PriorityCache::new(
                cache_size,
                word_byte,
                output_base_addr,
                b_matrix,
                psum_matrix,
            ),
            pes: vec![
                PE::new(
                    sb_size,
                    pb_size,
                    lane_num,
                    pop_num_per_lane,
                    sn_latency,
                    mt_latency
                );
                pe_num
            ],
            a_matrix,
            exec_cycle: 0,
        }
    }

    pub fn execute(&mut self) {
        // Reset the execution round counter.
        self.exec_cycle = 0;
        loop {
            trace_println!("\n---- cycle {}", self.exec_cycle);

            let prev_a_mem_read_count = self.a_matrix.read_count;
            let prev_b_mem_read_count = self.fiber_cache.b_mem.read_count;
            let prev_psum_mem_read_count = self.fiber_cache.psum_mem.read_count;
            let prev_psum_mem_write_count = self.fiber_cache.psum_mem.write_count;
            let prev_miss_count = self.fiber_cache.miss_count;
            let prev_b_evict_count = self.fiber_cache.b_evict_count;
            let prev_psum_evict_count = self.fiber_cache.psum_evict_count;
            let prev_cache_read_count = self.fiber_cache.read_count;
            let prev_cache_write_count = self.fiber_cache.write_count;

            // Fetch data stage.
            for pe_idx in 0..self.pe_num {
                // Assign new jobs if finished or init.
                if (self.pes[pe_idx].task.is_none()
                    || self
                        .scheduler
                        .is_window_finished(self.pes[pe_idx].task.as_ref().unwrap().window_token))
                    && self.pes[pe_idx].idle()
                {
                    // Label finished rows.
                    if self.pes[pe_idx].task.is_some() {
                        let prev_blk_tk = self.pes[pe_idx].task.as_ref().unwrap().block_token;
                        if self.scheduler.is_block_finished(prev_blk_tk) {
                            self.scheduler.label_finished_rows(prev_blk_tk);
                        }
                    }
                    // Swapout those finished rows.
                    self.swapout_finished_psums();
                    let task = self
                        .scheduler
                        .assign_jobs(&mut self.pes[pe_idx], &mut self.a_matrix);
                    self.pes[pe_idx].set_task(task);
                    trace_println!("---pe {} new task: {:?}", pe_idx, &self.pes[pe_idx].task);
                }
                if self.pes[pe_idx].task.is_some() {
                    let block_token = self.pes[pe_idx].task.as_ref().unwrap().block_token;
                    let block_tracker = &self.scheduler.block_tracker[&block_token];
                    let window_token = self.pes[pe_idx].task.as_ref().unwrap().window_token;
                    let window_tracker = &self.scheduler.window_tracker[&window_token];
                    trace_println!("-block anchor: {:?} block shape {:?} window anchor: {:?} window shape: {:?}",
                    block_tracker.anchor, block_tracker.shape, window_tracker.anchor, window_tracker.shape);
                    trace_println!("-arow_addr_pairs: {:?}", window_tracker.arow_addr_pairs);
                    trace_print!("-B assigned:");
                    for r_offset in 0..window_tracker.shape[0] {
                        for c_offset in 0..window_tracker.shape[1] {
                            let lane_pos = r_offset * window_tracker.shape[1] + c_offset;
                            match window_tracker.lane2idx[lane_pos] {
                                None => trace_print!("{:?} None  ", lane_pos),
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
                    trace_println!("");
                }
            }

            if self.scheduler.a_traversed && self.pes.iter().all(|p| p.idle() && p.task.is_none()) {
                break;
            }

            for pe_idx in 0..self.pe_num {
                // Stream buffer fetch data.
                for lane_idx in 0..self.lane_num {
                    let rb_num = self.pes[pe_idx].stream_buffer_size
                        - self.pes[pe_idx].stream_buffers[lane_idx].len();
                    let bs = self.stream_b_row(pe_idx, lane_idx, rb_num);
                    // trace_print!("-stream b {:?} to {}", &bs, lane_idx);
                    self.pes[pe_idx].push_stream_buffer(lane_idx, bs);
                }

                // Production phase.
                trace_print!("-Prod ");
                for lane_idx in 0..self.lane_num {
                    // Pop from stream buffer.
                    let b = self.pes[pe_idx].pop_stream_buffer(lane_idx);
                    // Set b element to multiplier.
                    // trace_print!("-mul b {:?} to {}", &b, lane_idx);
                    let prod = self.pes[pe_idx].multipliers[lane_idx].retrieve_c();
                    self.pes[pe_idx].multipliers[lane_idx].set_b(b);
                    self.pes[pe_idx].multipliers[lane_idx].multiply();
                    // Push prod to the psum buffer.
                    trace_print!("{}:{:?} ", lane_idx, prod.as_ref().map(|p| p.idx));
                    if prod.is_some() {
                        self.pes[pe_idx].push_psum_buffer(lane_idx, prod.unwrap());
                    }
                }
                trace_println!("");

                // Collect psum phase.
                self.pes[pe_idx].update_tail_flags();
                trace_println!("-tail flag: {:?}", &self.pes[pe_idx].tail_flags);
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
                trace_println!(
                    "-collected psums: {:?}",
                    &collected_psums
                        .iter()
                        .map(|p| p.as_ref().map(|_p| _p.idx))
                        .collect::<Vec<Option<[usize; 2]>>>()
                );
                if collected_psums.iter().any(|p| p.is_some()) {
                    self.pes[pe_idx]
                        .sorting_network
                        .push_elements(collected_psums);
                }

                // Sort & merge phase.
                let sorted_elements = self.pes[pe_idx].sorting_network.pop_elements();
                // trace_print!("sorted_elements: {:?} latency: {:?}", &self.pes[pe_idx].sorting_network.elements, &self.pes[pe_idx].sorting_network.latency_counter);
                if sorted_elements.len() > 0 {
                    self.pes[pe_idx].merge_tree.push_elements(sorted_elements);
                }
                // trace_print!("merge_tree: {:?} latency: {:?}", &self.pes[pe_idx].merge_tree.elements, &self.pes[pe_idx].merge_tree.latency_counter);
                let merged_psums = self.pes[pe_idx].merge_tree.pop_elements();
                trace_println!("-merged psum: {:?}", &merged_psums);
                self.write_psums(pe_idx, merged_psums);

                // When a window is finished, collect merge jobs.
                if self.pes[pe_idx].task.is_some()
                    && self
                        .scheduler
                        .is_window_finished(self.pes[pe_idx].task.as_ref().unwrap().window_token)
                {
                    self.scheduler.collect_pending_psums(
                        self.pes[pe_idx].task.as_ref().unwrap().window_token,
                    );
                }
            }

            trace_println!(
                "Cache read_count: + {} -> {}, write_count: + {} -> {}",
                self.fiber_cache.read_count - prev_cache_read_count,
                self.fiber_cache.read_count,
                self.fiber_cache.write_count - prev_cache_write_count,
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
                self.fiber_cache.miss_count - prev_miss_count, self.fiber_cache.miss_count,
                self.fiber_cache.b_evict_count - prev_b_evict_count, self.fiber_cache.b_evict_count,
                self.fiber_cache.psum_evict_count - prev_psum_evict_count, self.fiber_cache.psum_evict_count);
            trace_println!(
                "A mem: read_count: + {} -> {}",
                self.a_matrix.read_count - prev_a_mem_read_count,
                self.a_matrix.read_count
            );
            trace_println!(
                "B mem: read_count: + {} -> {}",
                self.fiber_cache.b_mem.read_count - prev_b_mem_read_count,
                self.fiber_cache.b_mem.read_count
            );
            trace_println!(
                "C mem: read_count: + {} -> {}, write_count: +{} -> {}",
                self.fiber_cache.psum_mem.read_count - prev_psum_mem_read_count,
                self.fiber_cache.psum_mem.read_count,
                self.fiber_cache.psum_mem.write_count - prev_psum_mem_write_count,
                self.fiber_cache.psum_mem.write_count
            );

            self.exec_cycle += 1;
        }
    }

    pub fn stream_b_row(&mut self, pe_idx: usize, lane_idx: usize, rb_num: usize) -> Vec<Element> {
        if self.pes[pe_idx].task.is_none() {
            return vec![];
        }
        let task = self.pes[pe_idx].task.as_ref().unwrap();

        let window_tracker = self
            .scheduler
            .window_tracker
            .get_mut(&task.window_token)
            .unwrap();
        let scalar_idx = window_tracker.lane2idx[lane_idx];
        if scalar_idx.is_none() {
            return vec![];
        }

        let scalar_idx = scalar_idx.unwrap();
        let b_col_idx = window_tracker.b_cols_assigned[lane_idx];
        trace_println!(
            "scalar_idx {:?} b_col_idx {} rb_num {}",
            scalar_idx,
            b_col_idx,
            rb_num
        );
        let elements = if self.pes[pe_idx].task.as_ref().unwrap().merge_mode {
            self.fiber_cache
                .consume_scalars(scalar_idx, b_col_idx, rb_num)
                .unwrap()
        } else {
            self.fiber_cache
                .read_scalars(scalar_idx, b_col_idx, rb_num)
                .unwrap()
        };
        window_tracker.b_cols_assigned[lane_idx] += elements.len();

        return elements;
    }

    pub fn write_psums(&mut self, pe_idx: usize, psums: Vec<Vec<Element>>) {
        if self.pes[pe_idx].task.is_none() {
            return;
        }
        let task = self.pes[pe_idx].task.as_ref().unwrap();

        for (gidx, ps) in psums.into_iter().enumerate() {
            if ps.len() == 0 {
                continue;
            }
            let mut csrrow = sorted_element_vec_to_csr_row(ps);
            let addr = self.scheduler.window_tracker[&task.window_token].arow_addr_pairs[gidx][1];
            // Assign the output address.
            csrrow.rowptr = addr;
            trace_println!("-write_psum: {:?}", &csrrow);
            self.scheduler
                .b_row_lens
                .entry(addr)
                .and_modify(|l| *l += csrrow.len())
                .or_insert(csrrow.len());
            self.fiber_cache.append_psum_to(addr, csrrow);
        }
    }

    pub fn swapout_finished_psums(&mut self) {
        trace_print!("finished a rows: {:?} ", &self.scheduler.finished_a_rows);
        let output_tracker = &mut self.scheduler.output_tracker;
        for id in self.scheduler.finished_a_rows.iter() {
            trace_print!("{:?} ", &output_tracker[id]);
        }
        trace_println!("");
        let swapable_rows = self
            .scheduler
            .finished_a_rows
            .drain_filter(|row| output_tracker.get(row).map_or(true, |ps| ps.len() == 1))
            .collect::<Vec<usize>>();
        trace_println!("swapable_rows: {:?}", &swapable_rows);
        for row in swapable_rows {
            self.fiber_cache.swapout(output_tracker[&row][0]);
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
        self.exec_cycle
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
                let addrs = self.scheduler.output_tracker.get(&rowid).unwrap();
                trace_println!(
                    "Get result: row: {} row len: {}",
                    raw_rowid,
                    self.fiber_cache.psum_mem.data[&addrs[0]].size() / 2
                );
                assert!(
                    addrs.len() == 1,
                    "Partially merged psums! {:?} of row {}",
                    &addrs,
                    raw_rowid
                );
                let addr = addrs[0];
                csrrow = match self.fiber_cache.psum_mem.data.get(&addr) {
                    Some(row) => row.clone(),
                    None => self.fiber_cache.rowmap.get(&addr).unwrap().clone(),
                };
                csrrow.rowptr = raw_rowid;
            }
            c.push(csrrow);
        }
        c.sort_by(|a, b| a.rowptr.cmp(&b.rowptr));
        return c;
    }
}
