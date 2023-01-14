use crate::scheduler::Task;
use crate::storage::Element;
use crate::{trace_print, trace_println};
use std::mem;

pub struct Multiplier {
    a_eles: Option<Vec<Element>>,
    b: Option<Element>,
    c: Option<Element>,
}

impl Multiplier {
    pub fn new() -> Multiplier {
        Multiplier {
            a_eles: None,
            b: None,
            c: None,
        }
    }

    pub fn set_as(&mut self, a_elements: Option<Vec<Element>>) {
        self.a_eles = a_elements;
    }

    pub fn set_b(&mut self, b: Option<Element>) {
        if b.is_none() || b.is_some() && b.as_ref().unwrap().idx == [usize::MAX; 2] {
            self.b = None;
        } else {
            self.b = b;
        }
    }

    pub fn retrieve_c(&mut self) -> Option<Element> {
        return self.c.clone();
    }

    pub fn multiply(&mut self) {
        if self.a_eles.is_none() || self.b.is_none() {
            self.c = None;
        } else {
            let b = self.b.as_ref().unwrap();
            match self
                .a_eles
                .as_ref()
                .unwrap()
                .iter()
                .find(|a| a.idx[1] == b.idx[0])
            {
                Some(a) => {
                    self.c = Some(Element::new([a.idx[0], b.idx[1]], a.value * b.value));
                }
                None => {
                    panic!("Mistach index b: {:?} a: {:?}", &b, &self.a_eles);
                }
            }
        }
    }

    pub fn idle(&self) -> bool {
        return self.a_eles.is_none() || self.b.is_none();
    }
}

pub struct Adder {
    cur: Option<Element>,
}

impl Adder {
    pub fn new() -> Adder {
        Adder { cur: None }
    }

    pub fn add(&mut self, input: Option<Element>) -> Option<Element> {
        if self.cur.is_some()
            && input.is_some()
            && self.cur.as_ref().unwrap().idx == input.as_ref().unwrap().idx
        {
            self.cur.as_mut().unwrap().value += input.as_ref().unwrap().value;
            None
        } else {
            mem::replace(&mut self.cur, input)
        }
    }

    pub fn idle(&self) -> bool {
        self.cur.is_none()
    }
}

pub struct MergeTree {
    pub tree_width: usize,
    pub tree_depth: usize,
    pub merge_tree: Vec<Vec<Option<Element>>>,
    pub drained: Vec<Vec<bool>>,
}

impl MergeTree {
    pub fn new(tree_width: usize) -> MergeTree {
        let mut layer_width = 1;
        let mut tree_depth = 0;
        let mut tree = vec![];
        let mut drained = vec![];
        while layer_width <= tree_width {
            tree.push(vec![None; layer_width]);
            drained.push(vec![false; layer_width]);
            tree_depth += 1;
            layer_width *= 2;
        }
        tree_depth -= 1;

        MergeTree {
            tree_width,
            tree_depth,
            merge_tree: tree,
            drained,
        }
    }

    pub fn reset(&mut self) {
        for lvl in 0..self.tree_depth + 1 {
            for node in 0..2usize.pow(lvl as u32) {
                self.drained[lvl][node] = false;
            }
        }
    }

    pub fn push_element(&mut self, leaf_idx: usize, element: Option<Element>) {
        if element.is_some() {
            if element.as_ref().unwrap().idx == [usize::MAX; 2] {
                self.drained[self.tree_depth][leaf_idx] = true;
                return;
            }
            assert!(
                mem::replace(&mut self.merge_tree[self.tree_depth][leaf_idx], element).is_none(),
                "Node at {} already has {:?}",
                leaf_idx,
                self.merge_tree[self.tree_depth][leaf_idx]
                    .as_ref()
                    .unwrap()
                    .idx
            );
        }
    }

    pub fn update(&mut self) -> Option<Element> {
        let popped = self.merge_tree[0][0].take();
        for lvl in 1..self.tree_depth + 1 {
            for left_node in (0..2usize.pow(lvl as u32)).step_by(2) {
                if self.merge_tree[lvl - 1][left_node / 2].is_some() {
                    continue;
                }
                let right_node = left_node + 1;
                // Elevate the smallest valid element.
                if (self.drained[lvl][right_node] || self.merge_tree[lvl][right_node].is_some())
                    && (self.drained[lvl][left_node] || self.merge_tree[lvl][left_node].is_some())
                {
                    let p = self.compare_pop(lvl, left_node, right_node);
                    self.merge_tree[lvl - 1][left_node / 2] = p;
                }
                // If the two child nodes are drained and empty, the parent node can be labelled drained.
                if self.drained[lvl][right_node]
                    && self.drained[lvl][left_node]
                    && self.merge_tree[lvl][left_node].is_none()
                    && self.merge_tree[lvl][right_node].is_none()
                {
                    self.drained[lvl - 1][left_node / 2] = true;
                }
            }
        }
        return popped;
    }

    pub fn compare_pop(
        &mut self,
        lvl: usize,
        left_node: usize,
        right_node: usize,
    ) -> Option<Element> {
        let le = &self.merge_tree[lvl][left_node];
        let re = &self.merge_tree[lvl][right_node];
        if re.is_none()
            || le.is_some() && le.as_ref().unwrap().idx[1] <= re.as_ref().unwrap().idx[1]
        {
            self.merge_tree[lvl][left_node].take()
        } else {
            self.merge_tree[lvl][right_node].take()
        }
    }

    pub fn idle(&self) -> bool {
        for layer in self.merge_tree.iter() {
            if layer
                .iter()
                .any(|n| n.is_some() && n.as_ref().unwrap().idx != [usize::MAX; 2])
            {
                return false;
            }
        }

        return true;
    }

    pub fn print_merge_tree(&self) {
        trace_println!("merge_tree:");
        for lvl in 0..self.tree_depth + 1 {
            trace_print!(
                "{:?} ",
                self.merge_tree[lvl]
                    .iter()
                    .map(|e| e.as_ref().map(|_e| _e.idx))
                    .collect::<Vec<Option<[usize; 2]>>>()
            );
        }
        trace_println!("");
        for lvl in 0..self.tree_depth + 1 {
            trace_print!("{:?}", self.drained[lvl]);
        }
    }

    pub fn is_leaf_node_empty(&self, leaf_node: usize) -> bool {
        self.merge_tree[self.tree_depth][leaf_node].is_none()
    }
}

pub struct AdderTree {
    pub pe_idx: usize,
    pub tree_width: usize,
    pub merge_tree: MergeTree,
    pub multiplier: Multiplier,
    pub adder: Adder,
    pub task: Option<Task>,
    pub mem_finish_cycle: Option<usize>,
}

impl AdderTree {
    pub fn new(pe_idx: usize, tree_width: usize) -> AdderTree {
        AdderTree {
            pe_idx,
            tree_width,
            merge_tree: MergeTree::new(tree_width),
            multiplier: Multiplier::new(),
            adder: Adder::new(),
            task: None,
            mem_finish_cycle: None,
        }
    }

    pub fn idle(&self) -> bool {
        let is_idle = self.merge_tree.idle() && self.multiplier.idle() && self.adder.idle();
        return is_idle;
    }

    pub fn set_task(&mut self, task: Option<(usize, Task)>) -> usize {
        if task.is_none() {
            self.multiplier.set_as(None);
            self.task = None;
            self.mem_finish_cycle = None;
            return 0;
        } else {
            let (a_latency, task) = task.unwrap();
            self.task = Some(task.clone());
            self.merge_tree.reset();
            self.multiplier
                .set_as(Some(task.a_eles.iter().filter_map(|e| e.clone()).collect()));
            self.mem_finish_cycle = None;
            return a_latency;
        }
    }
}
