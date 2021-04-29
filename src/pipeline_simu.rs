use crate::storage::Element;
use std::boxed::Box;
use std::rc::Rc;

pub trait Tickable {}

pub struct PipelineSimulator<'a> {
    pub cycle: i64,
    pub stages: Vec<Box<dyn Tickable + 'a>>,
}

impl<'a> PipelineSimulator<'a> {
    pub fn new() -> PipelineSimulator<'a> {
        PipelineSimulator {
            cycle: 0,
            stages: vec![],
        }
    }

    pub fn register_stage(&mut self, component: impl Tickable + 'a) -> usize {
        self.stages.push(Box::new(component));
        self.stages.len() - 1
    }

    pub fn unregister_stage(&mut self, stage_idx: usize) {
        self.stages.remove(stage_idx);
    }
}
