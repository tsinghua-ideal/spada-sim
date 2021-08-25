use std::any;
use rand::{distributions::Uniform, Rng};

fn get_type_name<T>(_: &T) -> String {
    format!("{}", any::type_name::<T>())
}

#[cfg(feature = "trace_exec")]
#[macro_export]
macro_rules! trace_print {
    ($( $args:expr ),*) => { println!( $( $args ),* ); }
}

// Non-debug version
#[cfg(not(feature = "trace_exec"))]
#[macro_export]
macro_rules! trace_print {
    ($( $args:expr ),*) => {}
}

pub fn gen_rands_from_range(low: usize, high: usize, num: usize) -> Vec<usize> {
    let mut rng = rand::thread_rng();
    let r = Uniform::new(low, high);
    return (0..num).map(|_| rng.sample(&r)).collect();
}