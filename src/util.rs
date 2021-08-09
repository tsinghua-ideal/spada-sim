use std::any;

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