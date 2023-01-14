#[cfg(feature = "trace_exec")]
#[macro_export]
macro_rules! trace_println {
    ($( $args:expr ),*) => { println!( $( $args ),* ); }
}

#[cfg(feature = "trace_exec")]
#[macro_export]
macro_rules! trace_print {
    ($( $args:expr ),*) => { print!( $( $args ),* ); }
}

// Non-debug version
#[cfg(not(feature = "trace_exec"))]
#[macro_export]
macro_rules! trace_print {
    ($( $args:expr ),*) => {};
}

#[cfg(not(feature = "trace_exec"))]
#[macro_export]
macro_rules! trace_println {
    ($( $args:expr ),*) => {};
}
