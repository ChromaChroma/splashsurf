use std::io;
use std::io::Write;

use opencl3::event::Event;

use crate::Real;

fn print_non_zero_values<R: Real>(values: Vec<R>) {
    for (i, x) in values.iter().enumerate() {
        if *x != R::zero() {
            print!("{}->{} ", i, x);
        }
    }
    println!();
}

fn print_nr_of_zero<R: Real>(values: Vec<R>) {
    println!("Len: {}, Zeros: {} ",
             values.clone().len(),
             values.clone().into_iter().filter(|x| *x == R::zero()).count()
    );
}


fn write_non_zero_values<R: Real>(file: String, values: Vec<R>) -> io::Result<()> {
    let mut file = std::fs::File::create(file)?;
    for (i, x) in values.iter().enumerate() {
        if *x != R::zero() {
            write!(file, "{}->{}\n", i, x)?;
        }
    }
    file.flush()?;

    Ok(())
}


/// Calculate the kernel duration, from the kernel_event
/// returns back the event
#[inline(always)]
pub fn log_kernel_exec_time(kernel_event: &Event) {
    let start_time = kernel_event.profiling_command_start().unwrap();
    let end_time = kernel_event.profiling_command_end().unwrap();
    let duration = end_time - start_time;
    println!(
        "kernel execution duration: {:.3} ms ({}ns)",
        duration as f64 / 100_000_f64,
        duration,
    );
}
#[macro_export]
macro_rules! exec_time {
    ( $x:expr ) => {
        log_kernel_exec_time($x)
    };
}