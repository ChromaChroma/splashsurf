use std::io;
use std::io::Write;

use opencl3::event::Event;

use crate::Real;

pub fn print_non_zero_values<R: Real>(values: Vec<R>) {
    for (i, x) in values.iter().enumerate() {
        if *x != R::zero() {
            print!("{}->{:?} ", i, x);
        }
    }
    println!();
}

pub fn print_nr_of_zero<R: Real>(values: Vec<R>) {
    println!("Len: {}, Zeros: {} ",
             values.clone().len(),
             values.clone().into_iter().filter(|x| *x == R::zero()).count()
    );
}

pub fn print_first_last_non_zero_index<R: Real>(values: Vec<R>) {
    let first_non_zero_index = values.clone()
        .into_iter()
        .enumerate()
        .find(|(i, x)| *x != R::zero())
        .unwrap()
        .0;
    let last_non_zero_index = values.clone()
        .into_iter()
        .enumerate()
        .rev()
        .find(|(i, x)| *x != R::zero())
        .unwrap()
        .0;

    println!("First idx: {}, Last idx: {}",
             first_non_zero_index,
             last_non_zero_index
    );
}


pub fn write_non_zero_values<R: Real>(file: String, values: Vec<R>) -> io::Result<()> {
    let mut file = std::fs::File::create(file)?;
    for (i, x) in values.iter().enumerate() {
        // if *x != R::zero() {
        write!(file, "{}->{}\n", i, x)?;
        // }
    }
    file.flush()?;

    Ok(())
}
pub fn write_non_zero_indexed_values<R: Real>(file: String, values: Vec<(usize, R)>) -> io::Result<()> {
    let mut file = std::fs::File::create(file)?;
    for (i, (idx, x)) in values.iter().enumerate() {
        if *x != R::zero() {
        write!(file, "{}->{}\n", idx, x)?;
        }
    }
    file.flush()?;

    Ok(())
}


/// Calculate the kernel duration, from the kernel_event
/// returns back the event
#[inline(always)]
pub fn log_kernel_exec_time(kernel_event: &Event, name: &str) {

    let start_time = kernel_event.profiling_command_start().unwrap();
    let end_time = kernel_event.profiling_command_end().unwrap();
    let duration = end_time - start_time;
    println!(
        "kernel ({}) execution duration: {:.3} ms ({}ns)",
        name,
        duration as f64 / 100_000_f64,
        duration,
    );
}
#[macro_export]
macro_rules! exec_time {
    ( $x:expr ) => {
        log_kernel_exec_time($x, "")
    };
    ($x:expr, $name:expr) => {
        log_kernel_exec_time($x, $name)
    };
}