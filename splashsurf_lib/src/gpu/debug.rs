use std::io;
use std::io::Write;
use crate::Real;

fn print_non_zero_values<R : Real>(values: Vec<R>) {
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


fn write_non_zero_values<R : Real>(file: String, values: Vec<R>) -> io::Result<()> {
    let mut file = std::fs::File::create(file)?;
    for (i, x) in values.iter().enumerate() {
        if *x != R::zero() {
            write!(file, "{}->{}\n", i, x)?;
        }
    }
    file.flush()?;

    Ok(())
}
