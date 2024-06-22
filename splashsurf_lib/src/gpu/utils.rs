use std::ptr;

use num_traits::Zero;
use opencl3::command_queue::{CL_NON_BLOCKING, CL_QUEUE_PROFILING_ENABLE, CommandQueue};
use opencl3::context::Context;
use opencl3::event::Event;
use opencl3::memory::{Buffer, cl_mem_flags, CL_MEM_READ_ONLY};
use opencl3::Result;


#[inline(always)]
pub fn as_read_buffer_non_blocking<R>(context: &Context, queue: &CommandQueue, arr: &[R]) -> Result<(Buffer<R>, Event)> {
    as_buffer_non_blocking(context, queue, arr, CL_MEM_READ_ONLY)
}

#[inline(always)]
pub fn as_buffer_non_blocking<R>(
    context: &Context,
    queue: &CommandQueue,
    arr: &[R],
    cl_mem_flags: cl_mem_flags,
) -> Result<(Buffer<R>, Event)> {
    let output_size: usize = arr.len();
    let mut buffer = unsafe {
        Buffer::<R>::create(&context, cl_mem_flags, output_size, ptr::null_mut())?
    };
    let _buffer_write_event = unsafe {
        queue.enqueue_write_buffer(&mut buffer, CL_NON_BLOCKING, 0, &arr, &[])?
    };

    Ok((buffer, _buffer_write_event))
}
//
// #[inline(always)]
// pub fn new_write_buffer<T: NumCast, U: NumCast>(
//     context: &Context,
//     queue: &CommandQueue,
//     data: &[T],
// ) -> (Buffer<U>, Event) {
//
//
//     let u_data: Vec<U>  = data
//         .to_vec()
//         .iter()
//         .clone()
//         .map(|&x| U::from(x).unwrap())
//         .collect();
//     unsafe {
//         let mut buffer = Buffer::<U>::create(
//             context, CL_MEM_WRITE_ONLY, u_data.len(), ptr::null_mut(),
//         ).expect("Could not create write buffer");
//
//         let write_event = queue.enqueue_write_buffer(
//             &mut buffer, CL_NON_BLOCKING, 0, &u_data, &[],
//         ).expect("Could not enqueue write buffer");
//
//         (buffer, write_event)
//     }
// }

#[inline(always)]
pub fn new_queue(context: &Context) -> CommandQueue {
    #[allow(deprecated)]
    CommandQueue::create_default(context, CL_QUEUE_PROFILING_ENABLE)
        .expect("CommandQueue::create_default failed")
}

#[inline(always)]
pub fn read_buffer_into<U: Zero>(
    queue: &CommandQueue,
    kernel_event: &Event,
    buffer_to_read: &Buffer<U>,
    results: &mut Vec<U>,
) {
    let read_event = unsafe {
        queue.enqueue_read_buffer(
            buffer_to_read,
            CL_NON_BLOCKING,
            0,
            results,
            &vec![kernel_event.get()]
        ).expect("Could not enqueue result read buffer")
    };

    // Wait for the read_event to complete.
    read_event.wait().expect("Could not read event for retrieving data from GPU buffer");
}