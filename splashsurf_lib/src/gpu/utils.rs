use std::ptr;
use opencl3::command_queue::{CL_NON_BLOCKING, CommandQueue};
use opencl3::context::Context;
use opencl3::event::Event;
use opencl3::memory::{Buffer, cl_mem_flags, CL_MEM_READ_ONLY};
use opencl3::Result;
use crate::Real;

#[inline(always)]
pub fn as_read_buffer_non_blocking<R : Real>(context: &Context, queue: &CommandQueue, arr: &[R],) -> Result<(Buffer<R>, Event)> {
    as_buffer_non_blocking(context, queue, arr, CL_MEM_READ_ONLY)
}

#[inline(always)]
pub fn as_buffer_non_blocking<R : Real>(
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