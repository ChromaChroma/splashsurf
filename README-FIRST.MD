

# Readme Teacher UU

## Building

To run the project, make sure to firstly make sure to have the rust toolchain (nightly) installed.
Depending on what version you are trying to run, there are different requirements. 

For our earlier SIMD implementation nightly mode is required, because we use portable simd, a nightly feature of rust. 

For our more recent optimizations, a GPU is required. 
Your environment should also have access to an OpenCL library (opencl.dll).
We have NVidia graphics cards, so we had to install the  [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=11)
(i.e. Windows, x86_64, version 11) and add the opencl.dll path to a LIB environement variable.
[Help reference](https://github.com/kenba/opencl3/issues/63#issuecomment-1793757742).


## Running

To run the application, we have made some build/run scripts with input to profile against.
The files used are _.vtk_ files that include the coordinates of water particles for that (vtk) frame. 
They can be found under the _/data_ directory

- Our single file script runs against the _cube_2366_particles.vtk_. Which is a single frame (file) with 2366 particles.
  (The output can be found under _/data/converted_)
- Our multi file script runs against 258 frames generated from two water columns crashing into each other. (They can be found under
_/data/DoubleDamBreakWithSphere/vtk_)
  (the output can be found under _/data/DoubleDamBreakWithSphere/surface-output_)

These (series of) files can be viewed in applications such as [ParaView](https://www.paraview.org/)

The latest commit is a version that runs our latest optimization attempt, without having to alter any code.
This version is not very compatible with multi file runs, due too long runtime. 


## Output 
Interesting output to look for are the total runtime (ms):
```
[15:27:29.817][INFO] reconstruct subcommand: 100.00%, 12970.69ms avg, 1 call (total: 12.971s)
```
And the reconstruction done within the complete execution:
```
[15:27:29.811][INFO]       reconstruction: 98.01%, 12281.80ms avg, 1 call (total: 12.282s)
[15:27:29.811][INFO]         subdomain reconstruction (dense): ≈99.74%, 4369.19ms avg, 4 calls (total: 17.477s)
[15:27:29.811][INFO]           density grid loop: 79.27%, 3463.52ms avg, 4 calls (total: 13.854s)
[15:27:29.812][INFO]           mc triangulation loop: 7.01%, 306.23ms avg, 4 calls (total: 1.225s)
```
Another one is the kernel execution time (ns). This is the execution time of the kernel itsel (aside from initialization and such):
```
kernel execution duration (ns): 2722816
```

  
