# Profiling
This document is meant to track measured profiling times, to compare performance of implemented optimizations.

## 0 Measurement
### Release mode. Single file

|                  | **Runs**    | **Reconstruction** |
|------------------|-------------|--------------------|
|                  | 108.53ms    | 34.35ms            |
|                  | 110.55ms    | 25.41ms            |
|                  | 111.70ms    | 21.15ms            |
|                  | 101.99ms    | 21.15ms            |
|                  | 100.74ms    | 23.73ms            |
|                  |             |                    |
| **Avg. Runtime** | **106,7ms** | **~23.5ms**        |

Most notable bottlenecks:
```shell
reconstruction: 91.37%, 65.09ms avg, 1 call (total: 0.065s)
  subdomain reconstruction (dense): ≈97.79%, 21.56ms avg, 4 calls (total: 0.086s)
    density grid loop: 70.75%, 15.26ms avg, 4 calls (total: 0.061s)
    mc triangulation loop: 27.18%, 5.86ms avg, 4 calls (total: 0.023s)
...
postprocessing: 24.18%, 24.25ms avg, 1 call (total: 0.024s)
   marching_cubes_cleanup: 57.48%, 13.94ms avg, 1 call (total: 0.014s)
     determine nearest grid points: 1.76%, 0.25ms avg, 1 call (total: 0.000s)
     mesh displacement: 98.20%, 13.69ms avg, 1 call (total: 0.014s)
       construct_half_edge_mesh: 46.89%, 6.42ms avg, 1 call (total: 0.006s)
       mesh displacement iteration: 19.75%, 1.35ms avg, 2 calls (total: 0.003s)
```

```shell
[21:09:20.887][INFO] reconstruct subcommand: 100.00%, 133.81ms avg, 1 call (total: 0.134s)
[21:09:20.887][INFO]  . surface reconstruction: 99.56%, 133.22ms avg, 1 call (total: 0.133s)
[21:09:20.887][INFO]  .  . loading particle positions: 0.25%, 0.34ms avg, 1 call (total: 0.000s)
[21:09:20.887][INFO]  .  . compute minimum enclosing aabb: 0.50%, 0.67ms avg, 1 call (total: 0.001s)
[21:09:20.887][INFO]  .  . surface reconstruction subdomain-grid: 68.19%, 90.84ms avg, 1 call (total: 0.091s)
[21:09:20.887][INFO]  .  .  . decomposition: 0.66%, 0.60ms avg, 1 call (total: 0.001s)
[21:09:20.887][INFO]  .  .  .  . classifying particles: 18.98%, 0.11ms avg, 1 call (total: 0.000s)
[21:09:20.888][INFO]  .  .  .  . merging TL per cell particle counters: 7.61%, 0.05ms avg, 1 call (total: 0.000s)
[21:09:20.888][INFO]  .  .  .  . initializing flat subdomain data and index mapping: 0.65%, 0.00ms avg, 1 call (total: 0.000s)
[21:09:20.888][INFO]  .  .  .  . copying particles to subdomains: 31.39%, 0.19ms avg, 1 call (total: 0.000s)
[21:09:20.888][INFO]  .  .  .  . sort subdomain particles: 16.46%, 0.10ms avg, 1 call (total: 0.000s)
[21:09:20.888][INFO]  .  .  . compute_global_density_vector: 4.55%, 4.13ms avg, 1 call (total: 0.004s)
[21:09:20.888][INFO]  .  .  .  . subdomain density computation: ≈100.00%, 0.59ms avg, 8 calls (total: 0.005s)
[21:09:20.888][INFO]  .  .  .  .  . collect subdomain data: 0.70%, 0.00ms avg, 8 calls (total: 0.000s)
[21:09:20.888][INFO]  .  .  .  .  . initialize particle filter: 0.47%, 0.00ms avg, 8 calls (total: 0.000s)
[21:09:20.889][INFO]  .  .  .  .  . neighborhood_search_spatial_hashing_flat_filtered: 77.02%, 0.45ms avg, 8 calls (total: 0.004s)
[21:09:20.889][INFO]  .  .  .  .  .  . sequential_generate_cell_to_particle_map: 18.82%, 0.09ms avg, 8 calls (total: 0.001s)
[21:09:20.889][INFO]  .  .  .  .  .  . write particle neighbors: 78.86%, 0.36ms avg, 8 calls (total: 0.003s)
[21:09:20.889][INFO]  .  .  .  .  . sequential_compute_particle_densities_filtered: 13.45%, 0.08ms avg, 8 calls (total: 0.001s)
[21:09:20.889][INFO]  .  .  .  .  . update global density values: 0.12%, 0.00ms avg, 8 calls (total: 0.000s)
[21:09:20.889][INFO]  .  .  .  .  . update global neighbor list: 4.97%, 0.03ms avg, 8 calls (total: 0.000s)
[21:09:20.889][INFO]  .  .  . reconstruction: 91.26%, 82.90ms avg, 1 call (total: 0.083s)
[21:09:20.889][INFO]  .  .  .  . subdomain reconstruction (dense): ≈97.93%, 26.93ms avg, 4 calls (total: 0.108s)
[21:09:20.890][INFO]  .  .  .  .  . density grid loop: 74.10%, 19.95ms avg, 4 calls (total: 0.080s)
[21:09:20.890][INFO]  .  .  .  .  . mc triangulation loop: 22.68%, 6.11ms avg, 4 calls (total: 0.024s)
[21:09:20.890][INFO]  .  .  .  . subdomain reconstruction (sparse): ≈2.07%, 0.57ms avg, 4 calls (total: 0.002s)
[21:09:20.890][INFO]  .  .  .  .  . density grid loop: 15.22%, 0.09ms avg, 4 calls (total: 0.000s)
[21:09:20.890][INFO]  .  .  .  .  . mc triangulation loop: 0.05%, 0.00ms avg, 4 calls (total: 0.000s)
[21:09:20.890][INFO]  .  .  . stitching: 2.28%, 2.07ms avg, 1 call (total: 0.002s)
[21:09:20.890][INFO]  .  .  .  . surface patch offset scan: 1.20%, 0.02ms avg, 1 call (total: 0.000s)
[21:09:20.890][INFO]  .  .  .  . copy interior verts/tris and deduplicate exterior verts: 14.37%, 0.30ms avg, 1 call (total: 0.000s)
[21:09:20.891][INFO]  .  . postprocessing: 25.48%, 33.94ms avg, 1 call (total: 0.034s)
[21:09:20.891][INFO]  .  .  . marching_cubes_cleanup: 54.91%, 18.64ms avg, 1 call (total: 0.019s)
[21:09:20.891][INFO]  .  .  .  . determine nearest grid points: 2.09%, 0.39ms avg, 1 call (total: 0.000s)
[21:09:20.891][INFO]  .  .  .  . mesh displacement: 97.88%, 18.24ms avg, 1 call (total: 0.018s)
[21:09:20.891][INFO]  .  .  .  .  . construct_half_edge_mesh: 46.66%, 8.51ms avg, 1 call (total: 0.009s)
[21:09:20.891][INFO]  .  .  .  .  . mesh displacement iteration: 26.13%, 2.38ms avg, 2 calls (total: 0.005s)
[21:09:20.891][INFO]  .  .  . initialize interpolator: 3.91%, 1.33ms avg, 1 call (total: 0.001s)
[21:09:20.891][INFO]  .  .  .  . build R-tree: 68.12%, 0.90ms avg, 1 call (total: 0.001s)
[21:09:20.892][INFO]  .  .  . compute smoothing weights: 12.23%, 4.15ms avg, 1 call (total: 0.004s)
[21:09:20.892][INFO]  .  .  .  . interpolate weighted neighbor counts: 84.17%, 3.49ms avg, 1 call (total: 0.003s)
[21:09:20.892][INFO]  .  .  .  .  . interpolate_quantity_inplace: 99.60%, 3.48ms avg, 1 call (total: 0.003s)
[21:09:20.892][INFO]  .  .  . mesh smoothing: 12.53%, 4.25ms avg, 1 call (total: 0.004s)
[21:09:20.892][INFO]  .  .  .  . laplacian_smoothing: 95.86%, 4.08ms avg, 1 call (total: 0.004s)
[21:09:20.892][INFO]  .  .  .  .  . laplacian_smoothing iter: 97.19%, 0.16ms avg, 25 calls (total: 0.004s)
[21:09:20.893][INFO]  .  .  . compute normals: 10.46%, 3.55ms avg, 1 call (total: 0.004s)
[21:09:20.893][INFO]  .  .  .  . mesh.par_vertex_normals: 30.31%, 1.08ms avg, 1 call (total: 0.001s)
[21:09:20.893][INFO]  .  .  .  . par_laplacian_smoothing_normals_inplace: 51.95%, 1.84ms avg, 1 call (total: 0.002s)
[21:09:20.893][INFO]  .  .  .  .  . smoothing iteration: 93.66%, 0.17ms avg, 10 calls (total: 0.002s)
[21:09:20.893][INFO]  .  . write surface mesh to file: 4.87%, 6.48ms avg, 1 call (total: 0.006s)
[21:09:20.893][INFO]  .  .  . writing mesh: 80.22%, 5.20ms avg, 1 call (total: 0.005s)
[21:09:20.894][INFO] Finished at 2024-06-06T21:09:20.894054+02:00.
[21:09:20.894][INFO] reconstruct subcommand: 100.00%, 133.81ms avg, 1 call (total: 0.134s)

```



---


### Release mode. Multi file

|                  | **Runs**       |
|------------------|----------------|
|                  | 42404.89ms     |
|                  | 42283.58ms     |
|                  | 41235.02ms     |
| **Avg. Runtime** | **41.974,5ms** |


Most notable bottlenecks:
```shell
reconstruct subcommand: 100.00%, 42283.58ms avg, 1 call (total: 42.284s)
surface reconstruction: 100.00%, 2748.78ms avg, 235 calls (total: 645.963s)
  ...
  surface reconstruction subdomain-grid: 40.28%, 1174.84ms avg, 224 calls (total: 263.165s)
  ...
  postprocessing: 51.64%, 1506.30ms avg, 224 calls (total: 337.411s)
```
---


## Optimization 1
Moved subdomain_ijk and cells_per_subdomain to outside of loop

|                  | **Runs** |
|------------------|----------|
|                  | 23.04ms  |
|                  | 21.69ms  |
|                  | 25.33ms  |
|                  |          |
|                  |          |
|                  |          |
| **Avg. Runtime** | ** **    |

Most notable bottlenecks:



## Optimization n
What optimized

### Release mode. Single File

|                  | **Runs** |
|------------------|----------|
|                  |          |
|                  |          |
|                  |          |
|                  |          |
|                  |          |
|                  |          |
| **Avg. Runtime** | ** ms**  |

Most notable bottlenecks:
```shell 

```

---
### Release mode. Multi File

|                  | **Runs** |
|------------------|----------|
|                  |          |
|                  |          |
|                  |          |
|                  |          |
|                  |          |
|                  |          |
| **Avg. Runtime** | ** ms**  |

Most notable bottlenecks:
```shell 

```
---