#! /bin/bash
cargo run --package splashsurf --bin splashsurf \
  -- \
  reconstruct ./data/cube_2366_particles.vtk \
  --output-dir ./data/converted \
  -r=0.025 -l=2.0 -c=0.5 -t=0.6 \
  --subdomain-grid=on \
  --mesh-cleanup=on  --mesh-smoothing-weights=on   --mesh-smoothing-iters=25 \
  --normals=on --normals-smoothing-iters=10