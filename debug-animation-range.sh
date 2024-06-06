#! /bin/bash
../target/debug/splashsurf.exe reconstruct ./data/DoubleDamBreakWithSphere/vtk/ParticleData_Fluid_{}.vtk \
 --start-index=0 --end-index=258 \
 --output-dir ./data/DoubleDamBreakWithSphere/surface-output \
 --quiet \
 -r=0.025 -l=2.0 -c=0.5 -t=0.6 \
 --subdomain-grid=on \
 --mesh-cleanup=on \
 --mesh-smoothing-weights=on \
 --mesh-smoothing-iters=25 \
 --normals=on \
 --normals-smoothing-iters=10  --mt-files=on --mt-particles=off
