# Run once: julia setup.jl. Then: julia --project=. run.jl
# Only Julia source is published; this creates local Project/Manifest files.
using Pkg
Pkg.activate(@__DIR__)
Pkg.add([PackageSpec(name="CUDA",version="6.5.1"),
         PackageSpec(name="KernelAbstractions",version="0.10.7")])
Pkg.precompile()
