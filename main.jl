# Current Lab 2 implementation: KernelAbstractions, explicit private/shared arrays.
# Setup: julia kernel_abstractions/setup.jl
# Run:   julia --project=kernel_abstractions main.jl
# The earlier CUDA version remains in the original files for comparison.
# Only Julia source is published. Results are recorded locally by run.jl.
include("kernel_abstractions/run.jl")
abspath(PROGRAM_FILE)==(@__FILE__) && run()
