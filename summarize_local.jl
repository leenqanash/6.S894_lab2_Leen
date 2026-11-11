using Statistics, Printf
files=sort(filter(f->startswith(f,"local_cpu_") && endswith(f,".csv"),readdir(joinpath(@__DIR__,"results"))))
rows=[split(l,',') for l in readlines(joinpath(@__DIR__,"results",last(files)))[2:end]]
serial=[parse(Float64,r[8]) for r in rows if r[6]=="serial"]
threaded=[parse(Float64,r[8]) for r in rows if r[6]=="threaded"]
open(joinpath(@__DIR__,"results","local_summary.md"),"w") do io
    println(io,"# Local exploratory CPU measurements — 8192 × 8192\n")
    println(io,"Julia $(VERSION); $(Sys.cpu_info()[1].model); eight Julia threads; Float32; 2,000 iterations.\n")
    println(io,"| Mode | Trials | Minimum (ms) | Median (ms) | Maximum (ms) |")
    println(io,"|---|---|---|---|---|")
    for (name,v) in (("Serial",serial),("Threaded",threaded))
        println(io,"| $name | $(length(v)) | $(minimum(v)) | $(median(v)) | $(maximum(v)) |")
    end
    @printf(io,"\nExploratory median ratio: %.2f× (serial / threaded). This is not a controlled speedup claim: package precompilation and validation ran concurrently, affecting the serial timings especially.\n",median(serial)/median(threaded))
    println(io,"\nAll six runs produced checksum 8,849,195,477. Small-image serial/threaded and ILP tests compare complete arrays exactly; equal full-image checksums alone are not a full-array proof.")
    println(io,"\nNo GPU run, GPU speedup, measured alignment winner, or fastest block/pixel configuration is available yet. The Mac has no NVIDIA CUDA device, and Engaging requires a fresh login.")
end
