# Build the human-readable report and a structured document payload in Julia.
include("analyze_results.jl")
include("analyze_profile.jl")
using Printf
function finalize(dir)
    analyze_profile(dir)
    analyze(dir)
    rows=readrows(joinpath(dir,"summary.csv"))
    @assert length(rows)==1272
    @assert all(r->r["mismatches"]=="0",rows)
    cpu=readrows(joinpath(dir,"cpu.csv")); @assert length(cpu)==8
    @assert all(r->r["mismatches"]=="0",cpu)
    confirmed=readrows(joinpath(dir,"confirmation.csv"))
    families=readrows(joinpath(dir,"family_confirmation.csv"))
    value(r)=parse(Float64,r["median_ms"])
    fmt(x)=@sprintf("%.2f",x)
    best=confirmed[argmin(value.(confirmed))]
    matchedp1=first(filter(r->r["threads"]==best["threads"] && r["blocks"]==best["blocks"] && r["pixels_per_thread"]=="1" && r["layout"]==best["layout"] && r["assignment"]==best["assignment"],confirmed))
    basemain=first(filter(r->r["threads"]=="128" && r["pixels_per_thread"]=="1" && r["assignment"]=="interleaved" && r["layout"]=="linear",confirmed))
    warp=first(filter(r->r["experiment"]=="single_warp" && r["size"]=="8192" && r["pixels_per_thread"]=="1",rows))
    warpilp=first(filter(r->r["experiment"]=="single_warp" && r["size"]=="8192" && r["pixels_per_thread"]=="2",rows))
    family(mode)=first(filter(r->r["experiment"]==mode,families))
    globalrow=family("coordinates_global");sharedrow=family("coordinates_shared")
    tile=family("tile_global");tileshared=family("tile_shared")
    cpuref=first(filter(r->r["mode"]=="serial" && r["pixels_per_thread"]=="1",cpu))
    cpubest=cpu[argmin(value.(cpu))]
    tables=Any[]
    gputable=[
        ["Julia GPU variant","Median (ms)","Compared with","Speedup"],
        ["One warp, one pixel/thread",fmt(value(warp)),"Baseline","1.00×"],
        ["One warp, two pixels/thread",fmt(value(warpilp)),"One-warp baseline",fmt(value(warp)/value(warpilp))*"×"],
        ["Full GPU: 128 threads × 568 blocks; P=1",fmt(value(basemain)),"One-warp baseline",fmt(value(warp)/value(basemain))*"×"],
        ["Tuned: 512 threads × 284 blocks; P=2; chunked",fmt(value(best)),"One-warp baseline",fmt(value(warp)/value(best))*"×"],
        ["Precomputed coordinates: direct global",fmt(value(globalrow)),"Compute coordinates on GPU",fmt(value(basemain)/value(globalrow))*"×*"],
        ["Direct global tile: 64 × 8",fmt(value(tile)),"Direct 1D global",fmt(value(globalrow)/value(tile))*"×"],
        ["Global → shared, linear",fmt(value(sharedrow)),"Matched direct global",fmt(value(globalrow)/value(sharedrow))*"×"],
        ["Global → shared, 64 × 8 tile",fmt(value(tileshared)),"Matched direct tile",fmt(value(tile)/value(tileshared))*"×"]]
    cputable=Vector{String}[["Julia CPU variant","Median (ms)","Compared with","Speedup"]]
    for r in cpu
        label=(r["mode"]=="serial" ? "One thread" : "Two Julia threads")*"; P="*r["pixels_per_thread"]
        push!(cputable,[label,fmt(value(r)),"One thread; P=1",fmt(value(cpuref)/value(r))*"×"])
    end
    memorytable=[
        ["Input access pattern","Offset (elements)","32-byte sectors/load request","Interpretation"],
        ["Contiguous; stride 1","0 (0 bytes)","4","Aligned, coalesced"],
        ["Contiguous; stride 1","1 (4 bytes)","5","25% more sectors"],
        ["Contiguous; stride 1","4 (16 bytes)","5","25% more sectors"],
        ["Contiguous; stride 1","8 (32 bytes)","4","Aligned, coalesced"],
        ["Contiguous; stride 1","16 (64 bytes)","4","Aligned, coalesced"],
        ["Stride 32","All tested offsets","32","8× the aligned sector count"]]
    tables=[Dict("marker"=>"GPU_RESULTS_TABLE","cells"=>gputable),Dict("marker"=>"CPU_RESULTS_TABLE","cells"=>cputable),Dict("marker"=>"MEMORY_RESULTS_TABLE","cells"=>memorytable)]
    findings=[
        "The matched, computed-coordinate GPU kernel reached $(fmt(value(best))) ms, a $(fmt(value(warp)/value(best)))× speedup over one warp. Its launch uses 512 threads/block, 284 blocks and two adjacent pixels per thread.",
        "The direct-global coordinate versions were effectively tied at about 5.43 ms. Their 0.006 ms difference is too small to establish a unique winner. Coordinate precomputation and transfer are excluded from those kernel timings.",
        "One-warp GPU ILP gave $(fmt(value(warp)/value(warpilp)))×: no meaningful improvement. CPU two-pixel ILP gave $(fmt(value(cpuref)/value(first(filter(r->r["mode"]=="serial" && r["pixels_per_thread"]=="2",cpu)))))× over one pixel at a time. At the same full-GPU launch (512 threads × 284 blocks, chunked), P=1 → P=2 reduced $(fmt(value(matchedp1))) → $(fmt(value(best))) ms, a $(fmt(value(matchedp1)/value(best)))× ILP speedup.",
        "The fastest tested CPU combination was two Julia threads and eight independent pixels, $(fmt(value(cpubest)/1000)) s, or $(fmt(value(cpuref)/value(cpubest)))× faster than the serial one-pixel control. This used a two-CPU Slurm allocation, not the old eight-core C++ comparison.",
        "The best tested shared tile took $(fmt(value(tileshared))) ms versus $(fmt(value(tile))) ms for its matched direct-global tile: $(fmt(100*(value(tileshared)/value(tile)-1)))% longer. Shared memory is implemented and verified, but is slower for this workload.",
        "Use adjacent Float32 elements across warp lanes and a warp starting address aligned to 32 bytes. The profiler measured 4 sectors/request when aligned, 5 when offset by 1 or 4 elements, and 32 with stride 32. The same input-load counts occurred before direct and shared paths: copying into shared memory does not repair a poorly coalesced global load."]
    title="Julia Mandelbrot — 8192 × 8192 — completed November 10, 2026"
    text=title*"\n\n"
    text*="Configuration and validation\nNVIDIA L40S on node1634 (142 SMs, sm_89), Intel Xeon Platinum 8562Y+, two allocated CPUs and two Julia threads. Julia 1.13.0; CUDA.jl/CUDACore 6.5.1; runtime 13.4.0 and compiler 13.5.37 artifacts; host driver 590.48.01. Full environment and source hashes are saved with the raw data.\n"
    text*="The image is 8192 × 8192 = 67,108,864 pixels, Float32, with a maximum of 2,000 iterations. It uses the same zoomed Lab 2 region. One UInt32 output occupies 268,435,456 bytes (256 MiB).\n"
    text*="All 1,272 recorded GPU/memory configurations had zero mismatches. Eight full-size CPU variants matched the GPU reference exactly. Both runnable renderer checks produced checksum 8,849,195,477. Local validation passed 582 CPU/index checks, 66 actual-kernel arithmetic checks using CPU index stubs, and eight allocation checks. CPU stubs are not a replacement for the completed GPU tests.\n\n"
    text*="GPU measurements\nGPU_RESULTS_TABLE\n*Precomputed-coordinate comparisons exclude creating and copying the two coordinate arrays; these total 64 KiB. The primary parallelism speedup uses computed coordinates in both the one-warp and full-GPU kernels. GPU values above are device-event medians, not end-to-end application timings.\n\n"
    text*="CPU measurements\nCPU_RESULTS_TABLE\nP is the number of independent pixel states in progress per thread. The Julia CPU code uses tuple unrolling and ordinary Julia threads; SIMD instruction selection is left to the compiler. These are not measurements of the original explicit AVX-vector C++ implementation. CPU values are wall-clock medians.\n\n"
    text*="Global-memory alignment: measured hardware counters\nMEMORY_RESULTS_TABLE\nOffsets are Float32 elements; each element is four bytes. The allocated input base was measured as zero modulo 128 bytes. Both direct and shared microbenchmark variants were profiled. These sector counts support coalescing directly; cached microbenchmark timings alone did not identify a unique best offset.\n\n"
    text*="Findings\n"
    findings_start=length(text)
    text*=join(findings,"\n")*"\n"
    findings_end=length(text)
    text*="\nWhy shared memory did not win\nThe reusable 64 × 8 tile cut global-load requests from 4,194,304 to 393,216 and L1 global-load sectors from 10,485,760 to 1,179,648. It used 288 bytes of static shared memory per block. Both tile variants used 30 registers/thread. The profiler reported barrier stalls for the shared version and none for the direct version. The small coordinate arrays are already cacheable, so fewer global-load instructions did not translate into a faster kernel. The stall measurements support synchronization overhead as a contributor; they do not prove it is the only cause.\n"
    text*="The main computed-coordinate kernel performed no global input loads: it calculates coordinates in registers and writes output counts. The separate precomputed-coordinate and memory-probe kernels make global → shared loading measurable. The tiled version cooperatively loads contiguous x coordinates and row y coordinates, synchronizes every thread, reuses x across rows, broadcasts y within rows and synchronizes before reusing the tile.\n"
    text*="Nsight Compute 2025.4.0 captured 25 representative launches. Its replay/clock/cache conditions differ from the ordinary benchmark; profiler durations were not substituted for the benchmark medians. Raw metrics, selected metrics, launch order and the complete .ncu-rep report are saved.\n\n"
    text*="Which pixel each thread handles\nThe output is a flat row-major array: zero-based p = row × width + column; the Julia array position is p + 1. This avoids confusing the image layout with Julia's column-major matrix layout.\n"
    text*="For the tuned computed-coordinate kernel, a block contains 512 threads and handles 1,024 pixels per batch. With chunked P=2 assignment, zero-based thread 0 handles pixels 0 and 1; thread 1 handles 2 and 3; lane 31 handles 62 and 63. Block 1 starts at pixel 1,024. The next grid-stride batch advances by 284 × 512 × 2 = 290,816 pixels.\n"
    text*="The interleaved comparison instead gives thread t the pixels tile + t and tile + t + threads_per_block. The transposed comparison assigns neighboring lanes to width-strided pixels. All variants cover each pixel exactly once; the mapping CSV lists concrete block/thread/warp/lane examples.\n"
    text*="The 64 × 8 shared tile uses 512 threads, adjacent warp lanes following the 64-column direction. Its x loads start on 256-byte tile boundaries and its y loads are cooperative. This implements the aligned global-to-shared path tested above.\n\n"
    text*="What was varied and recorded\nImage widths: 1024, 2048, 4096, 8192. Threads/block: 64, 128, 256, 512. Blocks: 1, 2, 4, 8 times the 142-SM count. Pixels/thread: 1, 2, 4, 8. Layouts: linear and transposed. Assignments: interleaved and chunked. Single-SM tests used 64, 128, 256, 512, 1024 threads in one block. Shared-coordinate tiles used widths 32/64 and heights 2/4/8. The independent memory probe used strides 1/2/4/8/32 and offsets 0/1/2/4/8/16/31 Float32 elements.\n"
    text*="Each sweep setting used compilation warmup, five timed trials and a correctness check. The main GPU configuration order was randomized with a recorded seed. Eleven leading/main-baseline configurations and four memory families received fresh 15-trial confirmations. The CPU phase saved five wall-time trials for each of eight variants. Device-event and synchronized wall times are stored separately. Allocation, validation and output transfer were outside the timed kernel samples.\n"
    transfer=strip(read(joinpath(dir,"transfer.txt"),String))
    text*="Output transfer was measured separately: $transfer Renderer logs also record cold coordinate setup and output transfer with allocation; neither was presented as a kernel-only speedup.\n"
    text*="Initial boxed tuple captures caused a GPU compilation error and excessive CPU allocation. Immutable typed state helpers fixed both paths. The allocating CPU attempt was paused before any usable CPU timing was accepted. The successful GPU sweep was retained; the corrected CPU, memory and confirmation phases were then completed. Original source snapshots, failed-attempt logs and completion provenance are preserved. All reported CPU results use the corrected implementation.\n"
    text*="The Mac-only exploratory runs are retained separately and are not the controlled Engaging results. They ran during setup activity and were not used for the final speedups.\n\n"
    text*="How to run and where to find the record\nLocal project: julia_lab2, opened in VS Code. Engaging project: /home/leenq/lab2/julia-8192-20261110. Primary completed data: results/20261110-142947-257797 on Engaging, copied locally to engaging_results/20261110-142947-257797.\n"
    text*="For one image on an allocated NVIDIA GPU: julia --project=. render.jl global\nFor the optimized global → shared tile: julia --project=. render.jl tile_shared\nFor a fresh full experiment: julia --project=. --threads=2 execute.jl\nThe Engaging scheduler wrapper is engaging.sbatch. SAVE_IMAGE=1 optionally saves a binary PGM. The final workflow records source hashes, CSV trials, hardware counters and readable reports. No C++ computation is used.\n\n"
    text*="References\nCUDA.jl kernel API: https://cuda.juliagpu.org/stable/api/kernel/\nCUDA.jl timing and profiling: https://cuda.juliagpu.org/stable/development/profiling/\nNVIDIA memory coalescing guidance: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/\n\n"
    text*="Julia implementation and reproducibility source\n"
    code_start=length(text)
    sourcefiles=["Project.toml","core.jl","gpu.jl","render.jl","run.jl","execute.jl","confirm.jl","confirm_families.jl","profile.jl","postprocess.jl","report.jl","analyze_results.jl","analyze_profile.jl","test_cpu.jl","test_kernel_logic.jl","mapping_samples.jl","setup.jl","engaging.sbatch"]
    for file in sourcefiles
        text*="\n$file\n"*read(joinpath(@__DIR__,file),String)*"\n"
    end
    text*="\nRecorded Engaging environment and source hashes\n"*read(joinpath(dir,"environment.txt"),String)
    payload=Dict("title"=>title,"text"=>text,"tables"=>tables,"findings"=>findings,"code_start_character_offset"=>code_start)
    mkpath(joinpath(@__DIR__,"documentation"))
    write(joinpath(@__DIR__,"documentation","document_payload.json"),json_value(payload))
    write(joinpath(@__DIR__,"documentation","appendix_final.txt"),text)
    open(joinpath(@__DIR__,"FINAL_RESULTS.md"),"w") do io
        println(io,"# Julia Mandelbrot: 8192 × 8192\n\nCompleted on Engaging's NVIDIA L40S, November 10, 2026. All 1,272 GPU/memory configurations and eight full-size CPU variants matched their references.\n")
        println(io,"- Main GPU kernel: **$(fmt(value(best))) ms**, **$(fmt(value(warp)/value(best)))×** versus one warp.\n- Adding two-pixel ILP at the matched full-GPU launch: **$(fmt(value(matchedp1)/value(best)))×** (6.41 → 5.84 ms).\n- Direct global-coordinate kernels: **about 5.43 ms**.\n- Best shared tile: **$(fmt(value(tileshared))) ms**, **$(fmt(100*(value(tileshared)/value(tile)-1)))% longer** than its matched direct tile.\n- CPU two-pixel ILP: **$(fmt(value(cpuref)/value(first(filter(r->r["mode"]=="serial" && r["pixels_per_thread"]=="2",cpu)))))×**. Best tested two-thread CPU: **$(fmt(value(cpubest)/1000)) s**, **$(fmt(value(cpuref)/value(cpubest)))×** versus serial one-pixel control.\n")
        for (heading,table) in (("GPU timings",gputable),("CPU timings",cputable),("Alignment counters",memorytable))
            println(io,"## $heading\n")
            println(io,"| ",join(table[1]," | ")," |")
            println(io,"| ",join(fill("---",length(table[1]))," | ")," |")
            for row in table[2:end];println(io,"| ",join(row," | ")," |");end
            println(io)
        end
        println(io,"GPU numbers are kernel-event medians. Precomputation, allocation and output copy are excluded; preallocated output copy median was 12.29 ms. CPU numbers are wall time. The CPU code uses independent pixel chains and compiler-selected SIMD. Hardware counters confirmed 4/5/32 sectors per load request for aligned/misaligned/stride-32 accesses. The complete .ncu-rep is stored separately from the lightweight ZIP.\n")
        println(io,"## Follow the code\n\n- [One-image renderer](render.jl)\n- [GPU kernels and shared tile](gpu.jl)\n- [CPU and recurrence](core.jl)\n- [Full experiment](execute.jl)\n- [Raw recorded results](engaging_results/20261110-142947-257797/)\n\nThe Google Doc contains the detailed results, explanation and source appendix. The original C++ files were preserved.\n")
    end
    println("Final report and document payload generated")
end
abspath(PROGRAM_FILE)==(@__FILE__) && finalize(only(ARGS))
