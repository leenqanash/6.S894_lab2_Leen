# One reproducible run: all GPU measurements, fresh winner confirmation, report.
include("run.jl")
include("report.jl")
include("confirm.jl")
include("postprocess.jl")
function execute()
    root=joinpath(@__DIR__,"results")
    before=Set(filter(isdir,readdir(root;join=true)))
    try
        main()
        added=setdiff(Set(filter(isdir,readdir(root;join=true))),before)
        dir=only(added)
        if get(ENV,"SIZE","8192")=="8192"
            confirm(dir)
            postprocess(dir)
        end
        write(joinpath(dir,"COMPLETE"),"All sweep and confirmation phases completed $(now())\n")
        report(dir)
        println("FINISHED_RESULTS_DIRECTORY=",dir)
    catch err
        added=setdiff(Set(filter(isdir,readdir(root;join=true))),before)
        for dir in added
            isfile(joinpath(dir,"summary.csv")) && try
                report(dir)
            catch report_error
                @warn "Partial report unavailable" exception=report_error
            end
        end
        rethrow()
    end
end
abspath(PROGRAM_FILE)==(@__FILE__) && execute()
