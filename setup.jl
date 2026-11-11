using Pkg
# Resolve on the target machine instead of assuming the Mac's manifest applies.
if isfile("Manifest.toml") && !isfile("Manifest.original-mac.toml")
    mv("Manifest.toml","Manifest.original-mac.toml")
end
if !isfile("Manifest.toml")
    Pkg.Registry.update()
end
Pkg.instantiate()
