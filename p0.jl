### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ b5a87946-291f-11ed-3fc3-1f8e9316607e
begin
	import Pkg
	Pkg.activate(Base.current_project())
end

# ╔═╡ a75431d9-ae2c-4804-a92d-d3d64505d412
begin
	using Downloads
	using Logging
	using YAML
	using ZipFile
end

# ╔═╡ ae8268a4-88c4-4873-8505-50f1078c7c47
md"""
## Data Preparation
Purpose of this file is to prepare the data folder and verify the existence of all essential data files or download them
"""

# ╔═╡ a32ebbda-3336-4d9b-bc23-06c7233e347c
begin
	sources_file = joinpath(pwd(), "sources.yml")
	sources = YAML.load_file(sources_file)
	nyc_sources = sources["data-sources"]["nyc"]

	data_destination = sources["output-destination"]
end

# ╔═╡ 3586e907-3244-4126-95f4-c3ae294a38ed
begin
	data_base = joinpath(data_destination, "data", "nyc")
	@info "Storing data to: " data_base
	mkpath(data_base)
end

# ╔═╡ a6823fd6-aa8e-4634-916f-7995f80bb332
for (key, value) in nyc_sources
	filename = key * "." * nyc_sources[key]["filetype"]
	fileurl = nyc_sources[key]["download"]
	file_size = nyc_sources[key]["filesize"]
	filepath = joinpath(data_base, filename)
	if ~isfile(filepath)
		@info "File not found. Downloading." filename file_size
		Downloads.download(fileurl, filepath)
	end
end

# ╔═╡ 0d1a027c-4ad3-4152-82d8-9c6bdca1fad2


# ╔═╡ 2759c602-9a3f-4921-8b3a-cea558a6292a
md"""
##### now managing the tmy files
"""

# ╔═╡ ed40de12-2f8f-477f-80d6-c0a41d4e492c
function unzip(file,exdir="")
    fileFullPath = isabspath(file) ?  file : joinpath(pwd(),file)
    basePath = dirname(fileFullPath)
    outPath = (exdir == "" ? basePath : (isabspath(exdir) ? exdir : joinpath(pwd(),exdir)))
    isdir(outPath) ? "" : mkdir(outPath)
    zarchive = ZipFile.Reader(fileFullPath)
    for f in zarchive.files
        fullFilePath = joinpath(outPath,f.name)
        if (endswith(f.name,"/") || endswith(f.name,"\\"))
            mkdir(fullFilePath)
        else
            write(fullFilePath, read(f))
        end
    end
    close(zarchive)
end

# ╔═╡ d51a6abd-cb87-4413-bd73-2ffde21a0775
# now unzip the US weather data, and I'm going to save a shapefile with the results from voronoi partitioning
begin
	tmy_zip_path = joinpath(data_base, "tmy-files.zip")
	tmy_path = mkpath(joinpath(data_base, "tmy-files"))
	
	unzip(tmy_zip_path, tmy_path)

	pluto_zip_path = joinpath(data_base, "pluto-data.zip")
	unzip(pluto_zip_path, data_base)
end

# ╔═╡ Cell order:
# ╠═b5a87946-291f-11ed-3fc3-1f8e9316607e
# ╠═a75431d9-ae2c-4804-a92d-d3d64505d412
# ╟─ae8268a4-88c4-4873-8505-50f1078c7c47
# ╠═a32ebbda-3336-4d9b-bc23-06c7233e347c
# ╠═3586e907-3244-4126-95f4-c3ae294a38ed
# ╠═a6823fd6-aa8e-4634-916f-7995f80bb332
# ╟─0d1a027c-4ad3-4152-82d8-9c6bdca1fad2
# ╟─2759c602-9a3f-4921-8b3a-cea558a6292a
# ╠═ed40de12-2f8f-477f-80d6-c0a41d4e492c
# ╠═d51a6abd-cb87-4413-bd73-2ffde21a0775
