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
end

# ╔═╡ 3586e907-3244-4126-95f4-c3ae294a38ed
begin
	data_base = joinpath(pwd(), "data", "nyc")
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

# ╔═╡ Cell order:
# ╠═b5a87946-291f-11ed-3fc3-1f8e9316607e
# ╠═a75431d9-ae2c-4804-a92d-d3d64505d412
# ╟─ae8268a4-88c4-4873-8505-50f1078c7c47
# ╠═a32ebbda-3336-4d9b-bc23-06c7233e347c
# ╠═3586e907-3244-4126-95f4-c3ae294a38ed
# ╠═a6823fd6-aa8e-4634-916f-7995f80bb332
