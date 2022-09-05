### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ ac97e0d6-2cfa-11ed-05b5-13b524a094e3
begin
	import Pkg
	Pkg.activate(Base.current_project())
	using CSV
	using DataFrames
	using DataStructures
	using Dates
	using Gadfly
	using Plots
	using Statistics
end;

# ╔═╡ 9b3790d3-8d5d-403c-8495-45def2c6f8ba
md"""
##### Purpose of this is to melt together building data with environmental data
"""

# ╔═╡ 020b96e3-d218-470d-b4b0-fc9b708ffdf3
begin
	data_path = joinpath(pwd(), "data", "nyc")
	input_dir = joinpath(data_path, "p1_o")
	input_dir_environmental = joinpath(data_path, "p2_o")
	output_dir = joinpath(data_path, "p3_o")
	mkpath(output_dir)
end

# ╔═╡ 9aa06073-d43e-4658-adb9-bbc11425978d
begin
	train = CSV.read(joinpath(input_dir, "train.csv"), DataFrame)
	validate = CSV.read(joinpath(input_dir, "validate.csv"), DataFrame)
	test = CSV.read(joinpath(input_dir, "test.csv"), DataFrame)
end;

# ╔═╡ 56f43ba6-568b-436d-85a5-a8da5a0a3956
begin
	# building the paths for all of the environmental features
	era5_p = joinpath(input_dir_environmental, "era5.csv")
	landsat8_p = joinpath(input_dir_environmental, "landsat8.csv")
	lst_aqua_p = joinpath(input_dir_environmental, "lst_aqua.csv")
	lst_terra_p = joinpath(input_dir_environmental, "lst_terra.csv")
	noaa_p = joinpath(input_dir_environmental, "noaa.csv")
	sentinel_1C_p = joinpath(input_dir_environmental, "sentinel_1C.csv")
	sentinel_2A_p = joinpath(input_dir_environmental, "sentinel_2A.csv")
	viirs_p = joinpath(input_dir_environmental, "viirs.csv")
end;

# ╔═╡ c65e56e9-0bde-4278-819c-f3148d71668b
begin
	# the _r prefix is meant to denote that these are closer to "raw" data
	date_f = "yyyy-mm-dd HH:MM:SS"
	era5_r = CSV.read(era5_p, DataFrame; dateformat=date_f)
	landsat8_r = CSV.read(landsat8_p, DataFrame; dateformat=date_f)
	lst_aqua_r = CSV.read(lst_aqua_p, DataFrame; dateformat=date_f)
	lst_terra_r = CSV.read(lst_terra_p, DataFrame; dateformat=date_f)
	lst_r = vcat(lst_aqua_r, lst_terra_r)
	
	noaa_r = CSV.read(noaa_p, DataFrame; dateformat=date_f)
	sentinel_1C_r = CSV.read(sentinel_1C_p, DataFrame; dateformat=date_f)
	sentinel_2A_r = CSV.read(sentinel_2A_p, DataFrame; dateformat=date_f)
	viirs_r = CSV.read(viirs_p, DataFrame; dateformat=date_f)
end;

# ╔═╡ 6b120bfc-e1e3-4b52-a99d-aa60518b32ad
# testing to see how the agg method might work
filter(term -> term ∉ ["Property Id", "date"], names(lst_r))

# ╔═╡ 348c4307-94dc-4d5f-82b0-77dc535c1650
function strip_month!(data::DataFrame)
	data[!,"date"] = Date.(Dates.Year.(data.date), Dates.Month.(data.date))
end

# ╔═╡ 09a4789c-cbe7-496e-98b5-a2c2db3102b6
begin
	# also want to get the training data in a uniform format for matching
	strip_month!(train)

	strip_month!(era5_r)
	strip_month!(landsat8_r)
	strip_month!(lst_r)
	strip_month!(noaa_r)
	strip_month!(sentinel_1C_r)
	strip_month!(sentinel_2A_r)
	strip_month!(viirs_r)
end;

# ╔═╡ ac31e0ac-b35b-494f-814c-3f9eaf26e8b1
function monthly_average(data::DataFrame, agg_terms::Vector{String})
	meanterms = filter(term -> term ∉ agg_terms, names(data))
	combine(groupby(data, agg_terms), meanterms .=> mean ∘ skipmissing, renamecols=false)
end

# ╔═╡ 637220ba-c76a-4210-8c08-fde56b86366a
begin
	agg_terms = 	["Property Id","date"]
	era5 = 			monthly_average(era5_r, agg_terms)
	landsat8 = 		monthly_average(landsat8_r, agg_terms)
	lst = 			monthly_average(lst_r, agg_terms)
	noaa = 			monthly_average(noaa_r, agg_terms)
	sentinel_1C = 	monthly_average(sentinel_1C_r, agg_terms)
	sentinel_2A = 	monthly_average(sentinel_2A_r, agg_terms)
	viirs = 		monthly_average(viirs_r, agg_terms)
end;

# ╔═╡ e4cda53e-7fe2-4cdb-8a9c-c9bdf67dd66f
md""" 
Getting a flavor of what kind of data we have now
"""

# ╔═╡ 39997f75-aa5e-4356-8353-b62b2ff01a98
# begin
# 	captured_nighttimes = filter( row -> ~ismissing(row.LST_Night_1km), lst_r )
# 	sample_nighttemp = filter( row -> row["Property Id"] == 1295925, captured_nighttimes)
# 	Plots.scatter(
# 		sample_nighttemp.date,
# 		sample_nighttemp.LST_Night_1km .* 0.02 .- 273.15,
# 		color=:indianred,
# 		markersize=3,
# 		markerstrokewidth=0
# 	)
# end

# ╔═╡ 7ddf461c-e167-4db0-98c7-940fc962bb6a
training_data = innerjoin(
	train, 
	era5, 
	noaa,
	lst,
	landsat8, 
	viirs, 
	on=agg_terms
)

# ╔═╡ b77aeb0f-3600-4483-a2a8-88aedff5c71a
training_data[:, "Property Id"]

# ╔═╡ fcc7ed57-3e9a-488c-80c7-3a09879aacbe
filter( row -> ~isnan(row.LST_Day_1km), training_data)

# ╔═╡ 3a473c8d-bd0c-4309-b637-84bde0389511
# restricting to just one building for interpretation
sample_training = filter( row -> row["Property Id"] == 1296056, training_data )

# ╔═╡ a85c6b86-b86d-431f-b2ca-e762565144d8
mean_temp_c = sample_training.mean_2m_air_temperature .- 273.15

# ╔═╡ 09a85206-3f09-4ba4-8fe3-85d32c8e8793
lst_means = sample_training.LST_Day_1km .* 0.02 .- 273.15

# ╔═╡ 4821dc90-f71b-4df1-9d31-19431a804048
lst_means .- mean_temp_c

# ╔═╡ f7c1bf95-be33-4f3e-9e29-6b9daba4427c
Gadfly.plot(
	sample_training,
	Gadfly.layer(x=:date, y=:TMP, Geom.point, Theme(default_color="indianred")),
	Gadfly.layer(x=:date, y=mean_temp_c, Geom.point),
	Gadfly.layer(x=:date, y=lst_means, Geom.point, Theme(default_color="orange"))
)

# ╔═╡ 87b55c18-6842-4321-b2c7-c1abd8fef6fc
Gadfly.plot(
	x = sample_training.date,
	y = lst_means .- mean_temp_c,
	Geom.point,
	Geom.line,
	Theme(default_color="indianred"),
	Guide.title("UHI Deviations from City Temperature")
)

# ╔═╡ ee641a2e-1fd3-46b2-af5e-757e8a4faa30


# ╔═╡ eaf8a806-7e31-49f9-bb62-0c9162c82d70


# ╔═╡ e4bf5dcd-d225-409a-ab97-277928fe4977


# ╔═╡ 3547c975-9df0-4377-bb5f-4ac02e037f0c
describe(training_data, :nmissing)

# ╔═╡ 99314fd6-69e2-489b-a9d9-196498b53285
dropmissing(training_data)

# ╔═╡ 26b4e386-8f09-4a2b-89d7-5071d70b52ea
begin
	leftjoin(train, [era5, lst], on=agg_terms)
end

# ╔═╡ 6b549025-4b60-4395-b4a8-3860ac949e90
# val_geojson = GeoJSON.read(read("/Users/thomas/Desktop/Work/Research/uil/postquals/ml_microclimate/data/nyc/p1_o/validate.geojson"))

# ╔═╡ c639b05d-a59e-4237-bed0-d8bda3203222
# georef(val, (:geometry))

# ╔═╡ fa506ceb-5145-4345-94fb-e06b751f4dab


# ╔═╡ afc32be7-6637-4ac6-841c-0954eae916e4
# viz(val.geometry)

# ╔═╡ 4aaad296-1d4e-46d2-8fc9-befa37937a9e
# RIO = GeoTables.gadm("BRA", "Rio de Janeiro", children=true)

# ╔═╡ 6666ff0e-67c4-463e-8a9b-b759e04a9967


# ╔═╡ 83f81b37-46ee-42f2-b6d6-dc96d7f62779
# Plots.plot(
# 	val.geometry
# )

# ╔═╡ Cell order:
# ╠═ac97e0d6-2cfa-11ed-05b5-13b524a094e3
# ╟─9b3790d3-8d5d-403c-8495-45def2c6f8ba
# ╠═020b96e3-d218-470d-b4b0-fc9b708ffdf3
# ╠═9aa06073-d43e-4658-adb9-bbc11425978d
# ╠═56f43ba6-568b-436d-85a5-a8da5a0a3956
# ╠═c65e56e9-0bde-4278-819c-f3148d71668b
# ╠═6b120bfc-e1e3-4b52-a99d-aa60518b32ad
# ╠═348c4307-94dc-4d5f-82b0-77dc535c1650
# ╠═09a4789c-cbe7-496e-98b5-a2c2db3102b6
# ╠═ac31e0ac-b35b-494f-814c-3f9eaf26e8b1
# ╠═637220ba-c76a-4210-8c08-fde56b86366a
# ╟─e4cda53e-7fe2-4cdb-8a9c-c9bdf67dd66f
# ╠═39997f75-aa5e-4356-8353-b62b2ff01a98
# ╟─7ddf461c-e167-4db0-98c7-940fc962bb6a
# ╠═b77aeb0f-3600-4483-a2a8-88aedff5c71a
# ╠═fcc7ed57-3e9a-488c-80c7-3a09879aacbe
# ╠═3a473c8d-bd0c-4309-b637-84bde0389511
# ╠═a85c6b86-b86d-431f-b2ca-e762565144d8
# ╠═09a85206-3f09-4ba4-8fe3-85d32c8e8793
# ╠═4821dc90-f71b-4df1-9d31-19431a804048
# ╟─f7c1bf95-be33-4f3e-9e29-6b9daba4427c
# ╟─87b55c18-6842-4321-b2c7-c1abd8fef6fc
# ╠═ee641a2e-1fd3-46b2-af5e-757e8a4faa30
# ╠═eaf8a806-7e31-49f9-bb62-0c9162c82d70
# ╠═e4bf5dcd-d225-409a-ab97-277928fe4977
# ╠═3547c975-9df0-4377-bb5f-4ac02e037f0c
# ╠═99314fd6-69e2-489b-a9d9-196498b53285
# ╠═26b4e386-8f09-4a2b-89d7-5071d70b52ea
# ╠═6b549025-4b60-4395-b4a8-3860ac949e90
# ╠═c639b05d-a59e-4237-bed0-d8bda3203222
# ╠═fa506ceb-5145-4345-94fb-e06b751f4dab
# ╠═afc32be7-6637-4ac6-841c-0954eae916e4
# ╠═4aaad296-1d4e-46d2-8fc9-befa37937a9e
# ╠═6666ff0e-67c4-463e-8a9b-b759e04a9967
# ╠═83f81b37-46ee-42f2-b6d6-dc96d7f62779
