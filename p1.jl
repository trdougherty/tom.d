### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ 1ea68209-380d-4b2e-9239-bedf850b8243
begin
	import Pkg
	Pkg.activate(Base.current_project())

	using ArchGDAL
	using CSV
	using DataFrames
	using Dates
	using GeoDataFrames
	using JSON
	using Logging
	using Plots
	using YAML
end;

# ╔═╡ 4ce9f094-273d-4497-9255-202726b00c11
md"""
## Chapter 1: New York City
"""

# ╔═╡ f6ba219d-9c7c-4947-a3fd-5a32e26e89a8
begin
	sources_file = joinpath(pwd(), "sources.yml")
	sources = YAML.load_file(sources_file)
	nyc_sources = sources["data-sources"]["nyc"]
	data_path = joinpath(pwd(), "data", "nyc")
	
	nyc_boundaries_path = joinpath(
		data_path, 
		"council-boundaries.geojson"
	)
	nyc_boundaries = GeoDataFrames.read(nyc_boundaries_path)
	Plots.plot(
		nyc_boundaries.geometry,
		color=:transparent,
		dpi=400
	)
end

# ╔═╡ f65a1dea-039e-4705-ae78-10a385f85b6b
md"""
The data we have for new york city:
"""

# ╔═╡ efdf735f-9576-4117-8003-bd78037bd4f9
nyc_sources

# ╔═╡ 8db8b2ca-ee25-44a8-91d2-26b7c7fb8c0d
md"""
#### Stage 1
1. Load the data used to link the energy data with building footprints
2. Load energy data, remove duplicated terms
3. Load footprint centroids data
4. Check which footprint centroids can be matched to the energy data, remove the ones which aren't
"""

# ╔═╡ f2013cd8-5ce1-47f0-8700-a0550612f943
md"""(1) *linking data*"""

# ╔═╡ e0eae347-e275-4737-b429-4dd2b04cb27c
begin
	linking_data = CSV.read(
		joinpath(
			data_path,
			"annual-energy.csv"
		),
		DataFrame
	)
	rename!(linking_data, "NYC Borough, Block and Lot (BBL)" => "bbl")
	select!(linking_data, ["Property Id", "bbl"])
end

# ╔═╡ 4197d920-ef37-4284-970a-7fd4331ad9b7
md"""(2) *Full monthly energy data between 2018 and 2020*"""

# ╔═╡ 6fc43b1e-bdd6-44dd-9940-9d96e25a285d
begin
	nyc_monthly_2020 = CSV.read(
		joinpath(
			data_path,
			"monthly-energy-2020.csv"
		),
		DataFrame
	)
	nyc_monthly_2019 = CSV.read(
		joinpath(
			data_path,
			"monthly-energy-2019.csv"
		),
		DataFrame
	)
	nyc_monthly_2018 = CSV.read(
		joinpath(
			data_path,
			"monthly-energy-2018.csv"
		),
		DataFrame
	)
	nyc_monthly = vcat(
		nyc_monthly_2020,
		nyc_monthly_2019,
		nyc_monthly_2018
	)
	nyc_monthly[!,"date"] = DateTime.(nyc_monthly.Month, "u-y") .+ Dates.Year(2000)
	select!(nyc_monthly, Not("Month"))
end

# ╔═╡ 5c1383bb-50ad-43d9-ac99-2e2b14d41f33
md"""(2) *unique monthly property ids - essentially only use the unique property ids seen*"""

# ╔═╡ 10f410f6-992e-4063-ab0c-9f016f51eb5d
property_ids_data = select(unique(nyc_monthly, "Property Id"), "Property Id");

# ╔═╡ 9a579e9d-8de1-4b82-9508-a378f08e2955
md"""(3) *NYC building centroids*"""

# ╔═╡ 595ce11b-7fac-4ccc-a7e3-9ef5d114ac0d
begin
	nyc_building_points = GeoDataFrames.read(
		joinpath(
			data_path,
			"building-points.geojson"
		)
	)
	rename!(
		nyc_building_points,
		:base_bbl => :bbl
	)
	select!(
		nyc_building_points,
		[
			"geometry",
			"bbl",
			"heightroof",
			"cnstrct_yr",
			"groundelev"
		]
	)
	dropmissing!(nyc_building_points)
	
	nyc_building_points[!,"heightroof"] = parse.(Float64, nyc_building_points[!,"heightroof"])
	nyc_building_points[!,"cnstrct_yr"] = parse.(Int64, nyc_building_points[!,"cnstrct_yr"])
	nyc_building_points[!,"groundelev"] = parse.(Int64, nyc_building_points[!,"groundelev"]);
end;

# ╔═╡ e54c18df-46fe-4425-bdfc-ca8c529d436a
md"""(4) *Cleaned nyc building centroids*"""

# ╔═╡ 009bc1d3-6909-4edf-a615-2b1c7bdb1ce8
begin
	nyc_building_points_ids = leftjoin(
		nyc_building_points,
		linking_data,
		on=:bbl
	)
	dropmissing!(
		nyc_building_points_ids,
		"Property Id"
	)
	select!(
		nyc_building_points_ids,
		Not(:bbl)
	)
	nyc_building_points_data = leftjoin(
		nyc_building_points_ids,
		property_ids_data,
		on="Property Id"
	)
	dropmissing!(nyc_building_points_data)
	unique!(nyc_building_points_data, ["Property Id"])

	@info "building data and locations" select(nyc_building_points_data, Not(:geometry))
end

# ╔═╡ 68b4bfb9-a1ce-4a3c-8f18-a9933dd3239d
Plots.plot(
	nyc_building_points_data.geometry, 
	dpi=400,
	color=:transparent,
	opacity=1,
	markersize=0.5
)

# ╔═╡ f600bad4-f853-433c-ac06-90a2a797b6bb


# ╔═╡ c97e405d-8601-4634-9a7c-654598dd1200
md"""
#### Stage 2
1. Get the footprints for each of the energy models.
2. Check to see if any of the points from the cleaned building centroid data are found in the boundaries of the energy model footprint. 
3. If it matches multiple footprints, throw away the datapoint."""

# ╔═╡ ab355e93-a064-40e2-a67a-e75cb54efb33
md"""(1) *Testing with just a sample of building footprints for now, here is what they might look like*"""

# ╔═╡ 8e462588-c0c1-4e96-a1fe-f2df50a8ec37
begin
	# want this file to change to the building shapefile before this is actually run
	energy_model_footprints = GeoDataFrames.read(
		joinpath(
			data_path,
			"largest_buildings.geojson"
		)
	)
	energy_model_footprints[!,"id"] = collect(1:nrow(energy_model_footprints))
	sample_building = energy_model_footprints[5,"geometry"]

	select!(energy_model_footprints, ["id","geometry"])
end

# ╔═╡ 14f9ac4d-e705-41fd-8561-4d1421e41126
Plots.plot(sample_building, color=:transparent)

# ╔═╡ 2d956f5a-7f40-4a7d-b491-f3de3340cb66
md"""
(2) Need to see if any of the energy points can be found in the buiding shapefile. First going to take the bounding box for all buidings to narrow down the scope of potential points which might be contained, prior to checking if it's actually inside the vectorized box.
"""

# ╔═╡ 630cbcdf-31c6-4a96-9c36-8fea5d64e2b9
md"""
- *gray:* candidate box
- *white:* true geometry
"""

# ╔═╡ 4e00b820-9f99-4134-afd7-2710424fe28e
begin
	Plots.plot(
		ArchGDAL.boundingbox(sample_building),
		opacity=0.15,
		color=:gray
	)
	Plots.plot!(
		sample_building,
		color=:white
	)
end

# ╔═╡ c4b107f9-421b-4c5e-b876-10b9efb4d4c6
begin
	points_range = length(nyc_building_points_data.geometry)
	buildings_range = length(energy_model_footprints.geometry)
	
	building_mapping = Array{Union{Missing, Int}}(missing, buildings_range)
	for i = 1:buildings_range
		# candidate geom - just a box
		candidate_geom = ArchGDAL.boundingbox(energy_model_footprints.geometry[i])
		
		for j = 1:points_range
			# candidate process to find something in the ballpark
			if ArchGDAL.contains(
				candidate_geom,
				nyc_building_points_data.geometry[j]
			)
				# if we pass the candidacy process
				if ArchGDAL.contains(
					energy_model_footprints.geometry[i], 
					nyc_building_points_data.geometry[j]
				)
					# if we already have a mapped value, it means multiple points
					# map to the same building. We need to toss it
					if ~ismissing(building_mapping[i])
						building_mapping[i] = missing
						break
					end
					building_mapping[i] = j
				end
			end
		end
	end
	building_mapping
end;

# ╔═╡ 11fd74bd-4ebc-4537-bbd9-958761ac7b01
md"""
Estimated time without candidate process (h):
"""

# ╔═╡ 91ac4f82-04ef-459d-b7ec-9b6d63c55c4e
2.8 * (1.1e6 / 50) / (60 * 60)

# ╔═╡ 7df29d94-63b7-46dd-8872-a8dbe0e81e38
md"""
Estimated time with candidate process:
"""

# ╔═╡ c5398a9d-e4fd-4703-9676-2e07334caeaf
0.963 * (1.1e6 / 50) / (60 * 60)

# ╔═╡ 5a35d191-76e7-447e-841d-881b47ab8d6c
md"""
After mapping to the data points we have for the buildings
	"""

# ╔═╡ 0c000a60-6fe3-464a-a8b6-906e1e94f02d
begin
	energy_model_footprints[!,"data_mapping"] = building_mapping
	energy_model_footprints
end

# ╔═╡ 329d43de-17b6-4d11-a0c8-4cd0a6980932
begin
	matched_buildings = dropmissing(energy_model_footprints, :data_mapping)
	matched_buildings
end

# ╔═╡ 9cfc27da-38b5-40cd-b41f-a1d2687011a8
matched_points = nyc_building_points_data[matched_buildings.data_mapping, :]

# ╔═╡ 69cd9bb6-01d7-447d-afbc-b349d01d9fbf
begin
	matched_points[:, "shapefile_id"] = matched_buildings.id
	id_mapping = select(matched_points, ["Property Id","shapefile_id"])
	@info "Matched building shapefiles with building IDs:" id_mapping
end

# ╔═╡ a85a3658-e1b5-4c57-a275-26a69dce572a
begin
	poi = 4
	Plots.plot(matched_buildings.geometry[poi], color=:transparent)
	Plots.plot!(matched_points.geometry[poi])
end

# ╔═╡ 2520a4ec-cebd-4f27-8e21-5a5f9b55c8a9
md"""
At this point we should just be able to extract a list of building ids for which we want to run simulations, and we can use the property id to collect the energy data
"""

# ╔═╡ f6e0da2e-1394-4945-aee7-67de4ade486c
matched_buildings.id

# ╔═╡ 39fc72af-2262-4597-9c44-c9ccd56b6ec9
matched_points[:, "Property Id"]

# ╔═╡ 3bd5037f-87cf-4988-888f-d62c10de3195
begin
	# now going to output a list of all the idf files we want to run
	@info "Creating queue of simulation files" id_mapping.shapefile_id
	
	output_dir = joinpath(data_path, "p1_o")
	mkpath(output_dir)

	simulation_output = joinpath(output_dir, "simulation_queue.txt")
	open(simulation_output,"w") do io
		for id in id_mapping.shapefile_id
	   		println(io,string(id)*".idf")
		end
	end
end

# ╔═╡ Cell order:
# ╠═1ea68209-380d-4b2e-9239-bedf850b8243
# ╟─4ce9f094-273d-4497-9255-202726b00c11
# ╟─f6ba219d-9c7c-4947-a3fd-5a32e26e89a8
# ╟─f65a1dea-039e-4705-ae78-10a385f85b6b
# ╟─efdf735f-9576-4117-8003-bd78037bd4f9
# ╟─8db8b2ca-ee25-44a8-91d2-26b7c7fb8c0d
# ╟─f2013cd8-5ce1-47f0-8700-a0550612f943
# ╟─e0eae347-e275-4737-b429-4dd2b04cb27c
# ╟─4197d920-ef37-4284-970a-7fd4331ad9b7
# ╠═6fc43b1e-bdd6-44dd-9940-9d96e25a285d
# ╟─5c1383bb-50ad-43d9-ac99-2e2b14d41f33
# ╠═10f410f6-992e-4063-ab0c-9f016f51eb5d
# ╟─9a579e9d-8de1-4b82-9508-a378f08e2955
# ╠═595ce11b-7fac-4ccc-a7e3-9ef5d114ac0d
# ╟─e54c18df-46fe-4425-bdfc-ca8c529d436a
# ╠═009bc1d3-6909-4edf-a615-2b1c7bdb1ce8
# ╟─68b4bfb9-a1ce-4a3c-8f18-a9933dd3239d
# ╟─f600bad4-f853-433c-ac06-90a2a797b6bb
# ╟─c97e405d-8601-4634-9a7c-654598dd1200
# ╟─ab355e93-a064-40e2-a67a-e75cb54efb33
# ╠═8e462588-c0c1-4e96-a1fe-f2df50a8ec37
# ╟─14f9ac4d-e705-41fd-8561-4d1421e41126
# ╟─2d956f5a-7f40-4a7d-b491-f3de3340cb66
# ╟─630cbcdf-31c6-4a96-9c36-8fea5d64e2b9
# ╟─4e00b820-9f99-4134-afd7-2710424fe28e
# ╠═c4b107f9-421b-4c5e-b876-10b9efb4d4c6
# ╟─11fd74bd-4ebc-4537-bbd9-958761ac7b01
# ╠═91ac4f82-04ef-459d-b7ec-9b6d63c55c4e
# ╟─7df29d94-63b7-46dd-8872-a8dbe0e81e38
# ╠═c5398a9d-e4fd-4703-9676-2e07334caeaf
# ╟─5a35d191-76e7-447e-841d-881b47ab8d6c
# ╠═0c000a60-6fe3-464a-a8b6-906e1e94f02d
# ╠═329d43de-17b6-4d11-a0c8-4cd0a6980932
# ╠═9cfc27da-38b5-40cd-b41f-a1d2687011a8
# ╠═69cd9bb6-01d7-447d-afbc-b349d01d9fbf
# ╟─a85a3658-e1b5-4c57-a275-26a69dce572a
# ╟─2520a4ec-cebd-4f27-8e21-5a5f9b55c8a9
# ╠═f6e0da2e-1394-4945-aee7-67de4ade486c
# ╠═39fc72af-2262-4597-9c44-c9ccd56b6ec9
# ╠═3bd5037f-87cf-4988-888f-d62c10de3195
