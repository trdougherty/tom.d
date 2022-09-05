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
	using Colors, ColorSchemes
	using DataFrames
	using Dates
	using GeoDataFrames
	using JSON
	using Logging
	using Plots
	using YAML
	using StatsBase
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
	nyc_boundaries[!,"coun_dist"] = parse.(Int64, nyc_boundaries[:,"coun_dist"])
	Plots.plot(
		nyc_boundaries.geometry,
		color=:transparent,
		dpi=400
	)
end

# ╔═╡ b8717e75-42d2-4120-b5ca-92269babaa8e
begin
	output_dir = joinpath(data_path, "p1_o")
	mkpath(output_dir)
end

# ╔═╡ d1ea0a90-7830-4d3f-8fb1-18cf88f3293c
nyc_boundaries

# ╔═╡ 0dc202b2-2130-482c-b8d4-c3e07e0a5ed9
md"""
##### Prepping functions for transforming the geometries between ESPG
"""

# ╔═╡ 91e95352-b96e-4246-9f44-fc9dec9755ed
begin
	function reproject_points!(
		geom_obj::ArchGDAL.IGeometry,
		source::ArchGDAL.ISpatialRef,
		target::ArchGDAL.ISpatialRef)
	
		ArchGDAL.createcoordtrans(source, target) do transform
			ArchGDAL.transform!(geom_obj, transform)
		end
	    return geom_obj
	end
	
	function reproject_points!(
		geom_obj::Union{
			Vector{ArchGDAL.IGeometry},
			Vector{ArchGDAL.IGeometry{ArchGDAL.wkbPoint}},
			Vector{ArchGDAL.IGeometry{ArchGDAL.wkbMultiPolygon}},
			Vector{ArchGDAL.IGeometry{ArchGDAL.wkbPolygon}}
		},
		source::ArchGDAL.ISpatialRef,
		target::ArchGDAL.ISpatialRef)
	
		ArchGDAL.createcoordtrans(source, target) do transform
			ArchGDAL.transform!.(geom_obj, Ref(transform))
		end
	    return geom_obj
	end
end

# ╔═╡ e6327ef0-c904-49c4-b43d-607ae3f199b4
begin
	source = ArchGDAL.importEPSG(4326)
	target = ArchGDAL.importEPSG(32618)
	pm = ArchGDAL.importEPSG(3857)
end;

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
		missingstring="Not Available",
		DataFrame
	)
	nyc_monthly_2019 = CSV.read(
		joinpath(
			data_path,
			"monthly-energy-2019.csv"
		),
		missingstring="Not Available",
		DataFrame
	)
	nyc_monthly_2018 = CSV.read(
		joinpath(
			data_path,
			"monthly-energy-2018.csv"
		),
		missingstring="Not Available",
		DataFrame
	)
	nyc_monthly = vcat(
		nyc_monthly_2020,
		nyc_monthly_2019,
		nyc_monthly_2018
	)
	nyc_monthly[!,"date"] = DateTime.(nyc_monthly.Month, "u-y") .+ Dates.Year(2000)
	select!(nyc_monthly, Not("Month"))

	select!(
		nyc_monthly, 
		Not(
			["Parent Property Id", "Property Name", "Parent Property Name"]
		)
	)
	rename!(
		nyc_monthly, 
		[
			"Electricity Use  (kBtu)" => "electricity_kbtu",
			"Natural Gas Use  (kBtu)" => "naturalgas_kbtu"
		]
	)
end

# ╔═╡ ccf8cb9f-88a4-4f41-a635-d02dc479e50d
nyc_idcount = countmap(nyc_monthly[:, "Property Id"])

# ╔═╡ 0f3a58c2-ae4b-4c89-9791-803cb89ab51e
# want to snip any of the weird data which might have snuk in
sneaky_ids = collect(keys(filter(t-> t.second <= 36, nyc_idcount)))

# ╔═╡ 78ace92e-81de-47e7-a19b-41a0c5f35bb4
filtering_idx = in.(nyc_monthly[:, "Property Id"], (sneaky_ids,))

# ╔═╡ bbc1d2c3-393d-4fb9-a039-d54ae414cf16
nyc_monthly_clean = nyc_monthly[filtering_idx, :]

# ╔═╡ 69c8b562-1ed2-4c6e-9b60-44a71ac62ed0
CSV.write(
	joinpath(output_dir, "monthly_energy_nyc.csv"),
	nyc_monthly_clean
)

# ╔═╡ 5c1383bb-50ad-43d9-ac99-2e2b14d41f33
md"""(2) *unique monthly property ids - essentially only use the unique property ids seen*"""

# ╔═╡ 9a579e9d-8de1-4b82-9508-a378f08e2955
md"""(3) *NYC building centroids*"""

# ╔═╡ 34db351a-d5cd-4069-add3-9aa3b4162585
begin
	footprints = GeoDataFrames.read(joinpath(data_path, "building-footprints.geojson"))

	rename!(
		footprints,
		:base_bbl => :bbl
	)
	select!(
		footprints,
		[
			"geometry",
			"bbl",
			"heightroof",
			"cnstrct_yr",
			"groundelev"
		]
	)
	dropmissing!(footprints)
	
	footprints[!,"heightroof"] = parse.(Float64, footprints[!,"heightroof"])
	footprints[!,"cnstrct_yr"] = parse.(Int64, footprints[!,"cnstrct_yr"])
	footprints[!,"groundelev"] = parse.(Int64, footprints[!,"groundelev"]);

	# want to reproject the coordinates into something which preserves meters
	reproject_points!(footprints.geometry, source, target)
	footprints[!,"area"] = ArchGDAL.geomarea.(footprints.geometry)
	reproject_points!(footprints.geometry, target, source)

	# finally drop all the buildings without a geometry provided
	filter!(row -> row.area > 0, footprints)
end

# ╔═╡ 6d6a3a4e-ac6f-41e5-8e8c-78f45eb4f618
property_ids_data = select(unique(nyc_monthly, "Property Id"), "Property Id");

# ╔═╡ d48973e8-b6a7-407e-b007-a7219ecaff2a
nyc_monthly

# ╔═╡ e54c18df-46fe-4425-bdfc-ca8c529d436a
md"""(4) *Cleaned nyc building centroids*"""

# ╔═╡ 009bc1d3-6909-4edf-a615-2b1c7bdb1ce8
begin
	nyc_building_points_ids = leftjoin(
		footprints,
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
	select(nyc_building_points_data, "Property Id", :)

	# @info "building data and locations" select(nyc_building_points_data, Not(:geometry))
end

# ╔═╡ 68c9d819-d948-4bcf-a625-64f2bdc26b62
begin
	regionlist::Vector{Union{Int64, Missing}} = 
		fill(missing, length(nyc_building_points_data.geometry))

	for (building_index, building_point) in enumerate(nyc_building_points_data.geometry)
		for (boundary_index, boundary_geom) in enumerate(nyc_boundaries.geometry)
			if GeoDataFrames.contains(boundary_geom, building_point)
				regionlist[building_index] = nyc_boundaries[boundary_index, "coun_dist"]
				break
			end
		end
	end
end

# ╔═╡ 79feaefc-42b3-49bc-b4eb-092e2b6525f4
begin
	nyc_building_points_data[!, "council_region"] = regionlist;
	dropmissing!(nyc_building_points_data);
end;

# ╔═╡ 2dea8532-505a-4b0e-8b98-966a47808804
begin
	council_countmap = countmap(dropmissing(nyc_building_points_data).council_region)
	council_counts = collect(values(council_countmap))
end;

# ╔═╡ 07aab43a-697f-4d6a-a8c4-4415116368a7
# want to find some councils for validation which roughly have this many buildings
begin
	council_std = std(council_counts)
	council_median = median(council_counts)
end

# ╔═╡ 878e4be3-1a1a-4f73-8823-fb69fd756437
begin
	Plots.histogram(
		council_counts, 
		bins=20, 
		color="transparent",
		label="Building Count - Council",
		dpi=400
	)
	Plots.vline!(
		median(council_counts, dims=1),
		line=(4, :dash, 1.0, :red),
		label="Median Line"
	)
end

# ╔═╡ 941b6f4a-1359-4084-859a-07b3b3fcc7a2
begin
	std_distance = 4
	candidate_councils = collect(keys(filter(
		t -> council_median + (council_std/std_distance) > t.second > council_median - (council_std/std_distance),
		council_countmap
	)))
end

# ╔═╡ 35d7d6d9-34bb-430a-9c74-001b446c9d09
council_count_df = DataFrame(
	coun_dist=collect(keys(council_countmap)), 
	building_count=collect(values(council_countmap))
);

# ╔═╡ 691a9409-ab03-4843-9743-5685c563af81
nyc_council_counts = dropmissing(leftjoin(nyc_boundaries, council_count_df, on="coun_dist"));

# ╔═╡ e7f3b286-fe7e-48ce-a01a-25e5e3743267
color_translation = nyc_council_counts.building_count / maximum(nyc_council_counts.building_count);

# ╔═╡ 5f548a26-a95e-4bc7-9dad-7bdefa41611c
council_colors = cgrad(:matter)[color_translation];

# ╔═╡ 9c877f38-ecf8-4acd-9849-200fb01bce00
begin
	Plots.plot(
		nyc_council_counts.geometry,
		color_palette=council_colors,
		title="Distribution of Building Counts",
		dpi=500
	)
end

# ╔═╡ 734c7a3b-10d5-4627-ae4d-1f787509e838
begin
	candidate_regions = filter(row -> row.coun_dist ∈ candidate_councils, nyc_council_counts)
	noncandidate_regions = filter(row -> row.coun_dist ∉ candidate_councils, nyc_council_counts)
end;

# ╔═╡ 70a44334-ae15-4225-9f2e-668f6ee2b965
md"""
#### Candidate Regions
"""

# ╔═╡ 7e0ecfd3-cdea-4470-a799-acbebc67e7dd
# this will be the validation and test set
candidate_councils

# ╔═╡ f66ee36c-b0c5-4536-ad9d-214c13b3a984
midpoint = length(candidate_councils) ÷ 2

# ╔═╡ a465b08a-26f6-4e7f-a189-df7d76395f80
validation_districts = candidate_councils[1:midpoint]

# ╔═╡ bab66f4b-1d1e-46ad-ae79-1c4f5fae6273
test_districts = candidate_councils[midpoint+1:end]

# ╔═╡ 0bfc6dec-97dd-4a7c-8af5-382068cddbc7
begin
	train_regions = filter(row -> row.coun_dist ∉ candidate_councils, nyc_council_counts)
	validate_regions = filter(row -> row.coun_dist ∈ validation_districts, nyc_council_counts)
	test_regions = filter(row -> row.coun_dist ∈ test_districts, nyc_council_counts)

	train_region_map = Plots.plot(
		train_regions.geometry,
		color=:white,
		dpi=400
	)
	validate_region_map = Plots.plot!(
		validate_regions.geometry,
		color=:lightblue,
		dpi=400	
	)
	test_region_map = Plots.plot!(
		test_regions.geometry,
		color=:indianred,
		dpi=400	
	)
end

# ╔═╡ f69cb982-573f-43c4-8fb0-a1aab6028021
nyc_building_points_data

# ╔═╡ 54ebbdff-f0ae-4265-bacf-a0301d5701b0
GeoDataFrames.write(joinpath(output_dir, "nyc_data.geojson"), nyc_building_points_data)

# ╔═╡ 2d234542-d4bf-41aa-8ed9-9d91b9b69284
begin
	nyc_data_stripped = select(nyc_building_points_data, Not(:geometry))
	nyc_train_buildings = filter(row -> row.council_region ∉ candidate_councils, nyc_data_stripped)

	nyc_validation_buildings = filter(row -> row.council_region ∈ validation_districts, nyc_data_stripped)

	nyc_test_buildings = filter(row -> row.council_region ∈ test_districts, nyc_data_stripped)
end;

# ╔═╡ ee175b30-0f05-4e91-9a0a-d9ef9f7d5a32
begin
	CSV.write(joinpath(output_dir, "train_buildings.csv"), nyc_train_buildings);
	CSV.write(joinpath(output_dir, "validate_buildings.csv"), nyc_validation_buildings);
	CSV.write(joinpath(output_dir, "test_buildings.csv"), nyc_test_buildings);
end;

# ╔═╡ 13e53eaf-a09e-45bd-ba81-8bbbd4f48b20
begin
	nyc_train = select(
		leftjoin(nyc_train_buildings, nyc_monthly, on="Property Id"),
		"Property Id", "date", :)
	dropmissing!(nyc_train, :date)
	
	nyc_validate = select(
		leftjoin(nyc_validation_buildings, nyc_monthly, on="Property Id"),
		"Property Id", "date", :)
	dropmissing!(nyc_validate, :date)
	
	nyc_test = select(
		leftjoin(nyc_test_buildings, nyc_monthly, on="Property Id"),
		"Property Id", "date", :)
	dropmissing!(nyc_test, :date)
end;

# ╔═╡ 309d3bcc-359d-4cf3-b2f1-28744f4d5cb0
# training percentage
nrow(dropmissing(nyc_train)) / nrow(nyc_train)

# ╔═╡ ea2eacce-d6a8-4f00-a04c-101d6918a471
# validation percentage
nrow(dropmissing(nyc_validate)) / nrow(nyc_validate)

# ╔═╡ 7f14239b-4fb4-4dd7-ab4d-be73bae4ed93
# validation percentage
nrow(dropmissing(nyc_test)) / nrow(nyc_test)

# ╔═╡ 36e0c27b-d77e-4616-a5f9-b11cc56c7f3f
md"""
##### Want to strip the geometries from the objects here and just store it as a csv to improve operation with other data types
"""

# ╔═╡ 6800885e-c030-46fd-9220-f75ea87dadb0
begin
	CSV.write(joinpath(output_dir, "train.csv"), nyc_train);
	CSV.write(joinpath(output_dir, "validate.csv"), nyc_validate);
	CSV.write(joinpath(output_dir, "test.csv"), nyc_test);
end;

# ╔═╡ c97e405d-8601-4634-9a7c-654598dd1200
md"""
#### Stage 2
1. Get the footprints for each of the energy models.
2. Check to see if any of the points from the cleaned building centroid data are found in the boundaries of the energy model footprint. 
3. If it matches multiple footprints, throw away the datapoint."""

# ╔═╡ ab355e93-a064-40e2-a67a-e75cb54efb33
md"""(1) *Testing with just a sample of building footprints for now, here is what they might look like*"""

# ╔═╡ 8e462588-c0c1-4e96-a1fe-f2df50a8ec37
# begin
# 	# want this file to change to the building shapefile before this is actually run
# 	energy_model_footprints = GeoDataFrames.read(
# 		joinpath(
# 			data_path,
# 			"largest_buildings.geojson"
# 		)
# 	)
# 	energy_model_footprints[!,"id"] = collect(1:nrow(energy_model_footprints))
# 	sample_building = energy_model_footprints[5,"geometry"]

# 	select!(energy_model_footprints, ["id","geometry"])
# end

# ╔═╡ 14f9ac4d-e705-41fd-8561-4d1421e41126
# Plots.plot(sample_building, color=:transparent)

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
# begin
# 	Plots.plot(
# 		ArchGDAL.boundingbox(sample_building),
# 		opacity=0.15,
# 		color=:gray
# 	)
# 	Plots.plot!(
# 		sample_building,
# 		color=:white
# 	)
# end

# ╔═╡ c4b107f9-421b-4c5e-b876-10b9efb4d4c6
# begin
# 	points_range = length(nyc_building_points_data.geometry)
# 	buildings_range = length(energy_model_footprints.geometry)
	
# 	building_mapping = Array{Union{Missing, Int}}(missing, buildings_range)
# 	for i = 1:buildings_range
# 		# candidate geom - just a box
# 		candidate_geom = ArchGDAL.boundingbox(energy_model_footprints.geometry[i])
		
# 		for j = 1:points_range
# 			# candidate process to find something in the ballpark
# 			if ArchGDAL.contains(
# 				candidate_geom,
# 				nyc_building_points_data.geometry[j]
# 			)
# 				# if we pass the candidacy process
# 				if ArchGDAL.contains(
# 					energy_model_footprints.geometry[i], 
# 					nyc_building_points_data.geometry[j]
# 				)
# 					# if we already have a mapped value, it means multiple points
# 					# map to the same building. We need to toss it
# 					if ~ismissing(building_mapping[i])
# 						building_mapping[i] = missing
# 						break
# 					end
# 					building_mapping[i] = j
# 				end
# 			end
# 		end
# 	end
# 	building_mapping
# end;

# ╔═╡ 11fd74bd-4ebc-4537-bbd9-958761ac7b01
md"""
Estimated time without candidate process (h):
"""

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
# begin
# 	energy_model_footprints[!,"data_mapping"] = building_mapping
# 	energy_model_footprints
# end

# ╔═╡ 329d43de-17b6-4d11-a0c8-4cd0a6980932
# begin
# 	matched_buildings = dropmissing(energy_model_footprints, :data_mapping)
# 	matched_buildings
# end

# ╔═╡ 9cfc27da-38b5-40cd-b41f-a1d2687011a8
# matched_points = nyc_building_points_data[matched_buildings.data_mapping, :]

# ╔═╡ 69cd9bb6-01d7-447d-afbc-b349d01d9fbf
# begin
# 	matched_points[:, "shapefile_id"] = matched_buildings.id
# 	id_mapping = select(matched_points, ["Property Id","shapefile_id"])
# 	@info "Matched building shapefiles with building IDs:" id_mapping
# end

# ╔═╡ a85a3658-e1b5-4c57-a275-26a69dce572a
# begin
# 	poi = 4
# 	Plots.plot(matched_buildings.geometry[poi], color=:transparent)
# 	Plots.plot!(matched_points.geometry[poi])
# end

# ╔═╡ 2520a4ec-cebd-4f27-8e21-5a5f9b55c8a9
md"""
At this point we should just be able to extract a list of building ids for which we want to run simulations, and we can use the property id to collect the energy data
"""

# ╔═╡ f6e0da2e-1394-4945-aee7-67de4ade486c
# matched_buildings.id

# ╔═╡ 39fc72af-2262-4597-9c44-c9ccd56b6ec9
# matched_points[:, "Property Id"]

# ╔═╡ 3bd5037f-87cf-4988-888f-d62c10de3195
# begin
# 	# now going to output a list of all the idf files we want to run
# 	@info "Creating queue of simulation files" id_mapping.shapefile_id
	
# 	output_dir = joinpath(data_path, "p1_o")
# 	mkpath(output_dir)

# 	simulation_output = joinpath(output_dir, "simulation_queue.txt")
# 	open(simulation_output,"w") do io
# 		for id in id_mapping.shapefile_id
# 	   		println(io,string(id)*".idf")
# 		end
# 	end
# end

# ╔═╡ Cell order:
# ╠═1ea68209-380d-4b2e-9239-bedf850b8243
# ╠═b8717e75-42d2-4120-b5ca-92269babaa8e
# ╟─4ce9f094-273d-4497-9255-202726b00c11
# ╟─f6ba219d-9c7c-4947-a3fd-5a32e26e89a8
# ╠═d1ea0a90-7830-4d3f-8fb1-18cf88f3293c
# ╟─0dc202b2-2130-482c-b8d4-c3e07e0a5ed9
# ╟─91e95352-b96e-4246-9f44-fc9dec9755ed
# ╠═e6327ef0-c904-49c4-b43d-607ae3f199b4
# ╟─f65a1dea-039e-4705-ae78-10a385f85b6b
# ╟─efdf735f-9576-4117-8003-bd78037bd4f9
# ╟─8db8b2ca-ee25-44a8-91d2-26b7c7fb8c0d
# ╟─f2013cd8-5ce1-47f0-8700-a0550612f943
# ╟─e0eae347-e275-4737-b429-4dd2b04cb27c
# ╟─4197d920-ef37-4284-970a-7fd4331ad9b7
# ╟─6fc43b1e-bdd6-44dd-9940-9d96e25a285d
# ╠═ccf8cb9f-88a4-4f41-a635-d02dc479e50d
# ╠═0f3a58c2-ae4b-4c89-9791-803cb89ab51e
# ╠═78ace92e-81de-47e7-a19b-41a0c5f35bb4
# ╠═bbc1d2c3-393d-4fb9-a039-d54ae414cf16
# ╠═69c8b562-1ed2-4c6e-9b60-44a71ac62ed0
# ╟─5c1383bb-50ad-43d9-ac99-2e2b14d41f33
# ╟─9a579e9d-8de1-4b82-9508-a378f08e2955
# ╟─34db351a-d5cd-4069-add3-9aa3b4162585
# ╠═6d6a3a4e-ac6f-41e5-8e8c-78f45eb4f618
# ╠═d48973e8-b6a7-407e-b007-a7219ecaff2a
# ╟─e54c18df-46fe-4425-bdfc-ca8c529d436a
# ╟─009bc1d3-6909-4edf-a615-2b1c7bdb1ce8
# ╠═68c9d819-d948-4bcf-a625-64f2bdc26b62
# ╠═79feaefc-42b3-49bc-b4eb-092e2b6525f4
# ╠═2dea8532-505a-4b0e-8b98-966a47808804
# ╠═07aab43a-697f-4d6a-a8c4-4415116368a7
# ╟─878e4be3-1a1a-4f73-8823-fb69fd756437
# ╠═941b6f4a-1359-4084-859a-07b3b3fcc7a2
# ╠═35d7d6d9-34bb-430a-9c74-001b446c9d09
# ╠═691a9409-ab03-4843-9743-5685c563af81
# ╠═e7f3b286-fe7e-48ce-a01a-25e5e3743267
# ╠═5f548a26-a95e-4bc7-9dad-7bdefa41611c
# ╟─9c877f38-ecf8-4acd-9849-200fb01bce00
# ╠═734c7a3b-10d5-4627-ae4d-1f787509e838
# ╟─70a44334-ae15-4225-9f2e-668f6ee2b965
# ╠═7e0ecfd3-cdea-4470-a799-acbebc67e7dd
# ╠═f66ee36c-b0c5-4536-ad9d-214c13b3a984
# ╠═a465b08a-26f6-4e7f-a189-df7d76395f80
# ╠═bab66f4b-1d1e-46ad-ae79-1c4f5fae6273
# ╟─0bfc6dec-97dd-4a7c-8af5-382068cddbc7
# ╠═f69cb982-573f-43c4-8fb0-a1aab6028021
# ╠═54ebbdff-f0ae-4265-bacf-a0301d5701b0
# ╠═2d234542-d4bf-41aa-8ed9-9d91b9b69284
# ╠═ee175b30-0f05-4e91-9a0a-d9ef9f7d5a32
# ╠═13e53eaf-a09e-45bd-ba81-8bbbd4f48b20
# ╠═309d3bcc-359d-4cf3-b2f1-28744f4d5cb0
# ╠═ea2eacce-d6a8-4f00-a04c-101d6918a471
# ╠═7f14239b-4fb4-4dd7-ab4d-be73bae4ed93
# ╟─36e0c27b-d77e-4616-a5f9-b11cc56c7f3f
# ╠═6800885e-c030-46fd-9220-f75ea87dadb0
# ╟─c97e405d-8601-4634-9a7c-654598dd1200
# ╟─ab355e93-a064-40e2-a67a-e75cb54efb33
# ╠═8e462588-c0c1-4e96-a1fe-f2df50a8ec37
# ╟─14f9ac4d-e705-41fd-8561-4d1421e41126
# ╟─2d956f5a-7f40-4a7d-b491-f3de3340cb66
# ╟─630cbcdf-31c6-4a96-9c36-8fea5d64e2b9
# ╟─4e00b820-9f99-4134-afd7-2710424fe28e
# ╠═c4b107f9-421b-4c5e-b876-10b9efb4d4c6
# ╟─11fd74bd-4ebc-4537-bbd9-958761ac7b01
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
