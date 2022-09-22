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
	using GeoDataFrames
	using Plots
	using Statistics
	using UnicodePlots

	import Cairo, Fontconfig
end;

# ╔═╡ 786c2441-7abb-4caa-9f50-c6078fff0f56
using ArchGDAL

# ╔═╡ b564d2d8-a55d-4900-9803-29cd24e2e879
using GLM

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
	# path for the weather station data
	epw_p = joinpath(input_dir, "epw.csv")
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

# ╔═╡ 8883d4ac-9ec4-40b5-a885-e1f3c5cbd4b9
epw_r = CSV.read(epw_p, DataFrame);

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

# ╔═╡ dcbcb589-b320-4e4d-842c-25daab5054d4
# Gadfly.plot(
# 	sample, 
# 	x=:date, 
# 	y="Drybulb Temperature (°C)", 
# 	Geom.smooth(method=:loess,smoothing=0.2)
# )

# ╔═╡ c5aaf1a9-fd38-4cb0-9ee8-e04ac0a2fecb
md"""
##### this section isn't quite as necessary, but to highlight discrepancies
"""

# ╔═╡ 731c8bcc-2350-484e-ac83-ad94e0b64d46


# ╔═╡ b897cbfd-484d-410e-b698-14196aaf8a73


# ╔═╡ 6b120bfc-e1e3-4b52-a99d-aa60518b32ad
# testing to see how the agg method might work
filter(term -> term ∉ ["Property Id", "date"], names(lst_r))

# ╔═╡ 348c4307-94dc-4d5f-82b0-77dc535c1650
function strip_month!(data::DataFrame)
	data[!,"date"] = Date.(Dates.Year.(data.date), Dates.Month.(data.date))
end

# ╔═╡ 21fac793-3617-437e-bc6d-85def0ae01c9
epw_r

# ╔═╡ 09a4789c-cbe7-496e-98b5-a2c2db3102b6
begin
	# also want to get the building data in a uniform format for matching
	strip_month!(train)
	strip_month!(validate)
	strip_month!(test)

	strip_month!(epw_r)

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

	epw = 			monthly_average(epw_r, agg_terms)
	
	era5 = 			monthly_average(era5_r, agg_terms)
	landsat8 = 		monthly_average(landsat8_r, agg_terms)
	lst = 			monthly_average(lst_r, agg_terms)
	noaa = 			monthly_average(noaa_r, agg_terms)
	sentinel_1C = 	monthly_average(sentinel_1C_r, agg_terms)
	sentinel_2A = 	monthly_average(sentinel_2A_r, agg_terms)
	viirs = 		monthly_average(viirs_r, agg_terms)
end;

# ╔═╡ c7011ece-6d0b-41b9-83c1-0d8e956fe515
max_epw_r = combine(groupby(epw_r, ["Property Id", "date"]), names(epw_r, Real) .=> maximum, renamecols=false);

# ╔═╡ e350f7f6-5757-4b8e-95bc-c6c098202d49
max_epw = monthly_average(
	max_epw_r,
	agg_terms
);

# ╔═╡ 87e7b997-6a8a-4395-8d0c-f6f59d34d96f
min_epw = monthly_average(
	combine(groupby(epw_r, ["Property Id", "date"]), names(epw_r, Real) .=> minimum, renamecols=false),
	agg_terms
);

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

# ╔═╡ 3b980465-9b75-404d-a41f-06ad351d12ae
@info "Training data points prior to merge" nrow(train)

# ╔═╡ a693145c-8552-44ff-8b46-485c8c8fb738
begin
	environmental_terms = [era5, noaa, lst, landsat8, viirs]
	@info "# Environmental Datasets" length(environmental_terms)
end

# ╔═╡ 7ddf461c-e167-4db0-98c7-940fc962bb6a
# this file actually spits out all of the data we need for ML in the next section
begin
	training_path = joinpath(output_dir, "training_environmental.csv")
	@info "Saving training data" training_path
	training_data = innerjoin(
		train, 
		environmental_terms..., 
		on=agg_terms)
	CSV.write(training_path, training_data)

	validating_path = joinpath(output_dir, "validating_environmental.csv")
	@info "Saving validation data" validating_path
	validating_data = innerjoin(
		validate,
		environmental_terms...,
		on=agg_terms)
	CSV.write(validating_path, validating_data)

	testing_path = joinpath(output_dir, "testing_environmental.csv")
	@info "Saving testing path" testing_path
	testing_data = innerjoin(
		test,
		environmental_terms...,
		on=agg_terms)
	CSV.write(testing_path, testing_data)
end;

# ╔═╡ aa25d6e8-0dab-4f57-bc9e-f26373538826
@info "Training points post merge" nrow(training_data)

# ╔═╡ 7c43c2eb-63e9-4508-8cae-75c48661f328
# this file actually spits out all of the data we need for ML in the next section
begin
	epw_training_path = joinpath(output_dir, "training_epw.csv")
	@info "Saving training data" epw_training_path
	epw_training_data = innerjoin(
		train, 
		epw, 
		on=agg_terms)
	CSV.write(epw_training_path, epw_training_data)

	epw_validating_path = joinpath(output_dir, "validating_epw.csv")
	@info "Saving validation data" epw_validating_path
	epw_validating_data = innerjoin(
		validate,
		epw,
		on=agg_terms)
	CSV.write(epw_validating_path, epw_validating_data)

	epw_testing_path = joinpath(output_dir, "testing_epw.csv")
	@info "Saving testing path" epw_testing_path
	epw_testing_data = innerjoin(
		test,
		epw,
		on=agg_terms)
	CSV.write(epw_testing_path, epw_testing_data)
end;

# ╔═╡ 091f7d95-eae6-474d-8bd0-29a12f99e3f8
md"""
Picking out terms which have recordings for LST
"""

# ╔═╡ fcc7ed57-3e9a-488c-80c7-3a09879aacbe
counter(filter( row -> ~isnan(row.LST_Day_1km), training_data)[:,"Property Id"])

# ╔═╡ 6b402147-68e1-4369-86bb-977304dedc01
# sample_id = 4406043

# ╔═╡ 6b1a360e-7d11-4809-ad20-8106241163d2
sort(unique(training_data.council_region))

# ╔═╡ e392b69c-abd4-454b-bf96-2cc3e4723b46
md"""
Maybe we want to just look at a sample in one council
"""

# ╔═╡ 5b461de9-40b3-40f1-bce9-a3b620474669
sample_id = filter( x -> x.council_region == 26, training_data )[1,"Property Id"]

# ╔═╡ 3a473c8d-bd0c-4309-b637-84bde0389511
# restricting to just one building for interpretation
sample_training = filter( row -> row["Property Id"] == sample_id, training_data );

# ╔═╡ 7529c522-c3ac-4f6c-9385-a91f8b640c45
sample_training_epw = filter( row -> row["Property Id"] == sample_id, epw_training_data );

# ╔═╡ d270993b-b9a0-4e21-a309-e3e9fcaf18e1
data_path

# ╔═╡ af4c25cb-228d-4a76-9bc8-78db0edc836d
begin
	# for the aid of visualization
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
	source_num = 4326
	target_num = 32618
	pm_num = 3857
	source = ArchGDAL.importPROJ4("+proj=longlat +datum=WGS84 +no_defs +type=crs")
	target = ArchGDAL.importEPSG(target_num)

	input_data_auxillary = joinpath(data_path, "p1_o")
	building_file = joinpath(input_data_auxillary, "buildings.geojson")
	buildings = GeoDataFrames.read(building_file)
	sample_building = filter( x -> x["Property Id"] == sample_id, buildings ).geometry
	councils = GeoDataFrames.read(joinpath(data_path, "council-boundaries.geojson"))
	reproject_points!(councils.geometry, source, target)
end;

# ╔═╡ f7030f91-7785-4355-83e9-3e4ea016ffe4
begin
sample_location = Plots.plot(
	ArchGDAL.centroid(sample_building),
	color="indianred",
	title="Location of Sample Point",
	dpi=400
)

Plots.plot!(
	councils.geometry,
	color="transparent"
)
end

# ╔═╡ 812d7cda-ce12-495c-be38-52c8f0f23747
Plots.savefig(joinpath(output_dir, "sample_location.png"))

# ╔═╡ 01e27ad1-9b45-40b3-8d03-26400cd153f9
sample_maxepw = filter( row -> row["Property Id"] == sample_id, max_epw );

# ╔═╡ 4eec8fe5-2c77-489d-8d1f-fef0ad388a50
sample_minepw = filter( row -> row["Property Id"] == sample_id, min_epw );

# ╔═╡ 09a85206-3f09-4ba4-8fe3-85d32c8e8793
begin
	lst_daymeans = sample_training.LST_Day_1km .* 0.02 .- 273.15
	lst_nightmeans = sample_training.LST_Night_1km .* 0.02 .- 273.15
end

# ╔═╡ 6ec97827-015d-4da4-8e58-4564db02fedf
default_colors = cgrad(:redblue, 5, categorical = true, rev=false);

# ╔═╡ 87b55c18-6842-4321-b2c7-c1abd8fef6fc
temp_plot = Gadfly.plot(
	Gadfly.layer(
		x = sample_training.date,
		y = lst_daymeans,
		Geom.point,
		Geom.line,
		Theme(
			default_color=default_colors[1],
			point_size=2pt
		)
	),
	Gadfly.layer(
		x = sample_maxepw.date,
		y = sample_maxepw[:,"Drybulb Temperature (°C)"],
		Geom.point,
		Geom.line,
		Theme(
			default_color=default_colors[2],
			point_size=2pt
		)
	),
	Gadfly.layer(
		x = sample_training.date,
		y = sample_training.TMP,
		Geom.point,
		Geom.line,
		Theme(
			default_color=default_colors[3],
			point_size=2pt
		)
	),
	Gadfly.layer(
		x = sample_training.date,
		y = lst_nightmeans,
		Geom.point,
		Geom.line,
		Theme(
			default_color=default_colors[4],
			point_size=2pt
		)
	),
	Gadfly.layer(
		x = sample_minepw.date,
		y = sample_minepw[:,"Drybulb Temperature (°C)"],
		Geom.point,
		Geom.line,
		Theme(
			default_color=default_colors[5],
			point_size=2pt
		)
	),
	Guide.ylabel("Measured Temperature (°C)"),
	Guide.xlabel("Date"),
	Guide.xticks(
		ticks=DateTime("2018-01-1"):Month(1):DateTime("2021-01-01"),
		orientation=:vertical
	),
	Guide.title("Temperature Measurements - Sample Location"),
	Guide.manual_color_key(
		"Data Type",
		[ 
			"MODIS Daytime LST", 
			"EPW Drybulb Temp Daily Maximum",
			"NOAA Reanalysis",
			"MODIS Nighttime LST", 
			"EPW Drybulb Temp Daily Minimum"
		],
		[
			default_colors[1], 
			default_colors[2], 
			default_colors[3], 
			default_colors[4], 
			default_colors[5]
		]
	)
)

# ╔═╡ e81eacd1-22ff-4890-989e-e4ec638f06b5
begin
sample_daylst_df = DataFrame(
	date=sample_training.date,
	lst_temp=lst_daymeans
)

lst_epw_tempdiff = select(
		innerjoin(
		sample_daylst_df,
		sample_maxepw,
		on="date"
		),
	["date","lst_temp","Drybulb Temperature (°C)"]
)
end;

# ╔═╡ 8ef0fc6c-2937-44bf-87b7-7914c8f2e41d
# Gadfly.plot(
# 	lst_epw_tempdiff,
# 	Gadfly.layer(
# 		x = :date,
# 		y = :lst_temp,
# 		Geom.point,
# 		Geom.line,
# 		Theme(
# 			default_color=default_colors[1]
# 		)
# 	),
# 	Gadfly.layer(
# 		x = :date,
# 		y = "Drybulb Temperature (°C)",
# 		Geom.point,
# 		Geom.line,
# 		Theme(
# 			default_color=default_colors[2]
# 		)
# 	)
# )

# ╔═╡ f37dd092-aa81-4669-8925-665415c90aaf
Gadfly.plot(
	Gadfly.layer(
		x=lst_epw_tempdiff.date,
		y=lst_epw_tempdiff.lst_temp .- lst_epw_tempdiff[:,"Drybulb Temperature (°C)"],
		Theme(default_color="black"),
		Geom.point,
		Geom.line
	),
	Guide.xlabel("Date"),
	Guide.ylabel("Temperature-Δ (°C)"),
	Guide.title("Temperature-Δ of Max Readings (MODIS LST - EPW)")
)

# ╔═╡ 512cad43-28ca-4b07-afa5-20571a31b311
draw(
	PNG(
		joinpath(output_dir, "temperature_deviation_example.png"), 
		20cm, 
		10cm,
		dpi=500
	), temp_plot
)

# ╔═╡ 36a789ff-eaf9-44b4-8d16-1893c8b6d39b
max_epw

# ╔═╡ b7d62ac5-1c30-4ec6-b307-5ae0d6bcaa29
epw_env_maxmix = leftjoin(
	select(max_epw, ["Property Id", "date","Drybulb Temperature (°C)"]),
	training_data,
	on=["Property Id", "date"]
)	

# ╔═╡ f3774051-976b-45db-a388-77757d23a299
training_data

# ╔═╡ bb59670b-9ca1-45b3-84f4-0c190734f02f
epw_env_maxmix[:,"epw_envtempdiff"] = ((epw_env_maxmix.LST_Day_1km .* 0.02).-273.15) .- epw_env_maxmix[:,"Drybulb Temperature (°C)"]

# ╔═╡ ce702ad5-a862-4bb0-805d-2bc4f6ce096d
begin
	sample_epw_envdiff = first(epw_env_maxmix, 10000)
	replace!(sample_epw_envdiff.epw_envtempdiff, NaN => missing)
end

# ╔═╡ 27a0b91c-7859-4193-939b-5ceac1afaeee
begin
	fm = @formula(epw_envtempdiff ~ weather_station_distance)
	linearRegressor = lm(fm, dropmissing(sample_epw_envdiff))
	intercept = coef(linearRegressor)[1]
	slope = coef(linearRegressor)[2]
end;

# ╔═╡ 3b71cbd9-39ee-4762-ab8e-12a6004835ab
linearRegressor

# ╔═╡ 6122733d-3184-4619-8c5c-cf994aceb2e7
slope

# ╔═╡ 28df8e0d-b4ee-478c-8ea9-a78a3bc6d144
slope * 1e3

# ╔═╡ 8216773d-0e11-435d-801d-476fe4a0191d
Gadfly.plot(
	sample_epw_envdiff,
	x=:weather_station_distance,
	y=:epw_envtempdiff,
	intercept=[intercept], slope=[slope], Geom.abline(color="red", style=:dash),
	Geom.point,
	Theme(default_color="black", point_size=1pt),
	Guide.xlabel("Distance Weather Station (m)"),
	Guide.ylabel("Temperature-Δ (°C)")
)

# ╔═╡ 10b04e08-3a86-45c0-9425-e3d80dda1b99


# ╔═╡ 388d9daf-d6e3-4551-84bb-56906f013900
# looking at general trends
temperature_trends = combine(
	groupby(
		epw_training_data,
		:date
	), :electricity_mwh .=> [mean∘skipmissing,std∘skipmissing] .=> ["mean","std"]
)

# ╔═╡ 99c5d986-0cdb-4321-82bf-49ced0502430
begin
	ymins = temperature_trends.mean .- (temperature_trends.std)
	ymaxs = temperature_trends.mean .+ (temperature_trends.std)

	Gadfly.plot(
		temperature_trends,
		x=:date,
		y=:mean,
		ymin = ymins,
		ymax = ymaxs,
		Geom.point,
		Geom.errorbar,
		Theme(default_color="black")
	)
end

# ╔═╡ 08701f76-1425-4ac0-8f57-07b8e12b8abe
# Gadfly.plot(
# 	epw_training_data,
# 	x="Drybulb Temperature (°C)",
# 	y="electricity_mwh"
# )

# ╔═╡ 75763bff-6b90-4c9f-9a4a-23581da228a2
# Plots.scatter(
# 	sample_training.TMP,
# 	sample_training.electricity_mwh,
# 	color="black"
# )

# ╔═╡ e2b8d9e9-a832-47d2-8e7b-11584a897712
sample_electricity = dropmissing(sample_training, :electricity_mwh)

# ╔═╡ dc7d7743-8463-4c1b-b068-9589fa8cb807
@info UnicodePlots.scatterplot(
	sample_electricity.TMP, 
	sample_electricity.electricity_mwh,
	marker=:cross,
	title="Electricity / Temperature - ID#:"*string(sample_id),
	xlabel="City Temperature",
	ylabel="(MWh)"
)

# ╔═╡ db005395-1652-421a-adbc-e4c5ca075f93
@info UnicodePlots.scatterplot(
	sample_electricity.LST_Day_1km .* 0.02 .- 273.15, 
	sample_electricity.electricity_mwh,
	marker=:circle,
	title="Electricity / Temperature - ID#:"*string(sample_id),
	xlabel="Localized Temperature",
	ylabel="(MWh)"
)

# ╔═╡ b92c6f37-2329-4ce8-bcf9-eb53f09f7266
md"""
#### now exploring a bit about this UHI effect
"""

# ╔═╡ e6cb1988-a03f-4931-84f6-a1bff6da1c4e
lhi_captured = filter( row -> ~isnan(row.LST_Day_1km), training_data )

# ╔═╡ 030c5f35-1f1d-4bc0-9bd6-c97c193ac835
lhi_captured.mean_2m_air_temperature

# ╔═╡ df7e5922-5d1e-4a55-8ae6-9e3e60e0fa9b
uhi_difference = ( lhi_captured.LST_Day_1km .* 0.02 .- 273.15 ) .- ( lhi_captured.mean_2m_air_temperature .- 273.15 )

# ╔═╡ 1cc9f9a5-1bd2-4949-b789-9f584dd1a8f0
@info UnicodePlots.histogram(
	uhi_difference, 
	nbins=30, 
	closed=:left,
	title="UHI Impact: MODIS LST minus Mean Temp",
	ylabel="UHI Difference (°C)"
)

# ╔═╡ 8c611f25-8632-4d2a-a2e9-02906262f9a4
sample_gas = dropmissing(sample_training, :naturalgas_mwh)

# ╔═╡ 2cbe4a5c-27b5-4b8b-94b9-8c8037e954cc
@info UnicodePlots.scatterplot(
	sample_gas.WIND, 
	sample_gas.naturalgas_mwh,
	title="Electricity / Wind - ID#:"*string(sample_id),
	xlabel="Wind Speed (m/s)",
	ylabel="(MWh)"
)

# ╔═╡ 3fd0d3b0-4824-4946-9275-1f75645cd26c
UnicodePlots.scatterplot(
	sample_electricity.TMP, 
	sample_electricity.electricity_mwh,
	marker=:circle,
	title="Electricity / Temperature - ID#:"*string(sample_id),
	xlabel="Temperature",
	ylabel="(MWh)"
)

# ╔═╡ e323534e-9603-41a5-be8e-4c22ae70bba7
Gadfly.plot(
	Gadfly.layer(
		x = sample_training.date,
		y = lst_nightmeans .- mean_temp_c,
		Geom.point,
		Geom.line
	),
	Gadfly.layer(
		x = sample_training.date,
		y = lst_daymeans .- mean_temp_c,
		Geom.point,
		Geom.line,
		Theme(default_color="indianred")
	),
	Guide.ylabel("Deviation from City Mean (°C)"),
	Guide.xlabel("Date")
)

# ╔═╡ fcb8ec47-74a0-4555-87ae-9519602fa300
# # want to get a sample of the full data behavior but don't want to overwhelm the computer
# begin
# 	n_terms = 100
# 	sample_idx = rand(1:nrow(training_data), n_terms)
# 	sample_data = training_data[sample_idx, :]
# end;

# ╔═╡ ee641a2e-1fd3-46b2-af5e-757e8a4faa30
# # now playing a bit with the relationship between temperature and electricity
# Gadfly.plot(
# 	x = sample_data.TMP,
# 	y = sample_data.electricity_mwh,
# 	Geom.point,
# 	Guide.xlabel("Temperature"),
# 	Guide.ylabel("Electrcity (MWh)")
# )

# ╔═╡ Cell order:
# ╠═ac97e0d6-2cfa-11ed-05b5-13b524a094e3
# ╠═786c2441-7abb-4caa-9f50-c6078fff0f56
# ╟─9b3790d3-8d5d-403c-8495-45def2c6f8ba
# ╠═020b96e3-d218-470d-b4b0-fc9b708ffdf3
# ╠═9aa06073-d43e-4658-adb9-bbc11425978d
# ╠═56f43ba6-568b-436d-85a5-a8da5a0a3956
# ╠═8883d4ac-9ec4-40b5-a885-e1f3c5cbd4b9
# ╠═c65e56e9-0bde-4278-819c-f3148d71668b
# ╠═dcbcb589-b320-4e4d-842c-25daab5054d4
# ╟─c5aaf1a9-fd38-4cb0-9ee8-e04ac0a2fecb
# ╠═731c8bcc-2350-484e-ac83-ad94e0b64d46
# ╠═b897cbfd-484d-410e-b698-14196aaf8a73
# ╠═6b120bfc-e1e3-4b52-a99d-aa60518b32ad
# ╠═348c4307-94dc-4d5f-82b0-77dc535c1650
# ╠═21fac793-3617-437e-bc6d-85def0ae01c9
# ╠═09a4789c-cbe7-496e-98b5-a2c2db3102b6
# ╠═ac31e0ac-b35b-494f-814c-3f9eaf26e8b1
# ╠═637220ba-c76a-4210-8c08-fde56b86366a
# ╠═c7011ece-6d0b-41b9-83c1-0d8e956fe515
# ╠═e350f7f6-5757-4b8e-95bc-c6c098202d49
# ╠═87e7b997-6a8a-4395-8d0c-f6f59d34d96f
# ╟─e4cda53e-7fe2-4cdb-8a9c-c9bdf67dd66f
# ╠═39997f75-aa5e-4356-8353-b62b2ff01a98
# ╠═3b980465-9b75-404d-a41f-06ad351d12ae
# ╠═a693145c-8552-44ff-8b46-485c8c8fb738
# ╠═aa25d6e8-0dab-4f57-bc9e-f26373538826
# ╠═7ddf461c-e167-4db0-98c7-940fc962bb6a
# ╠═7c43c2eb-63e9-4508-8cae-75c48661f328
# ╟─091f7d95-eae6-474d-8bd0-29a12f99e3f8
# ╠═fcc7ed57-3e9a-488c-80c7-3a09879aacbe
# ╠═6b402147-68e1-4369-86bb-977304dedc01
# ╠═6b1a360e-7d11-4809-ad20-8106241163d2
# ╟─e392b69c-abd4-454b-bf96-2cc3e4723b46
# ╠═5b461de9-40b3-40f1-bce9-a3b620474669
# ╠═3a473c8d-bd0c-4309-b637-84bde0389511
# ╠═7529c522-c3ac-4f6c-9385-a91f8b640c45
# ╠═d270993b-b9a0-4e21-a309-e3e9fcaf18e1
# ╟─af4c25cb-228d-4a76-9bc8-78db0edc836d
# ╟─f7030f91-7785-4355-83e9-3e4ea016ffe4
# ╠═812d7cda-ce12-495c-be38-52c8f0f23747
# ╠═01e27ad1-9b45-40b3-8d03-26400cd153f9
# ╠═4eec8fe5-2c77-489d-8d1f-fef0ad388a50
# ╠═09a85206-3f09-4ba4-8fe3-85d32c8e8793
# ╠═6ec97827-015d-4da4-8e58-4564db02fedf
# ╟─87b55c18-6842-4321-b2c7-c1abd8fef6fc
# ╟─e81eacd1-22ff-4890-989e-e4ec638f06b5
# ╟─8ef0fc6c-2937-44bf-87b7-7914c8f2e41d
# ╟─f37dd092-aa81-4669-8925-665415c90aaf
# ╠═512cad43-28ca-4b07-afa5-20571a31b311
# ╠═36a789ff-eaf9-44b4-8d16-1893c8b6d39b
# ╠═b7d62ac5-1c30-4ec6-b307-5ae0d6bcaa29
# ╠═f3774051-976b-45db-a388-77757d23a299
# ╠═bb59670b-9ca1-45b3-84f4-0c190734f02f
# ╠═ce702ad5-a862-4bb0-805d-2bc4f6ce096d
# ╠═b564d2d8-a55d-4900-9803-29cd24e2e879
# ╠═27a0b91c-7859-4193-939b-5ceac1afaeee
# ╠═3b71cbd9-39ee-4762-ab8e-12a6004835ab
# ╠═6122733d-3184-4619-8c5c-cf994aceb2e7
# ╠═28df8e0d-b4ee-478c-8ea9-a78a3bc6d144
# ╠═8216773d-0e11-435d-801d-476fe4a0191d
# ╠═10b04e08-3a86-45c0-9425-e3d80dda1b99
# ╠═388d9daf-d6e3-4551-84bb-56906f013900
# ╟─99c5d986-0cdb-4321-82bf-49ced0502430
# ╠═08701f76-1425-4ac0-8f57-07b8e12b8abe
# ╠═75763bff-6b90-4c9f-9a4a-23581da228a2
# ╠═e2b8d9e9-a832-47d2-8e7b-11584a897712
# ╠═dc7d7743-8463-4c1b-b068-9589fa8cb807
# ╠═db005395-1652-421a-adbc-e4c5ca075f93
# ╟─b92c6f37-2329-4ce8-bcf9-eb53f09f7266
# ╠═e6cb1988-a03f-4931-84f6-a1bff6da1c4e
# ╠═030c5f35-1f1d-4bc0-9bd6-c97c193ac835
# ╠═df7e5922-5d1e-4a55-8ae6-9e3e60e0fa9b
# ╠═1cc9f9a5-1bd2-4949-b789-9f584dd1a8f0
# ╟─8c611f25-8632-4d2a-a2e9-02906262f9a4
# ╠═2cbe4a5c-27b5-4b8b-94b9-8c8037e954cc
# ╠═3fd0d3b0-4824-4946-9275-1f75645cd26c
# ╠═e323534e-9603-41a5-be8e-4c22ae70bba7
# ╠═fcb8ec47-74a0-4555-87ae-9519602fa300
# ╟─ee641a2e-1fd3-46b2-af5e-757e8a4faa30
