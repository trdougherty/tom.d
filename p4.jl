### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ 0b706464-daee-4d19-8b08-4d24a9afae4c
begin
	import Pkg
	Pkg.activate(Base.current_project())

	using CSV
	using Dates
	using DataFrames
	using Missings
	using MLJ
	using Gadfly
	using Statistics
	using StatsBase
	import Cairo, Fontconfig
end;

# ╔═╡ 365b88bd-eead-4022-8a2c-6bd49ef82c41
using Plots

# ╔═╡ b70dfb57-7fad-40d3-a413-696221ac2d0e
begin
using Random
Random.seed!(1234)
end

# ╔═╡ 87a318da-2d4a-11ed-17be-3383e7a10f80
begin
	data_path = joinpath(pwd(), "data", "nyc")
	input_dir = joinpath(data_path, "p3_o")
	output_dir = joinpath(data_path, "p4_o")
	mkpath(output_dir)
end

# ╔═╡ 6c067da7-c69a-4dab-af75-7ad43482e96b
md"""
# Analysis
This is going to be split into two subcomponents. One for the analysis using environmental data and one for the analysis using the EPW files.
"""

# ╔═╡ 251075f8-c41c-4cab-b910-649de6d70637
md"""
#### Functional system of reading data and training
This should take a large block of data (X) and a large array of terms (Y) - normalize the X values, train the x values to predict the Y values
"""

# ╔═╡ ea8e4429-b797-4452-bb7d-73ebbd58af76
md"""
error functions, as defined by [ASHRAE](https://upgreengrade.ir/admin_panel/assets/images/books/ASHRAE%20Guideline%2014-2014.pdf)
"""

# ╔═╡ cf400726-f3dd-4629-85a1-d99760fd6118
cvstd(ŷ, y, p=1) = (100 * mean(y)) / (( sum( (y .- mean(y)).^2 ) / (length(y)-1))^0.5 )

# ╔═╡ 3b389516-480f-4a0a-aff1-3e37c5295f41
cvrmse(ŷ, y, p=1) = 100 * ( sum((y.-ŷ).^2) / (length(y)-p) )^0.5 / mean(y)

# ╔═╡ 8f44da0b-4fe9-4524-9811-8e655aeb2242
nmbe(ŷ, y, p=1) = 100 * sum(y.-ŷ) / (( length(y)-p ) * mean(y) )

# ╔═╡ dbb0ca89-7642-4f0f-9848-f211bd7b742c
function _test_suite(ŷ, y)
	@info "Test Suite" cvstd(ŷ, y) cvrmse(ŷ, y) nmbe(ŷ, y)
end

# ╔═╡ 7463dc36-310c-4d37-8a66-81f8aa086c2d
function _clean!(data)
	filter!(row -> all(x -> !(x isa Number && isnan(x)), row), data)
end

# ╔═╡ 98618170-0e02-4c3a-88e7-c8fb447e675f
function _normalize(data)
	normalizer = fit(ZScoreTransform, data[], dims=1)
	xnorm = StatsBase.transform(normalizer, data[])
	return (xnorm, normalizer)
end

# ╔═╡ cbe44127-80a1-4270-81f3-59743b8b4d51
function _train_pipeline(X, y, mach)
	MLJ.fit_only!(mach, X, y)
end

# ╔═╡ 884bf4ef-7032-4967-a2dd-e8c5d2ed4df0
function _validate_pipeline(X, y, mach)
	ŷ = MLJ.predict(mach, X)
	_test_suite(ŷ, y)
	ŷ
end

# ╔═╡ f44af340-b99d-464b-a320-508dd516a3ce
begin
	env_dropping_cols = [
		"Property Id",
		"date",
		"council_region",
		"weather_station_id",
		"weather_station_distance",
		"naturalgas_kbtu",
		"electricity_kbtu",
		"cnstrct_yr"
	]
	
	train_env = CSV.read(
		joinpath(input_dir, "training_environmental.csv"), 
		DataFrame
	)
	validate_env = CSV.read(
		joinpath(input_dir, "validating_environmental.csv"), 
		DataFrame
	)
	test_env = CSV.read(
		joinpath(input_dir, "testing_environmental.csv"), 
		DataFrame
	)

	_clean!(train_env)
	_clean!(validate_env)
	_clean!(test_env)
end;

# ╔═╡ 6881c609-0257-4730-9401-1347ba2aa3d9
begin
	epw_dropping_cols = [
		"Property Id",
		"date",
		"council_region",
		"weather_station_id",
		"weather_station_distance",
		"naturalgas_kbtu",
		"electricity_kbtu",
		"cnstrct_yr"
	]
	
	train_epw_full = CSV.read(
		joinpath(input_dir, "training_epw.csv"), 
		DataFrame
	)
	validate_epw = CSV.read(
		joinpath(input_dir, "validating_epw.csv"), 
		DataFrame
	)
	test_epw = CSV.read(
		joinpath(input_dir, "testing_epw.csv"), 
		DataFrame
	)

	random_sample_idx = shuffle(1:nrow(train_epw_full))[1:nrow(train_env)]
	train_epw = train_epw_full[random_sample_idx, :]

	_clean!(train_epw)
	_clean!(validate_epw)
	_clean!(test_epw)
end;

# ╔═╡ cbbc3e9d-f59b-4ac9-ba1c-f1e5543c5662
md"""
#### Starting to get into the ML system
----
1. clean the data
2. normalize the x terms
3. train the machine
4. validate against validation set
"""

# ╔═╡ 5444bd48-4dee-42f6-baf1-09523dce89e0
Tree = @load DecisionTreeRegressor pkg=DecisionTree verbosity=0

# ╔═╡ a9f2837c-651e-4498-8b0e-5a8d8282bf9e
model = Tree(max_depth=8)

# ╔═╡ fcc05219-b405-4dd2-be12-bf2306a34647
begin
step_size = 5000
distance_range = collect(step_size:step_size:25000)
distance_df = DataFrame(
	"distance" => distance_range,
	"distance_index" => collect(1:length(distance_range))
)
distance_range
end

# ╔═╡ cc3e7642-a69d-4190-99de-9563a916038c
md"""
###### Electricity - environmental terms
"""

# ╔═╡ c2f4541e-d453-4e44-99ed-229a126cda7e
begin
	train_electric_env = select(train_env, Not(:naturalgas_mwh))
	validate_electric_env = select(validate_env, Not(:naturalgas_mwh))
	test_electric_env = select(test_env, Not(:naturalgas_mwh))

	dropmissing!(train_electric_env)
	dropmissing!(validate_electric_env)
	dropmissing!(test_electric_env)

	train_env_electric_aux = select(train_electric_env, env_dropping_cols)
	validate_env_electric_aux = select(validate_electric_env, env_dropping_cols)
	test_env_electric_aux = select(test_electric_env, env_dropping_cols)

	select!(train_electric_env, Not(env_dropping_cols))
	select!(validate_electric_env, Not(env_dropping_cols))
	select!(test_electric_env, Not(env_dropping_cols))

	train_xe = select(train_electric_env, Not(:electricity_mwh))
	train_ye = train_electric_env.electricity_mwh

	validate_xe = select(validate_electric_env, Not(:electricity_mwh))
	validate_ye = validate_electric_env.electricity_mwh
end;

# ╔═╡ d504e9d4-12b8-4a90-9f7f-b4bdd418924a
mach_env = machine(model, train_xe, train_ye; cache=false)

# ╔═╡ 6fd78bf3-89b4-4270-8c44-71796d3e998b
evaluate!(mach_env, measure=l1)

# ╔═╡ 03149a9e-5a5b-4dbf-8441-4f9843e54de6
fit_only!(mach_env)

# ╔═╡ 2b13b70d-86ae-4215-858b-18d65b25c89a
begin
ŷ_env = MLJ.predict(mach_env,validate_xe)
_test_suite(ŷ_env, validate_electric_env.electricity_mwh)
end

# ╔═╡ 0f85cfe8-699f-4bf0-bcde-bfaa42411604
# env_importance = DataFrame(
# 	"variable" => names(select(train_electric_env, Not(:electricity_mwh))),
# 	"importance" => permute_importance(model_ee)
# );

# ╔═╡ 8f6fc575-1a10-4dc1-8622-8e88754ab29b
# sort(env_importance, :importance, rev=true)

# ╔═╡ a7a3bd37-37d9-4f8c-9ac6-fee8b51dea0f
# Pair(collect(1:length(names(train_electric_env))), names(train_electric_env))

# ╔═╡ 1bbb7b06-162c-4709-b492-858d0b767b0a
begin
dist_index = convert.(Int64, ceil.(validate_env_electric_aux.weather_station_distance ./ step_size))
end;

# ╔═╡ 8b786741-47e1-4e7d-ab8a-1f7c4bd341ae
begin
	validation_analysis_e = deepcopy(validate_env_electric_aux)
	validation_analysis_e.distance_index = dist_index
	validation_distances = leftjoin(
		validation_analysis_e,
		distance_df,
		on="distance_index"
	)
	validation_distances[!,"recorded"] = validate_ye
	validation_distances[!,"prediction"] = ŷ_env
	validation_distance_error_grouped = groupby(validation_distances, :distance)
	validation_distance_error_env = combine(validation_distance_error_grouped) do df
	(
	nmbe = nmbe(df.prediction, df.recorded),
	cvrmse = cvrmse(df.prediction, df.recorded),
	cvstd = cvstd(df.prediction, df.recorded)
	)
	end

	validation_distance_error_env[:,"model"] = repeat(["Remote Sensing"], nrow(validation_distance_error_env))
end;

# ╔═╡ bec80c0b-8edd-4cbf-ac87-bd4d163d435e
# begin
# Gadfly.plot(
# 	validation_distance_error_epw,
# 	x=:distance,
# 	y=:cvrmse,
# 	Geom.point,
# 	Geom.line,
# 	Guide.xlabel("Distance (m)"),
# 	Guide.ylabel("CVRMSE"),
# 	Guide.title("CVRMSE - Remote Sensing Data"),
# )
# end

# ╔═╡ f524548f-8774-4e30-a901-649e8e146556
md"""
##### Electricity - EPW files
"""

# ╔═╡ d395a123-32db-4783-a75b-c74c4bde85a9
begin
	train_electric_epw = select(train_epw, Not(:naturalgas_mwh))
	validate_electric_epw = select(validate_epw, Not(:naturalgas_mwh))
	test_electric_epw = select(test_epw, Not(:naturalgas_mwh))

	dropmissing!(train_electric_epw)
	dropmissing!(validate_electric_epw)
	dropmissing!(test_electric_epw)

	train_epw_electric_aux = select(train_electric_epw, env_dropping_cols)
	validate_epw_electric_aux = select(validate_electric_epw, env_dropping_cols)
	test_epw_electric_aux = select(test_electric_epw, env_dropping_cols)

	select!(train_electric_epw, Not(env_dropping_cols))
	select!(validate_electric_epw, Not(env_dropping_cols))
	select!(test_electric_epw, Not(env_dropping_cols))

	train_xe_epw = select(train_electric_epw, Not(:electricity_mwh))
	train_ye_epw = train_electric_epw.electricity_mwh

	validate_xe_epw = select(validate_electric_epw, Not(:electricity_mwh))
	validate_ye_epw = validate_electric_epw.electricity_mwh
end;

# ╔═╡ 6e37a6d4-e8bc-4978-b576-19d0c8d907b5
mach_epw = machine(model, train_xe_epw, train_ye_epw; cache=false);

# ╔═╡ 96ea99d9-4890-4605-9f19-fbf10abd451a
fit_only!(mach_epw)

# ╔═╡ 89ce438d-0caa-4fd8-882c-5df5322330ea
begin
ŷ_epw = MLJ.predict(mach_epw,validate_xe_epw)
_test_suite(ŷ_epw, validate_ye_epw)
end

# ╔═╡ 92964c4e-700d-492d-a707-c74ef06985f8
begin
	validation_analysis_p = deepcopy(validate_epw_electric_aux)
	validation_analysis_p.distance_index = 	convert.(Int64, ceil.(validate_epw_electric_aux.weather_station_distance ./ step_size))

	validation_distances_p = leftjoin(
		validation_analysis_p,
		distance_df,
		on="distance_index"
	)
	
	validation_distances_p[!,"recorded"] = validate_ye_epw
	validation_distances_p[!,"prediction"] = ŷ_epw

	validation_grouped = groupby(validation_distances_p, :distance)
	validation_distance_error_epw = combine(validation_grouped) do df
	(
	nmbe = nmbe(df.prediction, df.recorded),
	cvrmse = cvrmse(df.prediction, df.recorded),
	cvstd = cvstd(df.prediction, df.recorded)
	)
	end

	validation_distance_error_epw[:,"model"] = repeat(["Epw"], nrow(validation_distance_error_epw))
end;

# ╔═╡ a54e6db9-cc4b-4c37-9496-2b3cfc92399f
# begin
# combined_plot_epw = Gadfly.plot(
# 	stack(validation_distance_error_epw, 2:4), 
# 	ygroup="variable", 
# 	x="distance", 
# 	y="value",
# 	Geom.subplot_grid(Geom.line),
# )
# end;

# ╔═╡ 6d87c624-78c6-45ab-93df-e8177fe19143
validation_distance_error_epw;

# ╔═╡ bff43adb-d2a5-4b97-8570-87334b6141af
md"""
##### Now only looking at building information (size, area..)
"""

# ╔═╡ 04f84642-de6f-44c5-b7a3-e68dbac1b5d0
names(train_electric_epw)

# ╔═╡ d4afdc37-42db-4af1-98ea-6d33e3314c72
train_electric_basic = select(train_electric_epw, "heightroof", "groundelev", "area", "electricity_mwh");

# ╔═╡ 0be7c936-9d47-4fd8-b7c3-471119d888d5
validate_electric_basic = select(validate_electric_epw, "heightroof", "groundelev", "area", "electricity_mwh");

# ╔═╡ 33347a14-5630-437c-9eb2-aeca89896545
begin
train_xe_basic = select(train_electric_basic, Not(:electricity_mwh))
train_ye_basic = train_electric_basic.electricity_mwh

validate_xe_basic = select(validate_electric_basic, Not(:electricity_mwh))
validate_ye_basic = validate_electric_basic.electricity_mwh

model_ebasic = machine(model, train_xe_basic, train_ye_basic; cache=false)
fit_only!(model_ebasic)

ŷ_basic = MLJ.predict(model_ebasic, validate_xe_basic)
_test_suite(ŷ_basic, validate_ye_basic)


validation_analysis_basic = deepcopy(validate_epw_electric_aux)
validation_analysis_basic.distance_index = 	convert.(Int64, ceil.(validate_epw_electric_aux.weather_station_distance ./ step_size))

validation_analysis_basic[!,"recorded"] = validate_ye_basic
validation_analysis_basic[!,"prediction"] = ŷ_basic

validation_distances_basic = leftjoin(
	validation_analysis_basic,
	distance_df,
	on="distance_index"
)

validation_grouped_basic = groupby(validation_distances_basic, :distance)
validation_distance_error_basic = combine(validation_grouped_basic) do df
	(
	nmbe = nmbe(df.prediction, df.recorded),
	cvrmse = cvrmse(df.prediction, df.recorded),
	cvstd = cvstd(df.prediction, df.recorded)
	)
end

validation_distance_error_basic[:,"model"] = repeat(["Basic"], nrow(validation_distance_error_basic))
end;

# ╔═╡ a276e4f0-4a35-4163-9930-252e167df838
begin
plotterm = Symbol("cvrmse")
colorscheme = palette(:default)
	
Gadfly.plot(
Gadfly.layer(
	validation_distance_error_env,
	x=:distance,
	y=plotterm,
	Geom.point,
	Geom.line,
	Theme(default_color=colorscheme[1])
),
Gadfly.layer(
	validation_distance_error_epw,
	x=:distance,
	y=plotterm,
	Geom.point,
	Geom.line,
	Theme(default_color=colorscheme[2]),
),
Gadfly.layer(
	validation_distance_error_basic,
	x=:distance,
	y=plotterm,
	Geom.point,
	Geom.line,
	Theme(default_color=colorscheme[3])
),
Guide.xlabel("Distance from Weather Station (m)"),
Guide.ylabel(string(plotterm)),
Guide.title("Model Performance with Distance from Weather Station")
)
end

# ╔═╡ 10f7c1c0-88b1-4dba-b9bb-931b34a87fa4
full_distance_errors = vcat(
	stack(validation_distance_error_env, 2:4),
	stack(validation_distance_error_epw, 2:4),
	stack(validation_distance_error_basic, 2:4)
);

# ╔═╡ 0d1e512b-fd03-4ea9-a82b-5aa005d1e3be
begin
full_distanceplot = Gadfly.plot(
	full_distance_errors,
	color="model",
	ygroup="variable",
	x="distance", 
	y="value",
	Guide.title("Model Errors vs. Distance from Weather Station"),
	Guide.ylabel(""),
	Guide.xlabel("Distance (m)"),
	Geom.subplot_grid(
		Geom.point, 
		Geom.line, 
		free_y_axis=true,
		Guide.xticks(ticks=distance_range),
		Coord.cartesian(xmax=25e3)
	),
	Theme(point_size=2pt),
)
end

# ╔═╡ 47816f61-b6f8-4a4c-a9c3-89da41a06614
draw(
	PNG(
		joinpath(output_dir, "full_distance_errors.png"), 
		15cm, 
		12cm,
		dpi=700
	), 
	full_distanceplot
)

# ╔═╡ a6745d5a-da4f-40d7-8fb6-e5adc4646904
full_distance_means = combine(groupby(full_distance_errors, [:model, :variable, :distance]), :value => mean)

# ╔═╡ 77958186-598c-460e-bd0c-b593f4a81ddc
unstack(full_distance_means, :model, :value_mean)

# ╔═╡ ffb66045-ed91-4884-82b4-c9859957537c
unstack(combine(
	groupby(full_distance_errors, [:model, :variable]), 
	:value .=> [std]
), :variable, :value_std)

# ╔═╡ 84cd8369-e76c-40d0-902b-6729de372436
# Gadfly.plot(
# 	validation_distance_error_p,
# 	x=:distance,
# 	y=:mean_error,
# 	Geom.point,
# 	Geom.abline(color="indianred"),
# 	Guide.xlabel("Distance (m)"),
# 	Guide.ylabel("MAE"),
# 	Guide.title("MAE - EPW"),
# 	Theme(default_color="black"),
# )

# ╔═╡ 4c887e2c-05f2-468e-a80e-60604e8c2a5a
# Gadfly.plot(
# 	Gadfly.layer(
# 		validation_distance_error,
# 		x=:distance,
# 		y=:mean_error,
# 		ymin=validation_distance_error.mean_error .- validation_distance_error.std_error,
# 		ymax=validation_distance_error.mean_error .+ validation_distance_error.std_error,
# 		Geom.point,
# 		Theme(point_size=1pt)
# 		# Geom.errorbar,
# 		),
# 	Gadfly.layer(
# 		validation_distance_error_p,
# 		x=:distance,
# 		y=:mean_error,
# 		ymin=validation_distance_error_p.mean_error .- validation_distance_error_p.std_error,
# 		ymax=validation_distance_error_p.mean_error .+ validation_distance_error_p.std_error,
# 		Geom.point,
# 		Theme(default_color="indianred", point_size=1pt)
# 		# Geom.errorbar
# 	),
# 	Guide.xlabel("Distance (m)"),
# 	Guide.ylabel("NMBE"),
# 	Guide.title("NMBE - Remote Sensing")
# )

# ╔═╡ cf738544-d708-4b66-ac42-e63ed962dfb9
md"""
### 1. EPW Weather Station Analysis
Using values captured from nearby weather stations to make predictions of the building energy consumption
"""

# ╔═╡ f5fdd299-fa76-436b-8011-d898362e2cf6
begin
	# want to get the difference with the LST
	sample_analysis = deepcopy(train_env)
	train_lst = sample_analysis.LST_Day_1km .* 0.02 .- 273.15
	sample_analysis[!,"lst_diff"] = train_lst .- sample_analysis.TMP
end;

# ╔═╡ dfc69690-5f9c-4a99-bbdb-f4ad6768fe5b
begin
	train_temps = select(
		sample_analysis,
		["date","lst_diff"]
	)
	filter!(:lst_diff => x -> !any(f -> f(x), (ismissing, isnothing, isnan)), train_temps)
	train_temps[!,"month"] = month.(train_temps.date)
end

# ╔═╡ 15a7cd0b-4fca-42da-a1b0-2f3a6b742506
lst_date_diff = combine(
	groupby(
		train_temps,
		:date
	),
	:lst_diff .=> [mean, std]
)

# ╔═╡ c8d0b43d-2554-4587-9ac5-95169f293126
begin
	ymins = lst_date_diff.lst_diff_mean .- lst_date_diff.lst_diff_std
	ymaxs = lst_date_diff.lst_diff_mean .+ lst_date_diff.lst_diff_std
end;

# ╔═╡ 95d474da-67bb-4fde-aff4-4f97b36ea1f0
begin
	# plot demonstrating the typical difference between average temperature readings and measured land surface from MODIS
	lst_diffplot = Gadfly.plot(
		lst_date_diff,
		x=:date,
		y=:lst_diff_mean,
		ymin=ymins,
		ymax=ymaxs,
		Geom.point,
		Geom.errorbar,
		Guide.xlabel(nothing),
		Guide.ylabel("LSTΔ - °C"),
		Guide.title("Land Surface Temperature (MODIS) minus Regional Temperature"),
		Guide.yticks(ticks=0:3:15),
		Guide.xticks(ticks=DateTime("2018-01-1"):Month(1):DateTime("2020-06-01")),
		Theme(default_color="black")
	)
	draw(
		PNG(
			joinpath(output_dir, "lstdiff.png"), 
			6inch, 
			4inch,
			dpi=700
		), 
		lst_diffplot
	)
end

# ╔═╡ b56b6fa5-232e-4f01-8693-7840150d3b1b
md"""
### p1. Electric Analysis
---
"""

# ╔═╡ 211ed982-1d78-435a-84bf-aaff3cc8cb95
countmap(sample_analysis[:,"Property Id"])

# ╔═╡ fe6d0af8-c44b-4ead-99bf-e8d5d5b410fd
sample = filter( x -> x["Property Id"] == 6683659, train_env )

# ╔═╡ 2395c7da-4287-45e9-9876-35b1a5c54e61
Gadfly.plot(
	sample,
	x=:date,
	y=:electricity_mwh,
	Geom.point,
	Geom.line,
	Guide.title("Sample Electricity"),
	Guide.ylabel("Electricity MWh"),
	Guide.xlabel("Date")
)

# ╔═╡ a63b079f-6d10-4f91-abad-988d6945b8c2
sample_lst = sample.LST_Day_1km .* 0.02 .- 273.15

# ╔═╡ 49d60158-f0bf-47df-a69c-d4bdf5f5c9ec
temp_diff = sample_lst .- sample.TMP

# ╔═╡ ae677d3b-2f5f-45d1-abcb-fe35103a6d50
Gadfly.plot(
	sample,
	x=:date,
	y=temp_diff,
	Geom.point,
	Geom.line
)

# ╔═╡ 980b6bb8-a3a6-479a-910d-4b17639aa8a0
# annual_electricity = combine(
# 	groupby(select(train, ["cnstrct_yr", "electricity_mwh"]), :cnstrct_yr),
# 	:electricity_mwh .=> [mean, std]
# )

# ╔═╡ 5732f35c-4d90-4667-b08d-6f0e4bacaaed


# ╔═╡ 1df626cd-cd9c-4c8a-8c0e-1839da982ac0


# ╔═╡ 3922cac0-e46c-4601-a6e3-83564e20bb26
begin
	Gadfly.plot(
		sample,
		Gadfly.layer(
			x=:date,
			y=:TMP,
			Geom.point,
			Geom.line,
			Theme(default_color="black")
		),
		Gadfly.layer(
			x=:date,
			y=sample_lst,
			Geom.point,
			Geom.line,
			Theme(default_color="indianred")
		)
	)
end

# ╔═╡ 0e31c831-abb4-4cf1-9bce-fe5cbafb369b
# begin
# 	removing_features = [
# 		"Property Id",
# 		"date",
# 		"council_region",
# 		"electricity_mwh"
# 	]

# 	# this is for the training pipeline of the standardizer
	
# 	# need y values with no missing
# 	train_eclean = dropmissing(train, :electricity_mwh);
# 	train_ye = train_eclean.electricity_mwh;

# 	# likewise, want all x values with no nan
# 	train_electric = select(
# 		select(
# 			train, 
# 			Not([:naturalgas_mwh, :naturalgas_kbtu, :electricity_kbtu])
# 		), :electricity_mwh, :
# 	)
	
# 	train_electric_features = select(train_electric, Not(:electricity_mwh))
# 	train_eslim = filter(
# 		row -> all(x -> !(x isa Number && isnan(x)), row), 
# 		train_electric_features
# 	)

# 	# this is the raw data we need for training
# 	train_eclean
# 	train_ye
# end;

# ╔═╡ d73b7a0f-5c7b-4bd2-9b08-517ff8dee8fc
# length(train_ye)

# ╔═╡ c9bd2e90-4cb5-47f0-9a99-1f9f4a007329
# begin
# 	train_efeatures = select(
# 		train_eclean,
# 		Not([
# 			removing_features...,
# 			"electricity_kbtu",
# 			"naturalgas_kbtu",
# 			"naturalgas_mwh"
# 		])
# 	)
# 	nrow(train_efeatures)
# end

# ╔═╡ b4fcc76d-84e8-4ff3-b215-bc69fa7e7429


# ╔═╡ 2bdf8356-b54c-4a2d-9f8a-8fd2ba14a1a2
md"""
###### Note: we need to split into two separate analysis as we can't train the standardizer using missing values, which comes from the electricity and gas terms.
"""

# ╔═╡ 8f90f9ad-edc7-477c-990b-05daa5e2aace
# begin
# 	xtrain_std_mach = machine(Standardizer(), train_eslim);
# 	fit!(xtrain_std_mach);

# 	# electricity
# 	yetrain_std_mach = machine(UnivariateStandardizer(), train_ye);
# 	fit!(yetrain_std_mach);
# end;

# ╔═╡ b3d5522d-b1eb-4798-a9c4-2bc7655187f1
# # now going to apply the whitening to the full dataset with nans
# begin
# 	train_xnorm = MLJ.transform(xtrain_std_mach, train_electric_features);
# 	train_ynorm = MLJ.transform(yetrain_std_mach, train_electric.electricity_mwh);
# end

# ╔═╡ fabcb001-27cb-4278-931e-fcf72eabdc44
# fit!(train_x)

# ╔═╡ 6450f007-c12d-46b4-9d5f-4009caf7a097
# fit(ZScoreTransform, train_x, dims=2)

# ╔═╡ Cell order:
# ╠═0b706464-daee-4d19-8b08-4d24a9afae4c
# ╠═365b88bd-eead-4022-8a2c-6bd49ef82c41
# ╠═b70dfb57-7fad-40d3-a413-696221ac2d0e
# ╠═87a318da-2d4a-11ed-17be-3383e7a10f80
# ╟─6c067da7-c69a-4dab-af75-7ad43482e96b
# ╟─251075f8-c41c-4cab-b910-649de6d70637
# ╟─ea8e4429-b797-4452-bb7d-73ebbd58af76
# ╟─cf400726-f3dd-4629-85a1-d99760fd6118
# ╟─3b389516-480f-4a0a-aff1-3e37c5295f41
# ╟─8f44da0b-4fe9-4524-9811-8e655aeb2242
# ╟─dbb0ca89-7642-4f0f-9848-f211bd7b742c
# ╟─7463dc36-310c-4d37-8a66-81f8aa086c2d
# ╟─98618170-0e02-4c3a-88e7-c8fb447e675f
# ╠═cbe44127-80a1-4270-81f3-59743b8b4d51
# ╠═884bf4ef-7032-4967-a2dd-e8c5d2ed4df0
# ╟─f44af340-b99d-464b-a320-508dd516a3ce
# ╟─6881c609-0257-4730-9401-1347ba2aa3d9
# ╟─cbbc3e9d-f59b-4ac9-ba1c-f1e5543c5662
# ╠═5444bd48-4dee-42f6-baf1-09523dce89e0
# ╠═a9f2837c-651e-4498-8b0e-5a8d8282bf9e
# ╠═fcc05219-b405-4dd2-be12-bf2306a34647
# ╟─cc3e7642-a69d-4190-99de-9563a916038c
# ╠═c2f4541e-d453-4e44-99ed-229a126cda7e
# ╠═d504e9d4-12b8-4a90-9f7f-b4bdd418924a
# ╠═6fd78bf3-89b4-4270-8c44-71796d3e998b
# ╠═03149a9e-5a5b-4dbf-8441-4f9843e54de6
# ╠═2b13b70d-86ae-4215-858b-18d65b25c89a
# ╠═0f85cfe8-699f-4bf0-bcde-bfaa42411604
# ╠═8f6fc575-1a10-4dc1-8622-8e88754ab29b
# ╠═a7a3bd37-37d9-4f8c-9ac6-fee8b51dea0f
# ╠═1bbb7b06-162c-4709-b492-858d0b767b0a
# ╠═8b786741-47e1-4e7d-ab8a-1f7c4bd341ae
# ╟─bec80c0b-8edd-4cbf-ac87-bd4d163d435e
# ╟─f524548f-8774-4e30-a901-649e8e146556
# ╠═d395a123-32db-4783-a75b-c74c4bde85a9
# ╠═6e37a6d4-e8bc-4978-b576-19d0c8d907b5
# ╠═96ea99d9-4890-4605-9f19-fbf10abd451a
# ╠═89ce438d-0caa-4fd8-882c-5df5322330ea
# ╠═92964c4e-700d-492d-a707-c74ef06985f8
# ╠═a54e6db9-cc4b-4c37-9496-2b3cfc92399f
# ╠═6d87c624-78c6-45ab-93df-e8177fe19143
# ╟─bff43adb-d2a5-4b97-8570-87334b6141af
# ╠═04f84642-de6f-44c5-b7a3-e68dbac1b5d0
# ╠═d4afdc37-42db-4af1-98ea-6d33e3314c72
# ╠═0be7c936-9d47-4fd8-b7c3-471119d888d5
# ╠═33347a14-5630-437c-9eb2-aeca89896545
# ╠═a276e4f0-4a35-4163-9930-252e167df838
# ╠═10f7c1c0-88b1-4dba-b9bb-931b34a87fa4
# ╠═0d1e512b-fd03-4ea9-a82b-5aa005d1e3be
# ╠═47816f61-b6f8-4a4c-a9c3-89da41a06614
# ╠═a6745d5a-da4f-40d7-8fb6-e5adc4646904
# ╠═77958186-598c-460e-bd0c-b593f4a81ddc
# ╠═ffb66045-ed91-4884-82b4-c9859957537c
# ╠═84cd8369-e76c-40d0-902b-6729de372436
# ╠═4c887e2c-05f2-468e-a80e-60604e8c2a5a
# ╟─cf738544-d708-4b66-ac42-e63ed962dfb9
# ╠═f5fdd299-fa76-436b-8011-d898362e2cf6
# ╠═dfc69690-5f9c-4a99-bbdb-f4ad6768fe5b
# ╠═15a7cd0b-4fca-42da-a1b0-2f3a6b742506
# ╠═c8d0b43d-2554-4587-9ac5-95169f293126
# ╠═95d474da-67bb-4fde-aff4-4f97b36ea1f0
# ╟─b56b6fa5-232e-4f01-8693-7840150d3b1b
# ╠═211ed982-1d78-435a-84bf-aaff3cc8cb95
# ╟─fe6d0af8-c44b-4ead-99bf-e8d5d5b410fd
# ╟─2395c7da-4287-45e9-9876-35b1a5c54e61
# ╠═a63b079f-6d10-4f91-abad-988d6945b8c2
# ╠═49d60158-f0bf-47df-a69c-d4bdf5f5c9ec
# ╠═ae677d3b-2f5f-45d1-abcb-fe35103a6d50
# ╠═980b6bb8-a3a6-479a-910d-4b17639aa8a0
# ╠═5732f35c-4d90-4667-b08d-6f0e4bacaaed
# ╠═1df626cd-cd9c-4c8a-8c0e-1839da982ac0
# ╠═3922cac0-e46c-4601-a6e3-83564e20bb26
# ╠═0e31c831-abb4-4cf1-9bce-fe5cbafb369b
# ╠═d73b7a0f-5c7b-4bd2-9b08-517ff8dee8fc
# ╠═c9bd2e90-4cb5-47f0-9a99-1f9f4a007329
# ╠═b4fcc76d-84e8-4ff3-b215-bc69fa7e7429
# ╟─2bdf8356-b54c-4a2d-9f8a-8fd2ba14a1a2
# ╠═8f90f9ad-edc7-477c-990b-05daa5e2aace
# ╠═b3d5522d-b1eb-4798-a9c4-2bc7655187f1
# ╠═fabcb001-27cb-4278-931e-fcf72eabdc44
# ╠═6450f007-c12d-46b4-9d5f-4009caf7a097
