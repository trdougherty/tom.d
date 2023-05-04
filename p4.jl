### A Pluto.jl notebook ###
# v0.19.18

using Markdown
using InteractiveUtils

# ╔═╡ da3eed56-a0ea-11ed-381f-8fb87407934b
begin
import Pkg
Pkg.activate(Base.current_project())

using Cairo
using Fontconfig

using CSV
using DataFrames
using DataStructures
using Dates
using Gadfly
using GeoDataFrames

using MLJ
using ShapML
using Statistics
using UnicodePlots
using YAML
end;

# ╔═╡ 0f04176b-e090-46de-821f-e0e1dcb86aca
using MLJParticleSwarmOptimization

# ╔═╡ 57fa1025-1d94-4463-9fc7-7c4ebbd2f4b7
using Latexify

# ╔═╡ db32e6a6-8ace-4d7f-b8a4-0d875291bcfa
using Random

# ╔═╡ b06e9088-c511-48a7-96c8-4e10012f44e8
using Plots

# ╔═╡ ee536c4f-0273-4814-bf3a-6758cd13f968
using Compose

# ╔═╡ 9c906615-d33a-471d-a2af-393b9a963ba3
using ColorSchemes

# ╔═╡ 18d3f6a5-0004-4248-ad2b-3f28a1525252
@load EvoTreeRegressor pkg=EvoTrees verbosity=0

# ╔═╡ 1a90bee1-44c0-48e3-96e0-f8e919819064
begin
sources_file = joinpath(pwd(), "sources.yml")
sources = YAML.load_file(sources_file)
data_destination = sources["output-destination"]
data_path = joinpath(data_destination, "data", "nyc")
	
# modelname = "793244" # s1
# modelname = "689269" # s2
modelname = "746694" # s3
input_dir = joinpath(data_path, "p3_o", modelname)
machines_dir = joinpath(input_dir, "machines")
dataout_dir = joinpath(input_dir, "data_out")

output_dir = joinpath(data_path, "p4_o", modelname)
mkpath(output_dir)
end;

# ╔═╡ 44815554-9b54-4245-9241-4942521c67b7
md"""
Electricity Models and Datasets
"""

# ╔═╡ 4fb62bd6-4eb4-45d6-9732-232deecfeefe
begin
# null
m₀ = machine(joinpath(machines_dir, "m0.jlso"))
teₐ′₀ = CSV.read(joinpath(dataout_dir, "tea0.csv"), DataFrame)

# noaa
m₂ = machine(joinpath(machines_dir, "m2.jlso"))
teₐ′₂ = CSV.read(joinpath(dataout_dir, "tea2.csv"), DataFrame)

# noaa
m₃ = machine(joinpath(machines_dir, "m3.jlso"))
teₐ′₃ = CSV.read(joinpath(dataout_dir, "tea3.csv"), DataFrame)

# landsat
m₅ = machine(joinpath(machines_dir, "m5.jlso"))
teₐ′₅ = CSV.read(joinpath(dataout_dir, "tea5.csv"), DataFrame)

# # viirs
m₆ = machine(joinpath(machines_dir, "m6.jlso"))
teₐ′₆ = CSV.read(joinpath(dataout_dir, "tea6.csv"), DataFrame)

# # sar
m₇ = machine(joinpath(machines_dir, "m7.jlso"))
teₐ′₇ = CSV.read(joinpath(dataout_dir, "tea7.csv"), DataFrame)

# dynamic world
m₈ = machine(joinpath(machines_dir, "m8.jlso"))
teₐ′₈ = CSV.read(joinpath(dataout_dir, "tea8.csv"), DataFrame)

m₉ = machine(joinpath(machines_dir, "m9.jlso"))
teₐ′₉ = CSV.read(joinpath(dataout_dir, "tea9.csv"), DataFrame)

# full data
mₑ = machine(joinpath(machines_dir, "me.jlso"))
teₐ′ₑ = CSV.read(joinpath(dataout_dir, "teae.csv"), DataFrame)
end;

# ╔═╡ c69c3068-82fd-48f0-85c0-ce7fb049b904
begin
# epw used for the creation of the huge table in the appendix
# teₐ′₄ = CSV.read(
# 	joinpath(data_path, "p3_o", "415428", "data_out", "tea4.csv"),
# 	DataFrame
# )
# epw used for real analysis
# # epw
m₄ = machine(joinpath(machines_dir, "m4.jlso"))
teₐ′₄ = CSV.read(joinpath(dataout_dir, "tea4.csv"), DataFrame)

end;

# ╔═╡ c191ea0d-e3e7-48b1-99d4-d64889dd7889
md"""
Natural Gas Models and Datasets
"""

# ╔═╡ 22f1a2d1-f2a9-4c57-ba13-e888dd9deb3e
begin
mᵧ₀ = machine(joinpath(machines_dir, "mg0.jlso"))
teᵧ′₀ = CSV.read(joinpath(dataout_dir, "teg0.csv"), DataFrame)

mᵧ₂ = machine(joinpath(machines_dir, "mg2.jlso"))
teᵧ′₂ = CSV.read(joinpath(dataout_dir, "teg2.csv"), DataFrame)

mᵧ₃ = machine(joinpath(machines_dir, "mg3.jlso"))
teᵧ′₃ = CSV.read(joinpath(dataout_dir, "teg3.csv"), DataFrame)

mᵧ₄ = machine(joinpath(machines_dir, "mg4.jlso"))
teᵧ′₄ = CSV.read(joinpath(dataout_dir, "teg4.csv"), DataFrame)

mᵧ₅ = machine(joinpath(machines_dir, "mg5.jlso"))
teᵧ′₅ = CSV.read(joinpath(dataout_dir, "teg5.csv"), DataFrame)

mᵧ₆ = machine(joinpath(machines_dir, "mg6.jlso"))
teᵧ′₆ = CSV.read(joinpath(dataout_dir, "teg6.csv"), DataFrame)

mᵧ₇ = machine(joinpath(machines_dir, "mg7.jlso"))
teᵧ′₇ = CSV.read(joinpath(dataout_dir, "teg7.csv"), DataFrame)

mᵧ₈ = machine(joinpath(machines_dir, "mg8.jlso"))
teᵧ′₈ = CSV.read(joinpath(dataout_dir, "teg8.csv"), DataFrame)

mᵧ₉ = machine(joinpath(machines_dir, "mg9.jlso"))
teᵧ′₉ = CSV.read(joinpath(dataout_dir, "teg9.csv"), DataFrame)

# mᵧₑ = machine(joinpath(machines_dir, "mge.jlso"))
# teᵧ′ₑ = CSV.read(joinpath(dataout_dir, "tege.csv"), DataFrame)
end;

# ╔═╡ c8c3d6ad-e5a2-46ef-a05f-de82c71da2e5
teₐ′₄

# ╔═╡ 7f17363d-88d3-42ec-a88e-220e55e10ba1
test_terms = [teₐ′₃,teₐ′₈,teₐ′₀,teₐ′₂,teₐ′₄,teₐ′₅,teₐ′₉,teₐ′₇,teₐ′₆];

# ╔═╡ fa189aa8-3352-496d-a1be-ee50ad89bbe9


# ╔═╡ 3f871499-7309-456b-b54a-3060dfc783e4
# ╠═╡ disabled = true
#=╠═╡
test_termsᵧ = [teᵧ′₀,teᵧ′₂,teᵧ′₃,teᵧ′₄,teᵧ′₅,teᵧ′₆,teᵧ′₇,teᵧ′₈,teᵧ′₉];
  ╠═╡ =#

# ╔═╡ 3b800f80-c6b4-4e69-a0d5-2da30bac34ea
test_termsᵧ = [teᵧ′₀,teᵧ′₃,teᵧ′₅,teᵧ′₂,teᵧ′₄,teᵧ′₆,teᵧ′₉,teᵧ′₇,teᵧ′₈];

# ╔═╡ afbf700e-3078-4b5d-a712-b93248c6f79d
test_termsᵧ[7][1,"model"]

# ╔═╡ 8ab9d869-2f89-49cc-aa24-f623a3349eaf
teₐ′₃[1,"model"]

# ╔═╡ c1aa12bd-36cf-4cae-a7be-28bd0f29c463
teₐ′₈[1,"model"]

# ╔═╡ 75592dda-1ce4-4390-91c6-4f22de9fccec
md"""
##### Function definitions
"""

# ╔═╡ d0e572ce-932a-471d-ae69-47011189a6cd
begin
function electricaggregation(individual_results)
	ṫ = combine(groupby(individual_results, ["date","zipcode"]), [:prediction, :recorded] .=> sum, :model => first, renamecols=false)
	
	res = combine(groupby(ṫ, "zipcode")) do vᵢ
		(
		cvrmse = cvrmse(vᵢ.prediction, vᵢ.recorded),
		nmbe = nmbe(vᵢ.prediction, vᵢ.recorded),
		cvstd = cvstd(vᵢ.prediction, vᵢ.recorded),
		rmse = rmse(vᵢ.prediction, vᵢ.recorded),
		model = first(vᵢ.model)
		)
		end
	
	return ṫ, res
end
end

# ╔═╡ 54bccd79-dc16-4e98-8906-8381b4d35fd9
predict_function(model, data) = DataFrame(y_pred = MLJ.predict(model, data))

# ╔═╡ a883dd77-f8fd-4085-843f-31b85e4d9aea
sample_size = 100

# ╔═╡ 3a3e729a-4a69-40aa-b36e-1a4281331230
seed = 1

# ╔═╡ d3fb5ffe-f313-4160-a1f7-1702e94dfd5a
function splitdf(df, pct)
   @assert 0 <= pct <= 1
   ids = collect(axes(df, 1))
   shuffle!(MersenneTwister(seed), ids)
   sel = ids .<= nrow(df) .* pct
   return df[sel, :]#, view(df, .!sel, :)
end

# ╔═╡ de373688-1e78-49cb-b3f0-3b9329c3f780
begin
joining_terms = ["Property Id", "date"]
extra_omission_features = [
	"weather_station_distance",
	"zone",
	"zipcode",
	"month",
	"model",
	"recorded",
	"prediction",
	"electricity_mwh",
]
extra_omission_featuresᵧ = [
	"weather_station_distance",
	"zone",
	"zipcode",
	"month",
	"model",
	"recorded",
	"prediction",
	"naturalgas_mwh",
]

exclusion_terms = Not([joining_terms..., extra_omission_features...])
exclusion_termsᵧ = Not([joining_terms..., extra_omission_featuresᵧ...])
end;

# ╔═╡ 23196b9c-e2e4-4da7-b36f-a14491f49ae3
begin
# Compute stochastic Shapley values.
function compute_shap(mach, modeldata, gas=false)
	X = select(modeldata, exclusion_terms)
	explain = splitdf(X, 0.08)
	@info "Explaining Sample: " nrow(explain)

	data_shap = ShapML.shap(
		explain = explain,
	    reference = X,
		model = mach,
	    predict_function = predict_function,
		sample_size = sample_size,
		parallel = :features,
	    seed = 1,
		reconcile_instance=true
	)
end
end

# ╔═╡ 34f78bef-f706-4e57-bbb7-ff4d41a3b448
exclusion_terms

# ╔═╡ 58c5863f-6739-4a41-812c-747b86dfa66b
begin
# Compute stochastic Shapley values.
function compute_shapᵧ(mach, modeldata, gas=false)
	X = select(modeldata, exclusion_termsᵧ)
	explain = splitdf(X, 0.08)
	@info "Explaining Sample: " nrow(explain)

	data_shap = ShapML.shap(
		explain = explain,
	    reference = X,
		model = mach,
	    predict_function = predict_function,
		sample_size = sample_size,
		parallel = :features,
	    seed = 1,
		reconcile_instance=true
	)
end
end

# ╔═╡ cf430c83-380c-415e-8746-c573f85f7c18
function shap_totals(shapdf)
data_plot = rename(combine(groupby(shapdf, :feature_name), :shap_effect => x -> mean(abs.(x)), renamecols=false), :shap_effect => :mean_effect)
data_plot = sort(data_plot, order(:mean_effect, rev = true))

baseline = round(shapdf.intercept[1], digits = 1)
data_plot, baseline
end

# ╔═╡ ceb601ff-8f4c-41a6-9d7f-5cf27dc3d9e6
function plot_totals(shapdf, top_n=0)
@assert top_n <= nrow(shapdf)
data_plot, baseline = shap_totals(shapdf)
	
if top_n > 0
	top_n = top_n > nrow(data_plot) ? nrow(data_plot) : top_n
	data_plot = data_plot[1:top_n, :]
end

Gadfly.plot(
	data_plot, 
	y = :feature_name, 
	x = :mean_effect, 
	Coord.cartesian(yflip = true),
	Scale.y_discrete,
	Geom.bar(position = :dodge, orientation = :horizontal),
	Theme(default_color="black"),
	Guide.xlabel("|Shapley effect| (baseline = $baseline)"), 
	Guide.ylabel(nothing),
	Guide.title("Feature Importance - Mean Absolute Shapley Value")
)
end

# ╔═╡ 48d4d0a3-013a-4eca-b765-4899679c8087
function plot_term(data_shap, feature_name)
data_plot = data_shap[data_shap.feature_name .== feature_name, :]
baseline = round(data_shap.intercept[1], digits = 1)

Gadfly.plot(
	data_plot, 
	x = :feature_value, 
	y = :shap_effect,
	yintercept=[0],
	Geom.point,
	Geom.smooth(
		method = :loess, 
		smoothing = 0.3,
	),
	Geom.hline(color=["pink"], style=:solid),
	Guide.xrug,
	Theme(
		default_color = "black", 
		point_size = 0.3pt, 
		highlight_width = 0pt,
		alphas = [0.2],
		# rug_size=10pt,
	),
	Guide.yticks(ticks=collect(-0.5:0.1:0.5)),
	Guide.xlabel("Feature value"), 
	Guide.ylabel("Shapley effect (baseline = $baseline)"),
	Guide.title("Feature Effect - $feature_name"),
)
end

# ╔═╡ cf34a2e9-7444-4333-b4f9-afe304da737b
function plot_term(data_shap, feature_name::Vector{String})
data_plot = filter( x -> x.feature_name ∈ feature_name, data_shap )
baseline = round(data_shap.intercept[1], digits = 1)

Gadfly.plot(
	data_plot, 
	x = :feature_value, 
	y = :shap_effect,
	xgroup = :feature_name,
	# color= :feature_name,
	yintercept=[0],
	Geom.subplot_grid(
		# Geom.point,
		Geom.smooth(
			method = :loess, 
			smoothing = 0.2,
		),
		free_x_axis=true,
		Geom.hline(color=["pink"], style=:dash),
	),
	Theme(
		default_color = "black",
		line_width = 0.4mm,
		point_size=0.3pt,
		alphas=[0.2],
		highlight_width=0pt
	),
	# p_points, 
	Guide.xlabel("Feature value"), 
	Guide.ylabel("Shapley effect (baseline = $baseline)"),
	Guide.title("Feature Effect - $feature_name"),
)
end

# ╔═╡ 71745462-412b-4cc9-9f04-33afcdc0a0fb
function plot_siblings(data_shap, feature_name::Vector{String})
data_plot = filter( x -> x.feature_name ∈ feature_name, data_shap )
baseline = round(data_shap.intercept[1], digits = 1)

Gadfly.plot(
	data_plot, 
	x = :feature_value, 
	y = :shap_effect,
	color = :feature_name,
	# color= :feature_name,
	Geom.smooth(
		method = :loess, 
		smoothing = 0.1,
	),
	yintercept=[0],
	Geom.hline(color=["pink"], style=:dash),
	Theme(
		default_color = "black",
		line_width = 0.5mm
	),
	# p_points, 
	Guide.xlabel("Feature value"), 
	Guide.ylabel("Shapley effect (baseline = $baseline)"),
	Guide.title("Feature Effect - $feature_name"),
	# Scale.discrete_color_manual(palette(:grays, 3)...),
)
end

# ╔═╡ fd82f806-7787-4e3f-85b9-8fa7608ec87b
shap₆ = compute_shap(m₆, teₐ′₆);

# ╔═╡ b3c5372c-6fe4-448c-a4ae-de3aa283d954
shapvalues₆, baseline₆ = shap_totals(shap₆);

# ╔═╡ e6f29bfa-7f8b-4b7c-aabe-2cf8609baa28
Compose.circle

# ╔═╡ 2c2377c2-d369-4414-bdb3-85c1d29028eb
select(teᵧ′₂, r"^TMP.")

# ╔═╡ 42303c4d-91c5-4923-b350-c984452d8f86
shapᵧ₂ = compute_shapᵧ(mᵧ₂, teᵧ′₂);

# ╔═╡ 5a629447-c0c8-48a4-9569-3e1c4219d0ad
output_dir

# ╔═╡ 275b2b22-c60e-4bc4-a964-1a16575ea027
unique(shap₆.feature_name)

# ╔═╡ e86608e7-764d-43cc-9fbd-251d74ad596e
plot_term(shap₆, filter(x -> occursin(r"^avg\_rad*", x), unique(shap₆.feature_name)))

# ╔═╡ 4558fbb8-e4f2-430a-abbc-77fd1f8258ff
shap_temp =  shap₆

# ╔═╡ 2912ccbe-0831-424f-b8d4-eb254f248c56
shap_avg = sort(combine(groupby(shap_temp, :feature_name), :feature_value => mean, :shap_effect => mean, :shap_effect_sd => mean, renamecols=false), :shap_effect)

# ╔═╡ 42f850e9-2e91-496d-bf48-df10e700069d
# tplot = Gadfly.plot(
# 	shap_avg,
# 	x=:shap_effect,
# 	xmin=shap_avg.shap_effect .- shap_avg.shap_effect_sd,
# 	xmax=shap_avg.shap_effect .+ shap_avg.shap_effect_sd,	
# 	# color=:feature_value,
# 	xintercept=[0],
# 	# ygroup=:feature_name,
# 	y=:feature_name,
# 	Geom.point,
# 	Geom.errorbar,
# 	# Geom.beeswarm(padding=5mm),
# 	Geom.vline(color="pink"),
# 	Theme(point_size=2pt, default_color="black")
# )

# ╔═╡ db6c108f-c988-464a-adeb-974488c0e4a8
# draw(PNG(joinpath(output_dir, "sample_tornado.png"), 
# 	14cm, 
# 	8cm, 
# 	dpi=600
# ), tplot)

# ╔═╡ bceabf5a-b3bf-4533-b3fd-329161c628cf
function color_plot(shapdf, matching; title="", ylabel="", colorkeytitle="")
baseline = round(shapdf.intercept[1]; digits = 2, base = 10)
matched_shap = filter(x -> occursin(matching, x.feature_name), shapdf)
return Gadfly.plot(
	splitdf(matched_shap, 0.4),
	y=:feature_value,
	x=:shap_effect,
	color=:feature_value,
	xintercept=[0],
	ygroup=:feature_name,
	Guide.ylabel(ylabel),
	Guide.xlabel("SHAP Value Effect - Baseline: $(baseline)"),
	Guide.title(title),
	Geom.subplot_grid(
		Geom.point,
		Geom.vline(color="pink"),
		free_y_axis=true,
	),
	Theme(
		point_size=1.5pt, 
		default_color="black",
		highlight_width = 0pt,
		alphas=[0.7],
		key_position=:right,
	),
	Scale.ContinuousColorScale(
		palette -> get(ColorSchemes.curl, palette),
	),
	Guide.colorkey(title=colorkeytitle)
)
end

# ╔═╡ cecee0f8-2706-4086-9f68-9ff3ba4fe2f1
shap_temp

# ╔═╡ c3b2ad83-1b11-4a90-b555-13da332abc5d
tplot2 = color_plot(
	shap_temp, 
	r"^avg_rad";
	title="Nightlight radiance and Electricity Consumption",
	ylabel="Nightlight radiance (lux)",
	colorkeytitle="nW/cm²/sr"
)

# ╔═╡ 99d05200-c86e-4d00-9471-ac5b0ff842f5
draw(PNG(joinpath(output_dir, "wind_influence.png"), 
	14cm, 
	10cm, 
	dpi=600
), tplot2)

# ╔═╡ 04511cd6-f973-4906-b75c-22bf40f6e85c
shap₅ = compute_shap(m₅, teₐ′₅);

# ╔═╡ 1cfc45d4-e21c-44a4-8653-ed2720e2b192
plot_totals(shap₅, 15)

# ╔═╡ 435d8567-58ee-45f7-ab87-88376f2fa18e
begin
top_n = 10
data_plot, b1 = shap_totals(shap₅)

name_mapping = Dict(
	"area" => "Floor Area",
	"heightroof" => "Building Height",
	"cnstrct_yr" => "Construction Year",
	"groundelev" => "Ground Elevation",
	"SR_B7_f₁" => "Band 7 Reflectance - Q₅",
	"ST_DRAD_f₁" => "Downwelled Radiance - Q₅",
	"ST_URAD_f₅" => "Upwelled Radiance - Q₉₅",
	"ST_B10_f₅" => "Band 10 Surface Temperature - Q₉₅",
	"SR_B7_f₃" => "Band 7 Reflectance - Q₅₀",
	"NDVI_f₄" => "NDVI - Q₇₅"
)

data_plot.names = map( x -> x ∈ keys(name_mapping) ? name_mapping[x] : x, data_plot.feature_name )
	
if top_n > 0
	top_n = top_n > nrow(data_plot) ? nrow(data_plot) : top_n
	data_plot = data_plot[1:top_n, :]
end

landsat_values = Gadfly.plot(
	data_plot, 
	y = :names, 
	x = :mean_effect, 
	Coord.cartesian(yflip = true),
	Scale.y_discrete,
	Geom.bar(position = :dodge, orientation = :horizontal),
	Theme(
		default_color="gray", 
		line_width=5pt,
		plot_padding=[0mm,5mm,2mm,0mm],
		bar_spacing=0.5mm
		),
	Guide.annotation(
		compose(
			context(), 
			Compose.rectangle(0,4.5,0.8,1.0), 
			fill(nothing),
			Compose.linewidth(0.5mm),
			Compose.stroke("lightcoral"))
	),
	Guide.xlabel("Mean Absolute Shapley Effect"), 
	Guide.ylabel(nothing),
	Guide.title("Electricity Feature Importance - Landsat8")
)
end

# ╔═╡ d020780a-dd9c-48ef-a9b8-9512085763c6
draw(PNG(joinpath(output_dir, "landsat_values.png"), 
	14cm, 
	8cm, 
	dpi=600
), landsat_values)

# ╔═╡ 3c53faf0-5ea0-468b-8da4-95f9314d6cac
begin
data_plotᵧ, noaaᵧ = shap_totals(shapᵧ₂)
noaa_namemap = Dict(
	"area" => "Floor Area",
	"heightroof" => "Building Height",
	"cnstrct_yr" => "Construction Year",
	"groundelev" => "Ground Elevation",
	"HGT" => "Terrain Elevation",
	:PRES => "Pressure",
	"TMP_f₄" => "Temperature - Q₇₅",
	"TMP_f₁" => "Temperature - Q₅",
	"TMP_f₂" => "Temperature - Q₂₅",
	"TMP" => "Temperature",
	:DPT => "Dewpoint Temperature",
	:UGRD => "U-component Wind Speed",
	:VGRD => "V-component Wind Speed",
	"SPFH" => "Specific Humidity",
	"SPFH_f₅" => "Specific Humidity - Q₉₅",
	"SPFH_f₄" => "Specific Humidity - Q₇₅",
	"DPT" => "Dewpoint Temperature",
	:WDIR => "Wind Direction",
	:WIND => "Wind Speed",
	"GUST_f₁" => "Wind Speed (Gust) - Q₅",
	"GUST" => "Wind Speed (Gust)",
	:VIS => "Visability",
	"TCDC" => "Total Cloud Cover"
)
data_plotᵧ.names = map( x -> x ∈ keys(noaa_namemap) ? noaa_namemap[x] : x, data_plotᵧ.feature_name )

if top_n > 0
	# top_n = top_n > nrow(data_plotᵧ) ? nrow(data_plotᵧ) : top_n
	data_plotᵧ = data_plotᵧ[1:top_n, :]
end
	
noaa_values = Gadfly.plot(
	data_plotᵧ, 
	y = :names, 
	x = :mean_effect, 
	Coord.cartesian(yflip = true),
	Scale.y_discrete,
	Geom.bar(position = :dodge, orientation = :horizontal),
	Theme(
		default_color="gray", 
		line_width=5pt,
		plot_padding=[0mm,5mm,2mm,0mm],
		bar_spacing = 0.5mm
		),
	Guide.annotation(
		compose(
			context(), 
			Compose.rectangle(0,0.5,0.8,1.0), 
			fill(nothing),
			Compose.linewidth(0.5mm),
			Compose.stroke("lightcoral"))
	),
	Guide.xlabel("Mean Absolute Shapley Effect"), 
	Guide.ylabel(nothing),
	Guide.title("Natural Gas Feature Importance - NOAA")
)
end

# ╔═╡ 975eb5c4-2a11-436f-9997-dd432f768b29
data_plotᵧ

# ╔═╡ 215e0c82-138f-4323-b943-ca6cb4277967
draw(PNG(joinpath(output_dir, "noaa_values.png"), 
	11cm, 
	9cm, 
	dpi=600
), noaa_values)

# ╔═╡ 3e28dd4c-b4b8-4a42-8c00-dd8f808f77d5
color_plot(
	shap₅, 
	r"SR_B7_f₁";
	title="Vegetation and Electricity Consumption",
	ylabel="",
	colorkeytitle=""
)

# ╔═╡ f1c8e4c4-fd59-4c4d-9f7e-fef71e880fb1


# ╔═╡ a81c3d3a-1791-4ac1-9992-a053f9858884
begin
matched_shap = filter(x -> occursin("SR_B7_f₁", x.feature_name), shap₅)
baseline = round(shap₅.intercept[1]; digits = 2, base = 10)
shapmini = splitdf(matched_shap, 0.5)
shapmini.normfeature = shapmini.feature_value .* 2.75e-5 .- 0.2
filter!(x -> x.normfeature < 0.4, shapmini)

b7_imp = Gadfly.plot(
	shapmini,
	y=:normfeature,
	x=:shap_effect,
	color=:normfeature,
	xintercept=[0],
	Guide.yrug,
	# ygroup=:feature_name,
	Guide.ylabel("Band 7 Reflectance - Q₅"),
	Guide.xlabel("SHAP Value Effect - Baseline: $(baseline)MWh per day"),
	Guide.title("Shap Value Distibution of Infrared Measurements"),
	# Geom.subplot_grid(
	Geom.point,
	Geom.vline(color="pink", size=2pt),
		# free_y_axis=true,
	# ),
	Theme(
		point_size=1.5pt,
		line_width=0.03pt,
		default_color="gray",
		highlight_width = 0pt,
		alphas=[0.7],
		key_position=:right,
		minor_label_font_size=10pt,
		major_label_font_size=12pt,
	),
	Scale.ContinuousColorScale(
		palette -> get(reverse(ColorSchemes.curl), palette),
	),
	Guide.colorkey(title="Value")
)
end

# ╔═╡ c09712e9-7ffa-43f7-a34b-9f701e9428b8
begin
matched_noaa = filter(x -> occursin("TMP", x.feature_name), shapᵧ₂)
baselinenoaa = round(shapᵧ₂.intercept[1]; digits = 2, base = 10)
shapmininoaa = splitdf(matched_noaa, 0.75)

noaa_imp = Gadfly.plot(
	shapmininoaa,
	y=:feature_value,
	x=:shap_effect,
	color=:feature_value,
	xintercept=[0.0],
	yintercept=[26.5, 43],
	Guide.yrug,
	# ygroup=:feature_name,
	Guide.ylabel("Temperature °C"),
	Guide.xlabel("SHAP Value Effect - Baseline: $(baseline)MWh per day"),
	Guide.title("Shap Value Distibution of NOAA Temperature"),
	# Geom.subplot_grid(
	Guide.yticks(ticks=-10:10:40),
	# Guide.xticks(ticks=-0.3:0.1:0.3),
	Geom.point,
	Geom.vline(color="pink", size=2pt),
	# Geom.hline(color="lightgray", size=1pt),
	# Guide.annotation(compose(context(), Compose.text(-0.29, 44, "43°C"))),
	# Guide.annotation(compose(context(), Compose.text(-0.29, 27.5, "26.5°C"))),
		# free_y_axis=true,
	# ),
	Theme(
		point_size=1.5pt,
		line_width=0.03pt,
		default_color="gray",
		highlight_width = 0pt,
		alphas=[0.7],
		key_position=:right,
		minor_label_font_size=10pt,
		major_label_font_size=12pt,
	),
	Scale.ContinuousColorScale(
		palette -> get(reverse(ColorSchemes.curl), palette),
		minvalue=-5,
		maxvalue=30
	),
	Guide.colorkey(title="°C")
)
end

# ╔═╡ c2e8c9e8-4b38-441a-b887-3c9b5c555cf8
draw(PNG(joinpath(output_dir, "noaa_imp.png"), 
	11cm, 
	9cm, 
	dpi=800
), noaa_imp)

# ╔═╡ 8a705907-c504-45bd-ace9-d974a87585ec
draw(PNG(joinpath(output_dir, "landsat_b7.png"), 
	14cm, 
	8cm, 
	dpi=1000
), b7_imp)

# ╔═╡ 30f87011-b51b-4139-abe4-74406c918d99
color_plot(
	shap₅, 
	r"^ST_B10.";
	title="Landsat8 Surface Temperature - Shap Influence",
	ylabel="",
	colorkeytitle="°C"
)

# ╔═╡ 1be65fe0-1eec-47a9-a507-d41c8e4e7154
Compose.font

# ╔═╡ 126b3cfa-a492-48d5-83a3-3ef75ba12584
shap₅

# ╔═╡ c6b24844-9b36-445f-9938-651796034db6
begin
matched_shap10 = filter(x -> occursin("ST_B10_f₅", x.feature_name), shap₅)
baseline10 = round(shap₅.intercept[1]; digits = 2, base = 10)
shapmini10 = splitdf(matched_shap10, 0.5)

b10_imp = Gadfly.plot(
	shapmini10,
	y=:feature_value,
	x=:shap_effect,
	color=:feature_value,
	xintercept=[0],
	yintercept=[26.5, 43],
	Guide.yrug,
	# ygroup=:feature_name,
	Guide.ylabel("Surface Temperature °C - Q₉₅"),
	Guide.xlabel("SHAP Value Effect - Baseline: $(baseline)MWh per day"),
	Guide.title("Shap Value Distibution of Infrared Measurements"),
	# Geom.subplot_grid(
	Guide.yticks(ticks=-20:20:60),
	Guide.xticks(ticks=-0.3:0.2:0.3),
	Geom.point,
	Geom.vline(color="pink", size=2pt),
	Geom.hline(color="lightcoral", size=1pt),
	Guide.annotation(
		compose(
			context(), 
			Compose.text(-0.29, 44, "43°C"),
			fontsize(9pt),
			Compose.font("Arial"),
			fill(colorant"#564a55")
		)
	),
	Guide.annotation(
		compose(
			context(), 
			Compose.text(-0.29, 27.5, "26.5°C"),
			fontsize(9pt),
			Compose.font("Arial"),
			fill(colorant"#564a55")
		)),
		# free_y_axis=true,
	# ),
	Theme(
		point_size=1.5pt,
		line_width=0.03pt,
		default_color="gray",
		highlight_width = 0pt,
		alphas=[0.7],
		key_position=:right,
		minor_label_font_size=10pt,
		major_label_font_size=12pt,
	),
	Scale.ContinuousColorScale(
		palette -> get(ColorSchemes.curl, palette),
		minvalue=-60,
		maxvalue=60
	),
	Guide.colorkey(title="Value")
)
end

# ╔═╡ 4019a528-1fd3-425f-8385-8680180575d2
draw(PNG(joinpath(output_dir, "landsat_st10.png"), 
	14cm, 
	8cm, 
	dpi=600
), b10_imp)

# ╔═╡ bea1f1a0-0bb9-4754-93fc-fc6d8b200010
shapmini10

# ╔═╡ 36bb5b8f-5e83-4787-9f15-f48e6a6c75dd
# plot_term(shap₅, filter(x -> occursin(r"^ST\_B10", x), unique(shap₅.feature_name)))

# ╔═╡ 98b4271d-0dd9-444e-84b2-eedeb4cf2601
plot_term(shap₅, filter(x -> occursin(r"^ST_URAD", x), unique(shap₅.feature_name)))

# ╔═╡ dd4a536f-bad7-4553-9cfb-e0fbff1bc99b
teₐ′₇

# ╔═╡ 756e5f6f-945a-4caa-b816-c0cd4a2a21e2
# plot_term(shap₅, "NDVI_f₃")

# ╔═╡ 3a72130f-f2b8-419a-83b2-6070c0360414
shap₇ = compute_shap(m₇, teₐ′₇);

# ╔═╡ 694858dc-3ead-4634-9222-bb699af8decf
unique(shap₇.feature_name)

# ╔═╡ 185f99eb-f1b0-4993-bdbc-c65c1e8758de
shapvalues₇, baseline₇ = shap_totals(shap₇);

# ╔═╡ fe5aaa85-04f5-4d0f-aba4-60982ed79584
color_plot(
	shap₇, 
	r"^VH_f₅";
	title="SAR VH Reception - Shap Influence",
	ylabel="Vertical Transmit / Horizontal Receive",
	colorkeytitle=""
)

# ╔═╡ 622bf147-1ae2-456c-8baa-f832b5aa8778


# ╔═╡ e3b55f46-ff2d-4d2e-b95b-8b16c2b99877
md"""
Now for the appendix stuff
"""

# ╔═╡ 244e071e-6baa-4414-b843-81da37624e1d
common_terms = [
	"Property Id",
	"date",
	"month",
	"prediction",
	"recorded",
	"model",
	"zone",
	"zipcode",
	"weather_station_distance",
	"area",
	"heightroof",
	"electricity_mwh",
	"cnstrct_yr",
	"groundelev"
]

# ╔═╡ ed7c04bf-bbf7-42bf-94ff-fc07f6f9f870
common_termsᵧ = [
	"Property Id",
	"date",
	"month",
	"prediction",
	"recorded",
	"model",
	"zone",
	"zipcode",
	"weather_station_distance",
	"area",
	"heightroof",
	"naturalgas_mwh",
	"cnstrct_yr",
	"groundelev"
]

# ╔═╡ 908a5e29-e50e-4ff0-a857-eefe3ec71569
manual_namemapping = Dict(
	:date => "Date",
	:heightroof => "Building Height (m)",
	:cnstrct_yr => "Construction Year",
	:groundelev => "Ground Elevation (m)",
	:area => "Building Floor Area (m^{2})",
	:zone => "Zone",
	:month => :Month,
	:weather_station_distance => "Distance to Weather Station (km)",
	:zone => "Building Classification",
	:zipcode => "Zipcode",
	:electricity_mwh => "Daily Electricity (MWh)",
	:naturalgas_mwh => "Daily Natural Gas (MWh)",
	:pr => "Daily Precipitation",
	:pr_mean => "Daily Precipitation - Mean",
	:pr_quartile25 => "Daily Precipitation - 25th Quartile",
	:pr_median => "Daily Precipitation - Median",
	:pr_quartile75 => "Daily Precipitation - 75th Quartile",
	:tasmin => "Daily Minimum Temperature (C)",
	:tasmin_mean => "Daily Minimum Temperature - Mean",
	:tasmin_quartile25 => "Daily Minimum Temperature - 25th Quartile",
	:tasmin_median => "Daily Minimum Temperature - Median",
	:tasmin_quartile75 => "Daily Minimum Temperature - 75th Quartile",
	:tasmax => "Daily Maximum Temperature (^{\\circ}C)",
	:tasmax_mean => "Daily Maximum Temperature - Mean",
	:tasmax_quartile25 => "Daily Maximum Temperature - 25th Quartile",
	:tasmax_median => "Daily Maximum Temperature - Median",
	:tasmax_quartile75 => "Daily Maximum Temperature - 75th Quartile",
	:water => "Coverage by Water",
	:trees=> "Coverage by Trees",
	:grass=> "Coverage by Grass",
	:flooded_vegetation=> "Coverage by Flooded Vegetation",
	:crops=> "Coverage by Crops",
	:shrub_and_scrub=> "Coverage by Shrub",
	:built=> "Coverage by Built Environment",
	:bare=> "Coverage by Bare Land",
	:snow_and_ice=> "Coverage by Snow or Ice",
	:label=> "Likely Coverage Class",
	:HGT => "Terrain Elevation (m)",
	:PRES => "Pressure (Pa)",
	:TMP => "Temperature (C)",
	:DPT => "Dewpoint Temperature (C)",
	:UGRD => "U-component Wind Speed (m/s)",
	:VGRD => "V-component Wind Speed (m/s)",
	:SPFH => "Specific Humidity (kg/kg)",
	:WDIR => "Wind Direction (Deg.)",
	:WIND => "Wind Speed (m/s)",
	:GUST => "Wind Speed (Gust) (m/s)",
	:VIS => "Visability (m)",
	:TCDC => "Total Cloud Cover (\\%)",
	:ST_B10 => "Surface Temperature (^{\\circ}C)",
	:ST_QA => "Surface Temperature Uncertainty (C)",
	:ST_ATRAN => "Atmospheric Transmittance",
	:ST_CDIST => "Pixel Distance to Cloud (km)",
	:ST_DRAD => "Downwelled Radiance (\$W \\cdot m^{-2} \\cdot sr^{-1} \\cdot \\mu m^{-1} \$)",
	:ST_EMIS => "Emissivity of Band 10",
	:ST_EMSD => "Emissivity Standard Deviation",
	:ST_TRAD => "Thermal Band Converted to Radiance (\$W \\cdot m^{-2} \\cdot sr^{-1} \\cdot \\mu m^{-1} \$)",
	:ST_URAD => "Upwelled Radiance (\$W \\cdot m^{-2} \\cdot sr^{-1} \\cdot \\mu m^{-1} \$)",
	:SR_B1 => "Ultra Blue, Coastal Aerosol",
	:SR_B2 => "Blue",
	:SR_B3 => "Green",
	:SR_B4 => "Red",
	:SR_B5 => "Near Infrared",
	:SR_B6 => "Shortwave Infrared 1",
	:SR_B7 => "Shortwave Infrared 2",
	:B1 => "Aerosols",
	:B2 => "Blue",
	:B3 => "Green",
	:B4 => "Red",
	:B5 => "Red Edge 1",
	:B6 => "Red Edge 2",
	:B7 => "Red Edge 3",
	:B8 => "Near Infrared",
	:B8A => "Red Edge 4",
	:B9 => "Water Vapor",
	:B11 => "Shortwave Infrared 1",
	:B12 => "Shortwave Infrared 2",
	:NDVI_S => "NDVI",
	:VV => "Vertical Transmit / Vertical Receive (dB)",
	:VH => "Vertical Transmis / Horizontal Receive (dB)",
	:angle => "Incidence Angle from Ellipsoid (Deg.)",
	:pr => "Precipitation (kg \\cdot m^{-2}\\cdot s^{-1})",
	:avg_rad => "Average DNB Radiance (\$nW \\cdot cm^{-2} \\cdot sr^{-1}\$)",
	:cf_cvg => "Number of Cloud-free Observations Used",
	Symbol("Relative Humidity (%)_maximum") => "Relative Humidity_maximum",
	Symbol("Relative Humidity (%)_minimum") => "Relative Humidity_minimum",
	Symbol("Relative Humidity (%)_median") => "Relative Humidity_median"
)

# ╔═╡ 46ff8e97-88aa-4b5b-8e76-f7648f72ff52
function human_readable(x)
	kdis = 273.15
	x.weather_station_distance = x.weather_station_distance ./ 1000
	x.tasmin = x.tasmin .- kdis
	x.tasmax = x.tasmax .- kdis
	x.B1 = x.B1 .* 0.0001
	x.B2 = x.B2 .* 0.0001
	x.B3 = x.B3 .* 0.0001
	x.B4 = x.B4 .* 0.0001
	x.B5 = x.B5 .* 0.0001
	x.B6 = x.B6 .* 0.0001
	x.B7 = x.B7 .* 0.0001
	x.B8 = x.B8 .* 0.0001
	x.B8A = x.B8A .* 0.0001
	x.B9 = x.B9 .* 0.0001
	x.B11 = x.B11 .* 0.0001
	x.B12 = x.B12 .* 0.0001

	x.SR_B1 = x.SR_B1 .* 2.75e-5 .- 0.2
	x.SR_B2 = x.SR_B2 .* 2.75e-5 .- 0.2
	x.SR_B3 = x.SR_B3 .* 2.75e-5 .- 0.2
	x.SR_B4 = x.SR_B4 .* 2.75e-5 .- 0.2
	x.SR_B5 = x.SR_B5 .* 2.75e-5 .- 0.2
	x.SR_B6 = x.SR_B6 .* 2.75e-5 .- 0.2
	x.SR_B7 = x.SR_B7 .* 2.75e-5 .- 0.2
	x.ST_ATRAN = x.ST_ATRAN .* 0.0001
	x.ST_CDIST = x.ST_CDIST .* 0.01
	x.ST_DRAD = x.ST_DRAD .* 0.001
	x.ST_EMIS = x.ST_EMIS .* 0.0001
	x.ST_EMSD = x.ST_EMSD .* 0.0001
	x.ST_QA = x.ST_QA .* 0.01
	x.ST_TRAD = x.ST_TRAD .* 0.001
	x.ST_URAD = x.ST_URAD .* 0.001

	return x
end

# ╔═╡ 30da3dd7-3f77-4c26-b253-2cf575b5e61b
begin
E = CSV.read(joinpath(input_dir, "training_electric.csv"), DataFrame);
# select!(E, Not(:zone))

E = human_readable(E)
end;

# ╔═╡ 93b54d1c-f192-43b4-a5d8-f363348cb2f4
length(unique(E[:,"Property Id"]))

# ╔═╡ f5a5846f-96fd-4955-bb33-1820d70e89b2
nrow(E)

# ╔═╡ 80c6fd79-9307-4392-a3d2-3b8bed7e78c1
begin
Γ = human_readable(CSV.read(joinpath(input_dir, "training_gas.csv"), DataFrame))
end

# ╔═╡ d39af043-acc0-42d1-9926-9cd265fc0810
length(unique(Γ[:,"Property Id"]))

# ╔═╡ 7ed1e85e-54a5-4ad1-aa9e-ec600c009b35
nrow(Γ)

# ╔═╡ 5313b9b6-e319-438d-a909-32e245a65e36
begin
ntermₑ::Vector{String} = []
for term in test_terms
	modelname = term.model[1]
	modelvariables = filter( x -> x ∉ [common_terms..., ], names(term) )
	namelist = repeat([modelname], length(modelvariables))
	for (index, name) in enumerate(namelist)
		push!(ntermₑ, name)
	end
end
	
null_names = repeat(["Null"], (length(names(E))-length(ntermₑ)) )

Ė = describe(E, :all)
Ė.model = [null_names..., ntermₑ...]

newmodel_list = []

for (index, value) in enumerate(Ė.variable)
	if value ∈ [
				Symbol("month"),
				Symbol("weather_station_distance"),
				Symbol("zone"),
				Symbol("zipcode"),
				Symbol("electricity_mwh"),
				Symbol("Property Id"),
				Symbol("date")
			]
		push!(newmodel_list, "-")
	else
		push!(newmodel_list, Ė.model[index])
	end
end
Ė.model = newmodel_list

keyname_list = []
evarkeys = keys(manual_namemapping)
for variablename in Ė.variable
	if variablename ∈ evarkeys
		push!(keyname_list, String(manual_namemapping[variablename]))
	else
		push!(keyname_list, String(variablename))
	end
end
Ė.variable = keyname_list
	
Etex = select(Ė, "model", Not([:nmissing, :nunique, :first, :last, :eltype]))
sort!(Etex, [:model, :variable])

rename!(Etex, 
	"model" => "\\textbf{Model}",
	"variable" => "\\textbf{Variable}",
	"mean" => "\\textbf{Mean}",
	"std" => "\\textbf{Std. Dev}",
	"min" => "\\textbf{Min.}",
	"max" => "\\textbf{Max.}",
	"q25" => "\\textbf{Q25}",
	"median" => "\\textbf{Median}",
	"q75" => "\\textbf{Q75}",
)
end

# ╔═╡ 35c3218e-6682-47b0-8b84-1fe997fe8eae
test_termsᵧ[2][1,"model"]

# ╔═╡ 61e736a4-d376-4199-8b69-1c87579767aa
begin
ntermᵧ::Vector{String} = []
for term in test_termsᵧ
	modelname = term.model[1]
	modelvariables = filter( x -> x ∉ [common_termsᵧ..., ], names(term) )
	namelistᵧ = repeat([modelname], length(modelvariables))
	for (index, name) in enumerate(namelistᵧ)
		push!(ntermᵧ, name)
	end
end
	
null_namesᵧ = repeat(["Null"], (length(names(Γ))-length(ntermᵧ)) )

Γ̇ = describe(Γ, :all)
Γ̇.model = [null_namesᵧ..., ntermᵧ...]

newmodel_listᵧ = []

for (index, value) in enumerate(Γ̇.variable)
	if value ∈ [
				Symbol("month"),
				Symbol("weather_station_distance"),
				Symbol("zone"),
				Symbol("zipcode"),
				Symbol("naturalgas_mwh"),
				Symbol("Property Id"),
				Symbol("date")
			]
		push!(newmodel_listᵧ, "-")
	else
		push!(newmodel_listᵧ, Γ̇.model[index])
	end
end
Γ̇.model = newmodel_listᵧ

keyname_listᵧ = []
evarkeysᵧ = keys(manual_namemapping)
for variablename in Γ̇.variable
	if variablename ∈ evarkeysᵧ
		push!(keyname_listᵧ, String(manual_namemapping[variablename]))
	else
		push!(keyname_listᵧ, String(variablename))
	end
end
Γ̇.variable = keyname_listᵧ
	
Γtex = select(Γ̇, "model", Not([:nmissing, :nunique, :first, :last, :eltype]))
sort!(Γtex, [:model, :variable])

rename!(Γtex, 
	"model" => "\\textbf{Model}",
	"variable" => "\\textbf{Variable}",
	"mean" => "\\textbf{Mean}",
	"std" => "\\textbf{Std. Dev}",
	"min" => "\\textbf{Min.}",
	"max" => "\\textbf{Max.}",
	"q25" => "\\textbf{Q25}",
	"median" => "\\textbf{Median}",
	"q75" => "\\textbf{Q75}",
)
end

# ╔═╡ 29c13c9b-9eb0-4334-b194-2574f4c9bede
md"""
###### this is the electricity stuff
"""

# ╔═╡ d6ac7a52-d08e-4a91-9461-a15b2298d8d8
latexify(filter( x -> x["\\textbf{Model}"] ∈ ["Null","-","CMIP","Dynamic World","SAR","VIIRS"], Etex ), latex=false, fmt = FancyNumberFormatter(), env=:table)

# ╔═╡ e85612c8-c004-476e-a758-15bea7f06fa8
latexify(filter( x -> x["\\textbf{Model}"] ∈ ["NOAA","Sentinel-2","Landsat8"], Etex ), latex=false, fmt = FancyNumberFormatter(), env=:table)

# ╔═╡ 9c0fe06a-ddc9-4170-b033-33102310cff9
FancyNumberFormatter()

# ╔═╡ 147b9e45-0f5a-4447-b090-bb9df1d3941a
begin
epwₑ_clone = copy(filter( x -> x["\\textbf{Model}"] ∈ ["EPW"], Etex ))
epwₑ_clone[:,"\\textbf{Variable}"] = replace.(epwₑ_clone[:,"\\textbf{Variable}"], "_maximum" => " Daily Maxium")
epwₑ_clone[:,"\\textbf{Variable}"] = replace.(epwₑ_clone[:,"\\textbf{Variable}"], "_minimum" => " Daily Minimum")
epwₑ_clone[:,"\\textbf{Variable}"] = replace.(epwₑ_clone[:,"\\textbf{Variable}"], "_median" => " Daily Median")
end

# ╔═╡ 82b6bd5d-d21e-4af2-94cc-32843ca5ac7f
latexify(epwₑ_clone[1:39,:], latex=false, fmt = StyledNumberFormatter(), env=:table)

# ╔═╡ 3aaea654-c9aa-44ae-9a9e-303b62efbee2
latexify(epwₑ_clone[40:end,:], latex=false, fmt = StyledNumberFormatter(), env=:table)

# ╔═╡ d5ca7364-6735-41e2-91a1-fb09081249ce
filter( x -> x ∉ [common_terms..., ], names(teₐ′₇) )

# ╔═╡ 3b9e7d1b-8835-45eb-83a5-ee290ef9110b
md"""
###### this is the gas stuff
"""

# ╔═╡ 7c1a77ff-4604-406d-86e3-9c2f54f9040d
latexify(filter( x -> x["\\textbf{Model}"] ∈ ["Null","-","CMIP","Dynamic World","SAR","VIIRS"], Γtex ), latex=false, fmt = FancyNumberFormatter(), env=:table)

# ╔═╡ 366edacc-592d-444b-bb06-799fc1811ab3
latexify(filter( x -> x["\\textbf{Model}"] ∈ ["NOAA","Sentinel-2","Landsat8"], Γtex ), latex=false, fmt = FancyNumberFormatter(), env=:table)

# ╔═╡ 659e6b45-2d80-46b8-be0a-cc8d96ec636a
begin
epwᵧ_clone = copy(filter( x -> x["\\textbf{Model}"] ∈ ["EPW"], Γtex ))
epwᵧ_clone[:,"\\textbf{Variable}"] = replace.(epwᵧ_clone[:,"\\textbf{Variable}"], "_maximum" => " Daily Maxium")
epwᵧ_clone[:,"\\textbf{Variable}"] = replace.(epwᵧ_clone[:,"\\textbf{Variable}"], "_minimum" => " Daily Minimum")
epwᵧ_clone[:,"\\textbf{Variable}"] = replace.(epwᵧ_clone[:,"\\textbf{Variable}"], "_median" => " Daily Median")
end

# ╔═╡ 1232d3c4-c98a-49f5-ac1e-fe8812ee5a52
latexify(epwᵧ_clone[1:39,:], latex=false, fmt = StyledNumberFormatter(), env=:table)

# ╔═╡ c1e223b5-87c9-4c30-ab74-dcaa15d0d15b
latexify(epwₑ_clone[40:end,:], latex=false, fmt = StyledNumberFormatter(), env=:table)

# ╔═╡ Cell order:
# ╠═da3eed56-a0ea-11ed-381f-8fb87407934b
# ╠═0f04176b-e090-46de-821f-e0e1dcb86aca
# ╠═57fa1025-1d94-4463-9fc7-7c4ebbd2f4b7
# ╠═db32e6a6-8ace-4d7f-b8a4-0d875291bcfa
# ╠═b06e9088-c511-48a7-96c8-4e10012f44e8
# ╠═18d3f6a5-0004-4248-ad2b-3f28a1525252
# ╠═1a90bee1-44c0-48e3-96e0-f8e919819064
# ╟─44815554-9b54-4245-9241-4942521c67b7
# ╠═4fb62bd6-4eb4-45d6-9732-232deecfeefe
# ╠═c69c3068-82fd-48f0-85c0-ce7fb049b904
# ╟─c191ea0d-e3e7-48b1-99d4-d64889dd7889
# ╠═22f1a2d1-f2a9-4c57-ba13-e888dd9deb3e
# ╠═c8c3d6ad-e5a2-46ef-a05f-de82c71da2e5
# ╠═7f17363d-88d3-42ec-a88e-220e55e10ba1
# ╠═fa189aa8-3352-496d-a1be-ee50ad89bbe9
# ╠═afbf700e-3078-4b5d-a712-b93248c6f79d
# ╠═3f871499-7309-456b-b54a-3060dfc783e4
# ╠═3b800f80-c6b4-4e69-a0d5-2da30bac34ea
# ╠═8ab9d869-2f89-49cc-aa24-f623a3349eaf
# ╠═c1aa12bd-36cf-4cae-a7be-28bd0f29c463
# ╟─75592dda-1ce4-4390-91c6-4f22de9fccec
# ╟─d0e572ce-932a-471d-ae69-47011189a6cd
# ╠═54bccd79-dc16-4e98-8906-8381b4d35fd9
# ╠═a883dd77-f8fd-4085-843f-31b85e4d9aea
# ╠═3a3e729a-4a69-40aa-b36e-1a4281331230
# ╠═d3fb5ffe-f313-4160-a1f7-1702e94dfd5a
# ╠═de373688-1e78-49cb-b3f0-3b9329c3f780
# ╠═23196b9c-e2e4-4da7-b36f-a14491f49ae3
# ╠═34f78bef-f706-4e57-bbb7-ff4d41a3b448
# ╠═58c5863f-6739-4a41-812c-747b86dfa66b
# ╠═cf430c83-380c-415e-8746-c573f85f7c18
# ╟─ceb601ff-8f4c-41a6-9d7f-5cf27dc3d9e6
# ╠═48d4d0a3-013a-4eca-b765-4899679c8087
# ╟─cf34a2e9-7444-4333-b4f9-afe304da737b
# ╟─71745462-412b-4cc9-9f04-33afcdc0a0fb
# ╠═fd82f806-7787-4e3f-85b9-8fa7608ec87b
# ╠═b3c5372c-6fe4-448c-a4ae-de3aa283d954
# ╠═1cfc45d4-e21c-44a4-8653-ed2720e2b192
# ╠═ee536c4f-0273-4814-bf3a-6758cd13f968
# ╠═e6f29bfa-7f8b-4b7c-aabe-2cf8609baa28
# ╠═435d8567-58ee-45f7-ab87-88376f2fa18e
# ╠═d020780a-dd9c-48ef-a9b8-9512085763c6
# ╠═2c2377c2-d369-4414-bdb3-85c1d29028eb
# ╠═42303c4d-91c5-4923-b350-c984452d8f86
# ╠═975eb5c4-2a11-436f-9997-dd432f768b29
# ╠═3c53faf0-5ea0-468b-8da4-95f9314d6cac
# ╠═5a629447-c0c8-48a4-9569-3e1c4219d0ad
# ╠═215e0c82-138f-4323-b943-ca6cb4277967
# ╠═c09712e9-7ffa-43f7-a34b-9f701e9428b8
# ╠═c2e8c9e8-4b38-441a-b887-3c9b5c555cf8
# ╠═275b2b22-c60e-4bc4-a964-1a16575ea027
# ╠═e86608e7-764d-43cc-9fbd-251d74ad596e
# ╠═4558fbb8-e4f2-430a-abbc-77fd1f8258ff
# ╠═2912ccbe-0831-424f-b8d4-eb254f248c56
# ╠═42f850e9-2e91-496d-bf48-df10e700069d
# ╠═db6c108f-c988-464a-adeb-974488c0e4a8
# ╠═9c906615-d33a-471d-a2af-393b9a963ba3
# ╠═bceabf5a-b3bf-4533-b3fd-329161c628cf
# ╠═cecee0f8-2706-4086-9f68-9ff3ba4fe2f1
# ╠═c3b2ad83-1b11-4a90-b555-13da332abc5d
# ╠═99d05200-c86e-4d00-9471-ac5b0ff842f5
# ╠═04511cd6-f973-4906-b75c-22bf40f6e85c
# ╠═3e28dd4c-b4b8-4a42-8c00-dd8f808f77d5
# ╠═f1c8e4c4-fd59-4c4d-9f7e-fef71e880fb1
# ╠═a81c3d3a-1791-4ac1-9992-a053f9858884
# ╠═8a705907-c504-45bd-ace9-d974a87585ec
# ╠═30f87011-b51b-4139-abe4-74406c918d99
# ╠═1be65fe0-1eec-47a9-a507-d41c8e4e7154
# ╠═126b3cfa-a492-48d5-83a3-3ef75ba12584
# ╠═c6b24844-9b36-445f-9938-651796034db6
# ╠═4019a528-1fd3-425f-8385-8680180575d2
# ╠═bea1f1a0-0bb9-4754-93fc-fc6d8b200010
# ╠═36bb5b8f-5e83-4787-9f15-f48e6a6c75dd
# ╠═98b4271d-0dd9-444e-84b2-eedeb4cf2601
# ╠═dd4a536f-bad7-4553-9cfb-e0fbff1bc99b
# ╠═756e5f6f-945a-4caa-b816-c0cd4a2a21e2
# ╠═3a72130f-f2b8-419a-83b2-6070c0360414
# ╠═694858dc-3ead-4634-9222-bb699af8decf
# ╠═185f99eb-f1b0-4993-bdbc-c65c1e8758de
# ╠═fe5aaa85-04f5-4d0f-aba4-60982ed79584
# ╠═622bf147-1ae2-456c-8baa-f832b5aa8778
# ╟─e3b55f46-ff2d-4d2e-b95b-8b16c2b99877
# ╠═30da3dd7-3f77-4c26-b253-2cf575b5e61b
# ╠═80c6fd79-9307-4392-a3d2-3b8bed7e78c1
# ╠═244e071e-6baa-4414-b843-81da37624e1d
# ╠═ed7c04bf-bbf7-42bf-94ff-fc07f6f9f870
# ╠═908a5e29-e50e-4ff0-a857-eefe3ec71569
# ╠═93b54d1c-f192-43b4-a5d8-f363348cb2f4
# ╠═f5a5846f-96fd-4955-bb33-1820d70e89b2
# ╠═d39af043-acc0-42d1-9926-9cd265fc0810
# ╠═7ed1e85e-54a5-4ad1-aa9e-ec600c009b35
# ╠═46ff8e97-88aa-4b5b-8e76-f7648f72ff52
# ╠═5313b9b6-e319-438d-a909-32e245a65e36
# ╠═35c3218e-6682-47b0-8b84-1fe997fe8eae
# ╠═61e736a4-d376-4199-8b69-1c87579767aa
# ╠═29c13c9b-9eb0-4334-b194-2574f4c9bede
# ╠═d6ac7a52-d08e-4a91-9461-a15b2298d8d8
# ╠═e85612c8-c004-476e-a758-15bea7f06fa8
# ╠═9c0fe06a-ddc9-4170-b033-33102310cff9
# ╠═147b9e45-0f5a-4447-b090-bb9df1d3941a
# ╠═82b6bd5d-d21e-4af2-94cc-32843ca5ac7f
# ╠═3aaea654-c9aa-44ae-9a9e-303b62efbee2
# ╠═d5ca7364-6735-41e2-91a1-fb09081249ce
# ╠═3b9e7d1b-8835-45eb-83a5-ee290ef9110b
# ╠═7c1a77ff-4604-406d-86e3-9c2f54f9040d
# ╠═366edacc-592d-444b-bb06-799fc1811ab3
# ╠═659e6b45-2d80-46b8-be0a-cc8d96ec636a
# ╠═1232d3c4-c98a-49f5-ac1e-fe8812ee5a52
# ╠═c1e223b5-87c9-4c30-ab74-dcaa15d0d15b
