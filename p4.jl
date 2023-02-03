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

# ╔═╡ db32e6a6-8ace-4d7f-b8a4-0d875291bcfa
using Random

# ╔═╡ b06e9088-c511-48a7-96c8-4e10012f44e8
using Plots

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

modelname = "992009"
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

# epw
m₄ = machine(joinpath(machines_dir, "m4.jlso"))
teₐ′₄ = CSV.read(joinpath(dataout_dir, "tea4.csv"), DataFrame)

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

# full data
mₑ = machine(joinpath(machines_dir, "me.jlso"))
teₐ′ₑ = CSV.read(joinpath(dataout_dir, "teae.csv"), DataFrame)
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

mᵧₑ = machine(joinpath(machines_dir, "mge.jlso"))
teᵧ′ₑ = CSV.read(joinpath(dataout_dir, "tege.csv"), DataFrame)
end;

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
sample_size = 50

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
exclusion_terms = Not([joining_terms..., extra_omission_features...])
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
		palette -> get(ColorSchemes.matter, palette),
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
plot_totals(shap₅, 10)

# ╔═╡ ffa0f05f-b51e-4563-8946-6858e5d94c45
plot_totals(shap₅, 10)

# ╔═╡ 3e28dd4c-b4b8-4a42-8c00-dd8f808f77d5
color_plot(
	shap₅, 
	r"^NDVI";
	title="Vegetation and Electricity Consumption",
	ylabel="",
	colorkeytitle=""
)

# ╔═╡ 30f87011-b51b-4139-abe4-74406c918d99
color_plot(
	shap₅, 
	r"^ST\_B10";
	title="Landsat8 Surface Temperature - Shap Influence",
	ylabel="",
	colorkeytitle="°C"
)

# ╔═╡ 36bb5b8f-5e83-4787-9f15-f48e6a6c75dd
# plot_term(shap₅, filter(x -> occursin(r"^ST\_B10", x), unique(shap₅.feature_name)))

# ╔═╡ 98b4271d-0dd9-444e-84b2-eedeb4cf2601
plot_term(shap₅, filter(x -> occursin(r"^ST_URAD", x), unique(shap₅.feature_name)))

# ╔═╡ dd4a536f-bad7-4553-9cfb-e0fbff1bc99b


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
	r"^VH";
	title="SAR VH Reception - Shap Influence",
	ylabel="",
	colorkeytitle=""
)

# ╔═╡ Cell order:
# ╠═da3eed56-a0ea-11ed-381f-8fb87407934b
# ╠═db32e6a6-8ace-4d7f-b8a4-0d875291bcfa
# ╠═b06e9088-c511-48a7-96c8-4e10012f44e8
# ╠═18d3f6a5-0004-4248-ad2b-3f28a1525252
# ╠═1a90bee1-44c0-48e3-96e0-f8e919819064
# ╟─44815554-9b54-4245-9241-4942521c67b7
# ╠═4fb62bd6-4eb4-45d6-9732-232deecfeefe
# ╟─c191ea0d-e3e7-48b1-99d4-d64889dd7889
# ╠═22f1a2d1-f2a9-4c57-ba13-e888dd9deb3e
# ╟─75592dda-1ce4-4390-91c6-4f22de9fccec
# ╟─d0e572ce-932a-471d-ae69-47011189a6cd
# ╠═54bccd79-dc16-4e98-8906-8381b4d35fd9
# ╠═a883dd77-f8fd-4085-843f-31b85e4d9aea
# ╠═3a3e729a-4a69-40aa-b36e-1a4281331230
# ╠═d3fb5ffe-f313-4160-a1f7-1702e94dfd5a
# ╠═de373688-1e78-49cb-b3f0-3b9329c3f780
# ╠═23196b9c-e2e4-4da7-b36f-a14491f49ae3
# ╟─cf430c83-380c-415e-8746-c573f85f7c18
# ╟─ceb601ff-8f4c-41a6-9d7f-5cf27dc3d9e6
# ╠═48d4d0a3-013a-4eca-b765-4899679c8087
# ╟─cf34a2e9-7444-4333-b4f9-afe304da737b
# ╟─71745462-412b-4cc9-9f04-33afcdc0a0fb
# ╠═fd82f806-7787-4e3f-85b9-8fa7608ec87b
# ╠═b3c5372c-6fe4-448c-a4ae-de3aa283d954
# ╠═1cfc45d4-e21c-44a4-8653-ed2720e2b192
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
# ╠═ffa0f05f-b51e-4563-8946-6858e5d94c45
# ╠═3e28dd4c-b4b8-4a42-8c00-dd8f808f77d5
# ╠═30f87011-b51b-4139-abe4-74406c918d99
# ╠═36bb5b8f-5e83-4787-9f15-f48e6a6c75dd
# ╠═98b4271d-0dd9-444e-84b2-eedeb4cf2601
# ╠═dd4a536f-bad7-4553-9cfb-e0fbff1bc99b
# ╠═756e5f6f-945a-4caa-b816-c0cd4a2a21e2
# ╠═3a72130f-f2b8-419a-83b2-6070c0360414
# ╠═694858dc-3ead-4634-9222-bb699af8decf
# ╠═185f99eb-f1b0-4993-bdbc-c65c1e8758de
# ╠═fe5aaa85-04f5-4d0f-aba4-60982ed79584
