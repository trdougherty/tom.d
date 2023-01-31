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

# ╔═╡ 18d3f6a5-0004-4248-ad2b-3f28a1525252
@load EvoTreeRegressor pkg=EvoTrees verbosity=0

# ╔═╡ 1a90bee1-44c0-48e3-96e0-f8e919819064
begin
sources_file = joinpath(pwd(), "sources.yml")
sources = YAML.load_file(sources_file)
data_destination = sources["output-destination"]
data_path = joinpath(data_destination, "data", "nyc")

modelname = "920959"
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

# viirs
m₆ = machine(joinpath(machines_dir, "m6.jlso"))
teₐ′₆ = CSV.read(joinpath(dataout_dir, "tea6.csv"), DataFrame)

# sar
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
function predict_function(model, data)
  data_pred = DataFrame(y_pred = MLJ.predict(model, data))
  return data_pred
end

# ╔═╡ a883dd77-f8fd-4085-843f-31b85e4d9aea
sample_size = 20

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
	"electricity_mwh"
]
exclusion_terms = Not([joining_terms..., extra_omission_features...])
end;

# ╔═╡ 23196b9c-e2e4-4da7-b36f-a14491f49ae3
begin
# Compute stochastic Shapley values.
function compute_shap(mach, modeldata)
	X = select(modeldata, exclusion_terms)
	explain = splitdf(X, 0.05)
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

plot(
	data_plot, 
	y = :feature_name, 
	x = :mean_effect, 
	Coord.cartesian(yflip = true),
	Scale.y_discrete,
	Geom.bar(position = :dodge, orientation = :horizontal),
	Theme(bar_spacing = 1mm),
	Guide.xlabel("|Shapley effect| (baseline = $baseline)"), 
	Guide.ylabel(nothing),
	Guide.title("Feature Importance - Mean Absolute Shapley Value")
)
end

# ╔═╡ 48d4d0a3-013a-4eca-b765-4899679c8087
function plot_term(data_shap, feature_name)
data_plot = data_shap[data_shap.feature_name .== feature_name, :]
baseline = round(data_shap.intercept[1], digits = 1)

p_points = layer(
	data_plot, 
	x = :feature_value, 
	y = :shap_effect, 
	Geom.point()
)
p_line = layer(
	data_plot, 
	x = :feature_value, 
	y = :shap_effect, 
	Geom.smooth(
		method = :loess, 
		smoothing = 0.5
	),
    style(
		line_width = 0.75mm,
	), 
	Theme(default_color = "black")
)
Gadfly.plot(
	p_line, 
	p_points, 
	Guide.xlabel("Feature value"), 
	Guide.ylabel("Shapley effect (baseline = $baseline)"),
	Guide.title("Feature Effect - $feature_name")
)
end

# ╔═╡ df0f573f-ed84-4f09-9ae7-c40d012209c6
teₐ′₆[1:end, :]

# ╔═╡ fd82f806-7787-4e3f-85b9-8fa7608ec87b
shap₆ = compute_shap(m₆, teₐ′₆);

# ╔═╡ b3c5372c-6fe4-448c-a4ae-de3aa283d954
shapvalues₆, baseline₆ = shap_totals(shap₆);

# ╔═╡ 1cfc45d4-e21c-44a4-8653-ed2720e2b192
plot_totals(shap₆, 15)

# ╔═╡ e86608e7-764d-43cc-9fbd-251d74ad596e
plot_term(shap₆, "avg_rad_f₁")

# ╔═╡ 04511cd6-f973-4906-b75c-22bf40f6e85c
shap₅ = compute_shap(m₅, teₐ′₅);

# ╔═╡ 36bb5b8f-5e83-4787-9f15-f48e6a6c75dd
shap₅_totals = shap_totals(shap₅, 15)

# ╔═╡ d1918518-09ce-4c85-9c4a-33f74d210afe
shap_totals(shap₅)

# ╔═╡ Cell order:
# ╠═da3eed56-a0ea-11ed-381f-8fb87407934b
# ╠═db32e6a6-8ace-4d7f-b8a4-0d875291bcfa
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
# ╠═cf430c83-380c-415e-8746-c573f85f7c18
# ╠═ceb601ff-8f4c-41a6-9d7f-5cf27dc3d9e6
# ╠═48d4d0a3-013a-4eca-b765-4899679c8087
# ╠═df0f573f-ed84-4f09-9ae7-c40d012209c6
# ╠═fd82f806-7787-4e3f-85b9-8fa7608ec87b
# ╠═b3c5372c-6fe4-448c-a4ae-de3aa283d954
# ╠═1cfc45d4-e21c-44a4-8653-ed2720e2b192
# ╠═e86608e7-764d-43cc-9fbd-251d74ad596e
# ╠═04511cd6-f973-4906-b75c-22bf40f6e85c
# ╠═36bb5b8f-5e83-4787-9f15-f48e6a6c75dd
# ╠═d1918518-09ce-4c85-9c4a-33f74d210afe
