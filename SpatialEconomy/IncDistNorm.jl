using Distributions: cdf, pdf, rand, AliasTable, _normcdf, _normpdf
using StatsBase: Weights, sample, alias_sample!
using Random: GLOBAL_RNG, AbstractRNG, randn, shuffle!, randexp!, randexp
import Base: rand, length, iterate, maximum, minimum, ndims, axes, getindex, broadcastable
import Distributions: cdf, pdf, mean, var, entropy, median

####################################################################################
################## Income Distribution based on Normal Kernel #######################
struct IncDistNorm
    bdwth::Float64 ##Bandwidth
    obs_wt::Dict{Float64, Tuple{Float64, Float64}} ##Dictionary stocking the weight of each obeserved value: observation => (center, weight)
end

###To be able to broadcast the struct   
length(id_n::IncDistNorm) = 1
ndims(::Type{IncDistNorm}) = 0
axes(id_n::IncDistNorm) = ()
broadcastable(id_n::IncDistNorm) = id_n
getindex(id_n::IncDistNorm, ::CartesianIndex{0}) = id_n

###"Default" Constructor
function IncDistNorm(bdwth, points::AbstractArray{<: Real, 1}, weights::AbstractArray{<: Real, 1})
    bdwth < 0.0 && error("Bandwidth must be nonnegative.")
    id_n = IncDistNorm(bdwth, Dict{Float64, Tuple{Float64, Float64}}())
    sizehint!(id_n.obs_wt, length(points))
    ##Add points
    add_points!(id_n, points, weights)
    return id_n
end

###"Empty" constructor
function IncDistNorm(bdwth::Real)
    bdwth < 0.0 && error("Bandwidth must be nonnegative.")
    id_n = IncDistNorm(bdwth, Dict{Float64, Tuple{Float64, Float64}}())
    ##Add "insignificant" observation
    add_point!(id_n, 0.0, 10.0^-50)
    return id_n
end

###Constructor with equal weights
function IncDistNorm(bdwth, points::AbstractArray{<: Real, 1}, weight::Real=1.0)
    bdwth < 0.0 && error("Bandwidth must be nonnegative.")
    weight <= 0 && error("Weight must be positive.")
    any(x -> x < 0.0, points) && error("Points must be nonnegative.")
    id_n = IncDistNorm(bdwth, Dict{Float64, Tuple{Float64, Float64}}())
    sizehint!(id_n.obs_wt, length(points))
    ##Add observations
    for i=1:length(points)
        if haskey(id_n.obs_wt, points[i])
            id_n.obs_wt[points[i]] = id_n.obs_wt[points[i]] .+ (0.0, weight)
        else
            id_n.obs_wt[points[i]] = (_find_center(id_n.bdwth, points[i]), weight)
        end
    end
    return id_n
end

##To find the center that preserves the mean given the bandwith
####This is shaky: need to prove that there is no swinging
function _find_center(bdw::Real, pt::Real, tol::Real=10^-6)
    bdw == 0.0 && return pt
    pt == 0.0 && return - 6.0 * bdw
    ##General case
    pt_sc::Float64 = pt / bdw
    res::Float64 = pt_sc
    prev::Float64 = 0.1 * res + (res == 0.0)
    while abs((res - prev) / prev) > tol && res > - 6.0 ##Second condition: pt is like 0 if res goes too low
        prev = res
        res = (pt_sc - _normpdf(res)) / _normcdf(res)
    end
    isnan(res) && error("Point has NaN value.")
    return res * bdw
end

###Function to add a point to the distrbution
function add_point!(id_n::IncDistNorm, point::Real, weight::Real=1.0)
    weight <= 0.0 && error("Weight must be positive. Given: $weight")
    point < 0.0 && error("Point must be non-negative. Given: $point")
    if haskey(id_n.obs_wt, point)
        @inbounds id_n.obs_wt[point] = (id_n.obs_wt[point][1], id_n.obs_wt[point][2] + weight)
    else
        id_n.obs_wt[point] = (_find_center(id_n.bdwth, point), weight)
    end
    return nothing
end

###Function to add a point to the distrbution when center is given
function add_point_w_center!(id_n::IncDistNorm, 
                           point::Real, center::Real, weight::Real=1.0)
    weight <= 0.0 && error("Weight must be positive. Given: $weight")
    point < 0.0 && error("Point must be non-negative. Given: $point")
    if haskey(id_n.obs_wt, point)
        @inbounds id_n.obs_wt[point] = (id_n.obs_wt[point][1], id_n.obs_wt[point][2] + weight)
    else
        id_n.obs_wt[point] = (center, weight)
    end
    return nothing
end

###Function to add multiple points at once to the distrbution
function add_points!(id_n::IncDistNorm, points::AbstractArray{<: Real, 1}, weights::AbstractArray{<: Real, 1})
    ##Conformity checks
    length(points) != length(weights) && error("Vectors of weights and points must be of equal length.")
    any(x -> x <= 0.0, weights) && error("All weights must be positive.")
    any(x -> x < 0.0, points) && error("All points must be non-negative.")
    ##Add points
    for i=1:length(points)
        if haskey(id_n.obs_wt, points[i])
            id_n.obs_wt[points[i]] = id_n.obs_wt[points[i]] .+ (0.0, weights[i])
        else
            id_n.obs_wt[points[i]] = (_find_center(id_n.bdwth, points[i]), weights[i])
        end
    end
    return nothing
end

###Function to add multiple points with a same weight at once to the distrbution
function add_points!(id_n::IncDistNorm, points::AbstractArray{<: Real, 1}, weight::Real=1.0)
    ##Conformity checks
    weight <= 0.0 && error("Weight must be positive. Given: $weight")
    any(x -> x < 0.0, points) && error("All points must be non-negative.")
    ##Add points
    for i=1:length(points)
        if haskey(id_n.obs_wt, points[i])
            id_n.obs_wt[points[i]] = id_n.obs_wt[points[i]] .+ (0.0, weight)
        else
            id_n.obs_wt[points[i]] = (_find_center(id_n.bdwth, points[i]), weight)
        end
    end
    return nothing
end

##Reweights an observation or add it with the given weight if absent
function reweight_point!(id_n::IncDistNorm, point::Real, weight::Real)
    weight < 0.0 && error("Weight must be positive. Given: $weight")
    if weight > 0.0
        id_n.obs_wt[point] = (haskey(id_n.obs_wt, point) ? id_n.obs_wt[point][1] : _find_center(id_n.bdwth, point),
                              weight)
    else
        delete!(id_n.obs_wt, point)
        ###To avoid to have it empty
        isempty(id_n.obs_wt) && add_point!(id_n, 0.0, 10.0^-50)
    end
    return nothing
end

##Deletes an observation if exists
function del_point(id_n::IncDistNorm, point::Real)
    delete!(id_n.obs_wt, point)
    ###To avoid to have it empty
    isempty(id_n.obs_wt) && add_point!(id_n, 0.0, 10.0^-50)
    return nothing
end

##Returns the weight of an observation
get_weight(id_n::IncDistNorm, point::Real) = get(id_n.obs_wt, point, (0.0, 0.0))[2]

##Returns obervations and weight dictionnary in an array
get_obs_data(id_n::IncDistNorm) = [(k, id_n.obs_wt[k]) for k in keys(id_n.obs_wt)]

###Returns a reinitialized IncDistNorm (almost) without allocations
function reinitialize(id_n::IncDistNorm, new_bdw::Float64=id_n.bdwth)
    new_bdw < 0.0 && error("Bandwidth must be nonnegative.")
    empty!(id_n.obs_wt)
    return IncDistNorm(new_bdw, id_n.obs_wt)
end

###Returns a scaled transform of the distrbution
###Replace is to be used when the old distribution is going to be replaced by scaled i.e. old = scaled(old, s)
function scaled(id_n::IncDistNorm, s::Real)
    s < 0.0 && error("Scale must be nonnegative.")
    return IncDistNorm(s * id_n.bdwth, Dict((s * obs => (_find_center(s * id_n.bdwth, s * obs), cw[2]) for (obs, cw) in id_n.obs_wt)))
end

##Returns the sum of weights
sum_weights(id_n::IncDistNorm) = sum(x -> x[2], values(id_n.obs_wt))

###Function CDF : sig_int is the half width of the interval in which we considerable
_cdf_zero_bw(id_n::IncDistNorm, x::Real) = sum(p -> p[2] * (p[1] <= x), values(id_n.obs_wt)) / sum_weights(id_n)

###Only significant values of CDFs make a contrbution to the result
##sig_int is the half width of the interval in which we consider that the distribution is < 1
function _cdf_pos_bw(id_n::IncDistNorm, x::Real, sig_int::Real=6.5)
    x < 0 && return 0.0 ##Incomes are always positive
    res::Float64 = 0.0
    eval::Float64 = 0.0
    for (c, w) in values(id_n.obs_wt)
        eval = (x - c) / id_n.bdwth
        if eval >= sig_int
            res += w
        elseif eval > - sig_int
            res += w * _normcdf(eval)
        end
    end
    return res / sum_weights(id_n)
end

###evaluating the cdf of each component
cdf(id_n::IncDistNorm, x::Real, sig_int::Real=6.5) = id_n.bdwth == 0.0 ? _cdf_zero_bw(id_n, x) : _cdf_pos_bw(id_n, x, sig_int)

####Returns a "sampler" to select one of the obeservations based on weights
function _center_weight(id_n::IncDistNorm)
    ##Create vectors of observations and probabilities
    center = Array{Float64, 1}(undef, length(id_n.obs_wt))
    weight = Array{Float64, 1}(undef, length(id_n.obs_wt))
    i = 1
    for (c, w) in values(id_n.obs_wt)
        center[i] = c
        weight[i] = w
        i += 1
    end
    return center, weight
end

##Generate random samples
function rand(id_n::IncDistNorm)
    ##Collect weights and centers
    cen, wei = _center_weight(id_n)
    draw = randn() * id_n.bdwth + sample(cen, Weights(wei))
    return ifelse(draw > 0, draw, 0.0)
end

###Several samples
function rand(id_n::IncDistNorm, n::Int64)
    n < 0 && error("Sample size must be positive.")
    ##Create sampler
    cen, wei = _center_weight(id_n)
    ##Random draws
    draw::Array{Float64, 1} = sample_uos!(cen, wei, Array{Float64, 1}(undef, n))
    for i=1:n
        draw[i] += id_n.bdwth * randn()
        draw[i] *= draw[i] > 0.0 
    end
    return draw
end

###Adds random samples to an array add_to
function add_rand!(id_n::IncDistNorm, add_to)
    ##Create sampler
    cen, wei = _center_weight(id_n)
    ##Use alias table
    wei ./= sum(wei)
    selector = AliasTable(wei)
    ##Add realizations
    for i in eachindex(add_to)
        add_to[i] += randn() * id_n.bdwth + cen[rand(selector)]
    end
    return nothing
end

###Calculate mean
mean(id_n::IncDistNorm) = sum(x -> x[1]*x[2][2], id_n.obs_wt) / sum_weights(id_n)

###Extrema
minimum(id_n::IncDistNorm) = id_n.bdwth == 0.0 ? minimum(keys(id_n.obs_wt)) : 0.0
maximum(id_n::IncDistNorm) = id_n.bdwth == 0.0 ? maximum(keys(id_n.obs_wt)) : Inf64

###### To sample from weighted points
function _search_scale_a!(rng::AbstractRNG, a::AbstractArray, wv::AbstractArray, 
                          x::AbstractArray)    
    sum_wv = sum(wv)
    i = 1
    cw = wv[1] / sum_wv

    for s_x=1:length(x)
        while (cw < @inbounds x[s_x]) & (i < length(a))
            i += 1
            @inbounds cw += wv[i] / sum_wv
        end
        ###randomly choose location in (inspired from randperm in Random module)
        j = rand(rng, 1:s_x)
        @inbounds x[s_x] = x[j]
        @inbounds x[j] = a[i]
    end
    return nothing
end

function _search_scale_x!(rng::AbstractRNG, a::AbstractArray, wv::AbstractArray, 
                          x::AbstractArray)
    sum_wv = sum(wv)
    x .*= sum_wv

    i = 1
    cw = wv[1]
    
    for s_x=1:length(x)
        while (cw < @inbounds x[s_x]) & (i < length(a))
            i += 1
            @inbounds cw += wv[i]
        end
        j = rand(rng, 1:s_x)
        @inbounds x[s_x] = x[j]
        @inbounds x[j] = a[i]
    end
    return nothing
end

function sample_uos!(rng::AbstractRNG, a::AbstractArray,
                            wv::AbstractArray, x::AbstractArray)
    length(x) == 0 && x
    length(wv) == length(a) || throw(DimensionMismatch("Inconsistent lengths."))
    ###Sample sorted uniform from exponential ratios
    randexp!(rng, x)
    cumsum!(x, x)
    @inbounds x ./=  x[end] + randexp()

    if length(a) <= length(x)
        _search_scale_a!(rng, a, wv, x)
    else
        _search_scale_x!(rng, a, wv, x)
    end
    return x
end

sample_uos!(a::AbstractArray, wv::AbstractArray, x::AbstractArray) = sample_uos!(GLOBAL_RNG, a, wv, x)

