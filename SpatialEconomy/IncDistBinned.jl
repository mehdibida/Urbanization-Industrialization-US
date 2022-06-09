using Distributions: cdf, pdf, Normal, AliasTable
using StatsBase: Weights, sample, alias_sample!
using Random: GLOBAL_RNG, AbstractRNG, randn, shuffle!, randexp!, randexp
import Base: rand, length, iterate, maximum, minimum, ndims, axes, getindex, broadcastable, ht_keyindex, ht_keyindex2!, _setindex!
import Random: rand!
import Distributions: cdf, pdf, mean, var, entropy, median

####################################################################################
################## Income Distribution based on Binned Kernel Density #######################
################## Distrbution is zero-inflated

mutable struct Data_wp
    tot_weighted_point::Float64
    tot_weight::Float64
end

function add!(dwp::Data_wp, point::Float64, weight::Float64)
    dwp.tot_weighted_point += weight * point
    dwp.tot_weight += weight
    return nothing
end

function add_weighted!(dwp::Data_wp, weighted_point::Float64, weight::Float64)
    dwp.tot_weighted_point += weighted_point
    dwp.tot_weight += weight
    return nothing
end

function remove!(dwp::Data_wp, point::Float64, weight::Float64)
    dwp.tot_weighted_point -= weight * point
    dwp.tot_weight -= weight
    (dwp.tot_weight < -10^-9) && error("Weight to remove larger than total weight.")
    dwp.tot_weight = max(dwp.tot_weight, 0.0)
    if (dwp.tot_weight < 10^-9) & (dwp.tot_weighted_point > 10^-9)
        error("Total weight is 0, but not sum of weighted points.")
    end
    return nothing
end


##################### Binned kernel density estimation #######################
####Based on biweight kernel
struct IncDistBinned
    b_wth::Float64 ##Bin width
    ker_bdw::Float64 ##Kernel bandwidth / 2.0 (easier for calculations)
    bin_wgt::Dict{Int64, Float64} ##Weight of each bin
    points_wp::Data_wp ##Stores data average and total weight
    ##Technical paramater : which index is first of which distribution is non truncated
    first_non_truncated::Int64
end

###To be able to broadcast the struct   
length(::IncDistBinned) = 1
ndims(::Type{IncDistBinned}) = 0
axes(::IncDistBinned) = ()
broadcastable(id_n::IncDistBinned) = id_n
getindex(id_n::IncDistBinned, ::CartesianIndex{0}) = id_n
getindex(id_n::IncDistBinned) = id_n

################ Constructors ######################################
############### Always fill the dictionary with the ones that have average > center

###"Empty" constructor
function IncDistBinned(b_wth::Real, n_bins::Integer=100)
    b_wth < 0.0 && error("Bandwidth must be nonnegative.")
    ker_bdw = _ker_bdw(b_wth)
    id_n = IncDistBinned(b_wth, ker_bdw, Dict{Float64, Tuple{Float64, Float64}}(), Data_wp(0.0, 0.0), ceil(ker_bdw / b_wth))
    sizehint!(id_n.bin_wgt, n_bins)
    ##Add "insignificant" observation
    add_point!(id_n, 0.0, 10.0^-30)
    return id_n
end

###"Default" Constructor
###Add possibility to give dict size
function IncDistBinned(b_wth, points, weights, n_bins::Integer=length(points))
    b_wth < 0.0 && error("Bandwidth must be nonnegative.")
    ker_bdw = _ker_bdw(b_wth)
    length(points) == length(weights) || error("Points and weights vectors must be of same length.")
    id_n = IncDistBinned(b_wth, ker_bdw, Dict{Float64, Tuple{Float64, Float64}}(), Data_wp(0.0, 0.0), ceil(ker_bdw / b_wth))
    sizehint!(id_n.bin_wgt, n_bins)
    ##Add points
    add_points!(id_n, points, weights)
    return id_n
end

###Constructor with equal weights
function IncDistBinned(b_wth, points, weight::Real=1.0, n_bins::Integer=length(points))
    b_wth < 0.0 && error("Bandwidth must be nonnegative.")
    weight <= 0 && error("Weight must be positive.")
    any(x -> x < 0.0, points) && error("Points must be nonnegative.")
    ker_bdw = _ker_bdw(b_wth)
    id_n = IncDistBinned(b_wth, ker_bdw, Dict{Float64, Tuple{Float64, Float64}}(), Data_wp(0.0, 0.0), ceil(ker_bdw / b_wth))
    sizehint!(id_n.bin_wgt, n_bins)
    ##Add observations
    add_points!(id_n, points, weight)
    return id_n
end

##Kernel bandwidth given bin width
_ker_bdw(bin_width) = sqrt(bin_width)
###Bin mean
_bin_mean(id_n::IncDistBinned, b_ind::Int64) = _mean_biweight_positive(b_ind * id_n.b_wth, id_n.ker_bdw)
###Bin center of corresponding index
_center(id_n::IncDistBinned, ind::Int64) = id_n.b_wth * ind

###Do it with 0 is the center of the first bin, but only take positive values of the first bin
##Adds points without checking for errors
##Does not update sum(weights*points) and sum of weights
function _add_point!(id_n::IncDistBinned, point::Real, weight::Real)
    ###Find 2 closest bins in terms of average
    ###Find lower bin
    ic_low = convert(Int64, floor(Int64, point / id_n.b_wth)) ##max in case lower is 0.0
    if ic_low < id_n.first_non_truncated
        ##Check if kernel mean is above inferred bin (happens when bins distributions are truncated) and find bin in case 
        ##Truncated bins are included in the distribution beforehand so they should be found
        while (_bin_mean(id_n, ic_low) > point) & (ic_low > 0)
            ic_low -= 1
        end
    end
    ###Higher bin
    ic_high = ic_low + 1

    ###Calculate means
    mean_low = _bin_mean(id_n, ic_low)
    mean_high = _bin_mean(id_n, ic_high)
    ##Calculate weights
    weight_low = weight * min((mean_high - point) / (mean_high - mean_low), 1.0)
    ###Add weights to bins
    ind_key_low = ht_keyindex(id_n.bin_wgt, ic_low)
    if ind_key_low > 0
        @inbounds id_n.bin_wgt.vals[ind_key_low] += weight_low
    else
        ind_key_low = ht_keyindex2!(id_n.bin_wgt, ic_low)
        @inbounds _setindex!(id_n.bin_wgt, weight_low, ic_low, -ind_key_low)
    end
    ###Do calculations for weight hight if is it to be added
    if weight_low < weight
        weight_high = max(weight - weight_low, 0.0)
        ind_key_high = ht_keyindex(id_n.bin_wgt, ic_high)
        if ind_key_high > 0
            @inbounds id_n.bin_wgt.vals[ind_key_high] += weight_high
        else
            ind_key_high = ht_keyindex2!(id_n.bin_wgt, ic_high)
            @inbounds _setindex!(id_n.bin_wgt, weight_high, ic_high, -ind_key_high)
        end
    end
    return nothing
end

function _add_point_2bins!(id_n_1::IncDistBinned, id_n_2::IncDistBinned, point::Real, wgt_1::Real, wgt_2::Real)
    id_n_1.b_wth == id_n_2.b_wth || error("Distributions have incompatible binning.")
    ###Find 2 closest bins in terms of average
    ###Find lower bin
    ic_low = convert(Int64, floor(Int64, point / id_n_1.b_wth)) ##max in case lower is 0.0
    if ic_low < id_n_1.first_non_truncated
        ##Check if kernel mean is above inferred bin (happens when bins distributions are truncated) and find bin in case 
        ##Trunceted bins are included in the distrbution beforehand so they should be found
        while (_bin_mean(id_n_1, ic_low) > point) & (ic_low > 0) 
            ic_low -= 1
        end
    end
    ###Higher bin
    ic_high = ic_low + 1

    ###Calculate means
    mean_low = _bin_mean(id_n_1, ic_low)
    mean_high = _bin_mean(id_n_1, ic_high)
    ##Calculate weights
    share_w_low = ifelse(_bin_mean(id_n_1, 0) < point, (mean_high - point) / (mean_high - mean_low), 1.0)
    share_w_high = max(1.0 - share_w_low, 0.0)
    ###Add weights to bins
    ####Distribution 1
    ind_key_low_1 = ht_keyindex(id_n_1.bin_wgt, ic_low)
    if ind_key_low_1 > 0 #####Lower bin distribution 1
        @inbounds id_n_1.bin_wgt.vals[ind_key_low_1] += wgt_1 * share_w_low
    else
        ind_key_low_1 = ht_keyindex2!(id_n_1.bin_wgt, ic_low)
        @inbounds _setindex!(id_n_1.bin_wgt, wgt_1 * share_w_low, ic_low, -ind_key_low_1)
    end
    ind_key_high_1 = ht_keyindex(id_n_1.bin_wgt, ic_high)
    if ind_key_high_1 > 0 #####Higher bin distribution 1
        @inbounds id_n_1.bin_wgt.vals[ind_key_high_1] += wgt_1 * share_w_high
    else
        ind_key_high_1 = ht_keyindex2!(id_n_1.bin_wgt, ic_high)
        @inbounds _setindex!(id_n_1.bin_wgt, wgt_1 * share_w_high, ic_high, -ind_key_high_1)
    end
    ####Distrbution 2
    ind_key_low_2 = ht_keyindex(id_n_2.bin_wgt, ic_low)
    if ind_key_low_2 > 0 #####Lower bin distribution 2
        @inbounds id_n_2.bin_wgt.vals[ind_key_low_2] += wgt_2 * share_w_low
    else
        ind_key_low_2 = ht_keyindex2!(id_n_2.bin_wgt, ic_low)
        @inbounds _setindex!(id_n_2.bin_wgt, wgt_2 * share_w_low, ic_low, -ind_key_low_2)
    end
    ind_key_high_2 = ht_keyindex(id_n_2.bin_wgt, ic_high)
    if ind_key_high_2 > 0 #####Higher bin distribution 2
        @inbounds id_n_2.bin_wgt.vals[ind_key_high_2] += wgt_2 * share_w_high
    else
        ind_key_high_2 = ht_keyindex2!(id_n_2.bin_wgt, ic_high)
        @inbounds _setindex!(id_n_2.bin_wgt, wgt_2 * share_w_high, ic_high, -ind_key_high_2)
    end
    return nothing
end


###Function to add a point to the distrbution, checks for errors first
function add_point!(id_n::IncDistBinned, point::Real, weight::Real=1.0)
    weight <= 0.0 && error("Weight must be positive. Given: $weight")
    point < 0.0 && error("Point must be non-negative. Given: $point")
    ##Add point
    _add_point!(id_n, point, weight)
    add!(id_n.points_wp, point, weight)
    return nothing
end

###Function to add multiple points at once to the distrbution
function add_points!(id_n::IncDistBinned, points, weights)
    ##Conformity checks
    any(x -> x <= 0.0, weights) && error("All weights must be positive.")
    any(x -> x < 0.0, points) && error("All points must be non-negative.")
    ##Add points
    next = iterate(zip(points, weights))
    while next !== nothing
    ((p, w), (s_p, s_w)) = next
        _add_point!(id_n, p, w)
        if xor(isnothing(iterate(points, s_p)), isnothing(iterate(weights, s_w)))
            error("Points and weights of different lengths.")
        end
        next = iterate(zip(points, weights), (s_p, s_w))
    end
    add_weighted!(id_n.points_wp, sum(p*w for (p, w) in zip(points, weights)), sum(weights))
    return nothing
end

###Function to add multiple points with a same weight at once to the distrbution
function add_points!(id_n::IncDistBinned, points, weight::Real=1.0)
    ##Conformity checks
    weight <= 0.0 && error("Weight must be positive. Given: $weight")
    any(x -> x < 0.0, points) && error("All points must be non-negative.")
    ##Add points
    for p in points
        _add_point!(id_n, p, weight)
    end
    add_weighted!(id_n.points_wp, weight*sum(points), weight*length(points))
    return nothing
end

##To be used carefully, does not check if mean 
function _index_add!(id_n::IncDistBinned, index::Int64, weight::Float64)
    ind_key = ht_keyindex(id_n.bin_wgt, index)
    if ind_key > 0
        @inbounds id_n.bin_wgt.vals[ind_key] += weight
    else
        ind_key = ht_keyindex2!(id_n.bin_wgt, index)
        @inbounds _setindex!(id_n.bin_wgt, weight, index, -ind_key)
    end
    return nothing
end

###Adds a proportion of "added" to "id_n"
function add_from_IDB!(id_n::IncDistBinned, added::IncDistBinned, share_added_per_bin::Float64=1.0)
    (id_n.b_wth != added.b_wth) && error("Bins are incompatible: not same bin width.")
    for (i, w) in added.bin_wgt
        ind_key = ht_keyindex(id_n.bin_wgt, i)
        if ind_key > 0
            @inbounds id_n.bin_wgt.vals[ind_key] += share_added_per_bin * w
        else
            ind_key = ht_keyindex2!(id_n.bin_wgt, i)
            @inbounds _setindex!(id_n.bin_wgt, share_added_per_bin * w, i, -ind_key)
        end
    end
    add_weighted!(id_n.points_wp, 
                  share_added_per_bin * added.points_wp.tot_weighted_point, 
                  share_added_per_bin * added.points_wp.tot_weight)
    return nothing
end


function reweight!(id_n::IncDistBinned, rw_factor::Float64)
    map!(x -> rw_factor * x, values(id_n.bin_wgt))
    id_n.points_wp.tot_weighted_point *= rw_factor
    id_n.points_wp.tot_weight *= rw_factor
    return nothing
end

function reinitialize!(id_n::IncDistBinned)
    empty!(id_n.bin_wgt)
    id_n.bin_wgt[0] = 10^-30
    id_n.points_wp.tot_weighted_point = 0.0
    id_n.points_wp.tot_weight = 10^-30
    return nothing
end

##Returns the sum of weights
sum_weights(id_n::IncDistBinned) = id_n.points_wp.tot_weight

###evaluating the cdf of each component
function cdf(id_n::IncDistBinned, x::Real)
    res = 0.0
    for (c, w) in id_n.bin_wgt
        if (x - _center(id_n, c)) >= id_n.ker_bdw
            res += w
        elseif abs(x - _center(id_n, c)) < id_n.ker_bdw
            cdf_0 = c < id_n.first_non_truncated ? _cdf_biweight(_center(id_n, c), id_n.ker_bdw, 0.0) : 0.0
            res += w * (_cdf_biweight(_center(id_n, c), id_n.ker_bdw, x) - cdf_0) / (1.0 - cdf_0)
        end
    end
    return min(res / sum_weights(id_n), 1.0)
end

function pdf(id_n::IncDistBinned, x::Real)
    res = 0.0
    for (c, w) in id_n.bin_wgt
        if abs(x - _center(id_n, c)) < id_n.ker_bdw
            res += w * _pdf_biweight(_center(id_n, c), id_n.ker_bdw, x) /
                   (1.0 - _cdf_biweight(_center(id_n, c), id_n.ker_bdw, 0.0))
        end
    end
    return res / sum_weights(id_n)
end

###returns a pair of vectors of centers and corresponding weights
function _center_weight(id_n::IncDistBinned)
    cen_ind = collect(keys(id_n.bin_wgt))
    return cen_ind, [id_n.bin_wgt[c] for c in cen_ind]
end

##Generate random samples
function rand(id_n::IncDistBinned) 
    center_ind, weight = _center_weight(id_n)
    all(isequal(0.0), weight) && error("All weights are zero.")
    chosen_center = _center(id_n, sample(center_ind, Weights(weight)))
    res = id_n.ker_bdw * _rand_biweight() + chosen_center
    while (res < 0.0) & (chosen_center > 0) ##Case of center = 0.0 is treat by taking abs of r after
        res = id_n.ker_bdw * _rand_biweight() + chosen_center
    end
    return abs(res)
end

###Several samples
function rand!(id_n::IncDistBinned, res::AbstractArray{Float64, 1})
    isempty(res) && return res
    ##Use alias table to generate samples
    cen, wei = _center_weight(id_n)
    all(isequal(0.0), wei) && error("All weights are zero.")
    wei ./= sum(wei)
    selector = AliasTable(wei)
    @inbounds for i=1:length(res)
        chosen_center = _center(id_n, cen[rand(selector)])
        r = id_n.ker_bdw * _rand_biweight() + chosen_center
        while (r < 0.0) & (chosen_center > 0) ##Case of center = 0.0 is treat by taking abs of r after
            r = id_n.ker_bdw * _rand_biweight() + chosen_center
        end
        res[i] = abs(r)
    end
    return res
end

rand(id_n::IncDistBinned, n::Integer) = rand!(id_n, Array{Float64, 1}(undef, n))

###Adds random samples to an array add_to
function add_rand!(id_n::IncDistBinned, add_to::AbstractArray{<: Real, 1})
    isempty(add_to) && return add_to
    ##Use alias table to generate samples
    cen, wei = _center_weight(id_n)
    all(isequal(0.0), wei) && error("All weights are zero.")
    wei ./= sum(wei)
    selector = AliasTable(wei)
    @inbounds for i=1:length(add_to)
        chosen_center = cen[rand(selector)]
        r = id_n.ker_bdw * _rand_biweight() + _center(id_n, chosen_center)
        while (r < 0.0) & (chosen_center > 0) ##Case of center = 0.0 is treat by taking abs of r after
            r = id_n.ker_bdw * _rand_biweight() + _center(id_n, chosen_center)
        end
        add_to[i] += r
    end
    return add_to
end

mean(id_n::IncDistBinned) = id_n.points_wp.tot_weighted_point / id_n.points_wp.tot_weight
mean_from_ker(id_n::IncDistBinned) = sum(_bin_mean(id_n, k) * w for (k, w) in id_n.bin_wgt) / sum(values(id_n.bin_wgt))
minimum(id_n::IncDistBinned) = max(_center(id_n, minimum(keys(id_n.bin_wgt))) - id_n.ker_bdw, 0.0)

##Reweights an observation or add it with the given weight if absent
###  Doesn't work, /!\ Needs to be adapted
function remove_point!(id_n::IncDistBinned, point::Real, weight::Real)
    weight >= 0.0 || error("Weight must be positive. Given: $weight")
    ##Do nothing if weight of point to be removed is zero
    if weight > 0.0
        ###Find lower bin
        ic_low = floor(point / id_n.b_wth) ##max in case lower is 0.0
        ##Check if kernel mean is above inferred bin (happens when bins distributions are truncated) and find bin in case 
        ##Trunceted bins are included in the distrbution beforehand so they should be found
        while (id_n.bin_wgt[ic_low][1] > point) & (ic_low > 0.0)
            ic_low -= 1.0
        end
        ###Higher bin
        ic_high = ic_low + 1.0
        ###Weights to remove
        weight_low = id_n.bin_wgt[0][1] < point ? weight * (id_n.bin_wgt[ic_high][1] - point) / (id_n.bin_wgt[ic_high][1] - id_n.bin_wgt[ic_low][1]) : weight
        weight_high = max(weight - weight_low, 0.0)
        ###Error if contrbutions are higher than weights
        (weight_high / id_n.bin_wgt[ic_high][2] < 1.000_000_1) || error("Higher bin has smaller weight than point contribution.")
        (weight_low / id_n.bin_wgt[ic_low][2] < 1.000_000_1) || error("Lower bin has smaller weight than point contribution.")
        ###Reweight lower bin weight
        if (weight_low >= id_n.bin_wgt[ic_low][2]) & (ic_low >= id_n.first_non_truncated)
            delete!(id_n.bin_wgt, ic_low)
        elseif weight_low > 0.0
            id_n.bin_wgt[ic_low] = (id_n.bin_wgt[ic_low][1], max(id_n.bin_wgt[ic_low][2] - weight_low, 0.0))
        end
        ###Reweight higher bin weight
        if (weight_high >= id_n.bin_wgt[ic_high][2]) & (ic_high >= id_n.first_non_truncated)
            delete!(id_n.bin_wgt, ic_high)
        elseif weight_high > 0.0
            id_n.bin_wgt[ic_high] = (id_n.bin_wgt[ic_high][1], max(id_n.bin_wgt[ic_high][2] - weight_high, 0.0))
        end
        remove!(id_n.points_wp, point, weight)
    end
    return nothing
end


###################################################################
############### A vector based binned distribution ################
###################################################################

#To be used to create bins to store static income distributions
#The main goal of BinnedIncs is to serve as an auxiliary, to quickly fill
#the dictionary of IncDistBinned, as iterating over a dictionary is slow
struct BinnedIncs
    b_wth::Float64
    inc_wgtd_pts::Float64 #∑_i w_i*inc_i
    inc_tot_wgt::Float64 #∑_i w_i
    bin_ind_wgt::Array{Tuple{Int64, Float64}, 1}
end


###Assumes incomes are sorted
function BinnedIncs(b_wdth::Float64, incomes::AbstractArray{<: Real, 1}, each_inc_wgt::Float64)

    low_bin = 0 #Index of lower bin
    high_bin = 1 #Index of higher bin

    ###Bin means given truncated or not distributions
    low_bin_mean = _mean_biweight_positive(low_bin * b_wdth, _ker_bdw(b_wdth))
    high_bin_mean = _mean_biweight_positive(high_bin * b_wdth, _ker_bdw(b_wdth))

    ##bin total weight
    tot_wgt_low = 0.0
    tot_wgt_high = 0.0

    ###Vector containing indices and weights of bins
    b_w::Array{Tuple{Int64, Float64}, 1} = Tuple{Int64, Float64}[]

    curr_inc_ind = 1

    ###Go through all couples of consecutive bins and add the weights of the points that are
    ###in between according to their distances to the bin means
    while curr_inc_ind <= length(incomes)

        while curr_inc_ind <= length(incomes) && incomes[curr_inc_ind] < high_bin_mean
            ##Add weights to bins
            wgt_low = each_inc_wgt * min((high_bin_mean - incomes[curr_inc_ind]) / (high_bin_mean - low_bin_mean), 1.0)
            tot_wgt_low += wgt_low
            tot_wgt_high += max(each_inc_wgt - wgt_low, 0.0)

            curr_inc_ind += 1
        end
        ##Only add lower bin since upper one can still receive weights from the following points
        tot_wgt_low > 0.0 && push!(b_w, (low_bin, tot_wgt_low))
        
        low_bin = high_bin
        high_bin += 1

        low_bin_mean = high_bin_mean
        high_bin_mean = _mean_biweight_positive(high_bin * b_wdth, _ker_bdw(b_wdth))

        tot_wgt_low = tot_wgt_high
        tot_wgt_high = 0.0
    end

    tot_wgt_low > 0.0 && push!(b_w, (low_bin, tot_wgt_low))

    return BinnedIncs(b_wdth, sum(incomes) * each_inc_wgt, length(incomes) * each_inc_wgt, b_w)
end


###Adds the weights of BinnedIncs to a IncDistBinned
function add_from_BI!(id_n::IncDistBinned, bi::BinnedIncs, share_added_per_bin::Float64=1.0)
    (id_n.b_wth != bi.b_wth) && error("Bins are incompatible: not same bin width.")
    for (i, w) in bi.bin_ind_wgt
        ind_key = ht_keyindex(id_n.bin_wgt, i)
        if ind_key > 0
            @inbounds id_n.bin_wgt.vals[ind_key] += w * share_added_per_bin
        else
            ind_key = ht_keyindex2!(id_n.bin_wgt, i)
            @inbounds _setindex!(id_n.bin_wgt, w * share_added_per_bin, i, -ind_key)
        end
    end
    add_weighted!(id_n.points_wp, bi.inc_wgtd_pts * share_added_per_bin, bi.inc_tot_wgt * share_added_per_bin)
    return nothing
end

#################################################################
############# Tricube distribution functions ####################
#################################################################
function _cdf_tricube(x::Real)
    u = abs(x)
    u >= 1.0 && return ifelse(x >= 1.0, 1.0, 0.0)
    u = 0.5 + (-1.0)^(x < 0.0) * (0.864197530864197*u - 0.648148148148148*u^4 + 0.37037037037037*u^7 - 0.0864197530864197*u^10)
    return max(min(u, 1.0), 0.0)
end

_cdf_tricube(l::Real, s::Real, x::Real) = _cdf_tricube((x - l) / s)

_pdf_tricube(x::Real) = max(70.0 / 81.0 * (1.0 - abs(x)^3)^3, 0.0) * (abs(x) < 1.0)
 
_pdf_tricube(l::Real, s::Real, x::Real) = _pdf_tricube((x - l) / s) / s

####Computes the mean of a tricube RV restricted to positive values
@inline function _mean_tricube_positive(l::Float64, s::Float64)
    res = l
    if l < s
        int_lb = max(- (l + l / s) / s, -1.0)
        res = (l * (1.0 - _cdf_tricube(int_lb)) +
              s * (- 0.0785634118967452 * int_lb^11 - 0.324074074074074 * int_lb^8 - 0.518518518518518 * int_lb^5 - 0.432098765432099 * int_lb^2 + 0.159090909090909)) /
              (1.0 - _cdf_tricube(- l / s))
    end
    return res
end

###Use rejection method with normal(0.0, 1/2) as enveloppe and 1.25 as "fitting" factor
function _rand_tricube()
    ##Enveloppe
    d_env = Normal(0.0, 0.5)
    ##M is taken to be 1.25 and scale 0.5
    r_n = 0.5 * randn()
    r_u = rand()
    while r_u >= _pdf_tricube(r_n) / (pdf(d_env, r_n) * 1.25)
        r_n = 0.5 * randn()
        r_u = rand()
    end
    return r_n
end

_rand_tricube(location::Real, scale::Real) = scale * _rand_tricube() + location

_rand_tricube(n::Integer) = [_rand_tricube() for i=1:n]

function _rand_tricube(location::Real, scale::Real, n::Integer)
    res = _rand_tricube(n)
    res .= scale .* res .+ location
    return res
end

#############################################################
################## Biweight distribution functions ##########

_cdf_biweight(x::Real) = ifelse(abs(x) >= 1.0, convert(Float64, x >= 1.0), 
                                0.0625*(x + 1.0)^3 * evalpoly(x, (8.0, -9.0, 3.0)))

_cdf_biweight(l::Real, s::Real, x::Real) = _cdf_biweight((x - l) / s)

_pdf_biweight(x::Real) = ifelse(abs(x) < 1.0, 0.9375 * (1 - x*x)^2, 0.0)
_pdf_biweight(l::Real, s::Real, x::Real) = _pdf_biweight((x - l) / s) / s

####Computes the mean of a tricube RV restricted to positive values
@inline function _mean_biweight_positive(l::Float64, s::Float64)
    if l >= s
        return l
    else
        P = l * s
        l_2 = l*l; s_2 = s*s
        return 2.5 * (l + s) * (0.2*l_2 - 0.8*P + s_2) / (3.0*l_2 - 9.0*P + 8.0*s_2)
    end
end

###Use rejection method with normal(0.0, 0.43) as enveloppe and 1.12 as "fitting" factor
function _rand_biweight()
    ##Enveloppe
    d_env = Normal(0.0, 0.43)
    ##M is taken to be 1.12 and scale 0.43
    r_n = randn() * 0.43
    r_u = rand()
    while r_u >= _pdf_biweight(r_n) / (pdf(d_env, r_n) * 1.12)
        r_n = randn() * 0.43
        r_u = rand()
    end
    return r_n
end

_rand_biweight(location::Real, scale::Real) = scale * _rand_biweight() + location

_rand_biweight(n::Integer) = [_rand_biweight() for i=1:n]

function _rand_biweight(location::Real, scale::Real, n::Integer)
    res = _rand_biweight(n)
    res .= scale .* res .+ location
    return res
end
