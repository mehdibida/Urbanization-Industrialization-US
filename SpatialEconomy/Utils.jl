using StatsBase: cor, var, mean, Weights, sample
using Distributions: cdf, Beta, Gamma, FDist, LocationScale, DiscreteNonParametric, Distributions, AliasTable
using Base.Iterators: cycle as itercycle
using Graphs: barabasi_albert, desopo_pape_shortest_paths, AbstractGraph, nv, neighbors
using Random: randexp
using Base: Reverse, Perm
import Statistics.cov, Base.getindex


###Extend getindex for generators based on arrays
#getindex(g::Base.Generator{A, F}, ind) where {A <: AbstractArray, F} = g.f(g.iter[ind])

###utility functions (to make code look simpler)
###Sumrows and sumcols with a preallocated vector
function sumcols!(res::AbstractArray{<: Real, 1}, mat::AbstractArray{<: Real, 2})
    size(mat, 1) != length(res) && error("Vector length differs from matrix number of rows.")
    res .= view(mat, :, 1)
    for c=2:size(mat, 2)
        res .+= view(mat, :, c)
    end
    return res
end


function sumrows!(res::AbstractArray{<: Real, 1}, mat::AbstractArray{<: Real, 2})
    size(mat, 2) != length(res) && error("Vector length differs from matrix number of columns.")
    for c=1:size(mat, 2)
        res[c] = sum(mat[i, c] for i=1:size(mat, 1))
    end
    return res
end

sumcols(mat::AbstractArray{T, 2}) where{T <: Real} = sumcols!(Vector{T}(undef, size(mat, 1)), mat)
sumrows(mat::AbstractArray{T, 2}) where{T <: Real} = sumrows!(Vector{T}(undef, size(mat, 2)), mat)

###Extends a vector and assigns a value to the extension
function extend_to_val!(vec::AbstractArray{T, 1}, extension_size::Int64, val::T) where {T}
    extension_size < 0 && error("Extension size must be a non-negative integer.")
    if extension_size > 0
        old_size = length(vec)
        resize!(vec, old_size + extension_size)
        vec[(old_size + 1):end] .= val
    end
    return vec
end


###Case where new_val is a a value
function setiter!(old_val::AbstractArray{T, 1}, new_val::T, iter) where {T}
    for i in iter
        old_val[i] = new_val
    end
    return old_val
end
###Case where new_val is an array
function setiter!(old_val::AbstractArray{T, 1}, new_val::AbstractArray{R, 1}, iter) where {T, R}
    for i in iter
        old_val[i] = new_val[i]
    end
    return old_val
end

###Case where nwe_val is an array, and the iterator is different
function setiter!(old_val::AbstractArray{T, 1}, new_val::AbstractArray{R, 1}, iter_1, iter_2) where {T, R}
    next = iterate(zip(iter_1, iter_2))
    while next !== nothing
        ((i1, i2), (s1, s2)) = next
        old_val[i1] = new_val[i2]
        if xor(isnothing(iterate(iter_1, s1)), isnothing(iterate(iter_2, s2)))
            error("Iterators of different lengths.")
        end
        next = iterate(zip(iter_1, iter_2), (s1, s2))
    end
    return old_val
end

###Implements an "inifinite" array filled with only one value 
###(to be used with subset_assign and itr_fltr_on_val, findsubset! and findall!)
struct ScalarArray{T}
    scalar_val::T
end
getindex(sa::ScalarArray{T}, ::Integer) where {T} = sa.scalar_val


###Efficient assignment on a subset of indices of dest and b (generalization of setiter)
function subset_assign!(dest, inds, func::F, func_arg::Vararg{Any, N}) where {F, N}
    for i in inds
        dest[i] = func(broadcast(getindex, func_arg, i)...)
    end
    return dest
end

###Filters one iterator based on the values of other iterators
function itr_fltr_on_val(ind_itr, pred::F, val_itr::Vararg{Any, N}) where {F, N}
    M = length(val_itr)+1
    return (i[M] for i in Iterators.filter(pred, Iterators.zip(val_itr..., ind_itr)))
end

####Finds indices of ind_subset of which f(fun_arg.[i]) return true (return a view) in a preallocated vector pre
function findsubset!(pre::AbstractArray{Int64, 1}, f::F, ind_subset, func_arg::Vararg{Any, N}) where {F, N}
    pre_i::Int64 = 0
    for v_i in ind_subset
        if f(broadcast(getindex, func_arg, v_i)...)
            pre_i += 1
            pre[pre_i] = v_i
        end
    end
    ###returns the last index of which 
    ###slot was changed
    return pre_i
end


####Finds indices of ind_subset for which vec[i] are true in a preallocated vector pre
function findsubset!(pre::AbstractArray{Int64, 1}, ind_subset, vec::AbstractArray{Bool, 1})
    pre_i::Int64 = 0
    for v_i in ind_subset
        if vec[v_i]
            pre_i += 1
            pre[pre_i] = v_i
        end
    end
    ###returns the last index of which 
    ###slot was changed
    return pre_i
end

findall!(pre::AbstractArray{Int64, 1}, vec::AbstractArray{Bool, 1}) = findsubset!(pre, 1:length(vec), vec)


###Reorders source and puts the result in dest
function sample_wo_rplc!(source::Array{T, 1}, dest::Array{T, 1}) where {T}
    i::Int64 = length(source)
    for k=1:length(dest)
        j = rand(1:i)
        dest[k] = source[j]
        source[i], source[j] = source[j], source[i]
        i -= 1
    end
    return dest
end

###Returns maximum income in the system
function max_inc(inc_arr)
    length(inc_arr) == 0 && error("Empty array, cannot find max.")
    m = maximum(inc_arr[1])
    for i=2:length(inc_arr)
        maximum(inc_arr[i]) > m && (m = maximum(inc_arr[i]))
    end
    return m
end

###Returns unique values for vectors with values on which there is an order
###order means a < b is defined
function unique_ordered(val)
    length(val) == 0 && return copy(val)
    sorted_val = sort(val, alg=MergeSort)
    last_put = sorted_val[1]
    res = [last_put]
    for i=2:length(sorted_val)
        if sorted_val[i] != last_put
            last_put = sorted_val[i]
            push!(res, last_put)
        end
    end
    return res
end

###Gives the argmin of f.(arr) (without allocations)
function fargmin(f::Function, arr::AbstractArray{T, 1}) where T
    length(arr) == 0 && error("argmin not defined over an empty array.")
    @inbounds amin::T = arr[1]
    @inbounds famin = f(arr[1])
    @inbounds for i=2:length(arr)
        pot_famin = f(arr[i])
        amin = ifelse(pot_famin < famin, arr[i], amin)
        famin = ifelse(pot_famin < famin, pot_famin, famin)
    end
    return amin
end

###Define covariance for generators
function cov(x::Base.Generator, y::Base.Generator)
    m_x = mean(x); m_y = mean(y)
    res = 0.0
    n = 0.0
    next = iterate(zip(x, y))
    while next !== nothing
        ((i,j), (si, sj)) = next
        res += (i - m_x) * (j - m_y)
        n += 1.0
        if xor(isnothing(iterate(x, si)), isnothing(iterate(y, sj)))
            error("Iterables of different lengths.")
        end
        next = iterate(zip(x, y), (si, sj))
    end
    return res / (n - 1.0)
end

###To calculate the slope of a simple regression
simple_reg_slope(y, x) = cov(y, x) / var(x)
simple_reg_intercept(y, x) = mean(y) - simple_reg_slope(y, x) * mean(x)
simple_reg(y, x) = begin sl = simple_reg_slope(y, x); return (mean(y) - sl * mean(x), sl) end


###Simplify uniform random function
runif(a::Real, b::Real) = a + rand() * (b - a)


###Corrects sample mean by adding exponential realizations, only to be used when n is large enough
function correct_mean!(sample, target_mean)
    target_mean == 0.0 && return sample .*= 0.0
    ##Where the result will be stored
    while true
        mean_correction = target_mean - mean(sample)
        for i=1:length(sample)
            sample[i] = max(sample[i] + mean_correction * randexp(), 0.0)
        end
        any(x -> x > 0.0, sample) && break
    end
    sample .*= target_mean / mean(sample)
    return sample
end


###Directly samples with average = distribution mean
function rand_inc_fixed_mean(dist, n::Int64)
    minimum(dist) < 0.0 && error("Income distribution can't have negative values.")
    return correct_mean!(rand(dist, n), mean(dist))
end

####Function that logs positive and negative to ease plot readings and has f(1) = 1
logify(x::Float64)::Float64 = sign(x) * log(1.0 + (exp(1) - 1.0) * abs(x))

###Logistic function
lgstic(x) = 1.0 / (1.0 + exp(-x))

###Soft plus that doesn't go exponential on the negative side
ssm(x::Real) = (1.0 + abs(x))^sign(x)

###average that omits NaN
function func_nanrm(iter, func::F) where F <: Function
    return func([a for a in iter if !isnan(a)])
end


###Puts elements in inds at the beginning of the vector a
###inds must be sorted and unique
function shiftleft!(a::AbstractVector, inds)
    i = 1
    for k in inds
        a[i] = a[k]
        i += 1
    end
    return a
end

###"Complementary" of deleteat!, indices must be sorted and unique
keepat!(a::AbstractVector, inds) = resize!(shiftleft!(a, inds), length(inds))

#####Model functions
###Builds a scale free network of locations
function build_sc_net(n_loc::Int64)

    loc_net = barabasi_albert(n_loc, 1, 1)
    #matrix distance between locations
    loc_dist = zeros(Float64, n_loc, n_loc)
    ##Storing the path nodes is not worth it, becomes too slow and takes too much time
    ##Choice : Calculate it every time in the main
    for i=1:n_loc
        loc_dist[:,i] .= desopo_pape_shortest_paths(loc_net, i).dists
    end

    return loc_net, loc_dist

end

####Returns positive class of a modulo b
pos_mod(a::Int64, b::Int64)::Int64 = ifelse((a % b) == 0, b, a % b)

####Counts number of firms in each location
function count_firms_location(n_location::Int64, firm_location::AbstractArray{<:Integer, 1})
    res::Array{Int64, 1} = fill(0, n_location)
    for l in firm_location
        res[l] += 1
    end
    return res
end

##Production function corresponds to Leontief with increasing returns
####!!!!Careful: version with (x+1)^r - 1 has local maxima after 0 for unit cost
firm_production(labor::Real, capital::Real, ret_scale::Real, tech::Real) = tech * min(labor, capital)^ret_scale

firm_max_prod(capital::Real, ret_scale::Real, tech::Real) = tech * capital^ret_scale

##Returns the demand for a producer (returns capital id demand is higher than capital)
firm_labor_demand(plan_quant::Real, capital::Real, ret_scale::Real, tech::Real) = min(capital, (plan_quant / tech)^(1.0/ret_scale))

##Returns the cost of producing a unit
firm_unit_cost(lab::Float64, wage::Float64, cap::Float64, cap_cost::Float64,
               techn::Float64, ret_sc::Float64) = (lab * wage + cap_cost * cap) / firm_production(lab, cap, ret_sc, techn)


####Function to compute weight corresponding to profit (make it positive)
function pr_conv(x::Float64, q::Float64)::Float64
    if (x >= 0.0)
        return ((x + 1) ^ q)
    else
        return (1.0 / ((1.0 - x) ^ q))
    end
end

###Function that returns the nodes in the shortest path given the result of dijkstra_dst
###To bu used when shortest path is unique
function sp_nodes(dst::Int64, prev::AbstractArray{Int64,1}, 
                  exclude::Set{Int64}=Set(Int64[]))

    path::Array{Int64, 1} = Int64[dst]
    next_to_add::Int64 = prev[dst]

    while next_to_add != 0
        push!(path, next_to_add)
        next_to_add = prev[next_to_add]
    end
    return [n for n in path if !(n in exclude)]
end

###Get list of nodes that belong to at least one of the shortest paths between source and dst
###exclude : nodes to exclude from the lists
function sp_nodes(dst::Int64, prev::AbstractArray{T,1}, 
                  exclude::Set{Int64}=Set(Int64[])) where T <: AbstractArray{Int64, 1}

    nodes_on_path::Set{Int64} = Set(Int64[dst])
    to_check::Array{Int64, 1} = copy(prev[dst])
    next_to_check::Array{Int64, 1} = Int64[]

    while !isempty(to_check)
        for m in to_check
            if !(m in nodes_on_path) & (m != 0)
                push!(nodes_on_path, m)
                append!(next_to_check, prev[m])
            end
        end
        empty!(to_check)
        to_check, next_to_check = next_to_check, to_check
    end

    return [n for n in nodes_on_path if !(n in exclude)]
end


####Functions to compute the average, weighted average, and variance
##weighted average
function w_avg(values, weights=itercycle(1.0))
    length(values) == 0 && return NaN
    return sum(x -> x[1]*x[2], zip(values, weights)) / sum(x -> x[2], zip(values, weights))
end


##average
function avg(values)::Float64
    (length(values) == 0) && return NaN
    return (sum(values) / length(values))
end
##correlation
function crl(v1::AbstractArray{Float64, 1}, v2::AbstractArray{Float64, 1})::Float64
    return (sum((v1 .- mean(v1)) .* (v2 .- mean(v2))) / (length(v1) - 1.0) / (var(v1)*var(v2))^0.5)
end


###ready functions to by applied to groups
##grouped average
grp_avg(to_mean, group) = grp_fctn(to_mean, group, avg)
##grouped variance
grp_vrc(to_mean, group) = grp_fctn(to_mean, group, vrc)


#######To compress utilities
struct UtilityGroup
    grp_util_mean::Float64
    grp_inc_dist::BinnedIncs
end

wtd_incomes(ug::UtilityGroup) = ug.grp_inc_dist.inc_wgtd_pts
total_weight(ug::UtilityGroup) = ug.grp_inc_dist.inc_tot_wgt

####Compress utilities and put them in a preallocated vector
###Assumes utilities are sorted
function partition_util_inc!(util_groups::AbstractArray{UtilityGroup, 1}, utility::AbstractArray{<: Real, 1}, income::AbstractArray{<: Real, 1}, 
                             weight::Real, util_part_interval::Real, inc_part_interval::Real)
    isempty(util_groups) || empty!(util_groups) ###Empty the array if its not empty
    length(utility) > 0 || return util_groups
    length(utility) == length(income) || error("Income and utility of different lengths")

    j = 1
    @inbounds while j <= length(utility)
        util_hb = utility[j] + util_part_interval
        ind_included_low = j
        sum_util = utility[j]
        j += 1
        while (j <= length(utility)) & (utility[j] < util_hb)
            sum_util += utility[j]
            j += 1
        end
        ind_included_high = j - 1
        push!(util_groups,
              UtilityGroup(sum_util / (j - ind_included_low), 
                           BinnedIncs(inc_part_interval, sort!(view(income, ind_included_low:ind_included_high)), weight)))
    end
    return util_groups
end

####Compress utilities
###Assumes utilities are sorted
partition_util_inc(utility::AbstractArray{<: Real, 1}, income::AbstractArray{<: Real, 1}, weight::Real, 
                   util_part_interval::Real, inc_part_interval::Real) =  partition_util_inc!(UtilityGroup[], utility, income, weight, util_part_interval, inc_part_interval)


#####Functions that give the probability that an "index" is chosen randomly and uniformly
#####given that it has a certain probability to be in the consideration set (based on Lee 2019
##### and second order approximation of log(1-x)
###Needed to compute erf, needed for the second order approximation of log(1-x), from Ren & MacKenzie, 2007
pre_erf(x) = 2.751938393884109 / (3.1052299527891134*x + sqrt(pi*x^2 + 7.573164923733449))

###Probability that i is chosen based on second oreder approximation of log(1-x)
function p_cs_chosen(i, p_sel, prob_sum=sum(p_sel), prob_sq_sum=sum(x -> x^2, p_sel)) 
    p_sel[i] == 0.0 && return 0.0
    ps_i_1 = prob_sum - p_sel[i]
    ps_i_1 < 0.0001 && return p_sel[i]
    ##First order approximation is sufficient is ps_i_1 is small enough
    ps_i_1 < 0.05 && return p_sel[i] / ps_i_1 * (1 - exp(-ps_i_1))
    ###Second order otherwise
    ps_i_2 = (prob_sq_sum - p_sel[i]^2) / 2.0    
    s_ps_i_2 = sqrt(ps_i_2)
    lb = ps_i_1 / s_ps_i_2 / 2.0
    ###Computing erf is counter-productive, using only pre_erf is better (so that the exponentials cancel out)
    return 0.5 * p_sel[i] / s_ps_i_2 * sqrt(pi) * (pre_erf(lb) - pre_erf(lb + s_ps_i_2) * exp(- ps_i_1 - ps_i_2))
end

###Probability that i is chosen based on second oreder approximation of log(1-x)
function p_cs_chosen(p_sel_i, prob_sum, prob_sq_sum) 
    p_sel_i == 0.0 && return 0.0
    ps_i_1 = prob_sum - p_sel_i
    #ps_i_1 < 0.0001 && return p_sel_i
    ##First order approximation is sufficient is ps_i_1 is small enough
    ps_i_1 < 0.005 && return p_sel_i / ps_i_1 * (1 - exp(-ps_i_1))
    ###Second order otherwise
    ps_i_2 = (prob_sq_sum - p_sel_i^2) / 2.0    
    s_ps_i_2 = sqrt(ps_i_2)
    lb = ps_i_1 / s_ps_i_2 / 2.0
    ###Computing erf is counter-productive, using only pre_erf is better (so that the exponentials cancel out)
    return 0.5 * p_sel_i / s_ps_i_2 * sqrt(pi) * (pre_erf(lb) - pre_erf(lb + s_ps_i_2) * exp(- ps_i_1 - ps_i_2))
end


import Base.isless

###Consumers iteratively randomly look for sellers that sell at a price that will allow them
###to reach their satitation
####simpler only bcs there is no ranking of local prices but gives same results
###The last three vectors can be supplied to generate less allocations
function realize_cons_rq_simpler!(buy_realn_consume::AbstractArray{<:AbstractArray{Bool, 1}, 1}, buy_realn_inc::AbstractArray{<:AbstractArray{Float64, 1}, 1}, 
                          buy_realn_cons::AbstractArray{<:AbstractArray{Float64, 1}, 1}, buy_realn_rq::AbstractArray{<:AbstractArray{Float64, 1}, 1},
                          buy_pop::AbstractArray{Float64, 1}, satiation::Float64, buy_realn_sell_choice::AbstractArray{<:AbstractArray{Int64, 1}, 1},
                          sell_quant::AbstractArray{Float64, 1}, sell_buy_price::AbstractArray{Float64, 1}, sell_buy_ship::AbstractArray{Float64, 1},
                          tol::Float64=10^-9,
                          sell_ind::AbstractArray{Int64, 1}=Array{Int64, 1}(undef, length(sell_quant)), ##Indices of active sellers
                          sell_dem_share::AbstractArray{Float64, 1}=fill(0.0, length(sell_quant)), ##Share of demand satisfied by sellers
                          sell_buy_dem::AbstractArray{Float64, 1}=Array{Float64, 1}(undef, length(sell_quant))) ##Quantities demanded to sellers at each iteration
    
    ##Buyers satiation threshold
    buy_sat_threshold = satiation * (1.0 - tol)
    ###Store active buyers
    buy_ind::Array{Int64, 1} = Array{Int64, 1}(undef, maximum(length(x) for x in buy_realn_inc))
    ###To store reservation prices of buyers realizations (First if reservation price, second is buyer index)
    buy_res_price::Array{Tuple{Float64, Int64}, 1} = Array{Tuple{Float64, Int64}, 1}(undef, length(buy_ind))

    ###Indices of active sellers based on quantity
    selling = findsubset!(sell_ind, (q, s, t) -> (q > t) & (s < (1.0 - t) * q), 1:length(sell_quant), sell_quant, sell_buy_ship, ScalarArray(tol))

    ###Condition is only on sellers, the one on buyers comes later
    while selling > 0
        ###Minimum selling price
        min_price = minimum(sell_buy_price[s] for s in view(sell_ind, 1:selling))
        ###reinitialize intermediary shipments
        sell_buy_dem .= 0.0
        ##Begin search for seller
        for l=1:length(buy_realn_inc)
            ###Continue if the location has no population
            (buy_pop[l] < tol) && continue
            ##Realization weight
            buy_real_wgt = buy_pop[l] / length(buy_realn_inc[l])
            ###Reinitialize choice vector
            buy_realn_sell_choice[l] .= 0
            ###Find active buyers realizations
            consuming = findall!(buy_ind, buy_realn_consume[l])
            ##continue if no consumers
            consuming == 0 && continue
            ##Update reservation quantities to account for what has been consumed and satiation levels
            for c in view(buy_ind, 1:consuming); buy_realn_rq[l][c] = min(buy_realn_rq[l][c], satiation - buy_realn_cons[l][c]); end
            ##Find buyer reservation price (porblem of infinite res. price for those reaching satiation is solved by excluding them
            rp = 1
            for c in view(buy_ind, 1:consuming)
                buy_res_price[rp] = (ifelse(buy_realn_rq[l][c] > 0.0, buy_realn_inc[l][c] / buy_realn_rq[l][c], Inf64), c)
                rp += 1
            end
            sort!(view(buy_res_price, 1:consuming), alg = QuickSort, rev = true)
            ###Sellers concerned by the choice of consumers
            range_choice = selling
            ####Range of search (should only go decreasing bcs consumers are ordered by decreasing reservation price)
            for r=1:consuming
                c = buy_res_price[r][2]
                ###Shopping ends (for all others following) if c cannot buy
                if min_price > buy_res_price[r][1]
                    for nc=r:consuming; buy_realn_consume[l][buy_res_price[nc][2]] = false; end
                    break
                end
                while range_choice > 0
                    ##Randomly search seller: choose randomly an element from selling
                    choice = rand(1:range_choice)
                    if sell_buy_price[sell_ind[choice]] <= buy_res_price[r][1] ##Stop if satisfying seller found
                        buy_realn_sell_choice[l][c] = sell_ind[choice]
                        sell_buy_dem[sell_ind[choice]] += min(satiation - buy_realn_cons[l][c], ## quantity to reach satiation
                                                              buy_realn_inc[l][c] / sell_buy_price[sell_ind[choice]]) * ## quantity limited by budget
                                                          buy_real_wgt
                        break
                    end
                    ##Store non convenient in out of range and keep searching
                    sell_ind[range_choice], sell_ind[choice] = sell_ind[choice], sell_ind[range_choice]
                    range_choice -= 1
                end
            end
        end
        
        ###Stop process if no buyer is active
        all(x -> x < tol, sell_buy_dem) && break

        ##Update sellers
        ###Find active sellers
        selling = findsubset!(sell_ind, x -> x > 0.0, 1:length(sell_quant), sell_buy_dem)
        for s in view(sell_ind, 1:selling);
            ###update total quantity to sellers share
            sell_dem_share[s] =  min(1.0, max(0.0, sell_quant[s] - sell_buy_ship[s]) / sell_buy_dem[s])
            #####Update shipped quantities and seller left quantity
            sell_buy_ship[s] = min(sell_buy_ship[s] + sell_dem_share[s] * sell_buy_dem[s], sell_quant[s])
        end
        ###Find active sellers for the next iteration
        selling = findsubset!(sell_ind, (q, s, t) -> (q > t) & (s < (1.0 - t) * q), 1:length(sell_quant), sell_quant, sell_buy_ship, ScalarArray(tol))

        ###Update consumption and income, and reinitialize choice vectors
        for l=1:length(buy_realn_inc)
            ###Find indices to iterate on
            buying = findsubset!(buy_ind, x -> x != 0, 1:length(buy_realn_sell_choice[l]), buy_realn_sell_choice[l])
            for b_r in view(buy_ind, 1:buying)
                #####Keep chosen seller (for better code readability)
                c_s = buy_realn_sell_choice[l][b_r]
                ##Recover individual demand
                cons_real_dem = min(satiation - buy_realn_cons[l][b_r], ## quantity to reach satiation
                                    buy_realn_inc[l][b_r] / sell_buy_price[c_s]) ## quantity limited by budget
                ####Update consumption
                buy_realn_cons[l][b_r] += cons_real_dem * sell_dem_share[c_s]
                ####Update income
                buy_realn_inc[l][b_r] = max(buy_realn_inc[l][b_r] - cons_real_dem * sell_dem_share[c_s] * sell_buy_price[c_s], 0.0)
                ###Update active buyers
                buy_realn_consume[l][b_r] = (buy_realn_cons[l][b_r] <= buy_sat_threshold) & (buy_realn_inc[l][b_r] > tol)
            end
        end
    end
    
    return nothing
end


###Consumers iteratively randomly look for sellers that sell at a price that will allow them
###to reach their satitation
####simpler only bcs there is no ranking of local prices but gives same results
function realize_cons_rp_simpler!(buy_realn_consume::AbstractArray{<:AbstractArray{Bool, 1}, 1}, buy_realn_inc::AbstractArray{<:AbstractArray{Float64, 1}, 1}, 
                          buy_realn_cons::AbstractArray{<:AbstractArray{Float64, 1}, 1}, buy_realn_rp::AbstractArray{<:AbstractArray{Float64, 1}, 1},
                          buy_pop::AbstractArray{Float64, 1}, satiation::Float64, buy_realn_sell_choice::AbstractArray{<:AbstractArray{Int64, 1}, 1},
                          sell_quant::AbstractArray{Float64, 1}, sell_buy_price::AbstractArray{Float64, 1}, sell_buy_ship::AbstractArray{Float64, 1},
                          tol::Float64=10^-9)
    
    ##Indices of active sellers
    sell_ind::Array{Int64, 1} = Array{Int64, 1}(undef, length(sell_quant))
    
    ##Share of demand satisfied by sellers
    sell_dem_share = fill(0.0, length(sell_quant))

    ##Buyers satiation threshold
    buy_sat_threshold = satiation * (1.0 - tol)
    ###Store active buyers
    buy_ind::Array{Int64, 1} = Array{Int64, 1}(undef, maximum(length(x) for x in buy_realn_inc))
    ###To store reservation prices of buyers realizations (First if reservation price, second is buyer index)
    buy_res_price::Array{Tuple{Float64, Int64}, 1} = Array{Tuple{Float64, Int64}, 1}(undef, length(buy_ind))

    ###Intermediary shipped quantities
    sell_buy_dem = Array{Float64, 1}(undef, length(sell_quant))

    ###No more sorting of local prices

    ###Condition is only on sellers, the one on buyers comes later
    while any(x::Tuple{Float64, Float64} -> (x[1] > tol) && ((1.0 - x[2] / x[1]) > tol), zip(sell_quant, sell_buy_ship))
        ###reinitialize intermediary shipments
        sell_buy_dem .= 0.0
        ##Begin search for seller
        for l=1:length(buy_realn_inc)
            ###Continue if the location has no population
            buy_pop[l] < tol && continue
            ##Realization weight
            buy_real_wgt = buy_pop[l] / length(buy_realn_inc[l])
            ###Reinitialize choice vector
            buy_realn_sell_choice[l] .= 0
            ###Find active buyers realizations
            consuming = findall!(buy_ind, buy_realn_consume[l])
            ##continue if no consumers
            consuming == 0 && continue
            ##Find buyer reservation price (porblem of infinite res. price for those reaching satiation is solved by excluding them
            rp = 1
            for c in view(buy_ind, 1:consuming)
                buy_res_price[rp] = (buy_realn_rp[l][c], c)
                rp += 1
            end
            sort!(view(buy_res_price, 1:consuming), alg = QuickSort, rev = true)
            ##Update indices of: active and considered sellers
            selling = findsubset!(sell_ind, (q, s, q_0) -> (q - s) > (q_0 * q), 1:length(sell_quant), sell_quant, sell_buy_ship, ScalarArray(tol))
            selling = findsubset!(sell_ind, (p, res_p) -> p <= res_p, view(sell_ind, 1:selling), sell_buy_price, ScalarArray(buy_res_price[1][1]))
            ##continue if no sellers
            if selling == 0
                setiter!(buy_realn_consume[l], false, view(buy_ind, 1:consuming))
                continue
            end
            min_price = minimum(sell_buy_price[s] for s in view(sell_ind, 1:selling))
            range_choice = selling ####Once we have the length of selling, we can use sell_ind for random selection
            ####Range of search (should only go decreasing bcs consumers are ordered by decreasing reservation price)
            for r=1:consuming
                c = buy_res_price[r][2]
                ###Shopping ends (for all others following) if c cannot buy
                if min_price > buy_res_price[r][1]
                    for nc=r:consuming; buy_realn_consume[l][buy_res_price[nc][2]] = false; end
                    break
                end
                while range_choice > 0
                    ##Randomly search seller: choose randomly an element from selling
                    choice = rand(1:range_choice)
                    if sell_buy_price[sell_ind[choice]] <= buy_res_price[r][1] ##Stop if satisfying seller found
                        buy_realn_sell_choice[l][c] = sell_ind[choice]
                        sell_buy_dem[sell_ind[choice]] += min(satiation - buy_realn_cons[l][c], ## quantity to reach satiation
                                                              buy_realn_inc[l][c] / sell_buy_price[sell_ind[choice]]) * ## quantity limited by budget
                                                          buy_real_wgt
                        break
                    end
                    ##Store non convenient in out of range and keep searching
                    sell_ind[range_choice], sell_ind[choice] = sell_ind[choice], sell_ind[range_choice]
                    range_choice -= 1
                end
            end
        end
        
        ###Stop process if no buyer is active
        all(x -> x < (0.25 * tol), sell_buy_dem) && break

        ##Update sellers
        ###Find active sellers
        selling = findsubset!(sell_ind, x -> x > 0.0, 1:length(sell_buy_dem), sell_buy_dem)
        for s in view(sell_ind, 1:selling);
            ###update total quantity to sellers share
            sell_dem_share[s] =  min(1.0, max(0.0, sell_quant[s] - sell_buy_ship[s]) / sell_buy_dem[s])
            #####Update shipped quantities and seller left quantity
            sell_buy_ship[s] = min(sell_buy_ship[s] + sell_buy_dem[s] * sell_dem_share[s], sell_quant[s])
        end
          
        ###Update consumption and income, and reinitialize choice vectors
        for l=1:length(buy_realn_inc)
            ###Find indices to iterate on
            buying = findsubset!(buy_ind, x -> x != 0, 1:length(buy_realn_sell_choice[l]), buy_realn_sell_choice[l])
            for b_r in view(buy_ind, 1:buying)
                #####Keep chosen seller (for better code readability)
                c_s = buy_realn_sell_choice[l][b_r]
                ##Recover individual demand
                cons_real_dem = min(satiation - buy_realn_cons[l][b_r], ## quantity to reach satiation
                                    buy_realn_inc[l][b_r] / sell_buy_price[c_s]) ## quantity limited by budget
                ####Update consumption
                buy_realn_cons[l][b_r] += cons_real_dem * sell_dem_share[c_s]
                ####Update income
                buy_realn_inc[l][b_r] = max(buy_realn_inc[l][b_r] - cons_real_dem * sell_dem_share[c_s] * sell_buy_price[c_s], 0.0)
                ###Update active buyers
                buy_realn_consume[l][b_r] = (buy_realn_cons[l][b_r] <= buy_sat_threshold) & (buy_realn_inc[l][b_r] > tol)
            end
        end
    end
    
    return nothing
end




###Consumers iteratively randomly look for sellers that sell at a price that will allow them
###to reach their satitation
function realize_cons_CES!(buy_realn_consume::AbstractArray{<:AbstractArray{Bool, 1}, 1}, buy_realn_inc::AbstractArray{<:AbstractArray{Float64, 1}, 1}, 
                           buy_realn_cons::AbstractArray{<:AbstractArray{Float64, 1}, 1},
                           buy_pop::AbstractArray{Float64, 1}, satiation::Float64, price_sens::Float64, buy_realn_sell_choice::AbstractArray{<:AbstractArray{Int64, 1}, 1},
                           sell_quant::AbstractArray{Float64, 1}, sell_buy_price::AbstractArray{Float64, 2}, sell_buy_ship::AbstractArray{Float64, 2},
                           tol::Float64=10^-9)
    
    ##Indices of active sellers
    sell_act_ind = Array{Int64, 1}(undef, length(sell_quant))
    selling = findsubset!(sell_act_ind, x -> x > tol, 1:length(sell_quant), sell_quant)

    ##Share of demand satisfied by sellers
    sell_dem_share = fill(0.0, length(sell_quant))

    ##Buyers satiation threshold
    buy_sat_threshold = satiation * (1.0 - tol)
    
    ###Intermediary shipped quantities
    sell_buy_ship_int = Array{Float64, 2}(undef, size(sell_buy_ship))

    ##Used to store probability of selecting seller
    sell_prob = Array{Float64, 1}(undef, length(sell_quant))

    ###Condition is only on sellers, the one on buyers comes later
    while selling > 0
        ###Reinitialize sell prob
        sell_prob .= 0.0
        ###reinitialize intermediary shipments
        sell_buy_ship_int .= 0.0
        ##Begin search for seller
        for l=1:length(buy_realn_inc)
            ###Reinitialize choice vector
            buy_realn_sell_choice[l] .= 0
            ###Seller selector
            sell_prob[view(sell_act_ind, 1:selling)] .= view(sell_buy_price, view(sell_act_ind, 1:selling), l).^(-price_sens) ./
                                                        sum(x^(-price_sens) for x in view(sell_buy_price, view(sell_act_ind, 1:selling), l))
            sell_selector = AliasTable(sell_prob)
            for r=1:length(buy_realn_inc[l])
                ##Condition for buying: wanting to consume and not reaching threshold
                if buy_realn_consume[l][r] & (buy_realn_cons[l][r] <= buy_sat_threshold)
                    choice = rand(sell_selector)
                    buy_realn_sell_choice[l][r] = choice
                    sell_buy_ship_int[choice, l] += min(satiation - buy_realn_cons[l][r], ## quantity to reach satiation
                                                        buy_realn_inc[l][r] / sell_buy_price[choice, l]) ## quantity limited by budget
                end
            end
            ###Calculate real demand
            for s in view(sell_act_ind, 1:selling); sell_buy_ship_int[s, l] *= buy_pop[l] / length(buy_realn_inc[l]); end
        end

        ###Stop process if no buyer is active
        all(x -> x < (0.25 * tol), sell_buy_ship_int) && break

        ###Store TOTAL DEMAND in sell_dem_share
        sumcols!(sell_dem_share, sell_buy_ship_int)
        ###Find active sellers
        selling = findsubset!(sell_act_ind, x -> x > 0.0, 1:length(sell_dem_share), sell_dem_share)
        ###update total quantity to sellers share
        subset_assign!(sell_dem_share, view(sell_act_ind, 1:selling), (x, y) -> min(x / y, 1.0), sell_quant, sell_dem_share)
        #####Update shipped quantities and seller left quantity
        for l=1:length(buy_realn_inc); for s in view(sell_act_ind, 1:selling)
            sell_quant[s] = max(sell_quant[s] - sell_buy_ship_int[s, l], 0.0)
            sell_buy_ship[s, l] += sell_buy_ship_int[s, l] * sell_dem_share[s]
        end; end
        ###Find active sellers for the next iteration
        selling = findsubset!(sell_act_ind, x -> x > tol, 1:length(sell_quant), sell_quant)

        ###Update consumption and income, and reinitialize choice vectors
        for l=1:length(buy_realn_inc)
            ###Find indices to iterate on
            for b_r in (b for b=1:length(buy_realn_cons[l]) if buy_realn_sell_choice[l][b] != 0)
                #####Keep chosen seller (for better code readability)
                c_s = buy_realn_sell_choice[l][b_r]
                ##Recover individual demand
                cons_real_dem = min(satiation - buy_realn_cons[l][b_r], ## quantity to reach satiation
                                    buy_realn_inc[l][b_r] / sell_buy_price[c_s, l]) ## quantity limited by budget
                ####Update consumption
                buy_realn_cons[l][b_r] += cons_real_dem * sell_dem_share[c_s]
                ####Update income
                buy_realn_inc[l][b_r] = max(buy_realn_inc[l][b_r] - cons_real_dem * sell_dem_share[c_s] * sell_buy_price[c_s, l], 0.0)
                ###Update active buyers
                buy_realn_consume[l][b_r] = (buy_realn_cons[l][b_r] <= buy_sat_threshold) & (buy_realn_inc[l][b_r] > tol)
            end
        end
    end

    return nothing
end


###Version of findlast with a default value
function findlast_default(pred::Function, arr::Array{T, 1}, default::Int64=0) where T
    ind::Int64 = length(arr)
    @inbounds while ((ind > 0) & (!pred(arr[ind])))
        ind -= 1
    end
    return ifelse(ind == 0, default, ind)
end

function findfirst_default(pred::Function, arr::Array{T, 1}, default::Int64=length(arr)+1) where T
    ind::Int64 = 1
    @inbounds while ((ind <= length(arr)) & (!pred(arr[ind])))
        ind += 1
    end
    return ifelse(ind == (length(arr)+1), default, ind)
end

###CDF of exponential distrbution with lambda being the scale parameter
cdf_exp(lambda::Real, x::Real) = ifelse(x > 0.0, 1 - exp(-x/lambda), 0.0)

##Function that assigns labor to firms in cities
function assign_labor!(firm_labor_obt::AbstractArray{Float64, 1}, pop_man::Float64, res_wage, firm_wage::AbstractArray{Float64, 1}, firm_labor_demand::AbstractArray{Float64, 1}, 
                       tol::Float64=10^-9)

    ##Return result if no firms compete: lowest reservation wage is higher than highest firm wage
    ((length(firm_labor_demand) == 0) || (minimum(res_wage) >= maximum(firm_wage))) && return firm_labor_obt
    #Non-recruiting firms
    recruiting::Array{Bool, 1} = firm_labor_obt .<= (1.0 - tol) .* firm_labor_demand
    any(recruiting) || return firm_labor_obt
    ##Sort wages in decreasing order
    wage_sortperm::Array{Int64, 1} = sortperm(firm_wage, rev = true, alg = QuickSort)
    wage_order::Array{Int64, 1} = Array{Int64, 1}(undef, length(firm_wage))
    #Get position of each wage when sorted
    wage_order[wage_sortperm] .= 1:length(wage_order)

    ##############All following vectors are ordered by wage in decreasing order
    ###Amount of labor available to each firm for different wages intervals
    avail_labor::Array{Float64, 1} = pop_man .* cdf.(res_wage, view(firm_wage, wage_sortperm))
    @views avail_labor[1:(end-1)] .= avail_labor[1:(end-1)] .- avail_labor[2:end]
    ###Number of firms competing for labor for each intervals
    firm_n_comp::Array{Float64, 1} = cumsum(@view recruiting[wage_sortperm])
    ###Labor available to each firm
    avail_labor_firm::Array{Float64, 1} = Array{Int64, 1}(undef, length(firm_wage))
    ###Stores how much labor will be removed from the intervals at each iteration
    iter_hired_labor::Array{Float64, 1} = Array{Int64, 1}(undef, length(firm_wage))
    ##To be used to find stopping condition (finds first recruiting firm)
    first_rec = findfirst_default(x -> x > 0.0, firm_n_comp)

    ##Only start the loop if someone obtained something (meaning that there is population and that there are reservations that are low enough)
    while first_rec <= findlast_default(x -> x > tol, avail_labor)
        ###Re-initialize hired vector
        iter_hired_labor .= 0.0
        ##Quantity of labor available to each individual firm
        @views avail_labor_firm[first_rec:end] .= avail_labor[first_rec:end] ./ firm_n_comp[first_rec:end]

        ###Assign labor and update available
        @views for f=1:length(firm_wage) ; if recruiting[f]
            ##How much labor is available for the firm
            firm_tot_lab_avail = sum(avail_labor_firm[wage_order[f]:end])
            ##Share of labor available to firm that is taken
            share_labor_taken = min((firm_labor_demand[f] - firm_labor_obt[f]) / firm_tot_lab_avail, 1.0)
            ##Update quantity of labor obtained by f
            firm_labor_obt[f] += firm_tot_lab_avail * share_labor_taken
            ##Update quantity of hired labor for each interval
            iter_hired_labor[wage_order[f]:length(firm_wage)] .+= share_labor_taken .* avail_labor_firm[wage_order[f]:length(firm_wage)]
        end ; end
        ##Update available labor (max is to correct numerical errors)
        avail_labor .= max.(avail_labor .- iter_hired_labor, 0.0)

        #Update recruiting firms
        subset_assign!(recruiting, (f for f=1:length(firm_wage) if recruiting[f]),
                       (x, y, t) -> x <= (1.0 - t) * y, firm_labor_obt, firm_labor_demand, ScalarArray(tol))

        ##Update number of competitors
        cumsum!(firm_n_comp, @view recruiting[wage_sortperm])
        ###Find first index of recruiting in terms of sorted wages (use as a condition for stopping)
        first_rec = findfirst_default(x -> x > 0.0, firm_n_comp)
    end

    ###Correct numerical errors
    firm_labor_obt .= min.(firm_labor_obt, firm_labor_demand) .* (firm_labor_obt .> 0.0)

    return firm_labor_obt
end


##Function that assigns labor to firms following a CES-like formula
function assign_labor_CES(pop_man::Float64, wage_sens::Real, firm_wage::AbstractArray{Float64, 1}, firm_labor_demand::AbstractArray{Float64, 1}, 
                          tol::Float64=10^-9)
    ##Return empty array if no firms compete
    length(firm_labor_demand) == 0 && return Float64[]    
    
    ###Available labor
    labor_avail::Float64 = pop_man
    #Non-recruiting firms
    recruiting::Array{Bool, 1} = firm_labor_demand .> tol
    ###Proportion of labor choosing each firm
    firm_labor_obt::Array{Float64, 1} = fill(0.0, length(firm_labor_demand))
    
    ##Only start the loop if someone obtained something (meaning that there is population and that there are reservations that are low enough)
    while any(recruiting) && labor_avail > tol
        ###Labor available to each firm
        firm_labor_obt .+= labor_avail * recruiting .* firm_wage.^wage_sens / 
                           sum(x -> x[1]*x[2]^wage_sens, zip(recruiting, firm_wage))
        ###Correct obtained for firm labor demand
        firm_labor_obt .= min.(firm_labor_obt, firm_labor_demand)
        ##Update available labor
        labor_avail = max(labor_avail - sum(firm_labor_obt), 0.0)
        #Update recruiting firms
        subset_assign!(recruiting, (f for f=1:length(firm_wage) if recruiting[f]),
                       (x, y) -> (1.0 - x / y) >= tol, firm_labor_obt, firm_labor_demand)
    end
    return firm_labor_obt
end