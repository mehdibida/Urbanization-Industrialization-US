#####Contains the functions that update the model
using Distributions: rand!, shuffle!, Truncated, Exponential, Normal, LocationScale,
                     cdf, Beta, Categorical, Gamma, mean, Binomial, TriangularDist, Multinomial
using Base.Iterators: cycle, zip, product as iterprod
using Random: randexp!, randexp

function grow_pop!(SE::SpatialEconomy)
    SE.loc_pop .*= 1.0 + SE.pop_grth_rate
    for l=1:(3*SE.N_location); reweight!(SE.loc_income[l], 1.0 + SE.pop_grth_rate); end
    SE.firm_assets[SE.firm_active] .*= 1.0 + SE.pop_grth_rate
end

###Updates the distance between locations
###Default values correspond to a factor 15 reduction after 240 steps and a factor 20 after 480
function update_distance!(SE::SpatialEconomy, red_coef::Real, final_red_factor::Real)
    ##red_coef is the distance reduction coefficient
    SE.loc_dist .= red_coef .* SE.loc_dist .+ SE.loc_dist_init .* final_red_factor .* (1.0 - red_coef)
end

####Update distance given location distance matrix and parents in shortest path
function update_distance!(SE::SpatialEconomy, node_dist::AbstractArray{Float64, 2},
                          node_parent, nodes_exclude::Set{Int64})

    SE.loc_dist .= node_dist

    ##Put distances to scale
    SE.loc_dist .*= SE.dist_init_cost

    ###Stores nodes on shortest paths
    empty!(SE.loc_sp_node)
    for (c, v) in combinations(1:SE.N_location, 2)
        push!(SE.loc_sp_node, (c, v, sp_nodes(v, view(node_parent, :, c), nodes_exclude)))
    end

    return nothing
end

###### Adds new firms to the economy
## possible n_potential new firms per location can enter each step
function firms_entry!(SE::SpatialEconomy, n_potential::Int64, firm_min_capital::Float64, iter::Int64)   
    #####Probability varies depending on location
    #=
    mean_pot_rec_pop = sum(SE.share_agr_to_man * max(SE.loc_pop[3*l-2] - SE.pop_fixed, 0.0) + SE.loc_pop[3*l-1] for l=1:SE.N_location)
    mean_wage = sum(SE.share_agr_to_man * max(SE.loc_pop[3*l-2] - SE.pop_fixed, 0.0) * 
                    (SE.loc_housing_price[l] * SE.housing_sat + SE.agr_end_share * SE.agr_sat * (1.0 + SE.distr_markup) + SE.loc_avg_agr_wage_prev[l]) +
                    SE.loc_pop[3*l-1] * SE.loc_avg_man_wage_prev[l] for l=1:SE.N_location) / mean_pot_rec_pop
    mean_pot_rec_pop /= sum((SE.loc_pop[3*l-2] - SE.pop_fixed > 0.0) | (SE.loc_pop[3*l-1] > 0.0) for l=1:SE.N_location)
    ###Probability that each location is chosen by entering firms
    loc_prob_chosen = [(SE.share_agr_to_man * max(SE.loc_pop[3*l-2] - SE.pop_fixed, 0.0) + SE.loc_pop[3*l-1]) / mean_pot_rec_pop / 
                       (1.0 + SE.loc_avg_man_wage_prev[l] / mean_wage) for l=1:SE.N_location]
    loc_prob_chosen .+= 0.1*mean(loc_prob_chosen) ###To give a chance to other locations
    loc_prob_chosen ./= sum(loc_prob_chosen)
    =#
    ###Market potential
    loc_prob_chosen = loc_man_qsold(SE)
    loc_prob_chosen ./= (ifelse(k > 0, k, 1.0) for k in loc_tot_cap(SE))
    loc_prob_chosen .+= SE.man_tech * firm_min_capital^(SE.man_ret_scale - 1.0)
    loc_prob_chosen .+= 0.1 / 0.9 * mean(loc_prob_chosen) ###To give a chance to other locations
    loc_prob_chosen ./= sum(loc_prob_chosen)
    rand!(Multinomial(n_potential, loc_prob_chosen), SE.loc_firm_entry)

    SE.loc_new_firm .= SE.loc_firm_entry .> 0

    ##Number of slots available to fill from the inactive firms slots
    n_slots_avail = isempty(SE.firm_assets) ? 0 : length(SE.firm_assets) - sum(x -> x >= 0.0, SE.firm_assets)
    slot_to_fill = 1

    ##Adding firms to the list
    n_to_add = sum(SE.loc_firm_entry) ##number of firms to add
    ent_loc = 1 ##To find location of next entrant
    
    ###Use firm_select to store indices of entrants
    empty!(SE.firm_select)

    ##Firm active will be updated once wages and capital are chosen
    while (n_to_add > 0) & (n_slots_avail > 0)
        ##Increment ent_index if the above finding process ended
        ent_loc = findnext(n -> n > 0, SE.loc_firm_entry, ent_loc)
        slot_to_fill = findnext(x -> x < 0.0, SE.firm_assets, slot_to_fill)
        push!(SE.firm_select, slot_to_fill)
        SE.loc_firm_entry[ent_loc] -= 1
        SE.firm_loc[slot_to_fill] = ent_loc ##Location
        SE.firm_entry_time[slot_to_fill] = iter ### Entry time
        SE.firm_capital[slot_to_fill] = 0.0 ###To be changed below
        SE.firm_q_sold[slot_to_fill] = 0.0 ### total sold quantity
        SE.firm_wage[slot_to_fill] = 0.0 ### Firm wage
        SE.firm_labor_dem[slot_to_fill] = 0.0 ### Firm labor demand
        SE.firm_labor_use[slot_to_fill] = 0.0 ### Firm current labor usage 
        SE.firm_price[slot_to_fill] = 1.0 ### Firm price
        SE.firm_assets[slot_to_fill] = 0.0 ### Firm equity (financial assets)
        n_to_add -= 1
        n_slots_avail -= 1 ##index for finding current slot of inactive firm
    end
    ###Reinitialize matrices of new entrants
    SE.firm_loc_price[SE.firm_select, :] .= 1.0
    SE.firm_loc_prod[SE.firm_select, :] .= 0.0
    SE.firm_loc_ship[SE.firm_select, :] .= 0.0

    ###Total number of firms to add as an extension of the current vectors
    total_out_to_add = n_to_add
    if n_to_add > 0
        new_start_ind = length(SE.firm_assets) + 1
        ##Firms location
        resize!(SE.firm_loc, length(SE.firm_loc) + n_to_add)
        while n_to_add > 0
            ent_loc = findnext(n -> n > 0, SE.loc_firm_entry, ent_loc)
            SE.firm_loc[new_start_ind:(new_start_ind + SE.loc_firm_entry[ent_loc] - 1)] .= ent_loc
            new_start_ind += SE.loc_firm_entry[ent_loc]
            n_to_add -= SE.loc_firm_entry[ent_loc]
            SE.loc_firm_entry[ent_loc] = 0
        end
    end
    
    ##Update number of firms to add and number of available slots
    ##Extend vectors if no slots are available
    if total_out_to_add > 0
        ##Firms location: no update needed since already done in previous loop
        extend_to_val!(SE.firm_entry_time, total_out_to_add, iter) ### Entry time
        extend_to_val!(SE.firm_capital, total_out_to_add, 0.0) ##Firm capital
        extend_to_val!(SE.firm_q_sold, total_out_to_add, 0.0) ### total sold quantity
        extend_to_val!(SE.firm_wage, total_out_to_add, 0.0) ### Firm wage
        extend_to_val!(SE.firm_labor_dem, total_out_to_add, 0.0) ##Firm labor demand
        extend_to_val!(SE.firm_labor_use, total_out_to_add, 0.0) ###Firm current labor usage 
        extend_to_val!(SE.firm_price, total_out_to_add, 1.0) ###Firm price
        extend_to_val!(SE.firm_assets, total_out_to_add, 0.0) ###Firm equity (financial assets)
        #Shipments to each location
        SE.firm_loc_price = vcat(SE.firm_loc_price, fill(1.0, total_out_to_add, size(SE.firm_loc_price, 2)))
        SE.firm_loc_prod = vcat(SE.firm_loc_prod, fill(0.0, total_out_to_add, size(SE.firm_loc_prod, 2)))
        SE.firm_loc_ship = vcat(SE.firm_loc_ship, fill(0.0, total_out_to_add, size(SE.firm_loc_ship, 2)))
    end

    ########## Choice of capital for entering firms ###########
    ############################################################
    ####Computing number of firms in each location
    #Select active firms that entered in the previous iteration to base new capitals on local average
    empty!(SE.firm_select)
    append!(SE.firm_select, f for f in SE.firm_active if SE.firm_entry_time[f] < iter)

    ##Choosing capital and wages for new entrants
    for e_l=1:SE.N_location
        pop_recruitable = SE.loc_pop[3*e_l-1] + SE.share_agr_to_man * max(SE.loc_pop[3*e_l-2] - SE.pop_fixed, 0.0)
        if SE.loc_new_firm[e_l] & (pop_recruitable > firm_min_capital) ##Locations with population
            ##Add new firms to list of active firms
            append!(SE.firm_active, f for f=1:length(SE.firm_loc) if (SE.firm_loc[f] == e_l) & (SE.firm_entry_time[f] == iter))
            ##Choice of capital and wages
            if any(SE.firm_loc[f] == e_l for f in SE.firm_select) ##Case there are incumbent firms present
                ############### Capital Choice
                ###Find minimum, maximum and average capital to use them as reference
                loc_max_cap = maximum(SE.firm_capital[f] for f in SE.firm_select if SE.firm_loc[f] == e_l)
                #n_new_firm = count((SE.firm_loc[f] == e_l) & (SE.firm_entry_time[f] == iter) for f in SE.firm_active)
                for w_f in SE.firm_active ; if ((SE.firm_loc[w_f] == e_l) & (SE.firm_entry_time[w_f] == iter))
                    ###Choice of capital
                    #cap_choice = rand()
                    SE.firm_capital[w_f] = max(firm_min_capital, rand(Beta(1.0, 3.0)) * min(1.85 * loc_max_cap, pop_recruitable))
                    ###find closest neighbor in terms of capital
                    f_closest = fargmin(x -> ifelse(SE.firm_loc[x] == e_l, abs(SE.firm_capital[x] - SE.firm_capital[w_f]), Inf64),
                                        SE.firm_select)
                    SE.firm_wage[w_f] = SE.firm_wage[f_closest] * 
                                        (1.0 + runif(0.0, 0.25))^sign(SE.firm_capital[w_f] - SE.firm_capital[f_closest])
                end ; end
            else ###Case where no incumbent firms are present
                ###Number of new firms appearing (needed for capital calculation)
                #n_new_firm = count((SE.firm_loc[f] == e_l) & (SE.firm_entry_time[f] == iter) for f in SE.firm_active)
                for w_f in SE.firm_active ; if ((SE.firm_loc[w_f] == e_l) & (SE.firm_entry_time[w_f] == iter))
                    ####Wage should be larger than wages in agriculture in order to attract workers
                    SE.firm_wage[w_f] = (SE.loc_pop[3*e_l-1] * SE.loc_avg_man_wage_prev[e_l] +
                                         SE.share_agr_to_man * max(SE.loc_pop[3*e_l-2] - SE.pop_fixed, 0.0) * 
                                         (SE.loc_housing_price[e_l] * SE.housing_sat + SE.agr_end_share * SE.agr_sat * (1.0 + SE.distr_markup) + SE.loc_avg_agr_wage_prev[e_l])) /
                                        pop_recruitable * 1.2^runif(-1.0, 1.0)
                    ####Firm capital takes into account the availability of labor
                    #cap_choice = rand(firm_cap_select)
                    SE.firm_capital[w_f] = max(firm_min_capital, rand(Beta(1.0, 3.0)) * pop_recruitable)
                    #min(max_cap_inc, max_cap_sat, SE.loc_pop[4*e_l-2] / n_new_firm) ###Take into account labor availability
                end ; end
            end
        end
    end
    ##Sort firm active for faster iteration and to use keepat!
    sort!(SE.firm_active)
    allunique(SE.firm_active) || error("firms_entry!: active firm indices are not unique.")

    ###Try to clean if vectors get too big (to see if it works faster)
    if 1.25 * length(SE.firm_active) < length(SE.firm_loc)
        keepat!(SE.firm_loc, SE.firm_active);
        keepat!(SE.firm_entry_time, SE.firm_active);
        keepat!(SE.firm_capital, SE.firm_active); 
        keepat!(SE.firm_q_sold, SE.firm_active);
        keepat!(SE.firm_wage, SE.firm_active);
        resize!(SE.firm_labor_dem, length(SE.firm_active));
        keepat!(SE.firm_labor_use, SE.firm_active);
        keepat!(SE.firm_price, SE.firm_active);
        keepat!(SE.firm_assets, SE.firm_active);
        SE.firm_loc_price = SE.firm_loc_price[SE.firm_active, :]
        SE.firm_loc_prod = SE.firm_loc_prod[SE.firm_active, :]
        SE.firm_loc_ship = SE.firm_loc_ship[SE.firm_active, :]
        SE.firm_active .= 1:length(SE.firm_active)
    end

    return nothing
end


##### Producers decide on production quantities and wages
function set_quantities_wages!(SE::SpatialEconomy, curr_iter::Int64, wage_sens::Float64, q_s_agr::Float64, q_s_man::Float64, expl_share::Float64)

    ###Change loc_agr_ship so that it accounts for previous manufacturing quantities sold 
    ###This will be used to determine the production of new firms
    SE.loc_agr_man_ship .-= SE.loc_agr_ship

    ###Indices of non inhabited locations
    non_inh_ind = [l for l=1:SE.N_location if !(SE.loc_inhabited[3*l-2] | SE.loc_inhabited[3*l-1] | SE.loc_inhabited[3*l])]
    ###Agriculture sector sets quantities for the next period
    for (a, l) in iterprod(SE.loc_agr_inh_ind, SE.loc_any_inh_ind)
        if SE.loc_agr_prod[a, l] > SE.tol
            SE.loc_agr_prod[a, l] = ifelse(SE.loc_agr_ship[a, l] >= (1.0 - SE.tol) * SE.loc_agr_prod[a, l], 
                                           (1.0 + rand() * q_s_agr) * SE.loc_agr_prod[a, l], 
                                           (1.0 - q_s_agr) * SE.loc_agr_ship[a, l] + q_s_agr * SE.loc_agr_prod[a, l])
        end
    end
    ##No production to uninhabited locations
    SE.loc_agr_prod[SE.loc_agr_inh_ind, non_inh_ind] .= 0.0
    ####Explore markets and adjust quantities w.r.t. production capacities
    for a in SE.loc_agr_inh_ind
        n_markets_a_absent = sum(SE.loc_agr_prod[a, l] <= SE.tol for l in SE.loc_any_inh_ind)
        if n_markets_a_absent > 0
            a_expl_quant = expl_share / (1.0 - expl_share) * sum(SE.loc_agr_prod[a, l] for l in SE.loc_any_inh_ind) / n_markets_a_absent
            for l in SE.loc_any_inh_ind; if SE.loc_agr_prod[a, l] <= SE.tol
                SE.loc_agr_prod[a, l] = a_expl_quant * rand()
            end; end
        end
    end
    ################# Firms decide on next production quantities
    ############## Incumbent firms decide on wages ##################
    loc_sold_prev = map(x -> any(y -> y > 0.0, x), eachrow(SE.loc_agr_man_ship))

    for (f, l) in iterprod(SE.firm_active, SE.loc_any_inh_ind)
        if SE.firm_q_sold[f] > SE.tol
            if SE.firm_loc_prod[f, l] > SE.tol
                SE.firm_loc_prod[f, l] = ifelse(SE.firm_loc_ship[f, l] >= (1.0 - SE.tol) * SE.firm_loc_prod[f, l],
                                                (1.0 + rand() * q_s_man) * SE.firm_loc_prod[f, l],
                                                (1.0 - q_s_man) * SE.firm_loc_ship[f, l] + q_s_man * SE.firm_loc_prod[f, l])
            end
        else
            #if SE.firm_loc[f] == l
                SE.firm_loc_prod[f, l] = loc_sold_prev[l] ? 
                                         SE.loc_agr_man_ship[SE.firm_loc[f], l] * runif(0.8, 1.2) : 
                                         ifelse(SE.firm_loc[f] == l, 1.0, 0.0)
                ##This will be corrected later according to total production capacity
            #end
        end
    end
    ##No production to uninhabited locations
    SE.firm_loc_prod[SE.firm_active, non_inh_ind] .= 0.0
    ##Firms explore new markets
    for f in SE.firm_active
        n_markets_f_absent = sum(SE.firm_loc_prod[f, l] <= SE.tol for l in SE.loc_any_inh_ind)
        if n_markets_f_absent > 0
            f_expl_quant = expl_share / (1.0 - expl_share) * sum(SE.firm_loc_prod[f, l] for l in SE.loc_any_inh_ind) / n_markets_f_absent
            for l in SE.loc_any_inh_ind; if SE.firm_loc_prod[f, l] <= SE.tol
                SE.firm_loc_prod[f, l] = f_expl_quant * rand()
            end; end
        end
        ###For new entrants or non selling, adjust production so that it equals total capacity
        if SE.firm_q_sold[f] <= SE.tol
            SE.firm_loc_prod[f, :] .*= firm_max_prod(SE.firm_capital[f], SE.man_ret_scale, SE.man_tech) /
                                       sum(SE.firm_loc_prod[f, l] for l in SE.loc_any_inh_ind)
        end
        ###Firms adjust their labor demand and decide on wages
        SE.firm_labor_dem[f] = firm_labor_demand(sum(SE.firm_loc_prod[f, l] for l in SE.loc_any_inh_ind), SE.firm_capital[f], SE.man_ret_scale, SE.man_tech)
        ###Incumbent firms multiply their previous wage by 1 ± sens * unif(0,1) depending on whether they recruited what they needed or not
        if SE.firm_entry_time[f] < curr_iter
            SE.firm_wage[f] *= (1.0 + rand() * SE.wage_sens)^ifelse(SE.firm_labor_use[f] >= (1.0 - SE.tol) * SE.firm_labor_dem[f], .-1.0, 1.0)
        end
    end
    return nothing
end

function match_labor!(SE::SpatialEconomy)
    #### Labor matching and production
    for c=1:SE.N_location
        if any(x -> x[1] == x[2], zip(view(SE.firm_loc, SE.firm_active), cycle(c)))
            ###select relevant firms
            empty!(SE.firm_select)
            append!(SE.firm_select, f for f in SE.firm_active if SE.firm_loc[f] == c)
            ##Reinitialize obtained labor
            SE.firm_labor_use[SE.firm_select] .= 0.0
            ##Matching
            #res_wage = SE.loc_housing_price[c]*SE.housing_sat + SE.agr_sat * (1.0 + SE.distr_markup)
            res_wage = SE.loc_avg_man_wage_prev[c] #max(res_wage, SE.loc_avg_man_wage_prev[c])
            if SE.loc_pop[3*c-1] > SE.tol
                assign_labor!(view(SE.firm_labor_use, SE.firm_select),
                              SE.loc_pop[3*c-1],
                              #Exponential(SE.tol + SE.loc_avg_man_wage_prev[c]),
                              Truncated(Normal(res_wage, SE.tol + 0.15 * res_wage), 0.0, Inf64),
                              view(SE.firm_wage, SE.firm_select),
                              view(SE.firm_labor_dem, SE.firm_select), 
                              SE.tol)
            else
                SE.firm_labor_use[SE.firm_select] .= 0.0
            end
            any(x -> x[1] > x[2], zip(view(SE.firm_labor_use, SE.firm_select), view(SE.firm_labor_dem, SE.firm_select))) && error("Obtained > demanded")
            SE.loc_unempl[c] = max(SE.loc_pop[3*c-1] - sum(view(SE.firm_labor_use, SE.firm_select)), 0.0)
        else
            SE.loc_unempl[c] = SE.loc_pop[3*c-1]
        end
    end
    return nothing
end

function match_labor_from_agr!(SE::SpatialEconomy)
    reorder = false ###Should location indices be reordered ? In case some new are added
    #### Labor matching and production
    for c=1:SE.N_location
        if ((SE.share_agr_to_man * max(SE.loc_pop[3*c-2] - SE.pop_fixed, 0.0)) > SE.tol) &
           any((SE.firm_loc[f] == c) & (SE.firm_labor_dem[f] > SE.tol) for f in SE.firm_active)
            ###select relevant firms
            empty!(SE.firm_select)
            append!(SE.firm_select, f for f in SE.firm_active if (SE.firm_loc[f] == c) & (SE.firm_labor_dem[f] > SE.tol))
            ###To calculate how much was recruited from agriculture
            share_recruited = sum(SE.firm_labor_use[i] for i in SE.firm_select)
            ##Matching
            assign_labor!(view(SE.firm_labor_use, SE.firm_select),
                          SE.share_agr_to_man * max(SE.loc_pop[3*c-2] - SE.pop_fixed, 0.0),
                          LocationScale(SE.loc_housing_price[c]*SE.housing_sat + SE.agr_end_share * SE.agr_sat * (1.0 + SE.distr_markup), 
                                        1.0, Gamma(2.0, SE.tol + 0.5*SE.loc_avg_agr_wage_prev[c])),
                          view(SE.firm_wage, SE.firm_select),
                          view(SE.firm_labor_dem, SE.firm_select),
                          SE.tol)

            any(x -> x[1] > x[2], zip(view(SE.firm_labor_use, SE.firm_select), view(SE.firm_labor_dem, SE.firm_select))) && error("Obtained > demanded")
            
            share_recruited = (sum(SE.firm_labor_use[i] for i in SE.firm_select) - share_recruited) / SE.loc_pop[3*c-2]
            if share_recruited > 0.0
                ###Update populations assuming that wealth is independent from reservation wage
                add_from_IDB!(SE.loc_income[3*c-1], SE.loc_income[3*c-2], share_recruited)
                reweight!(SE.loc_income[3*c-2], (1.0 - share_recruited))
                ###Recompute populations
                SE.loc_pop[3*c-2] = sum_weights(SE.loc_income[3*c-2])
                SE.loc_pop[3*c-1] = sum_weights(SE.loc_income[3*c-1])
                ##Update active locations (agriculture remains always active because of the fixed_pop threshold)
                if (SE.loc_pop[3*c-1] > 0.0) & (!SE.loc_inhabited[3*c-1])
                    push!(SE.loc_man_inh_ind, c)
                    push!(SE.loc_inh_ind, 3*c-1)
                    SE.loc_inhabited[3*c-1] = true
                    reorder = true
                end
            elseif share_recruited < 0.0
                error("Share recruted at $c should be non-negative. Obtained: $share_recruited.")
            end
        end
    end
    if reorder
        sort!(SE.loc_man_inh_ind)
        sort!(SE.loc_inh_ind)
    end
    return nothing
end


function match_labor_CES!(SE::SpatialEconomy, w_sens::Real)
    #### Labor matching and production
    for c=1:SE.N_location
        if any(x -> x[1] == x[2], zip(view(SE.firm_loc, SE.firm_active), cycle(c)))
            ###select relevant firms
            empty!(SE.firm_select)
            append!(SE.firm_select, f for f in SE.firm_active if SE.firm_loc[f] == c)

            ##Matching
            SE.firm_labor_use[SE.firm_select] = assign_labor_CES(SE.loc_pop[3*c-1],
                                                                 w_sens, 
                                                                 view(SE.firm_wage, SE.firm_select),
                                                                 view(SE.firm_capital, SE.firm_select), 
                                                                 10.0^-5)

            SE.loc_unempl[c] = max(SE.loc_pop[3*c-1] - sum(view(SE.firm_labor_use, SE.firm_select)), 0.0)
        else
            SE.loc_unempl[c] = SE.loc_pop[3*c-1]
        end
    end
    return nothing
end


###Updates SE with firms production
function production!(SE::SpatialEconomy)
    ### Production
    for f in SE.firm_active
        if SE.firm_labor_use[f] <= SE.tol
            SE.firm_loc_prod[f, SE.loc_any_inh_ind] .= 0.0
        else 
            f_planned_prod = sum(SE.firm_loc_prod[f, l] for l in SE.loc_any_inh_ind)
            f_eff_prod = firm_production(SE.firm_labor_use[f], SE.firm_capital[f], SE.man_ret_scale, SE.man_tech)
            if f_eff_prod < (1.0 - SE.tol) * f_planned_prod
                SE.firm_loc_prod[f, SE.loc_any_inh_ind] .*= f_eff_prod / f_planned_prod
            end
        end
    end
    ###Agricultural sector adjusts its production according to labor available
    for a in SE.loc_agr_inh_ind
        agr_prod_cap = max(0.0, SE.agr_tech * SE.loc_pop[3*a-2]^SE.agr_ret_scale - SE.agr_end_share * SE.agr_sat * SE.loc_pop[3*a-2])
        tot_agr_plan = sum(SE.loc_agr_prod[a, l] for l in SE.loc_any_inh_ind)
        if tot_agr_plan > agr_prod_cap
            SE.loc_agr_prod[a, :] .*= agr_prod_cap / tot_agr_plan
        end
    end
    ### Update list of firms that are active and produced
    empty!(SE.firm_act_prod)
    append!(SE.firm_act_prod, f for f in SE.firm_active if SE.firm_labor_use[f] > SE.tol)
    return nothing
end

function firms_set_prices!(SE::SpatialEconomy, share_assets_tc)
    SE.firm_price .= 1.0 ###Just a value so that we don't have Inf*0.0 when computing incomes for non producing firms
    SE.firm_loc_price .= 1.0 ###Just a value so that we don't have Inf*0.0 when computing incomes for non producing firms

    ###Determine selling price for the firms that produced
    for f in SE.firm_act_prod
        ##Firms prices before export
        f_tc = SE.firm_labor_use[f] * SE.firm_wage[f] + SE.c_capital * SE.firm_capital[f]
        f_tot_prod = firm_production(SE.firm_labor_use[f], SE.firm_capital[f], SE.man_ret_scale, SE.man_tech)
        SE.firm_price[f] = f_tc / f_tot_prod * (1.0 + min(SE.markup, max(0.0, share_assets_tc - SE.firm_assets[f] / f_tc)))
    end
    ##Firms prices after export
    for l in SE.loc_any_inh_ind
        for f in SE.firm_act_prod
            SE.firm_loc_price[f, l] = (SE.firm_price[f] + SE.loc_dist[SE.firm_loc[f], l]) * (1.0 + SE.distr_markup)
        end
    end
    
    return nothing
end

function consume_housing!(SE::SpatialEconomy)
    for l=1:(3*SE.N_location)
        SE.loc_real_h_cons[l] .= 0.0
    end
    ##initialize spending to 0
    SE.loc_hous_spending .= 0.0
    
    ### Housing consumption and Income Drawing
    for l in SE.loc_inh_ind
        ####Draw incomes
        correct_mean!(rand!(SE.loc_income[l], SE.loc_real_inc[l]), mean(SE.loc_income[l]))
        ####Housing consumption (only for urban sectors)
        if (l % 3) != 1 ##case: manufacturing and services
            hp_ind = 1 + (l - 1) ÷ 3
            ###Update consumption
            SE.loc_real_h_cons[l] .= min.((SE.housing_share / SE.loc_housing_price[hp_ind]) .* SE.loc_real_inc[l], SE.housing_sat)
            ###Update location total spending on housing
            SE.loc_hous_spending[l] = sum(x -> x[1]*x[2], zip(SE.loc_real_h_cons[l], cycle(SE.loc_housing_price[hp_ind]))) *
                                      SE.loc_pop[l] / length(SE.loc_real_h_cons[l])
            ###Update income
            SE.loc_real_inc[l] .= max.((1.0 - SE.housing_share) .* SE.loc_real_inc[l], 
                                       SE.loc_real_inc[l] .- (SE.loc_housing_price[hp_ind] .* SE.housing_sat))
        else ##case: rural area 
            ###Update consumption
            SE.loc_real_h_cons[l] .= SE.housing_sat
        end
    end

    return nothing

end

function consume_agr_rq!(SE::SpatialEconomy, res_inertia::Float64)
    ###reinitialize realizations and flows
    for l=1:(3*SE.N_location)
        #Choices are reinitialized by the consume function
        SE.loc_real_a_cons[l] .= ifelse(l % 3 == 1, 
                                        min(SE.agr_end_share * SE.agr_sat, SE.agr_tech * SE.loc_pop[l]^(SE.agr_ret_scale-1.0)), 
                                        0.0)
    end
    for l in SE.loc_inh_ind
        SE.loc_real_consume[l] .= true
    end
    #SE.loc_agr_spending .= 0.0
    SE.loc_agr_ship .= 0.0

    ###Sample reservation quantities
    for l in SE.loc_inh_ind
        rand!(Truncated(Normal(SE.loc_agr_avg_cons_prev[(l - 1) ÷ 3 + 1], 
                               SE.tol + 0.15 * SE.loc_agr_avg_cons_prev[(l - 1) ÷ 3 + 1]), 
                        0.0, SE.agr_sat), SE.loc_real_rq[l])
    end
    
    ####No need to sort incomes since housing consumption preserves order
    ############### Agriculture   #############
    ###Prepare reservation price for update
    SE.loc_agr_avg_cons_prev .*= res_inertia

    l_inh = Int64[] ##inhabited "sectors" in l
    ##Allocate vectors to make consumption function more efficient
    sell_ind = Array{Int64, 1}(undef, length(SE.loc_agr_inh_ind)) ##Indices of active sellers
    sell_dem_share = fill(0.0, length(SE.loc_agr_inh_ind)) ##Share of demand satisfied by sellers
    sell_buy_dem = Array{Float64, 1}(undef, length(SE.loc_agr_inh_ind)) ##Quantities demanded to sellers at each iteration
    
    for l in SE.loc_any_inh_ind
        append!(l_inh, il for il=(3*l-2):(3*l) if SE.loc_inhabited[il])

        realize_cons_rq_simpler!(view(SE.loc_real_consume, l_inh),
                                 view(SE.loc_real_inc, l_inh), ###Incomes do not need to be in increasing order
                                 view(SE.loc_real_a_cons, l_inh),
                                 view(SE.loc_real_rq, l_inh),
                                 view(SE.loc_pop, l_inh),
                                 SE.agr_sat,
                                 view(SE.loc_real_choice, l_inh),
                                 view(SE.loc_agr_prod, SE.loc_agr_inh_ind, l), #Is not changed by the function
                                 (1.0 .+ view(SE.loc_dist, SE.loc_agr_inh_ind, l)) .* (1.0 + SE.distr_markup),
                                 view(SE.loc_agr_ship, SE.loc_agr_inh_ind, l),
                                 SE.tol,
                                 sell_ind, sell_dem_share, sell_buy_dem)
        
        #Reinitialize vector of satisfied demand shares (the two others are reinitialized by the function)
        sell_dem_share .= 0.0
        ###Update average consumptions
        SE.loc_agr_avg_cons_prev[l] += (1.0 - res_inertia) * 
                                        sum(view(SE.loc_agr_ship, SE.loc_agr_inh_ind, l)) / 
                                        sum(SE.loc_pop[il] for il in l_inh)
        empty!(l_inh)
    end

    return nothing
end

function consume_man_rq!(SE::SpatialEconomy, res_inertia::Float64)
    ###reinitialize realizations and flows
    for l=1:(3*SE.N_location)
        #Choices are reinitialized by the consume function
        SE.loc_real_m_cons[l] .= 0.0
    end

    SE.firm_loc_ship .= 0.0

    ##Only consumer if there is production
    if length(SE.firm_act_prod) > 0
        ##Threshold starting from which we consider that satiation of agricultural products is reached
        a_cons_threshold = (1.0 - SE.tol) * SE.agr_sat
        h_cons_threshold = (1.0 - SE.tol) * SE.housing_sat
        ##Update buyers that consume based on previous consumption of agriculture
        for l in SE.loc_inh_ind
            SE.loc_real_consume[l] .= (SE.loc_real_a_cons[l] .>= a_cons_threshold) .& (SE.loc_real_h_cons[l] .>= h_cons_threshold)
        end

        ###Sample reservation quantities
        for l in SE.loc_inh_ind
            rand!(Truncated(Normal(SE.loc_man_avg_cons_prev[(l - 1) ÷ 3 + 1], 
                                   SE.tol + 0.15 * SE.loc_man_avg_cons_prev[(l - 1) ÷ 3 + 1]), 
                            0.0, SE.man_sat), SE.loc_real_rq[l])
        end

        SE.loc_man_avg_cons_prev .*= res_inertia 

        ####Consumption and reservations update
        l_inh = Int64[]; ##inhabited "sectors" in l
        ##Allocate vectors to make consumption function more efficient
        sell_ind = Array{Int64, 1}(undef, length(SE.firm_act_prod)) ##Indices of active sellers
        sell_dem_share = fill(0.0, length(SE.firm_act_prod)) ##Share of demand satisfied by sellers
        sell_buy_dem = Array{Float64, 1}(undef, length(SE.firm_act_prod)) ##Quantities demanded to sellers at each iteration

        for l in SE.loc_any_inh_ind
            append!(l_inh, il for il=(3*l-2):(3*l) if SE.loc_inhabited[il])

            realize_cons_rq_simpler!(view(SE.loc_real_consume, l_inh),
                                     view(SE.loc_real_inc, l_inh), ###Incomes do not need to be in increasing order
                                     view(SE.loc_real_m_cons, l_inh),
                                     view(SE.loc_real_rq, l_inh),
                                     view(SE.loc_pop, l_inh),
                                     SE.man_sat,
                                     view(SE.loc_real_choice, l_inh),
                                     view(SE.firm_loc_prod, SE.firm_act_prod, l), #Is not changed by the function
                                     view(SE.firm_loc_price, SE.firm_act_prod, l),
                                     view(SE.firm_loc_ship, SE.firm_act_prod, l),
                                     SE.tol,
                                     sell_ind, sell_dem_share, sell_buy_dem)

            #Reinitialize vector of satisfied demand shares (the two others are reinitialized by the function)
            sell_dem_share .= 0.0
            ###Update average consumption
            SE.loc_man_avg_cons_prev[l] += (1.0 - res_inertia) *
                                           sum(view(SE.firm_loc_ship, SE.firm_act_prod, l))  /
                                           sum(SE.loc_pop[il] for il in l_inh)
            empty!(l_inh)
        end
    end

    return nothing
end


function cons_utility!(SE::SpatialEconomy, agr_threshold::Real=(1.0 - SE.tol) * SE.agr_sat, hous_threshold::Real=(1.0 - SE.tol) * SE.housing_sat, 
                       util_inert::Float64=0.0)
    ####Correct negative consumptions if they are under the threshold
    for l in SE.loc_inh_ind
        SE.loc_real_h_cons[l] .*= SE.loc_real_h_cons[l] .> 0
    end
    for l in SE.loc_inh_ind
        SE.loc_real_a_cons[l] .*= SE.loc_real_a_cons[l] .> 0
    end
    for l in SE.loc_inh_ind
        SE.loc_real_m_cons[l] .*= SE.loc_real_m_cons[l] .> 0
    end
    
    ##Reweight past utility
    SE.loc_avg_util .*= util_inert

    for l in SE.loc_inh_ind
        r_w = (1.0 - util_inert) / length(SE.loc_real_util[l])
        for b_r=1:length(SE.loc_real_util[l])
            ###Agriculture and housing
            SE.loc_real_util[l][b_r] = (SE.loc_real_h_cons[l][b_r] / SE.housing_sat)^SE.housing_share * 
                                       (SE.loc_real_a_cons[l][b_r] / SE.agr_sat)^SE.agr_share
            ###Manufacturing
            if (SE.loc_real_a_cons[l][b_r] >= agr_threshold) & (SE.loc_real_h_cons[l][b_r] >= hous_threshold)
                SE.loc_real_util[l][b_r] += log(1.0 + SE.loc_real_m_cons[l][b_r])
            end

            SE.loc_avg_util[l] += SE.loc_real_util[l][b_r] * r_w
        end
    end
    return nothing
end


function update_income_realizations!(SE::SpatialEconomy, hp_inertia::Float64=0.0, res_wage_inertia::Float64=0.0, loc_profit_inertia::Float64=0.0)
    ###Preallocate a vector to store wage realizations
    w_r_pre = Array{Float64, 1}(undef, length(SE.loc_real_inc[1]))
    ###Service sector: start with housing revenue
    @views SE.loc_ser_tot_inc .= SE.loc_hous_spending[2:3:(3*SE.N_location)] .+ SE.loc_hous_spending[3:3:(3*SE.N_location)]
    ####To compute income of services: agriculture transportation
    SE.loc_agr_man_ship .= 0.0
    @views SE.loc_agr_man_ship[SE.loc_agr_inh_ind, SE.loc_any_inh_ind] .= SE.loc_agr_ship[SE.loc_agr_inh_ind, SE.loc_any_inh_ind]
    ####To compute income of services: manufactring transportation and capital servicing
    for (f, l) in iterprod(SE.firm_active, SE.loc_any_inh_ind)
        SE.loc_agr_man_ship[SE.firm_loc[f], l] += SE.firm_loc_ship[f, l]
    end
    ####Update quantity sold by firms
    SE.firm_q_sold .= 0.0
    sumcols!(view(SE.firm_q_sold, SE.firm_act_prod), view(SE.firm_loc_ship, SE.firm_act_prod, SE.loc_any_inh_ind))
    ###Try with AR
    SE.loc_avg_man_wage_prev .*= res_wage_inertia ##Re-initialize average wages
    ###Updating average wages and income to services
    for f in SE.firm_active #####capital servicing cost goes to local service sector
        firm_tot_cost = SE.firm_wage[f] * SE.firm_labor_use[f] + SE.firm_capital[f] * SE.c_capital
        firm_asset = SE.firm_price[f] * SE.firm_q_sold[f] + SE.firm_assets[f]
        asset_to_cost = min(firm_asset / firm_tot_cost, 1.0)
        SE.loc_ser_tot_inc[SE.firm_loc[f]] += SE.c_capital * SE.firm_capital[f] * asset_to_cost
        ###Add wage to the percieved average wages vector
        if SE.firm_labor_use[f] > 0.0
            SE.loc_avg_man_wage_prev[SE.firm_loc[f]] += (1.0 - res_wage_inertia) * SE.firm_labor_use[f] * SE.firm_wage[f] * 
                                                        asset_to_cost / SE.loc_pop[3 * SE.firm_loc[f] - 1]
        end
    end
    ## Income from transportation to service sector
    for (c, v, nodes) in SE.loc_sp_node
        ##transportation earnings is dependent on the service population: we assume that they have more labor available
        ##for transportation
        s_t_cost = (SE.loc_agr_man_ship[c, v] + SE.loc_agr_man_ship[v, c]) *
                    SE.loc_dist[v, c] / sum(SE.loc_pop[3*n] for n in nodes)
        for n in nodes
            SE.loc_inhabited[3*n] && (SE.loc_ser_tot_inc[n] += s_t_cost * SE.loc_pop[3*n])
        end
    end
    ###Service income from distrbution
    for l in SE.loc_ser_inh_ind
        if !isempty(SE.loc_agr_inh_ind) ###From selling agriculture
            SE.loc_ser_tot_inc[l] += SE.distr_markup * 
                                     sum(SE.loc_agr_ship[sl, l] * (1.0 + SE.loc_dist[sl, l]) for sl in SE.loc_agr_inh_ind)
        end
        if !isempty(SE.firm_act_prod) ###From selling manufacturing
            SE.loc_ser_tot_inc[l] += SE.distr_markup / (1.0 + SE.distr_markup) *
                                     sum(SE.firm_loc_ship[f, l] * SE.firm_loc_price[f, l] for f in SE.firm_act_prod)
        end
    end
    
    ####Updating income distributions and housing price
    SE.loc_housing_price .*= hp_inertia ##Reinitialize housing prices
    agr_inc_all = 0.0
    wage_dist = Gamma(2.0, 1.0/2.0) ###Wage distribution (mean = 1) for sectors that are not modelled at micro-level
    ###Agriculture  
    for l in SE.loc_agr_inh_ind
        agr_tot_inc = sum(SE.loc_agr_ship[l, b] for b in SE.loc_any_inh_ind)
        ###Update expected wage (weighted mean of curr mean wage and past expectation)
        SE.loc_avg_agr_wage_prev[l] = res_wage_inertia * SE.loc_avg_agr_wage_prev[l] + 
                                      (1.0 - res_wage_inertia) * agr_tot_inc / SE.loc_pop[3*l-2]
        if agr_tot_inc > 0.0
            agr_inc_all += agr_tot_inc
            ###We assume that wages follow a Gamma distrbution (see Salem & Mount, 1974, McDonald & Ransom, 1979)
            ##update wage realizations
            SE.loc_real_inc[3*l-2] .+= agr_tot_inc ./ SE.loc_pop[3*l-2] .* 
                                       correct_mean!(rand!(wage_dist, w_r_pre), mean(wage_dist))
        end
    end
    ###Manufacturing
    for l in SE.loc_man_inh_ind
        n_firm_act_l = length(SE.firm_act_prod) > 0 ? sum(SE.firm_loc[f] == l for f in SE.firm_act_prod) : 0
        ##Only update wages if there are any paying firms
        if n_firm_act_l > 0
            ##Reinitialize wage distribution (We cheat to go faster)
            reinitialize!(SE.man_wage_dist)
            ##set unemployment
            if SE.loc_unempl[l] > 0
                add_point!(SE.man_wage_dist, 0.0, SE.loc_unempl[l])
            end
            for f in SE.firm_act_prod; if SE.firm_loc[f] == l
                firm_asset = SE.firm_price[f] * SE.firm_q_sold[f] + SE.firm_assets[f]
                firm_tot_cost = SE.firm_wage[f] * SE.firm_labor_use[f] + SE.firm_capital[f] * SE.c_capital
                ##set wage and employment
                add_point!(SE.man_wage_dist,
                           SE.firm_wage[f] * min(firm_asset / firm_tot_cost, 1.0), 
                           SE.firm_labor_use[f])
            end; end
            SE.loc_real_inc[3*l-1] .+= correct_mean!(rand!(SE.man_wage_dist, w_r_pre), mean(SE.man_wage_dist))
            ##Update housing price
            SE.loc_housing_price[l] += (1.0 - hp_inertia) * SE.housing_share *
                                       mean(SE.man_wage_dist) * SE.loc_pop[3*l-1] / (SE.loc_pop[3*l-1] + SE.loc_pop[3*l])
        end
    end
    ##Urban services
    for l in SE.loc_ser_inh_ind; if SE.loc_ser_tot_inc[l] > 0.0
        ###We assume that wages follow a Gamma distrbution (see Salem & Mount, 1974, McDonald & Ransom, 1979)
        ##update wage realizations
        SE.loc_real_inc[3*l] .+= SE.loc_ser_tot_inc[l] ./ SE.loc_pop[3*l] .* 
                                 correct_mean!(rand!(wage_dist, w_r_pre), mean(wage_dist))
        SE.loc_housing_price[l] +=  (1.0 - hp_inertia) * SE.housing_share *
                                    SE.loc_ser_tot_inc[l] / (SE.loc_pop[3*l-1] + SE.loc_pop[3*l]) ##Add to wage to housing price computation
        
    end; end
    ###Update firm profits and market exit
    ##Set equity of non active firms to a negative value
    n_firm_loc = count_firms_location(SE.N_location, view(SE.firm_loc, SE.firm_active))
    SE.loc_avg_firm_profit .*= loc_profit_inertia
    for f in SE.firm_active
        f_profit = SE.firm_q_sold[f] * SE.firm_price[f] - 
                   SE.c_capital * SE.firm_capital[f] -
                   SE.firm_wage[f] * SE.firm_labor_use[f]
        ##Update firm asset
        SE.firm_assets[f] += f_profit
        ###Update location average profit
        SE.loc_avg_firm_profit[SE.firm_loc[f]] += (1.0 - loc_profit_inertia) * f_profit / n_firm_loc[SE.firm_loc[f]]
    end

    ##Sorting vector of active firms
    keepat!(SE.firm_active, [i for i=1:length(SE.firm_active) if SE.firm_assets[SE.firm_active[i]] > - SE.tol])
    ##Correct errors due to numerical imprecision
    @views SE.firm_assets[SE.firm_active] .= max.(0.0, SE.firm_assets[SE.firm_active])
    
    return nothing
end

##Implements migration with consideration sets
##Partitions utilities into groups to go faster
###Uses Binned Distribution for incomes
function migrate_cs_part!(SE::SpatialEconomy, util_interval::Float64, mig_disutil::Float64=0.0, service_perm::Float64=1.0)
    ###Reinitialize income distributions
    for l in SE.loc_inh_ind
        reinitialize!(SE.loc_income[l])
    end

    ###Sort locations by utility (only permutation)
    loc_sp_util = sortperm(SE.loc_avg_util)
    ##Vector for utility realizations sorted permutations
    ru_sortperm = Array{Int64, 1}(undef, maximum(x -> length(x), SE.loc_real_util))
    ###Used to intermediary calculations of probabilities
    mig_attract = Array{Float64, 1}(undef, length(SE.loc_mig_attract))
    ###Preallocate an array that will contain the utility groups
    util_groups::Array{UtilityGroup, 1} = UtilityGroup[]

    ###Choice of locations
    for l in SE.loc_inh_ind
        ###Order income and utilities
        sortperm!(view(ru_sortperm, 1:length(SE.loc_real_util[l])), SE.loc_real_util[l])

        ###Probability that each location is considered
        ######Probabiliy due to distance
        l_service = (l % 3) == 0
        for d=1:3
            SE.loc_mig_attract[d:3:(3*SE.N_location)] .= ifelse((l_service & (d % 3 != 0)) | ((d % 3 == 0) & (!l_service)), service_perm, 1.0) .*
                                                         lgstic.(SE.mig_size_effect .* view(SE.loc_pop, d:3:(3*SE.N_location)) ./ sum(SE.loc_pop) .+
                                                                 SE.mig_link_intens .* view(SE.loc_mig_link, l, d:3:(3*SE.N_location)) .- 
                                                                 SE.mig_dist_fric .* view(SE.loc_dist, (l - 1) ÷ 3 + 1, :))
        end
        
        ###Remove "self-migration"
        SE.loc_mig_attract[l] = 0.0

        if any(x -> x > 0.0, SE.loc_mig_attract)
            ###Sums of attractiveness to each location: needed to calculate probabilities of individual choices
            sp_1 = sum(SE.loc_mig_attract)
            sp_2 = sum(x -> x * x, SE.loc_mig_attract)
            ##prepare miglink for update
            SE.loc_mig_link[l, :] .*= SE.mig_link_inert
            ###Partition utilities
            partition_util_inc!(util_groups, view(SE.loc_real_util[l], view(ru_sortperm, 1:length(SE.loc_real_util[l]))), 
                                             view(SE.loc_real_inc[l], view(ru_sortperm, 1:length(SE.loc_real_util[l]))), 
                                             SE.loc_pop[l] / length(SE.loc_real_util[l]), util_interval, SE.bin_width)
            ####Go through locations by utility to find migrations (realizations go to locations that have higher utilities)
            ####because location with lower utilities are not considered
            lu_dest = 1 ###Lowest utility destination (i.e lowest utility higher than the one of the migrant)
            for ug in util_groups
                stay_util = ug.grp_util_mean + mig_disutil
                prev_lu_dest = lu_dest
                while lu_dest <= length(SE.loc_avg_util) && SE.loc_avg_util[loc_sp_util[lu_dest]] <= stay_util
                    lu_dest += 1
                end
                ####Correct sums needed to compute the probabilities
                if prev_lu_dest < lu_dest
                    sp_1 = max(sp_1 - sum(SE.loc_mig_attract[loc_sp_util[i]] for i=prev_lu_dest:(lu_dest-1)), 0.0)
                    sp_2 = max(sp_2 - sum(SE.loc_mig_attract[loc_sp_util[i]]^2.0 for i=prev_lu_dest:(lu_dest-1)), 0.0)
                end
                if sp_1 > SE.tol ###No migration if sp_1 = 0.0
                    ###Calculate probabilities that each location is chosen
                    mig_attract[lu_dest:length(SE.loc_avg_util)] .= p_cs_chosen.((SE.loc_mig_attract[loc_sp_util[i]] for i=lu_dest:length(SE.loc_avg_util)), sp_1, sp_2)
                    ###Normalize approximations if sum is larger than 1
                    s_a = sum(mig_attract[i] for i=lu_dest:length(SE.loc_avg_util))
                    s_a > 1.0 && (mig_attract[lu_dest:length(SE.loc_avg_util)] ./= s_a)
                    ####Share of staying
                    ssp = 1.0
                    ##Share of population ready to move (store calculation that does not depend on destination)
                    share_motile = max(1.0 - ifelse(l % 3 != 2, SE.pop_fixed / SE.loc_pop[l], 0.0), 0.0) * SE.motility
                    for dest=lu_dest:length(mig_attract); if mig_attract[dest] > 0.0
                        share_to_dest = share_motile * mig_attract[dest]
                        ##Only significant population moves are considered (to speed up calculations)
                        if share_to_dest * total_weight(ug) > 0.0001 * min(SE.loc_pop[loc_sp_util[dest]], SE.loc_pop[l])
                            add_from_BI!(SE.loc_income[loc_sp_util[dest]], ug.grp_inc_dist, share_to_dest)
                            ###Update staying population
                            ssp -= share_to_dest
                            ###Update migration link
                            SE.loc_mig_link[l, loc_sp_util[dest]] += (1.0 - SE.mig_link_inert) * share_to_dest *
                                                                     total_weight(ug) / SE.loc_pop[l]
                        end
                    end; end
                    ###Staying population
                    add_from_BI!(SE.loc_income[l], ug.grp_inc_dist, ssp)
                else
                    ##Non movers corresponding to utility group
                    add_from_BI!(SE.loc_income[l], ug.grp_inc_dist)
                end
            end
            empty!(util_groups) ##Empty the vector of utility groups
        else
            ##Add those that stayed in the location
            add_points!(SE.loc_income[l], SE.loc_real_inc[l], SE.loc_pop[l] / length(SE.loc_real_inc[l]))
        end
    end
    return nothing
end


##Implements migration with complete information
##Partitions utilities into groups to go faster
###Uses Binned Distribution for incomes
function migrate_CI_part!(SE::SpatialEconomy, util_interval::Float64, mig_disutil::Float64=0.0, fixed_service=false)
    ###Reinitialize income distributions
    for l in SE.loc_inh_ind
        reinitialize!(SE.loc_income[l])
    end

    ###Sort locations by utility (only permutation)
    loc_sp_util = sortperm(SE.loc_avg_util)
    ##Vector for utility realizations sorted permutations
    ru_sortperm = Array{Int64, 1}(undef, maximum(x -> length(x), SE.loc_real_util))
    ###Vector that will store the utility groups§
    util_groups::Array{UtilityGroup, 1} = UtilityGroup[]

    ###Choice of locations
    for l in SE.loc_inh_ind
        ###Order income and utilities
        sortperm!(view(ru_sortperm, 1:length(SE.loc_real_util[l])), SE.loc_real_util[l])
        
        for i=1:3
            SE.loc_mig_attract[i:3:(3*SE.N_location)] .=  (1.0 .+ SE.loc_pop ./ sum(SE.loc_pop)) .*
                                                          (1.0 .+ view(SE.loc_mig_link, l, i:3:(3*SE.N_location))).^SE.mig_link_intens ./
                                                          (1.0 .+ view(SE.loc_dist, (l - 1) ÷ 3 + 1, :)).^SE.mig_dist_fric
        end
                                         
        ###Remove "self-migration"
        SE.loc_mig_attract[l] = 0.0
        ###separate service and non-service if needed
        if fixed_service
            ##Below condition is true if one of the locations is not service and other is
            for d in SE.loc_inh_ind; if ((l % 3 == 0) & (d % 3 in (1, 2))) | ((d % 3 == 0) & (l % 3 in (1, 2)))
                SE.loc_mig_attract[d] = 0.0
            end; end
        end

        if any(x -> x > 0.0, SE.loc_mig_attract)
            ###Make probabilities out of attractivities
            SE.loc_mig_attract ./= sum(SE.loc_mig_attract)
            ##prepare miglink for update
            SE.loc_mig_link[l, :] .*= SE.mig_link_inert
            ###Partition utilities
            partition_util_inc!(util_groups, view(SE.loc_real_util[l], view(ru_sortperm, 1:length(SE.loc_real_util[l]))), 
                                             view(SE.loc_real_inc[l], view(ru_sortperm, 1:length(SE.loc_real_util[l]))), 
                                             SE.loc_pop[l] / length(SE.loc_real_util[l]), util_interval, SE.bin_width)

            ####Go through locations by utility to find migrations (realizations go to locations that have higher utilities)
            ####because location with lower utilities are not considered
            lu_dest = 1 ###Lowest utility destination (i.e lowest utility higher than the one of the migrant)
            for ug in util_groups
                stay_util = ug.grp_util_mean + mig_disutil
                while lu_dest <= length(SE.loc_avg_util) && SE.loc_avg_util[loc_sp_util[lu_dest]] <= stay_util
                    lu_dest += 1
                end
                if lu_dest <= length(SE.loc_avg_util)
                    ####Share of staying
                    ssp = 1.0
                    ##Share of population ready to move (store calculation that does not depend on destination)
                    share_motile = max(1.0 - ifelse(l % 3 != 2, SE.pop_fixed / SE.loc_pop[l], 0.0), 0.0) * SE.motility
                    for dest=lu_dest:length(SE.loc_mig_attract)
                        share_to_dest = share_motile * SE.loc_mig_attract[loc_sp_util[dest]]
                        ##Only significant population moves are considered (to speed up calculations)
                        if share_to_dest * total_weight(ug) > 0.0001 * min(SE.loc_pop[loc_sp_util[dest]], SE.loc_pop[l])
                            add_from_BI!(SE.loc_income[loc_sp_util[dest]], ug.grp_inc_dist, share_to_dest)
                            ###Update staying population
                            ssp -= share_to_dest
                            ###Update migration link
                            SE.loc_mig_link[l, loc_sp_util[dest]] += (1.0 - SE.mig_link_inert) * share_to_dest *
                                                                     total_weight(ug) / SE.loc_pop[l]
                        end
                    end
                    ###Staying population
                    add_from_BI!(SE.loc_income[l], ug.grp_inc_dist, ssp)
                else
                    ##Non movers cooresponding to utility group
                    add_from_BI!(SE.loc_income[l], ug.grp_inc_dist)
                end
            end
        else
            ##Add those that stayed in the location
            add_points!(SE.loc_income[l], SE.loc_real_inc[l], SE.loc_pop[l] / length(SE.loc_real_inc[l]))
        end
    end
    return nothing
end


function update_population!(SE::SpatialEconomy)
    ##update populations
    SE.loc_pop .= sum_weights.(SE.loc_income)
    SE.loc_pop .= ifelse.(SE.loc_pop .> SE.tol, SE.loc_pop, 0.0)

    ############ Update inhabited locations #################
    SE.loc_inhabited .= SE.loc_pop .> SE.tol
    for l=1:(3*SE.N_location) ; if !SE.loc_inhabited[l]
        reinitialize!(SE.loc_income[l])
    end ; end
    ###Update indices of inhabited locations
    ##Boolean indices
    ####Integer indices of all locations
    resize!(SE.loc_inh_ind, sum(SE.loc_inhabited))
    findall!(SE.loc_inh_ind, SE.loc_inhabited)
    ###Integer indices for locations inhabited by any sector
    empty!(SE.loc_any_inh_ind)
    append!(SE.loc_any_inh_ind, l for l=1:SE.N_location if any(@view SE.loc_inhabited[(3*l-2):(3*l)]))
    ####Integer indices for rural population (indices: 3k-2)
    empty!(SE.loc_agr_inh_ind)
    append!(SE.loc_agr_inh_ind, l for l=1:SE.N_location if SE.loc_inhabited[3*l-2])
    ####Integer indices for manufacturing popularion (indices: 3k-1)
    empty!(SE.loc_man_inh_ind)
    append!(SE.loc_man_inh_ind, l for l=1:SE.N_location if SE.loc_inhabited[3*l-1])
    ####Integer indices for service population (indices 3k)
    empty!(SE.loc_ser_inh_ind)
    append!(SE.loc_ser_inh_ind, l for l=1:SE.N_location if SE.loc_inhabited[3*l])

    return nothing
end
