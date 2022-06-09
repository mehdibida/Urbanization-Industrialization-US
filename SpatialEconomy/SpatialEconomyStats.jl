using Base.Iterators: cycle, zip, product as iterprod 

## Total employment
tot_man_empl(SE::SpatialEconomy) = @views isempty(SE.firm_active) ? 0.0 : sum(SE.firm_labor_use[SE.firm_active])

tot_man_unempl(SE::SpatialEconomy) = sum(SE.loc_unempl)

###Scaling man empl, nonagr size
function sc_urb_man(SE::SpatialEconomy)
    ###Population working in services in urban area
    if @views sum(x -> x > 0.0, SE.loc_pop[2:3:(3*SE.N_location)]) > 1
        return simple_reg_slope((log(SE.loc_pop[3*l-1]) for l=1:SE.N_location if SE.loc_pop[3*l-1] > 0.0),
                                (log(SE.loc_pop[3*l-1] + SE.loc_pop[3*l]) for l=1:SE.N_location if SE.loc_pop[3*l-1] > 0.0))
    else
        return NaN
    end
end

## Correlation wage capital
cor_wage_capital(SE::SpatialEconomy) = @views length(SE.firm_active) > 1 ? cor(SE.firm_wage[SE.firm_active], SE.firm_capital[SE.firm_active]) : NaN

## Correlation between manufacturing employment and "urban" size
cor_man_ser(SE::SpatialEconomy) = cor(view(SE.loc_pop, 2:3:(3*SE.N_location)) .- SE.loc_unempl, view(SE.loc_pop, 3:3:(3*SE.N_location)))

## Urban share
urban_share(SE::SpatialEconomy) = (sum(view(SE.loc_pop, 2:3:((3*SE.N_location)))) + sum(view(SE.loc_pop, 3:3:(3*SE.N_location)))) /
                                   sum(SE.loc_pop)

## Urban share (a location is not a city if it contains less than tot_pop/N_loc which corresponds to full dispersion)
function urban_share(SE::SpatialEconomy, threshold = tot_pop / SE.N_location)
    tot_pop = sum(SE.loc_pop)
    urb_pop = 0.0
    for l=1:SE.N_location
        l_p = SE.loc_pop[3*l-1] + SE.loc_pop[3*l]
        urb_pop += l_p * (l_p > threshold)
    end
    return urb_pop / tot_pop
end

function share_dist_ser(SE::SpatialEconomy)
    dist_share = fill(0.0, SE.N_location)
    for l in SE.loc_ser_inh_ind
        if !isempty(SE.loc_agr_inh_ind) ###From selling agriculture
            dist_share[l] = SE.distr_markup * 
                            sum(SE.loc_agr_ship[sl, l] * (1.0 + SE.loc_dist[sl, l]) for sl in SE.loc_agr_inh_ind)
        end
        if !isempty(SE.firm_act_prod) ###From selling manufacturing
            dist_share[l] += SE.distr_markup / (1.0 + SE.distr_markup) *
                             sum(SE.firm_loc_ship[f, l] * SE.firm_loc_price[f, l] for f in SE.firm_act_prod)
        end
    end
    dist_share ./= SE.loc_ser_tot_inc
    return dist_share
end

function share_transportation_ser(SE::SpatialEconomy)
    inc_share_trans = fill(0.0, SE.N_location)
    for (c, v, nodes) in SE.loc_sp_node
        ##transportation earnings is dependent on the service population: we assume that they have more labor available
        ##for transportation
        s_t_cost = (SE.loc_agr_man_ship[c, v] + SE.loc_agr_man_ship[v, c]) *
                    SE.loc_dist[v, c] / sum(SE.loc_pop[3*n] for n in nodes)
        for n in nodes
            SE.loc_inhabited[3*n] && (inc_share_trans[n] += s_t_cost * SE.loc_pop[3*n])
        end
    end
    inc_share_trans ./= SE.loc_ser_tot_inc

    return inc_share_trans
end

sc_pop_productivity(SE::SpatialEconomy) = @views sum(x -> x[1] > x[2], zip(SE.loc_pop[2:3:(3*SE.N_location)], SE.loc_unempl)) > 1 ?
                                          simple_reg_slope((log(firm_production(SE.firm_labor_use[f], SE.firm_wage[f], SE.man_ret_scale, SE.man_tech) * 
                                                                (SE.firm_wage[f] + SE.c_capital) / (SE.firm_labor_use[f] * SE.firm_wage[f] + SE.c_capital * SE.firm_capital[f])) for f in SE.firm_act_prod),
                                                           (log(SE.loc_pop[3 * SE.firm_loc[f] - 1]) for f in SE.firm_act_prod)) :
                                          NaN


###Agriculture share
agr_share(SE) = @views sum(SE.loc_pop[1:3:(3*SE.N_location)]) / sum(SE.loc_pop)
###Manufacturing share
man_share(SE) = @views sum(SE.loc_pop[2:3:(3*SE.N_location)]) / sum(SE.loc_pop)
###Services share
ser_share(SE) = @views sum(SE.loc_pop[3:3:(3*SE.N_location)]) / sum(SE.loc_pop)

###Average capital
avg_capital(SE::SpatialEconomy) = @views isempty(SE.firm_active) ? 0.0 : mean(SE.firm_capital[SE.firm_active])
###Average capital for each location
avg_capital_loc(SE::SpatialEconomy, loc::Integer) = any(x -> x[1] == x[2], zip(loc, view(SE.firm_loc, SE.firm_active))) ? mean(SE.firm_capital[i] for i in SE.firm_active if SE.firm_loc[i] == i) : 0.0

###Average wage
avg_wage(SE::SpatialEconomy) = @views w_avg(SE.firm_wage[SE.firm_active], SE.firm_labor_use[SE.firm_active])

tot_loc_pop(SE::SpatialEconomy) = sum.(Iterators.partition(SE.loc_pop, 3))

###Total income for agriculture
tot_inc_agr(SE::SpatialEconomy) = sum(x -> mean(x[1]) * x[2], zip(view(SE.loc_income, 1:3:(3*SE.N_location)), 
                                                                  view(SE.loc_pop, 1:3:(3*SE.N_location))))
###Total income for manufacturing
tot_inc_man(SE::SpatialEconomy) = sum(x -> mean(x[1]) * x[2], zip(view(SE.loc_income, 2:3:(3*SE.N_location)), 
                                                                  view(SE.loc_pop, 2:3:(3*SE.N_location))))
###Total income for services
tot_inc_ser(SE::SpatialEconomy) = sum(x -> mean(x[1]) * x[2], zip(view(SE.loc_income, 3:3:(3*SE.N_location)), 
                                                                  view(SE.loc_pop, 3:3:(3*SE.N_location))))


##Total quantity of manufactured goods sold per location
function loc_man_qsold(SE::SpatialEconomy)
    res = fill(0.0, SE.N_location)
    for (f, l) in iterprod(SE.firm_active, SE.loc_any_inh_ind)
        res[SE.firm_loc[f]] += SE.firm_loc_ship[f, l]
    end
    return res
end

##Total quantity of manufactured goods exported per location
function loc_man_qexport(SE::SpatialEconomy)
    res = fill(0.0, SE.N_location)
    for (f, l) in iterprod(SE.firm_active, SE.loc_any_inh_ind)
        l != SE.firm_loc[f] && (res[SE.firm_loc[f]] += SE.firm_loc_ship[f, l])
    end
    return res
end

###Average transported distance by manufacturing products
function loc_man_avg_qtrsp(SE::SpatialEconomy)
    a_d = fill(0.0, SE.N_location)
    s_d = fill(0.0, SE.N_location)
    for (f, l) in iterprod(SE.firm_active, SE.loc_any_inh_ind)
        a_d[SE.firm_loc[f]] += SE.firm_loc_ship[f, l] * SE.loc_dist_init[SE.firm_loc[f], l]
        s_d[SE.firm_loc[f]] += SE.firm_loc_ship[f, l]
    end
    a_d ./= s_d
    return a_d
end


##Total sold quantity
tot_man_qsold(SE::SpatialEconomy) = @views sum(SE.firm_loc_ship[SE.firm_act_prod, :])

##Total quantity of manufactured goods sold per location
loc_agr_qsold(SE::SpatialEconomy) = sumcols(SE.loc_agr_ship)

##Total quantity of manufactured goods exported per location
function loc_agr_qexport(SE::SpatialEconomy)
    res = loc_agr_qsold(SE)
    res .-= (SE.loc_agr_ship[l, l] for l=1:SE.N_location)
    return res
end

###Average transported distance by manufacturing products
function loc_agr_avg_qtrsp(SE::SpatialEconomy)
    a_d = fill(0.0, SE.N_location)
    s_d = fill(0.0, SE.N_location)
    for (a, l) in iterprod(SE.loc_agr_inh_ind, SE.loc_any_inh_ind)
        a_d[a] += SE.loc_agr_ship[a, l] * SE.loc_dist_init[a, l]
        s_d[a] += SE.loc_agr_ship[a, l]
    end
    a_d ./= s_d
    return a_d
end

function loc_tot_cap(SE::SpatialEconomy)
    res = fill(0.0, SE.N_location)
    for f in SE.firm_active
        res[SE.firm_loc[f]] += SE.firm_capital[f]
    end
    return res
end


###Average cost for producing one unit at each location
function loc_avg_unit_cost(SE::SpatialEconomy)
    a_uc = fill(0.0, SE.N_location)
    s_c = fill(0.0, SE.N_location)
    for f in SE.firm_active
        a_uc[SE.firm_loc[f]] += SE.firm_capital[f] * SE.c_capital + SE.firm_labor_use[f] * SE.firm_wage[f]
        s_c[SE.firm_loc[f]] += firm_production(SE.firm_labor_use[f], SE.firm_capital[f], SE.man_ret_scale, SE.man_tech)
    end
    a_uc ./= s_c
    return a_uc
end

##Mean income for agriculture
avg_inc_agr(SE::SpatialEconomy) = tot_inc_agr(SE) / sum(view(SE.loc_pop, 1:3:(3*SE.N_location)))
##Mean income for manufacturing
avg_inc_man(SE::SpatialEconomy) = tot_inc_man(SE) / sum(view(SE.loc_pop, 2:3:(3*SE.N_location)))
##Mean income for services
avg_inc_ser(SE::SpatialEconomy) = tot_inc_ser(SE) / sum(view(SE.loc_pop, 3:3:(3*SE.N_location)))

##Mean utility for agriculture
avg_util_agr(SE::SpatialEconomy) = @views w_avg(mean.(SE.loc_real_util[1:3:(3*SE.N_location)]), SE.loc_pop[1:3:(3*SE.N_location)])
##Mean utility for manufacturing
avg_util_man(SE::SpatialEconomy) = @views w_avg(mean.(SE.loc_real_util[2:3:(3*SE.N_location)]), SE.loc_pop[2:3:(3*SE.N_location)])
##Mean utility for rural services
avg_util_ser(SE::SpatialEconomy) = @views w_avg(mean.(SE.loc_real_util[3:3:(3*SE.N_location)]), SE.loc_pop[3:3:(3*SE.N_location)])
####Transported agr quantity
transported_agr(SE::SpatialEconomy) = @views sum(x -> ifelse(x[2] > 0.0, x[1], 0.0), zip(SE.loc_agr_ship[:, SE.loc_any_inh_ind], SE.loc_dist[:, SE.loc_any_inh_ind]), init = 0.0)
###Transported man
transported_man(SE::SpatialEconomy) = @views sum(x -> ifelse(x[2] > 0.0, x[1], 0.0), 
                                                 zip(SE.firm_loc_ship[SE.firm_act_prod, SE.loc_any_inh_ind], 
                                                     SE.loc_dist[SE.firm_loc[SE.firm_act_prod], SE.loc_any_inh_ind]), init = 0.0)
