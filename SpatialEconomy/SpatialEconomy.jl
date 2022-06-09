using Distributions: Gamma
using Graphs: desopo_pape_shortest_paths, enumerate_paths, grid, neighbors
using Combinatorics: combinations
using DataFrames
using CSV: CSV.write as write_csv, read as read_csv

###Store the model in a struct
##The struct contains a snapshot of the state of the model (at a certain time step)
###TL is 3 times the number of locations
mutable struct SpatialEconomy
    bin_width::Float64
    tol::Float64
    N_location::Int64
    dist_init_cost::Float64 ##Initial dist scale

    pop_fixed::Float64
    pop_grth_rate::Float64
    share_agr_to_man::Float64 ###Share of population working in agr recruitable by man firms directly
    motility::Float64
    mig_size_effect::Float64
    mig_dist_fric::Float64
    mig_link_intens::Float64
    mig_link_inert::Float64

    housing_sat::Float64 ; agr_sat::Float64 ; agr_end_share::Float64 ; #Last: endowment of agriculture workers in terms of share of agr_sat
    man_sat::Float64; 
    housing_share::Float64 ; agr_share::Float64
    
    agr_tech::Float64 ; agr_ret_scale::Float64 ### Agriculture technology and return to scale

    man_tech::Float64 ; man_ret_scale::Float64 ##Technology & returns to scale
    markup::Float64; wage_sens::Float64
    c_capital::Float64 ##Cost of capital

    distr_markup::Float64 ##Price markup put by the distrbution sector (part of serices) 

    loc_pop::Array{Float64, 1} #Population at each location
    
    ###### Indices to designate the different populations
    ##Tells whether a location is inhabited or not 
    loc_inhabited::Array{Bool, 1}
    ###Integer indices of inhabited locations (takes sectors and location into account: agr, man, rural_ser, urb_ser), length = 4*N_location
    loc_inh_ind::Array{Int64, 1}
    ###Integer indices of locations of which at least one sector is has population
    loc_any_inh_ind::Array{Int64, 1}
    ####Integer indices for rural population (indices: 4k-3)
    loc_agr_inh_ind::Array{Int64, 1}
    ####Integer indices for manufacturing popularion (indices: 4k-2)
    loc_man_inh_ind::Array{Int64, 1}
    ####Integer indices for urban service population (indices 4k)
    loc_ser_inh_ind::Array{Int64, 1}

    ####Population of each city
    ####Number of firms in each location
    loc_new_firm::Array{Bool, 1}

    ####Vectors of random realizations of (to estimate deamnds, utilities, etc.)
    loc_real_inc::Array{Array{Float64, 1}, 1} ###incomes
    loc_real_rq::Array{Array{Float64, 1}, 1} ##Reservation quantities
    loc_real_consume::Array{Array{Bool, 1}, 1} ##whether realizations of consumers are consuming agr and man or not
    loc_real_h_cons::Array{Array{Float64, 1}, 1} ##housing
    loc_real_a_cons::Array{Array{Float64, 1}, 1} ##agriculture
    loc_real_m_cons::Array{Array{Float64, 1}, 1} ##manufacturing
    loc_real_util::Array{Array{Float64, 1}, 1} ###utility
    
    loc_real_choice::Array{Array{Int64, 1}, 1} ###Consumer choice: used for location and producer
    
    loc_income::Array{IncDistBinned, 1} ##Vector of income distrbutions
    
    loc_housing_price::Array{Float64, 1}  ###Location housing price
    loc_agr_avg_cons_prev::Array{Float64, 1} ###Location previous mean agriculture price
    loc_man_avg_cons_prev::Array{Float64, 1} ###Location previous mean manufacturing price
    loc_agr_spending::Array{Float64,1} ### Amount spent on agriculture per location
    loc_hous_spending::Array{Float64,1} ### Amount spent on housing for each location
    loc_ser_tot_inc::Array{Float64,1} ##Total income of the urban service sector
    loc_avg_util::Array{Float64,1} ###Mean utility for each location
    loc_unempl::Array{Float64, 1} ###Amount of manufacturing labor left without a job
    loc_avg_man_wage_prev::Array{Float64, 1} ##Average wages in manufacturing by location (amounts paid, not announced by firms): takes unemployment into account, considered as wage=0
    loc_avg_agr_wage_prev::Array{Float64, 1} ##Average wages in agriculture by location (amounts paid, not announced by firms): takes unemployment into account, considered as wage=0
    loc_avg_firm_profit::Array{Float64, 1}

    loc_mig_attract::Array{Float64, 1}### Attractiveness of locations for a given location (~ mig_link / dist)

    loc_mig_link::Array{Float64, 2}###Stores migration link
    
    ###They are arrays of arrays for compatibility with firms' arrays - indexation: arr[recipient, sender]
    loc_agr_prod::Array{Float64, 2} ##Amount of agricultural products sent to each location
    loc_agr_ship::Array{Float64, 2} ### Location agricultural shipments (to calculate income from transport cost of man goods)
    loc_agr_man_ship::Array{Float64, 2}### Quantity of manufacturing goods sold from location in column to location in row (to calculate income from transport cost of goods)

    ###Storing shortest paths and parents (in rows are the consumers, columns are for sellers)
    loc_dist::Array{Float64, 2}
    loc_dist_init::Array{Float64, 2} 
    
    ###Stores nodes on shortest paths
    loc_sp_node::Array{Tuple{Int64, Int64, Array{Int64, 1}}, 1}
    
    ###Wage distribution of manufacturing
    man_wage_dist::IncDistBinned
    #loop_loc_employment::Array{Float64, 1}
    #loop_loc_wage::Array{Float64, 1} #Wages at a given location (to be used in loops on locations)
    
    #path_nodes::Array{Int64, 1} ##To use later for shortest path calculations
    
    ###System variables concerning firms
    loc_firm_entry::Array{Int64,1} ###Initialize entring firms
    ####Firms characteristics
    firm_capital::Array{Float64,1} ##Firms capital
    firm_loc::Array{Int64,1} ##Firms location
    firm_entry_time::Array{Int64,1} ### Entry time
    
    firm_active::Array{Int64,1} ### Are firms active
    firm_act_prod::Array{Int64,1} ##If firm is active and producing
    firm_select::Array{Int64,1} ###Vector to select the firms to be changed

    ### Variable characteristics
    ### When there are 3 dimensions, the last dimension is for temporality: 1 for t-2, 2 for t-1, and 3 for t
    ##In coding we use "end" to signal the contemporaneous values, so that end-1 and end-2 coincide w/ t-1 and t-2
    firm_labor_dem::Array{Float64,1} ##Firm current labor demand
    firm_labor_use::Array{Float64,1} ##Firm current labor usage
    firm_q_sold::Array{Float64,1} ###Total quantity sold by each firm 
    firm_wage::Array{Float64, 1} ##Firm wage
    firm_price::Array{Float64,1} ##Firm price
    firm_assets::Array{Float64,1} ###Firm financial assets

    ##Matrices of interaction between firms and locations - indexation: arr[location, firm]
    ### Firm shipments (firms are in columns, cities are in rows)
    firm_loc_price::Array{Float64, 2}
    ### produced quantity sent to each location (firms are in columns, cities are in rows)
    firm_loc_prod::Array{Float64,2}
    ### Firm shipments (firms are in columns, cities are in rows)
    firm_loc_ship::Array{Float64, 2}
end


###Builds a spatial economy on an arbitrary network
###node_dist is a square matrix with distances between locations
###loc_sp_parent is the matrix of nodes found in the shortest paths
function SpatialEconomy(; node_dist::AbstractArray{Float64, 2}, loc_sp_parent,
                        dist_init_cost::Float64, pop_grth_rate::Float64, share_agr_to_man::Float64,
                        man_sat::Float64, housing_share::Float64, housing_sat::Float64,
                        agr_share::Float64, agr_sat::Float64, agr_end_share::Float64, mean_inc_agr::Float64, mean_inc_ser::Float64, loc_pop::Array{Float64, 1},
                        agr_tech::Float64, agr_ret_scale::Float64,
                        man_tech::Float64, man_ret_scale::Float64, c_capital::Float64, markup::Float64,
                        wage_sens::Float64, distr_markup::Float64, motility::Float64, mig_size_effect::Float64, mig_dist_fric::Float64, mig_link_intens::Float64, mig_link_inert::Float64,
                        pop_fixed::Float64=mean(loc_pop)*10^-3, 
                        bin_width::Float64, wc_sample_size::Int64=50*4*N_location,
                        sp_nodes_exclude::Set{Int64} = Set(Int64[]), tol::Float64=10^-9)

    0 <= motility <= 1 || error("Motility must be between 0 and 1.")
   
    N_location = size(node_dist)[1]

    loc_inhabited = [loc_pop[i] > ifelse((i % 3) != 2, pop_fixed, tol) for i=1:(3*N_location)]
    
    ##Income distrbution
    loc_inc_dist = vcat([[IncDistBinned(bin_width, rand_inc_fixed_mean(Gamma(2.0, 0.5 * mean_inc_agr), wc_sample_size), loc_pop[3*i-2] / wc_sample_size),
                          loc_pop[3*i-1] > 0.0 ? IncDistBinned(bin_width, [0.0], loc_pop[3*i-1]) : IncDistBinned(bin_width),
                          IncDistBinned(bin_width, rand_inc_fixed_mean(Gamma(2.0, 0.5 * mean_inc_ser), wc_sample_size), loc_pop[3*i] / wc_sample_size)] for i=1:N_location]...)

    loc_h_price = [(mean(loc_inc_dist[3*l-1]) * loc_pop[3*l-1] + mean(loc_inc_dist[3*l]) * loc_pop[3*l]) / 
                    (loc_pop[3*l-1] + loc_pop[3*l]) * housing_share for l=1:N_location]

    loc_distance::Array{Float64, 2} = Array{Float64, 2}(undef, N_location, N_location)

    loc_distance .= node_dist
    
    ##Put distances to scale
    loc_distance .*= dist_init_cost

    ###Stores nodes on shortest paths
    loc_sp_node::Array{Tuple{Int64, Int64, Array{Int64, 1}}, 1} = Tuple{Int64, Int64, Array{Int64, 1}}[]
    for (c, v) in combinations(1:N_location, 2)
        push!(loc_sp_node, (c, v, sp_nodes(v, view(loc_sp_parent, :, c), sp_nodes_exclude)))
    end

    ##loc_agr_prod
    loc_agr_prod = fill(0.0, N_location, N_location)
    for l=1:N_location; loc_agr_prod[l, l] = agr_tech * loc_pop[3*l-2]^agr_ret_scale; end

    return SpatialEconomy(bin_width, tol, N_location, dist_init_cost,
                            pop_fixed, pop_grth_rate, share_agr_to_man, 
                            motility, mig_size_effect, mig_dist_fric, mig_link_intens, mig_link_inert,
                            housing_sat, agr_sat, agr_end_share, man_sat, housing_share, agr_share, 
                            agr_tech, agr_ret_scale, man_tech, man_ret_scale, 
                            markup, wage_sens, c_capital,
                            distr_markup,
                            loc_pop,
                            loc_inhabited,
                            findall(loc_inhabited), #loc_inh_ind
                            [l for l=1:N_location if any(@view loc_inhabited[(3*l-2):(3*l)])], #loc_any_inh_ind
                            [i for i=1:N_location if (1.0 - loc_pop[3*i - 2] / pop_fixed) < tol], #loc_agr_inh_ind
                            [i for i=1:N_location if (1.0 - loc_pop[3*i - 1] / pop_fixed) < tol], #loc_man_inh_ind
                            [i for i=1:N_location if (1.0 - loc_pop[3*i] / pop_fixed) < tol], #loc_ser_inh_ind
                            fill(false, N_location), #loc_new_firm
                            [fill(0.0, wc_sample_size) for i=1:(3*N_location)], ##loc_real_inc
                            [fill(0.0, wc_sample_size) for i=1:(3*N_location)], ##loc_real_rq
                            [fill(true, wc_sample_size) for i=1:(3*N_location)], ##loc_real_consume
                            [fill(0.0, wc_sample_size) for i=1:(3*N_location)], ##loc_real_h_cons
                            [fill(0.0, wc_sample_size) for i=1:(3*N_location)], ##loc_real_a_cons
                            [fill(0.0, wc_sample_size) for i=1:(3*N_location)], ##loc_real_m_cons
                            [fill(0.0, wc_sample_size) for i=1:(3*N_location)], ##loc_real_util
                            [fill(0, wc_sample_size) for i=1:(3*N_location)], ##loc_real_choice
                            loc_inc_dist, ##loc_income
                            loc_h_price, ##loc_housing_price
                            fill(agr_sat, N_location), ###loc_agr_avg_cons_prev
                            fill(0.0, N_location), ###loc_man_avg_cons_prev
                            fill(0.0, 3 * N_location), ##loc_agr_spending
                            fill(0.0, 3 * N_location), ##loc_hous_spending
                            [mean(loc_inc_dist[l]) * loc_pop[l] * loc_inhabited[l] for l=3:3:(3*N_location)], ##loc_ser_tot_inc
                            fill(0.0, 3*N_location), ##loc_avg_util
                            loc_pop[2:3:(3*N_location-1)], ##loc_unempl
                            fill(0.0, N_location), ##loc_avg_man_wage_prev
                            mean.(@view loc_inc_dist[1:3:(3*N_location)]), ##loc_avg_agr_wage_prev
                            fill(log(2.0), N_location), #loc_avg_firm_profit
                            fill(0.0, 3*N_location), #loc_mig_attract
                            fill(0.0, 3*N_location, 3*N_location), ##loc_mig_link
                            loc_agr_prod, ##loc_agr_prod
                            fill(0.0, N_location, N_location), ##loc_agr_ship
                            fill(0.0, N_location, N_location), ##loc_agr_man_ship
                            loc_distance,
                            copy(loc_distance), ##loc_dist_init
                            loc_sp_node,
                            IncDistBinned(bin_width, wc_sample_size), #man_wage_dist::
                            fill(0, N_location), #loc_firm_entry
                            Float64[], #firm_capital
                            Int64[], Int64[], #firm_loc, firm_entry_time
                            Int64[], Int64[], Int64[],
                            #firm_active, firm_act_prod, firm_act_n_prod, firm_select
                            Float64[], Float64[], Float64[],
                            #firm_labor_dem, firm_labor_use, firm_q_sold
                            Float64[], Float64[], Float64[],
                            #firm_wage, firm_price, firm_assets
                            Array{Float64, 2}(undef, 0, N_location), #firm_loc_price
                            Array{Float64, 2}(undef, 0, N_location), #firm_loc_prod
                            Array{Float64, 2}(undef, 0, N_location) #firm_loc_ship
                            )
end


###Basic spatial economy on a squared grid
function Simple_SpatialEconomy(;N_location::Int64, dist_init_cost::Float64, pop_grth_rate::Float64, share_agr_to_man::Float64,
                                man_sat::Float64, housing_share::Float64, housing_sat::Float64,
                                agr_share::Float64, agr_sat::Float64, agr_end_share::Float64, mean_inc_agr::Float64, mean_inc_ser::Float64, loc_pop::Array{Float64, 1},
                                agr_tech::Float64, agr_ret_scale::Float64,
                                man_tech::Float64, man_ret_scale::Float64, c_capital::Float64, markup::Float64,
                                wage_sens::Float64, distr_markup::Float64, 
                                motility::Float64, mig_size_effect::Float64, mig_dist_fric::Float64, mig_link_intens::Float64, mig_link_inert::Float64,
                                pop_fixed::Float64=mean(loc_pop)*10^-3, 
                                bin_width::Float64, wc_sample_size::Int64=50*4*N_location, tol::Float64=10^-9)
    ##Grid mut be a square
    floor(sqrt(N_location)) != sqrt(N_location) && error("Grid must be a square.")
    0 <= motility <= 1 || error("Motility must be between 0 and 1.")
    
    ###Location network topology
    loc_net = grid([convert(Int64, N_location^0.5); convert(Int64, N_location^0.5)]);
    ###building distance matrix
    net_weight = fill(0.0, N_location, N_location);
    #Distance between pair of connected locations is random uniform
    for i=1:N_location
        for j in neighbors(loc_net, i)
            if j < i
                net_weight[i,j] = 0.75 + 0.5 * rand(Beta(1.0, 0.5))
                net_weight[j,i] = net_weight[i,j]
            end
        end
    end
    ##To store distances and parents in shortest path
    node_dist = fill(0.0,  N_location, N_location)
    sp_parent = fill(0,  N_location, N_location)
    ###Storing shortest paths and parents (in rows are the consumers, columns are for sellers)
    for l=1:N_location
        s_p_c = desopo_pape_shortest_paths(loc_net, l, net_weight)
        node_dist[:, l] .= s_p_c.dists
        sp_parent[:, l] .= s_p_c.parents
    end
    
    SE = SpatialEconomy(; node_dist = node_dist, loc_sp_parent = sp_parent,
                        dist_init_cost = dist_init_cost, pop_grth_rate = pop_grth_rate, share_agr_to_man = share_agr_to_man,
                        man_sat = man_sat, housing_share = housing_share, housing_sat = housing_sat,
                        agr_share = agr_share, agr_sat = agr_sat, agr_end_share = agr_end_share, 
                        mean_inc_agr = mean_inc_agr, mean_inc_ser = mean_inc_ser,
                        loc_pop = loc_pop, agr_tech = agr_tech, agr_ret_scale = agr_ret_scale,
                        man_tech = man_tech, man_ret_scale = man_ret_scale, c_capital = c_capital, markup = markup,
                        wage_sens = wage_sens, distr_markup = distr_markup, 
                        motility = motility, mig_size_effect = mig_size_effect, mig_dist_fric = mig_dist_fric, 
                        mig_link_intens = mig_link_intens, mig_link_inert = mig_link_inert,
                        pop_fixed = pop_fixed, bin_width = bin_width, wc_sample_size = wc_sample_size, tol = tol)

    return SE;
end


##Saves the whole SpatialEconomy into a file
function dump_state(SE::SpatialEconomy, file_path)
    serialize(file_path, SE)
end

####Allows to have a time moving average
mutable struct SEDemoData ###Contains demographic data of SE
    AGR_POP::Array{Float64, 1}
    MAN_POP::Array{Float64, 1}
    SER_POP::Array{Float64, 1}
    SER_TP_SHARE::Array{Float64, 1} ##Share of population working in transportation
    SER_DIST_SHARE::Array{Float64, 1} ##Share of population working in distribution
    SER_HOUS_SHARE::Array{Float64, 1} ##Share of population working in housing
    n_avg::Int64 ###Number of observations added, to divide by to have average
end

###Constructor
function SEDemoData(SE::SpatialEconomy)
    SEDemoData(SE.loc_pop[1:3:(3*SE.N_location)], 
               SE.loc_pop[2:3:(3*SE.N_location)], 
               SE.loc_pop[3:3:(3*SE.N_location)], 
               share_transportation_ser(SE),
               share_dist_ser(SE),
               (view(SE.loc_hous_spending, 2:3:(3*SE.N_location)) .+ view(SE.loc_hous_spending, 3:3:(3*SE.N_location))) ./ SE.loc_ser_tot_inc,
               1)
end

function add_obs!(dd::SEDemoData, SE::SpatialEconomy)
    dd.AGR_POP .+= view(SE.loc_pop, 1:3:(3*SE.N_location))
    dd.MAN_POP .+= view(SE.loc_pop, 2:3:(3*SE.N_location))
    dd.SER_POP .+= view(SE.loc_pop, 3:3:(3*SE.N_location))
    dd.SER_TP_SHARE .+= share_transportation_ser(SE)
    dd.SER_DIST_SHARE .+= share_dist_ser(SE)
    dd.SER_HOUS_SHARE .+= (view(SE.loc_hous_spending, 2:3:(3*SE.N_location)) .+ view(SE.loc_hous_spending, 3:3:(3*SE.N_location))) ./ SE.loc_ser_tot_inc
    dd.n_avg += 1
    return nothing
end

function export_data(dd::SEDemoData, ts, file_path::String, appnd::Bool=false)
    (length(file_path) > 3 && file_path[end-2:end] == ".gz") || error("File should have .gz extension.")
    loc_data = DataFrame(LOC_ID = 1:length(dd.AGR_POP), 
                         AGR_POP = dd.AGR_POP ./ dd.n_avg, 
                         MAN_POP = dd.MAN_POP ./ dd.n_avg, 
                         SER_POP = dd.SER_POP ./ dd.n_avg,
                         SER_TP_SHARE = dd.SER_TP_SHARE ./ dd.n_avg,
                         SER_DIST_SHARE = dd.SER_DIST_SHARE ./ dd.n_avg,
                         SER_HOUS_SHARE = dd.SER_HOUS_SHARE ./ dd.n_avg,
                         TIME_STEP = ts)
    write_csv(file_path, loc_data, compress = true, append = appnd)
    return nothing
end

mutable struct SEEcoData
    MAN_EMPL::Array{Float64, 1}
    MAN_UNEMPL::Array{Float64, 1}
    MEAN_CAP::Array{Float64, 1}
    MEAN_UC::Array{Float64, 1}
    N_FIRM::Array{Int64, 1}
    MEAN_MAN_WAGE_EMPL::Array{Float64, 1}
    MEAN_AGR_WAGE::Array{Float64, 1}
    MEAN_SER_WAGE::Array{Float64, 1}
    AGR_AVG_WEALTH::Array{Float64, 1}
    MAN_AVG_WEALTH::Array{Float64, 1}
    SER_AVG_WEALTH::Array{Float64, 1}
    AGR_AVG_UTILITY::Array{Float64, 1}
    MAN_AVG_UTILITY::Array{Float64, 1}
    SER_AVG_UTILITY::Array{Float64, 1}
    HOUSING_PRICE::Array{Float64, 1}
    AGR_TOT_QSOLD::Array{Float64, 1}
    MAN_TOT_QSOLD::Array{Float64, 1}
    AGR_TOT_QEXP::Array{Float64, 1}
    MAN_TOT_QEXP::Array{Float64, 1}
    AGR_AVG_DST_TRSP::Array{Float64, 1}
    MAN_AVG_DST_TRSP::Array{Float64, 1}
    n_avg::Int64
end

###Constructor
function SEEcoData(SE::SpatialEconomy)
    agr_qsold = loc_agr_qsold(SE)

    SEEcoData(view(SE.loc_pop, 2:3:(3*SE.N_location)) .- SE.loc_unempl, ##MAN_EMPL
              copy(SE.loc_unempl), ##MAN_UNEMPL
             [any(SE.firm_loc[f] == l for f in SE.firm_active) ? 
             mean(SE.firm_capital[f] for f in SE.firm_active if SE.firm_loc[f] == l) : 
             NaN for l=1:SE.N_location], ###MEAN_CAP
             loc_avg_unit_cost(SE), ###MEAN_UC
             isempty(SE.firm_active) ? fill(0, SE.N_location) : 
             [sum(SE.firm_loc[f] == l for f in SE.firm_active) for l=1:SE.N_location], ###N_FIRM
             isempty(SE.firm_act_prod) ? fill(NaN, SE.N_location) :
             [any(SE.firm_loc[f] == l for f in SE.firm_act_prod) ? 
             sum(SE.firm_wage[f]*SE.firm_labor_use[f] for f in SE.firm_act_prod if SE.firm_loc[f] == l) / 
             sum(SE.firm_labor_use[f] for f in SE.firm_act_prod if SE.firm_loc[f] == l) : 
              NaN for l=1:SE.N_location], ###MEAN_MAN_WAGE_EMPL
             sumcols(SE.loc_agr_ship) ./ view(SE.loc_pop, 1:3:(3*SE.N_location)), #MEAN_AGR_WAGE
             SE.loc_ser_tot_inc ./ view(SE.loc_pop, 3:3:(3*SE.N_location)), #MEAN_SER_WAGE
             mean.(@view SE.loc_income[1:3:(3*SE.N_location)]), #AGR_AVG_WEALTH
             mean.(@view SE.loc_income[2:3:(3*SE.N_location)]), #MAN_AVG_WEALTH
             mean.(@view SE.loc_income[3:3:(3*SE.N_location)]), #SER_AVG_WEALTH
             SE.loc_avg_util[1:3:(3*SE.N_location)], #AGR_AVG_UTILITY
             SE.loc_avg_util[2:3:(3*SE.N_location)], #MAN_AVG_UTILITY
             SE.loc_avg_util[3:3:(3*SE.N_location)], #SER_AVG_UTILITY
             copy(SE.loc_housing_price), #HOUSING_PRICE
             agr_qsold, #AGR_TOT_QSOLD
             loc_man_qsold(SE), #MAN_TOT_QSOLD
             agr_qsold .- (SE.loc_agr_ship[l, l] for l=1:SE.N_location), #AGR_TOT_QEXP
             loc_man_qexport(SE), #MAN_TOT_QEXP
             loc_agr_avg_qtrsp(SE),#AGR_AVG_DST_TRSP
             loc_man_avg_qtrsp(SE),#MAN_AVG_DST_TRSP
             1)

end

function add_obs!(ed::SEEcoData, SE::SpatialEconomy)
    ed.MAN_EMPL .+= view(SE.loc_pop, 2:3:(3*SE.N_location)) .- SE.loc_unempl ##MAN_EMPL
    ed.MAN_UNEMPL .+= SE.loc_unempl ##MAN_UNEMPL
    for l=1:SE.N_location
        if any(SE.firm_loc[f] == l for f in SE.firm_active)
            ed.MEAN_CAP[l] += mean(SE.firm_capital[f] for f in SE.firm_active if SE.firm_loc[f] == l)
            ed.N_FIRM[l] += sum(SE.firm_loc[f] == l for f in SE.firm_active)
            ed.MEAN_MAN_WAGE_EMPL[l] += sum(SE.firm_wage[f]*SE.firm_labor_use[f] for f in SE.firm_active if (SE.firm_loc[f] == l) & (SE.firm_labor_use[f] > 0.0)) / 
                                        sum(SE.firm_labor_use[f] for f in SE.firm_active if (SE.firm_loc[f] == l) & (SE.firm_labor_use[f] > 0.0))
        else
            ed.MEAN_CAP[l] = NaN
            ed.MEAN_MAN_WAGE_EMPL[l] = NaN
        end
    end ##MEAN_MAN_WAGE_EMPL, MEAN_CAP
    ed.MEAN_UC .+= loc_avg_unit_cost(SE)
    ed.MEAN_AGR_WAGE .+= sumcols(SE.loc_agr_ship) ./ view(SE.loc_pop, 1:3:(3*SE.N_location)) #MEAN_AGR_WAGE
    ed.MEAN_SER_WAGE .+= SE.loc_ser_tot_inc ./ view(SE.loc_pop, 3:3:(3*SE.N_location)) #MEAN_SER_WAGE
    @views ed.AGR_AVG_WEALTH .+= mean.(SE.loc_income[1:3:(3*SE.N_location)]) #AGR_AVG_WEALTH
    @views ed.MAN_AVG_WEALTH .+= mean.(SE.loc_income[2:3:(3*SE.N_location)]) #MAN_AVG_WEALTH
    @views ed.SER_AVG_WEALTH .+= mean.(SE.loc_income[3:3:(3*SE.N_location)]) #SER_AVG_WEALTH
    @views ed.AGR_AVG_UTILITY .+= SE.loc_avg_util[1:3:(3*SE.N_location)] #AGR_AVG_UTILITY
    @views ed.MAN_AVG_UTILITY .+= SE.loc_avg_util[2:3:(3*SE.N_location)] #MAN_AVG_UTILITY
    @views ed.SER_AVG_UTILITY .+= SE.loc_avg_util[3:3:(3*SE.N_location)] #SER_AVG_UTILITY
    ed.HOUSING_PRICE .+= SE.loc_housing_price #HOUSING_PRICE
    agr_qsold = loc_agr_qsold(SE)
    ed.AGR_TOT_QSOLD .+= agr_qsold #TOT_AGR_QSOLD
    ed.MAN_TOT_QSOLD .+= loc_man_qsold(SE) #TOT_MAN_QSOLD
    ed.AGR_TOT_QEXP .+= agr_qsold .- (SE.loc_agr_ship[l, l] for l=1:SE.N_location) #TOT_AGR_QEXP
    ed.MAN_TOT_QEXP .+= loc_man_qexport(SE) #TOT_MAN_QEXP
    ed.AGR_AVG_DST_TRSP .+= loc_agr_avg_qtrsp(SE)
    ed.MAN_AVG_DST_TRSP .+= loc_man_avg_qtrsp(SE)
    ed.n_avg += 1
    return nothing
end

function export_data(ed::SEEcoData, ts, file_path::String, appnd::Bool=false)
    (length(file_path) > 3 && file_path[end-2:end] == ".gz") || error("File should have .gz extension.")

    @views loc_data = DataFrame(LOC_ID = 1:length(ed.MAN_EMPL),
                                MAN_EMPL = ed.MAN_EMPL ./ ed.n_avg,
                                MAN_UNEMPL = ed.MAN_UNEMPL ./ ed.n_avg,
                                MEAN_CAP = ed.MEAN_CAP ./ ed.n_avg,
                                MEAN_UC = ed.MEAN_UC ./ ed.n_avg,
                                N_FIRM = ed.N_FIRM ./ ed.n_avg,
                                MEAN_MAN_WAGE_EMPL = ed.MEAN_MAN_WAGE_EMPL ./ ed.n_avg,
                                MEAN_AGR_WAGE = ed.MEAN_AGR_WAGE ./ ed.n_avg,
                                MEAN_SER_WAGE = ed.MEAN_SER_WAGE ./ ed.n_avg,
                                AGR_AVG_WEALTH = ed.AGR_AVG_WEALTH ./ ed.n_avg,
                                MAN_AVG_WEALTH = ed.MAN_AVG_WEALTH ./ ed.n_avg, 
                                SER_AVG_WEALTH = ed.SER_AVG_WEALTH ./ ed.n_avg,
                                AGR_AVG_UTILITY = ed.AGR_AVG_UTILITY ./ ed.n_avg, 
                                MAN_AVG_UTILITY = ed.MAN_AVG_UTILITY ./ ed.n_avg, 
                                SER_AVG_UTILITY = ed.SER_AVG_UTILITY ./ ed.n_avg,
                                HOUSING_PRICE = ed.HOUSING_PRICE ./ ed.n_avg,
                                AGR_TOT_QSOLD = ed.AGR_TOT_QSOLD ./ ed.n_avg,
                                MAN_TOT_QSOLD = ed.MAN_TOT_QSOLD ./ ed.n_avg,
                                AGR_TOT_QEXP = ed.AGR_TOT_QEXP ./ ed.n_avg,
                                MAN_TOT_QEXP = ed.MAN_TOT_QEXP ./ ed.n_avg,
                                AGR_AVG_DST_TRSP = ed.AGR_AVG_DST_TRSP ./ ed.n_avg,
                                MAN_AVG_DST_TRSP = ed.MAN_AVG_DST_TRSP ./ ed.n_avg,
                                TIME_STEP = ts)
    
    write_csv(file_path, loc_data, compress = true, append = appnd)
    return nothing
end
