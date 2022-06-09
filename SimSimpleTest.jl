#using BenchmarkTools
using StatsBase: var, quantile, mean, cor
using Gadfly
using Plots: scatter, scatter!, plot as plt, plot! as plt!, histogram, histogram!
#using Plots: histogram, histogram!, scatter, scatter!
#using LightGraphs: degree, enumerate_paths, dijkstra_shortest_paths, nv, erdos_renyi, 
#                   nv, neighbors, desopo_pape_shortest_paths, floyd_warshall_shortest_paths,
#                   edges, src, dst
using Distributions: quantile, Uniform, SymTriangularDist
using Images: RGB
#using Serialization
#using GraphPlot: gplot, spectral_layout
#using CatViews
#using Profile
#using Fontconfig, Cairo
#using Compose
#using Compose: @colorant_str, LCHab, Gray

#For debugging
#/(a::Float64, b::Float64) = (a == 0.0) & (b == 0.0) ? error("got 0/0") : Base.:/(a,b)
#/(a::R, b::S) where {R <: Real, S <: Real} = (a == zero(R)) & (b == zero(S)) ? error("got 0/0") : Base.:/(a,b)

cd("/Volumes/Donnees/switchdrive/These_moi/Code/UrbIndUS")

folder_name = "SpatialEconomy_AF"
#include("SpatialEconomy-Ju1.3/IncDistNorm.jl")
include(folder_name * "/IncDistBinned.jl")
include(folder_name * "/Utils.jl")
include(folder_name * "/SpatialEconomy.jl")
include(folder_name * "/SpatialEconomyUpdate.jl")
include(folder_name * "/SpatialEconomyStats.jl")

##Moving average
ma(val, delay::Int64) = [mean(view(val, i:(i+delay))) for i=1:(length(val)-delay)]
##Calculates slope of limple linear regression for y ~ x
N_location = 8 * 8;
pop_orig = vcat([[1000.0 * rand(), 0.0, 200.0 * rand()] for i=1:N_location]...);


###Choose theme for plotting
#Gadfly.push_theme(Theme(panel_fill = "white", background_color = "white", key_position = :bottom))

include(folder_name * "/SpatialEconomyUpdate.jl")

####Two things : reduce man_tech
####Raise the migration threshold


###Parameters to explore:
###a_rs, m_rs, man_sat, inc_mult, agr_sat, motility, mig_dist_fric, mig_link_intens, c_cap,
###a_sat, init_dist, h_share, dest_f_mult

a_sat = 3.0; a_precons_sh = 0.0;
a_rs = .96; m_rs = 1.4;
gr = 0.00;
h_share = 0.3; h_sat = 0.8; inc_mult = 1.8;
c_cap = 2.0; di_m = 0.1;
pop = copy(pop_orig);
tot_pop = sum(pop);
max_iter = 1000; dist_iter = 450 #130
##Minimum agriculture technology to sustain a fully dispersed population
m_a_tech = a_sat; #* ((1+gr)^max_iter * (sum(pop) - sum(pop[3 .* (1:N_location)])) / N_location)^(1-a_rs);
migr_thresh = 0.15;


SE = Simple_SpatialEconomy(N_location = N_location, dist_init_cost= 1.0, pop_grth_rate = gr, share_agr_to_man = 0.002,
                           man_sat = 100000.0, housing_share = h_share, housing_sat = h_sat,
                           agr_share = 1.0 - h_share, agr_sat = a_sat, agr_end_share = a_precons_sh,
                           agr_tech = 2.0 * m_a_tech, agr_ret_scale = a_rs,
                           mean_inc_agr = inc_mult * (1.0 - a_precons_sh) * a_sat * (1 + di_m),
                           mean_inc_ser = inc_mult * a_sat * (1 + di_m) / (1.0 - h_sat*h_share), loc_pop = pop,
                           man_ret_scale = m_rs, man_tech = 4.0,
                           c_capital = c_cap, distr_markup = di_m,  markup = 0.1, wage_sens = 0.05, 
                           motility = 0.1, mig_size_effect = 0.0, mig_link_intens = 0.0, mig_dist_fric = .1, mig_link_inert = 0.0,
                           pop_fixed = mean(pop)*10^-3, wc_sample_size = 450, tol = 10^-7,
                           bin_width = 0.1 * inc_mult * a_sat);
##Remove man sat, large man_sat seems to cause a problem, check that!


##Remove previous link effect of migration 
begin
urb_share = Float64[];
     unempl = Float64[];
     man_empl = Float64[];
     a_share = Float64[]; sr_share = Float64[]; su_share = Float64[];
     avg_cap = Float64[]; avg_w = Float64[];
     c_w_c = Float64[]; s_na_m = Float64[];
     c_m_us = Float64[];
     a_u_a = Float64[]; a_u_m = Float64[]; a_u_rs = Float64[]; a_u_us = Float64[];
     q_p = Float64[]; q_s = Float64[];
     trsp_agr = Float64[]; trsp_man = Float64[];
     loc_prod_shift = Float64[]; sc_prod = Float64[]; sc_uc = Float64[];
end

iter = 1;
@time while iter <= max_iter
     grow_pop!(SE)
     print("Iteration: $iter ; ")
     push!(c_m_us, cor(SE.loc_pop[2:3:(3*SE.N_location)], SE.loc_pop[3:3:(3*SE.N_location)]))
     iter > dist_iter && update_distance!(SE, 0.99, 0.06) #(SE, 0.975, 0.15) 
     print("firm entry ; ")
     iter > 1 && firms_entry!(SE, 50, 0.5, iter)
     print("MIN ACTIVE CAPITAL: ", isempty(SE.firm_active) ? NaN : minimum(SE.firm_capital[SE.firm_active]), " ")
     print("LENGTH FIRM LOC: ", length(SE.firm_loc), " ; ")
     #firms_set_quantities_wages!(SE, iter)
     iter > 1 && set_quantities_wages!(SE, iter, 0.07, 0.33333, 0.33333, 0.1)
     #display(SE.firm_loc_prod[SE.firm_active, :])
     #readline()
     #iter % 20 == 0 && @views scatter(SE.firm_capital[SE.firm_active], SE.firm_wage[SE.firm_active], xaxis = :log, yaxis = :log)
     print("labor matching ; ")
     match_labor!(SE)
     recruited = @views sum(SE.firm_labor_use[SE.firm_active])
     match_labor_from_agr!(SE)
     print("Recruited from man : ", recruited, " ; ")
     print("Recruited from agr: ", sum(SE.firm_labor_use[SE.firm_active]) - recruited, " ; ")
     #match_labor_CES!(SE, 3.0)
     print("Total employed labor: ", sum(SE.firm_labor_use[SE.firm_active]), " ; ")
     println("Total unemployment labor: ", tot_man_unempl(SE))
     #push!(s_na_m, sc_nonagr_manempl(SE))
     production!(SE)
     firms_set_prices!(SE, .15)
     print("Consumption : ")
     push!(sc_prod, sc_pop_productivity(SE))
     consume_housing!(SE)
     print("H done ; ")
     consume_agr_rq!(SE, 0.75)
     print("A done ; ")
     #println("Mean agr consumption: ", mean.(view(SE.loc_real_a_cons, SE.loc_inh_ind)))
     consume_man_rq!(SE, 0.75)
     print("M done ; ")
     #println("Mean man consumption: ", mean.(view(SE.loc_real_m_cons, SE.loc_inh_ind)))
     #consume_CES!(SE, 0.5, 3.0)
     push!(trsp_agr, transported_agr(SE) / sum(SE.loc_agr_ship))
     push!(trsp_man, transported_man(SE) / @views sum(SE.firm_loc_ship[SE.firm_act_prod, :]))
     #print("total money supply AC: ", sum(SE.loc_hous_spending) + sum(SE.loc_agr_spending) + sum(SE.firm_loc_price[SE.firm_act_prod, SE.loc_any_inh_ind] .* SE.firm_loc_ship[SE.firm_act_prod, SE.loc_any_inh_ind]) +
     #                                   sum(SE.firm_assets[SE.firm_active]) + sum(mean.(SE.loc_real_inc[SE.loc_inh_ind]) .* SE.loc_pop[SE.loc_inh_ind]), " ; ")
     cons_utility!(SE, SE.agr_sat * (1.0 - SE.tol), SE.housing_sat * (1.0 - SE.tol), 0.75)
     push!(q_s, tot_man_qsold(SE)) ; push!(q_p, tot_man_qsold(SE))
     push!(a_u_a, avg_util_agr(SE)) ; push!(a_u_m, avg_util_man(SE)) ; push!(a_u_us, avg_util_ser(SE))
     print("Income update ; ")
     #println("Producing firms: ",SE.firm_act_prod)
     #println("Previous wages: ", SE.loc_avg_man_wage_prev)
     update_income_realizations!(SE, 0.75, 0.75, 0.75)
     println("Sold by firms: ", isempty(SE.firm_active) ? 0.0 : sum(SE.firm_q_sold[f] for f in SE.firm_act_prod))
     push!(man_empl, tot_man_empl(SE))
     push!(unempl, tot_man_unempl(SE))
     push!(a_share, @views sum(SE.loc_pop[1:3:(3*SE.N_location)]) / sum(SE.loc_pop))
     push!(su_share, @views sum(SE.loc_pop[3*l] for l=1:SE.N_location) / sum(SE.loc_pop))
     push!(urb_share, urban_share(SE))
     println("total money supply : ", sum(SE.firm_assets[SE.firm_active]) + sum(mean.(SE.loc_real_inc[SE.loc_inh_ind]) .* SE.loc_pop[SE.loc_inh_ind]))
     push!(avg_cap, avg_capital(SE))
     push!(avg_w, avg_wage(SE))
     push!(c_w_c, cor_wage_capital(SE))
     length(SE.firm_active) > 0 && print("Scaling coeff capital wage: ",
               simple_reg_slope(log.(SE.firm_capital[SE.firm_active]), log.(SE.firm_wage[SE.firm_active])), " ; ")
     print("Population: $(sum(SE.loc_pop)) ; ")
     print("Migration ; ")
     migrate_cs_part!(SE, 0.05 * migr_thresh, migr_thresh, 0.01)
     #migrate_CI!(SE, 0.1 * SE.agr_sat^SE.agr_share * SE.housing_sat^SE.housing_share, 3.0)
     update_population!(SE)
     #println("Net mig man: ", SE.loc_pop[4 .* (1:SE.N_location) .- 2] .- pop_pre_mig[4 .* (1:SE.N_location) .- 2])
     print("Population: $(sum(SE.loc_pop)) ; ")
     println("total money supply : ", sum(SE.firm_assets[SE.firm_active]) + sum(mean.(SE.loc_income[SE.loc_inh_ind]) .* SE.loc_pop[SE.loc_inh_ind]))
     println("Share of Money in equity: ", sum(SE.firm_assets[SE.firm_active]) / (sum(SE.firm_assets[SE.firm_active]) + sum(mean.(SE.loc_income[SE.loc_inh_ind]) .* SE.loc_pop[SE.loc_inh_ind])))
     println("total firm active", length(SE.firm_active))
     println("Length vector of firms: $(length(SE.firm_loc))" )
     println("inc : ",sum(mean.(SE.loc_income[2:3:(3*SE.N_location)]) .* SE.loc_pop[2:3:(3*SE.N_location)]), " pop : ",
             sum(SE.loc_pop[3:3:(3*SE.N_location)]))
     iter += 1
end
#145, 137
##Returns to scale and wage : people with lower capital can afford lower wages
##do in the end they actually work at increasing returns 
delay = 15;

####Urban Share
#plot(y=ma(urb_share, delay), Guide.title("Urbanization share"), Geom.line,
#     layer(xintercept = [dist_iter - delay], Geom.vline(style=:dot), Theme(default_color=colorant"black")))
###Employment
plot(layer(xintercept = [dist_iter - delay], Geom.vline(style=:dot), Theme(default_color=colorant"black")), 
     layer(y=ma(unempl, delay), Theme(default_color=colorant"red"), Geom.line),
     layer(y=ma(man_empl, delay), Theme(default_color=colorant"green"), Geom.line),  
     Guide.manual_color_key("", ["Employment", "Unemployment"], ["green", "red"]),
     Guide.title("Employment"))
####Manufacturing share
#scatter(SE.firm_capital[SE.firm_active], SE.firm_wage[SE.firm_active], yaxis = :log)
#scatter(SE.firm_labor_use[SE.firm_active] ./ SE.firm_capital[SE.firm_active], SE.firm_wage[SE.firm_active])
plot(y=avg_w, Geom.line)

scatter(mean.(SE.loc_income), groups = repeat(collect(1:3), SE.N_location))

plot(y=ma(man_empl .+ unempl, delay) ./ ma(sum(SE.loc_pop) ./ (1+gr).^((iter-1):-1:1), delay), Geom.line,  Guide.title("Manufacturing share"),
     layer(xintercept = [dist_iter - delay], Theme(default_color=colorant"black"), Geom.vline(style=:dot)))
####Average capital
plot(y=ma(avg_cap, delay), Geom.line,  Guide.title("Average capital"),
     layer(xintercept = [dist_iter], Theme(default_color=colorant"black"), Geom.vline(style=:dot)))
####Populations
plot(layer(y=ma((man_empl .+ unempl) ./ (sum(SE.loc_pop) ./ (1+gr).^((iter-1):-1:1)), delay), Theme(default_color=colorant"blue"), Geom.line),
     layer(y=ma(a_share, delay), Theme(default_color=colorant"green"), Geom.line),
     layer(y=ma(su_share, delay), Theme(default_color=colorant"red"), Geom.line),
     Guide.manual_color_key("", ["Agr.", "Man.", "Ser."], ["green", "blue", "red"]),
     layer(xintercept = [dist_iter - delay], Theme(default_color=colorant"black"), Geom.vline(style=:dot)),
     Guide.title("Share of populations"))

####Utilities per capita
plot(layer(y=ma(a_u_a, delay), Theme(default_color=colorant"green"), Geom.line),
     layer(y=ma(a_u_m, delay), Theme(default_color=colorant"blue"), Geom.line),
     layer(y=ma(a_u_rs, delay), Theme(default_color=colorant"yellow"), Geom.line),
     layer(y=ma(a_u_us, delay), Theme(default_color=colorant"red"), Geom.line),
     Guide.manual_color_key("", ["Agr.", "Man", "R. Ser.", "U. Ser."], ["green", "blue", "yellow", "red"]),
     layer(xintercept = [dist_iter - delay], Theme(default_color=colorant"black"), Geom.vline(style=:dot)),
     Guide.title("Utilities per capita"))

###Correlation manufacturing urban services
plot(y = c_m_us, Geom.line,
     layer(xintercept = [dist_iter - delay], Theme(default_color=colorant"black"), Geom.vline(style=:dot)))

####Manufacturing consumption
plot(layer(y=ma(q_p, delay), Theme(default_color=colorant"green"), Geom.line),
     layer(y=ma(q_s, delay), Theme(default_color=colorant"red"), Geom.line),
     layer(xintercept = [dist_iter - delay], Geom.vline(style=:dot), Theme(default_color=colorant"black")), 
     layer(x = [1], y= [2.0], label = ["Max cons: $(SE.man_sat * sum(SE.loc_pop))"],  Geom.label),
     Guide.title("Manufacturing consumption"))

####Rank-size
@views reg = simple_reg(log.(sort([sum(SE.loc_pop[[(3i-1),(3i)]]) for i=1:SE.N_location], rev = true)),
                        log.(1:SE.N_location))

@views plot(layer(x= 1:SE.N_location, y = sort([sum(SE.loc_pop[[(3i-1),(3i)]]) for i=1:SE.N_location], rev = true), Geom.point),
            layer(x = 1:SE.N_location, y = exp(reg[1]) .* (1:SE.N_location).^reg[2], Geom.line),
            Scale.x_log10, Scale.y_log10,
            layer(x = [1], y= [exp(reg[1]) .* SE.N_location.^reg[2]], label = ["Slope coeff: $(reg[2])"],  Geom.label))
@views plot(layer(x= 1:SE.N_location, y = sort([SE.loc_pop[3i-1] for i=1:SE.N_location], rev = true), Geom.point),
            Scale.x_log10, Scale.y_log10)

####Share of Quantities transported
plot(layer(y=trsp_agr, Theme(default_color=colorant"green"), Geom.line),
     layer(y=trsp_man, Theme(default_color=colorant"blue"), Geom.line),
     layer(xintercept = [dist_iter - delay], Geom.vline(style=:dot), Theme(default_color=colorant"black")),
     Scale.y_continuous(minvalue=0.0, maxvalue=1.0))

scatter(SE.loc_avg_util, groups = repeat(collect(1:3), SE.N_location), color = [:green :blue :red])

histogram(SE.firm_entry_time[SE.firm_active])

plot(y=SE.loc_housing_price, x=SE.loc_pop[2:3:(3*SE.N_location)].+SE.loc_pop[3:3:(3*SE.N_location)], 
     Scale.x_log10, Scale.y_log10, Geom.point)

plot(y=SE.loc_housing_price, x=SE.loc_pop[2:3:(3*SE.N_location)].+SE.loc_pop[3:3:(3*SE.N_location)], 
     Geom.point)



N_l_sr = convert(Int64, SE.N_location^0.5);

sq_tot_pop = reshape(SE.loc_pop[3 .* (1:N_location)] .+ 
                     SE.loc_pop[3 .* (1:N_location) .- 1] .+ 
                     SE.loc_pop[3 .* (1:N_location) .- 2], N_l_sr, N_l_sr);

sq_max_pop = maximum(SE.loc_pop[3 .* (1:N_location)] .+ 
                     SE.loc_pop[3 .* (1:N_location) .- 1] .+ 
                     SE.loc_pop[3 .* (1:N_location) .- 2]);

sq_max_pop_c = maximum.((SE.loc_pop[3 .* (1:N_location)] .+ 
                         SE.loc_pop[3 .* (1:N_location) .- 1] .+ 
                         SE.loc_pop[3 .* (1:N_location) .- 2]));

RGB.(reshape(SE.loc_pop[3 .* (1:N_location)], N_l_sr, N_l_sr) ./ sq_max_pop_c[1],
     reshape(SE.loc_pop[3 .* (1:N_location) .- 2], N_l_sr, N_l_sr) ./ sq_max_pop_c[3],
     reshape(SE.loc_pop[3 .* (1:N_location) .- 1], N_l_sr, N_l_sr) ./ sq_max_pop_c[2])

###Firms distrbutions
plot(y=sort(SE.firm_capital[SE.firm_active], rev = true), x=1:length(SE.firm_active), Geom.point,
     Scale.x_log10, Scale.y_log10)


histogram(vcat(SE.loc_real_inc[1:4:(4*N_location)]...))
histogram!(vcat(SE.loc_real_inc[2:4:(4*N_location)]...))
histogram!(vcat(SE.loc_real_inc[4:4:(4*N_location)]...))



###Total population vs employment
scatter(SE.loc_pop[2:3:(3*N_location)], [sum(SE.man_tech * SE.firm_labor_use[f]^SE.man_ret_scale for f in SE.firm_act_prod if SE.firm_loc[f] == l; init=0.0) for l=1:SE.N_location],
        xaxis = :log, yaxis = :log)
print(SE.loc_pop[2:3:(3*N_location)])
print([sum(SE.man_tech * SE.firm_labor_use[f]^SE.man_ret_scale for f in SE.firm_act_prod if SE.firm_loc[f] == l; init=0.0) for l=1:SE.N_location])

plot!(1.0:1.0:40, 1.0:1.0:40, legend = false)


simple_reg_slope([log(SE.man_tech * sum(SE.firm_labor_use[f]^SE.man_ret_scale for f in SE.firm_act_prod if SE.firm_loc[f] == l; init=0.0)) for l=1:SE.N_location], log.(SE.loc_pop[2:3:(3*N_location)]))


sum(SE.firm_q_prod[SE.firm_act_prod])


histogram(SE.firm_entry_time[SE.firm_active])
histogram(SE.firm_entry_time[SE.firm_act_prod])

scatter(SE.firm_entry_time[SE.firm_act_prod], SE.firm_q_prod[SE.firm_act_prod])

scatter(SE.firm_entry_time[SE.firm_act_prod], SE.firm_wage[SE.firm_act_prod])


simple_reg_slope(log.(tot_loc_pop(SE)), log.(SE.loc_pop[3 .* (1:SE.N_location) .- 2]))

argmax(log.(tot_loc_pop(SE)))

#####Share of pop that has income less than 0.001
scatter(cdf.(SE.loc_income[3 .* (1:N_location) .- 2], 0.001))
scatter!(cdf.(SE.loc_income[3 .* (1:N_location) .- 1], 0.001))
scatter!(cdf.(SE.loc_income[3 .* (1:N_location)], 0.001))
#####Mean utilities for wach location-sector
scatter([u for u in mean.(SE.loc_real_util[3 .* (1:N_location) .- 2])])
scatter!([u for u in mean.(SE.loc_real_util[3 .* (1:N_location) .- 1])])
scatter!([u for u in mean.(SE.loc_real_util[3 .* (1:N_location)])])
#####Share of pop that has utility less than 0.001
scatter([sum.(x -> x < 0.01, SE.loc_real_util[3 .* (1:N_location) .- 2])...])
scatter!([sum.(x -> x < 0.01, SE.loc_real_util[3 .* (1:N_location) .- 1])...])
scatter!([sum.(x -> x < 0.01, SE.loc_real_util[3 .* (1:N_location)])...])


dists = [Matrix{Float64}(read_csv("/Volumes/Donnees/switchdrive/These_moi/Part1-UrbInd/Data/Simulation_Input/DistancesNoSea/TN_Dist_$y.gz", DataFrame, header = false)) for y=1800:10:1920]

histogram(dists[12][1:end])

mean(SE.loc_dist)

plt(1800:10:1920, mean.(dists))


maximum(m)

(-1)^-true


mul(ws, b::Bool) = (1.0 + ws)^((-1)^b*rand())

mul(0.1, true)
mul(0.1, false)

l = [log(reduce(*, mul.(0.05, rand(1000) .> 0.5))) for i=1:10000]

mean(l)

reduce(*, mul.(0.004, rand(1000) .> 0.5))

mul(0.075, 1.0, 1.0)
mul(0.075, .7, 1.0)


Matrix{Int64}(read_csv("DistancesNoSea/TN_Parents_$(min(s_year, max_year_dist_net)).gz", DataFrame, header = false)))




population = read_csv("/Volumes/Donnees/switchdrive/These_moi/Part1-UrbInd/Data/Input_Simulation/County_Pop_Urb_NonUrb_85largest.csv", DataFrame, 
                      types = Dict("TOT_COUNTY_POP" => Float64, "URB_COUNTY_POP" => Float64, "TOT_POP" => Float64))


pop_mac = combine(groupby(select(population, [:TOT_COUNTY_POP, :YEAR]), :YEAR), :TOT_COUNTY_POP => sum)

mac_gr = diff(pop_mac.TOT_COUNTY_POP_sum) ./ pop_mac.TOT_COUNTY_POP_sum[1:end-1]

print(mac_gr[2:end])