import os
import logging
from itertools import repeat, product
import multiprocessing as mp

import pandas as pd

from oemof.network.graph import create_nx_graph
from oemof import solph
import pyomo.environ as po

from q100opt import plots as plots
from q100opt.scenario_tools import DistrictScenario
from q100opt.setup_model import load_csv_data
from q100opt import postprocessing as qpp
import postprocessing as pp

table_collection = load_csv_data("data")

timesteps = 10
ds = DistrictScenario(
    name="Scenario1",
    table_collection=table_collection,
    number_of_time_steps=timesteps,
    year=2018,
    emission_limit=10 * 50500,
)

ds.solve(solver="gurobi")
optimal_obj_val = ds.model.objective()
optimal_results = pp.get_optimal_results(ds, ds.results["main"])


qpp.get_all_sequences(ds.results["main"])


epsilon_range = [0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1]
result_list = []
for eps in epsilon_range:
    for sense in [po.minimize, po.maximize]:
        result_collection = {}
        for invest_var in ds.model.InvestmentFlow.invest.keys():

            # Add near-optimal-solution constraint
            ds.model.InvestmentFlow.del_component("investment_costs")
            ds.model.GenericInvestmentStorageBlock.del_component(
                "investment_costs"
            )
            objective = sum(
                [
                    block._objective_expression()
                    for block in ds.model.component_data_objects()
                    if hasattr(block, "_objective_expression")
                ]
            )
            ds.model.near_opt_costraints = po.Constraint(
                expr=(objective - optimal_obj_val * (1 + eps) <= 0)
            )
            # replace old objective
            ds.model.del_component("objective")
            # maximize / minimize technology
            ds.model.objective = po.Objective(
                expr=ds.model.InvestmentFlow.invest[invest_var], sense=sense
            )
            # solve model
            ds.model.solve(solver="gurobi")
            # get results and extract investment
            results = solph.processing.results(ds.model)
            result_collection[(invest_var[0].label, str(eps), str(sense))] = {}
            for flow in ds.model.InvestmentFlow.invest.keys():
                result_collection[(invest_var[0].label, str(eps), str(sense))][
                    flow[0].label
                ] = results[flow]["scalars"]["invest"]
            for node in ds.model.GenericInvestmentStorageBlock.invest.keys():
                result_collection[(invest_var[0].label, str(eps), str(sense))][
                    node.label
                ] = results[(node, None)]["scalars"]["invest"]
            ds.model.del_component("near_opt_costraints")
        result_list.append(result_collection)

for eps in epsilon_range:
    for sense in [po.minimize, po.maximize]:
        result_collection = {}
        for invest_var in ds.model.GenericInvestmentStorageBlock.invest.keys():
            result_collection[(invest_var.label, str(eps), str(sense))] = {}
            # Add near-optimal-solution constraint
            ds.model.InvestmentFlow.del_component("investment_costs")
            ds.model.GenericInvestmentStorageBlock.del_component(
                "investment_costs"
            )
            objective = sum(
                [
                    block._objective_expression()
                    for block in ds.model.component_data_objects()
                    if hasattr(block, "_objective_expression")
                ]
            )
            ds.model.near_opt_costraints = po.Constraint(
                expr=(objective - optimal_obj_val * (1 + eps) <= 0)
            )
            # replace old objective
            ds.model.del_component("objective")
            # maximize / minimize technology
            ds.model.objective = po.Objective(
                expr=ds.model.GenericInvestmentStorageBlock.invest[invest_var],
                sense=sense
            )
            # solve model
            ds.model.solve(solver="gurobi")
            # get results and extract investment
            results = solph.processing.results(ds.model)
            for flow in ds.model.InvestmentFlow.invest.keys():
                result_collection[(invest_var.label, str(eps), str(sense))][
                    flow[0].label
                ] = results[flow]["scalars"]["invest"]
            for node in ds.model.GenericInvestmentStorageBlock.invest.keys():
                result_collection[(invest_var.label, str(eps), str(sense))][
                    node.label
                ] = results[(node, None)]["scalars"]["invest"]
            ds.model.del_component("near_opt_costraints")
        result_list.append(result_collection)

df = pd.concat([optimal_results.T] + [pd.DataFrame(i).T for i in result_list])
if not os.path.exists("results"):
    os.makedirs("results")
df.to_csv("results/capacities.csv")

#
# def find_nos(optimal_obj_val, epsense):
#     """ """
#     eps = epsense[0]  # eps in eps_range={0.01...0.1}
#     sense = epsense[1]  # po.minimize or po.maximize
#
#     ds = DistrictScenario(
#         name="Scenario1",
#         table_collection=table_collection,
#         number_of_time_steps=timesteps,
#         year=2018,
#         emission_limit=10 * 50500,
#     )
#     ds.table2es()
#     ds.create_model()
#     ds.add_emission_constr()
#
#     result_collection = {}
#     for invest_var in ds.model.InvestmentFlow.invest.keys():
#         result_collection[(invest_var[0].label, str(eps), str(sense))] = {}
#         # Add near-optimal-solution constraint
#         ds.model.InvestmentFlow.del_component("investment_costs")
#         ds.model.GenericInvestmentStorageBlock.del_component(
#             "investment_costs"
#         )
#         objective = sum(
#             [
#                 block._objective_expression()
#                 for block in ds.model.component_data_objects()
#                 if hasattr(block, "_objective_expression")
#             ]
#         )
#         ds.model.near_opt_costraints = po.Constraint(
#             expr=(objective - optimal_obj_val * (1 + eps) <= 0)
#         )
#         # replace old objective
#         ds.model.del_component("objective")
#         # maximize / minimize technology
#         ds.model.objective = po.Objective(
#             expr=ds.model.InvestmentFlow.invest[invest_var], sense=sense
#         )
#         # solve model
#         ds.model.solve(solver="gurobi")
#         # get results and extract investment
#         results = solph.processing.results(ds.model)
#         for flow in ds.model.InvestmentFlow.invest.keys():
#             result_collection[(invest_var[0].label, str(eps), str(sense))][
#                 flow[0].label
#             ] = results[flow]["scalars"].values[0]
#         ds.model.del_component("near_opt_costraints")
#
#     return result_collection
#
#
# epsilon_range = [0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1]
# pool = mp.Pool(processes=4)
# final = pool.starmap(
#     find_nos,
#     [
#         i
#         for i in zip(
#             repeat(optimal_obj_val),
#             product(epsilon_range, [po.maximize, po.minimize]),
#         )
#     ],
# )
# df = pd.concat([pd.DataFrame(i).T for i in final])
# df = pd.concat([df, optimal_results])
#
# if not os.path.exists("results"):
#     os.makedirs("results")
# df.to_csv("results/capacities.csv")




try:
    import networkx as nx

    grph = create_nx_graph(ds.es)
    pos = nx.drawing.nx_agraph.graphviz_layout(grph, prog='neato')
    plots.plot_graph(pos, grph)
    logging.info('Energy system Graph OK.')

    # plot esys graph II (oemof examples)
    graph = create_nx_graph(ds.es)
    plt.figure(3,figsize=(10,10))
    plots.draw_graph(
        grph=graph,
        plot=True,
        layout="neato",
        node_size=500,
        node_color={"b_heat_gen": "#cd3333", "b_el_ez": "#cd3333"},
    )

except ImportError:
    print("Need to install networkx to create energy system graph.")
