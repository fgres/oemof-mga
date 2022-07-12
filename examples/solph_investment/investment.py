# -*- coding: utf-8 -*-
"""
"""
from oemof.tools import logger
from oemof.tools import economics
from oemof import solph

import logging
import os
import pandas as pd
import pprint as pp
from oemof.solph.modelling import MGA

number_timesteps = 8760
timeindex = pd.date_range("1/1/2012", periods=number_timesteps, freq="H")
energysystem = solph.EnergySystem(timeindex=timeindex)

# Read data file
full_filename = os.path.join("examples", "investment_data.csv")
data = pd.read_csv(full_filename, sep=",")

price_gas = 0.04

# If the period is one year the equivalent periodical costs (epc) of an
# investment are equal to the annuity. Use oemof's economic tools.
epc_onshore = economics.annuity(capex=1000, n=20, wacc=0.05)
epc_offshore = economics.annuity(capex=1400, n=20, wacc=0.05)
epc_pv = economics.annuity(capex=800, n=20, wacc=0.05)
epc_battery = economics.annuity(capex=6*100+94, n=20, wacc=0.05)
epc_h2 = economics.annuity(capex=27, n=30, wacc=0.05)

bel = solph.Bus(label="DE")

excess = solph.components.Sink(label="excess_bel", inputs={bel: solph.Flow()})
shortage = solph.components.Source(
    label="shortage_bel", inputs={bel: solph.Flow(variable_cost=10e6)})


onshore = solph.components.Source(
    label="onshore",
    outputs={
        bel: solph.Flow(
            fix=data["onshore"], investment=solph.Investment(ep_costs=epc_onshore)
        )
    },
)
offshore = solph.components.Source(
    label="offshore",
    outputs={
        bel: solph.Flow(
            fix=data["offshore"], investment=solph.Investment(ep_costs=epc_offshore)
        )
    },
)
pv = solph.components.Source(
    label="pv",
    outputs={
        bel: solph.Flow(
            fix=data["pv"], investment=solph.Investment(ep_costs=epc_pv)
        )
    },
)

demand = solph.components.Sink(
    label="demand",
    inputs={bel: solph.Flow(fix=data["demand_el"], nominal_value=1017)},
)


storage = solph.components.GenericStorage(
    label="battery_storage",
    inputs={bel: solph.Flow(variable_costs=0.0001)},
    outputs={bel: solph.Flow()},
    loss_rate=0.00,
    initial_storage_level=0,
    invest_relation_input_capacity=1 / 6,
    invest_relation_output_capacity=1 / 6,
    inflow_conversion_factor=1,
    outflow_conversion_factor=0.8,
    investment=solph.Investment(ep_costs=epc_storage),
)

storage = solph.components.GenericStorage(
    label="h2_storage",
    inputs={bel: solph.Flow(variable_costs=0.0001)},
    outputs={bel: solph.Flow()},
    loss_rate=0.00,
    initial_storage_level=0,
    invest_relation_input_capacity=1 / 168,
    invest_relation_output_capacity=1 / 168,
    inflow_conversion_factor=1,
    outflow_conversion_factor=0.4,
    investment=solph.Investment(ep_costs=epc_storage),
)

energysystem.add(excess, shortage, wind, pv, demand, storage)

model = solph.Model(energysystem)

# create MGA object to explore near-optimal solution space
nos = MGA(model, solver="gurobi")

nos.calculate_base_solution()

nos.explore_near_optimal_space()
