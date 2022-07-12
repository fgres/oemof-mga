import os
import pandas as pd

from q100opt import plots as plots
from q100opt.scenario_tools import DistrictScenario
from q100opt.setup_model import load_csv_data

from plotting import plot_capacities, plot_envelopes
from mga import MGA

table_collection = load_csv_data("examples/q100opt/data")

timesteps = 12

ds = DistrictScenario(
    name="Scenario1",
    table_collection=table_collection,
    number_of_time_steps=timesteps,
    year=2018,
    emission_limit=50000,
)
ds.table2es()
ds.create_model()
ds.add_emission_constr()


nos = MGA(ds.model, solver="gurobi")
#nos.calculate_base_solution()


# nos.calculate_base_solution()
nos.explore_near_optimal_space(
    epsilon_range=[0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1]
)
nos.generate_invest_table()
nos.store_sequences()

df = nos.read_invest_table()

# ----------------------------------------------------------------------------
# Plot results
# ----------------------------------------------------------------------------
select = [
    "t_boiler",
    "t_hp_geo",
    "t_hp_air",
    "t_hp_ely",
    "t_chp",
    "pv",
    "t_H2_storage_compressor",
    "s_elec",
    "s_heat_d30",
]
techs = [
    (flow[0].label, flow[1].label)
    for (flow, f) in nos._model.es.flows().items()
    if flow[0].label in select and f.investment is not None
]

plot_capacities(df, techs=techs)
plot_envelopes(
    df,
    techs=techs,
    unit="kW",
    nicenames=dict(
        zip(techs, [i[0].replace("_storage_compressor", "") for i in techs])
    ),
)

storage_techs = [
    (node.label, node.label)
    for node in nos._model.es.nodes
    if hasattr(node, "investment") and node.investment is not None
]
plot_envelopes(
    df,
    techs=storage_techs,
    unit="kWh",
    nicenames=dict(zip(storage_techs, [i[0] for i in storage_techs])),
)

# ----------------------------------------------------------------------------
# Plot capacities
# ----------------------------------------------------------------------------

#df.sort_index(inplace=True)
eps = [i for i in df.index.get_level_values(2).unique()][:-1]
epsmap = dict(zip(range(len(eps)), eps))
import matplotlib.pyplot as plt

fig, axs = plt.subplots(2, len(eps), figsize=(15, 5), sharex=True, sharey=True)

kwargs = dict(kind="bar", stacked=True, legend=False, width=0.9, cmap="Set1")
for idx, eps in epsmap.items():
    tech_df = df.xs(eps, level=2, axis=0)[techs].unstack()
    tech_df.xs(1, level=2, axis=1).loc[techs].plot(ax=axs[1, idx], **kwargs)

    tech_df.xs(-1, level=2, axis=1).loc[techs].plot(ax=axs[0, idx],**kwargs)


    axs[1, idx].set_xticklabels([i[0].replace("_storage_compressor", "") for i in techs], fontsize=12)
    axs[0, idx].set_title(r"$\epsilon=$" + str(eps))

axs[0, 0].set_ylabel("Max")
axs[1, 0].set_ylabel("Min")

handles, labels = axs[0, 0].get_legend_handles_labels()
legend = fig.legend(
    handles[::-1],
    [i[0].replace("_storage_compressor", "") for i in techs][::-1], #labels[::-1],
    ncol=8,
    loc="upper left",
    bbox_to_anchor=(0.2, -0.11),
    frameon=False,
)
axs[0, 0].add_artist(legend)
