from matplotlib import pyplot as plt

from q100opt.scenario_tools import ParetoFront
from q100opt.setup_model import load_csv_data

table_collection = load_csv_data("examples/q100opt/data")
pf = ParetoFront(
    number_of_time_steps=8760,
    table_collection=table_collection,
    number_of_points=5,
)

pf.calc_pareto_front(solver='cbc', tee=True)

pf.results["pareto_front"].plot(x='emissions', y='costs', kind='scatter')
plt.xlabel('emissions')
plt.ylabel('costs')
plt.show()

pf.emission_limits
