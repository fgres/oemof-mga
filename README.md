# oemof-mga

Modelling to Generate Alternatives (MGA) to explore near-optimal solutions with oemof.solph


## How to use

```
# use any oemof solph model to instantiate MGA-object
nos = MGA(model, solver="gurobi")

# get the optimal solution (optional)
nos.calculate_base_solution()

# explore near optimal solution space
nos.explore_near_optimal_space(
    epsilon_range=[0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1]
)

# get and store result data

# get all investment values (stores automatically)
nos.generate_invest_table()

# stores all sequences for generated solutions
nos.store_sequences()
```
