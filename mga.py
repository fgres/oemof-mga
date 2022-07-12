# -*- coding: utf-8 -*-
"""Tools to perform Modelling to Generate Alternatives (MGA) with oemof solph.

SPDX-FileCopyrightText: Simon Hilpert

SPDX-License-Identifier: MIT
"""
import logging
import warnings
from logging import getLogger
import os

import pandas as pd

import oemof.solph as solph
from oemof.solph import views
from pyomo import environ as po


class MGA:
    """ Class to perform Modelling to Generate Alternative (MGA), by exploring
    the near-optimal solution space a linear `oemof.solph` optimisation model.

    Parameters
    -----------
    model : oemof.solph._models.Model
        An oemof solph model instance for which alternative solutions are
        to be generated
    solver : string
        A solver name for the solver to be used

    Attributes
    ----------
    base_model : oemof.solph._models.Models
        An oemof solph model. This object will be altered during the process
    base_objective_expr : pyomo.Expression
        A valid pyomo expression of the base model provided by the user
    solutions : dict
        Dictionary with oemof.solph result dicts for each generated solution
    objectives : dict
        Dictionary with the objectiv function expression for each
        generated solution

    """

    def __init__(self, model, solver, **kwargs):
        """
        """
        self._model = model
        self.base_objective_expr = model.objective.expr
        self.base_solution = None
        self.solver = solver
        self.solutions = {}
        self.objectives = {}
        self.results_dir = kwargs.get(
            "results_dir",
            os.path.join(os.path.expanduser("~"), "oemof-mga", "results")
        )

        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

    def _build_base_model(self):
        """(Re)Builds the base model.
        """
        self._model.del_component("objective")
        self._model.del_component("near_optimal_costraint")
        self._model.objective = po.Objective(expr=self.base_objective_expr)

        return True

    def calculate_base_solution(self, **kwargs):
        """Calculate the solution for the base model provided by the user

        """
        if self.base_solution:
            raise AttributeError(
                "Base solution already exists and can not be generated again."
                " Please create a new instance of the MGA class."
            )
            ## TODO: Here we could also re-create the initial model by
            # dropping the near-optimal-constraint and using the
            # self.base_objective_expr as expression
        self._model.solve(solver=self.solver, **kwargs)
        self.base_objective_value = self._model.objective()
        self.base_solution = self._model.results()
        self.solutions.update(
            {(("base_solution", "base_solution"), 0, 1): self.base_solution,
             (("base_solution", "base_solution"), 0, -1): self.base_solution}
        )

    def calculate_near_optimal_solution(
        self, objective=None, eps=0.01, sense=po.minimize,
        solution_name=None):
        """ Methdo to get a near optimal solution to the base problem.

        Parameters
        ----------
        objective: dict
            Dictionary with the objective expression with the structure of
            {"name": name_of_expression, "expr": valid_pyomo_expression}
        eps : float
            Value to define the deviation from the optimal objective value as
            fraction of one, 0 <= eps <= 1
        sense : integer
            Sense of optimisation, i.e. minimize -> 1 , maximize -1
        """
        if solution_name is None:
            solution_name = (objective["name"], eps, sense)

        if self.base_solution is None:
            self.calculate_base_solution()

        self._model.del_component("near_optimal_costraint")

        self._model.InvestmentFlow.del_component(
            "investment_costs")
        self._model.GenericInvestmentStorageBlock.del_component(
            "investment_costs")

        self._model.near_optimal_costraint = po.Constraint(
            expr=(
                self.base_objective_expr
                - self.base_objective_value * (1 + eps)
                <= 0
            )
        )
        # replace old objective
        self._model.del_component("objective")
        # maximize / minimize to find new solution
        self._model.objective = po.Objective(
            expr=objective["expr"], sense=sense, name=objective["name"]
        )
        self._model.solve(solver=self.solver)

        # store results and objective expression
        self.solutions[solution_name] = self._model.results()
        self.objectives[solution_name] = objective["expr"]


    def explore_near_optimal_space(
        self,
        objectives=None,
        epsilon_range=[0.01, 0.02, 0.03],
        optimisation_sense=(po.minimize, po.maximize),
    ):
        """
        Method to generate a set of near optimal solutions.

        Parameters
        -----------
        objectives : dict
            A dictionary containing objective functions to explore. Values must
            be valid pyomo-expressions. Keys are names and can be chosen by
            the user. If no objectives are provided by the user, the objectives
            to explore are all investment variables of the `oemof.solph`
        epsilon_range : list (optional)
            List with values between 0 and 1
            Default is: [0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1]
        optimisation : tuple (optional)
            Default is (minimize, maximize)
        """

        if self.base_solution is None:
            self.calculate_base_solution()

        if objectives is None:
            objectives = {
                (k[0].label, k[1].label): v
                for (k, v) in self._model.InvestmentFlow.invest.items()
            }
            objectives.update(
                {   # needs to k,k because investment is at node not flow..
                    (k.label, k.label): v
                    for (
                        k,
                        v,
                    ) in self._model.GenericInvestmentStorageBlock.invest.items()
                }
            )

        for eps in epsilon_range:
            for sense in [po.minimize, po.maximize]:
                for (
                    objective_name,
                    objective_expr,
                ) in objectives.items():
                    solution_name = (objective_name, eps, sense)

                    self.calculate_near_optimal_solution(
                        {"name": objective_name, "expr": objective_expr},
                        eps=eps, sense=sense, solution_name=solution_name)


    def get_all_sequences(self, results):
        """
        Copied from J. Röder q100opt repository. Thank you!
        """

        d_node_types = {
            'sink': solph.Sink,
            'source': solph.Source,
            'transformer': solph.Transformer,
            'storage_flow': solph.GenericStorage,
        }

        l_df = []

        for typ, solph_class in d_node_types.items():
            group = {
                k: v["sequences"]
                for k, v in results.items()
                if k[1] is not None
                if isinstance(k[0], solph_class) or isinstance(k[1], solph_class)
            }

            df = views.convert_to_multiindex(group)
            df_mi = df.columns.to_frame()
            df_mi.reset_index(drop=True, inplace=True)
            df_mi['from'] = [x.label for x in df_mi['from']]
            df_mi['to'] = [x.label for x in df_mi['to']]
            df_mi['type'] = typ
            df.columns = pd.MultiIndex.from_frame(df_mi[['type', 'from', 'to']])

            l_df.append(df)

        df_results = pd.concat(l_df, axis=1)

        # add storage content with extra type=storage_content
        group = {
            k: v["sequences"]
            for k, v in results.items()
            if k[1] is None
            if isinstance(k[0], solph.GenericStorage) or isinstance(
                k[1], solph.GenericStorage)
        }
        df = views.convert_to_multiindex(group)
        df_mi = df.columns.to_frame()
        df_mi.reset_index(drop=True, inplace=True)
        df_mi['from'] = [x.label for x in df_mi['from']]
        df.columns = pd.MultiIndex.from_frame(df_mi[['type', 'from', 'to']])

        df_results = pd.concat([df_results, df], axis=1)

        return df_results


    def generate_invest_table(self, filename="invest_table.csv"):
        """ Get all investment values for the generated solutions
        """
        if bool(self.solutions) is False:
            raise AttributeError(
                "No solutions found."
                " Run .explore_near_optimal_space() to generate solutions."
            )
        invest_results = {}
        for k, v in self.solutions.items():
            temp_dict = {}
            for x in v.keys():
                if hasattr(v[x]["scalars"], "invest"):
                    if x[1] is None:
                        key = (x[0].label, x[0].label)
                    else:
                        key = (x[0].label, x[1].label)
                    temp_dict.update({key: v[x]["scalars"]["invest"]})
            invest_results[k] = temp_dict

        invest_df = pd.DataFrame(invest_results).T
        invest_df.index.names = ["objective", "eps", "sense"]
        invest_df.sort_index(inplace=True, axis=1)
        invest_df.sort_index(inplace=True, axis=0)

        # convert tuple index to columns to create multiindex without tuples
        invest_df.reset_index(0, inplace=True)
        invest_df[['from', 'to']] = pd.DataFrame(
            invest_df['objective'].tolist(), index=invest_df.index)

        invest_df = invest_df.set_index(["from", "to"], append=True)
        # to get: "from, to, eps, sense"
        invest_df = invest_df.swaplevel(1,3).swaplevel(0,2)

        invest_df.drop("objective", axis=1, inplace=True)

        if filename:
            invest_df.to_csv(
                os.path.join(self.results_dir, filename))

        return invest_df

    def store_sequences(self, folder=None):
        """Write all sequences from all solutions to csv files
        """
        if folder is None:
            folder = os.path.join(self.results_dir, "sequences")

        if not os.path.exists(folder):
            os.makedirs(folder)

        for name, result in self.solutions.items():
            df = self.get_all_sequences(result)
            df.to_csv(os.path.join(folder, str(name) + ".csv"))

    def read_invest_table(self, filepath=None):
        """
        """
        if filepath is None:
            filepath = os.path.join(self.results_dir, "invest_table.csv")
            
        if not os.path.exists(filepath):
            raise ValueError("File `{}` not found".format(filepath))
        df = pd.read_csv(
                filepath,
                index_col=[0, 1, 2, 3], header=[0, 1])
        return df
