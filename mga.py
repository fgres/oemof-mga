# -*- coding: utf-8 -*-
"""Tools to perform Modelling to Generate Alternatives (MGA) with oemof solph.

SPDX-FileCopyrightText: Simon Hilpert

SPDX-License-Identifier: MIT
"""
import logging
import warnings
from logging import getLogger
import pandas as pd

import oemof.solph as solph
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
            {("base_solution", 0, 1): self.base_solution}
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

        self._model.InvestmentFlowBlock.del_component(
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
                k: v
                for (k, v) in self._model.InvestmentFlowBlock.invest.items()
            }
            objectives.update(
                {
                    k: v
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


    def get_invest_values(self):
        """ Get all investment values for the generated solutions
        """
        if bool(self.solutions) is False:
            raise AttributeError(
                "No solutions found."
                " Run .explore_near_optimal_space() to generate solutions."
            )

        invest_results = {}
        for k, v in self.solutions.items():
            if isinstance(
                k[0], solph.components._generic_storage.GenericStorage):
                k = ((k[0], k[0]), k[1], k[2])
            # need to do str conversion for None in storage label

            temp_dict = {}
            for x in v.keys():
                if hasattr(v[x]["scalars"], "invest"):
                    if x[1] is None:
                        key = (x[0], x[0])
                    else:
                        key = (x[0], x[1])
                    temp_dict.update({key: v[x]["scalars"]["invest"]})
            invest_results[k] = temp_dict

        invest_df = pd.DataFrame(invest_results).T
        invest_df.index.names = ["objective", "eps", "sense"]
        invest_df.sort_index(inplace=True, axis=1)
        invest_df.sort_index(inplace=True, axis=0)

        return invest_df
