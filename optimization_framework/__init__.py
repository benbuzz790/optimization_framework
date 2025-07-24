"""
Optimization Framework - A pure Python optimization library for ARM64 Windows.

This framework provides a complete optimization solution without external dependencies,
specifically designed for environments where scipy cannot be compiled (ARM64 Windows).

Key Components:
- Variables: ContinuousVariable, IntegerVariable, BinaryVariable
- Functions: ObjectiveFunction, ConstraintFunction  
- Problems: Problem class for bundling objectives and constraints
- Solvers: GreedySearchSolver, GeneticAlgorithmSolver, SimulatedAnnealingSolver
- Solutions: Solution class with complete optimization history

Example Usage:
    from optimization_framework import *

    # Define variables
    x = ContinuousVariable("x", bounds=(-10, 10))
    y = ContinuousVariable("y", bounds=(-10, 10))

    # Define objective
    def objective_func(var_dict):
        return var_dict["x"]["value"]**2 + var_dict["y"]["value"]**2

    objective = ObjectiveFunction(objective_func)

    # Create and solve problem
    problem = Problem(objective, variables=[x, y])
    solver = GeneticAlgorithmSolver()
    solution = solver.solve(problem)
"""

__version__ = "1.0.0"
__author__ = "Optimization Framework Team"
__email__ = "team@optimization-framework.com"

# Import all public classes and functions
from .variables import (
    Variable,
    ContinuousVariable, 
    IntegerVariable,
    BinaryVariable
)

from .functions import (
    ObjectiveFunction,
    ConstraintFunction
)

from .problems import Problem

from .solvers import (
    BaseSolver,
    GreedySearchSolver,
    GeneticAlgorithmSolver, 
    SimulatedAnnealingSolver
)

from .solutions import Solution

from .utils.assertions import production_assert

# Define public API
__all__ = [
    # Variables
    "Variable",
    "ContinuousVariable",
    "IntegerVariable", 
    "BinaryVariable",

    # Functions
    "ObjectiveFunction",
    "ConstraintFunction",

    # Problems
    "Problem",

    # Solvers
    "BaseSolver",
    "GreedySearchSolver",
    "GeneticAlgorithmSolver",
    "SimulatedAnnealingSolver",

    # Solutions
    "Solution",

    # Utilities
    "production_assert",
]