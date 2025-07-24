"""
Barebones Optimization Example - Minimal Working Example

Solves: Minimize f(x, y) = (x - 2)² + (y - 3)² + 5
Subject to: x + y <= 8, x >= 0, y >= 0

Expected solution: x=2, y=3, f=5
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimization_framework import (
    ContinuousVariable, ObjectiveFunction, ConstraintFunction, 
    Problem, GeneticAlgorithmSolver
)

def main():
    # 1. Define variables
    x = ContinuousVariable("x", bounds=(0.0, 10.0))
    y = ContinuousVariable("y", bounds=(0.0, 10.0))
    variables = [x, y]

    # 2. Define objective function
    def objective_func(var_dict):
        x_val = var_dict["x"]["value"]
        y_val = var_dict["y"]["value"]
        return (x_val - 2.0)**2 + (y_val - 3.0)**2 + 5.0

    objective = ObjectiveFunction(objective_func, name="quadratic")

    # 3. Define constraints
    def constraint_func(var_dict):
        return var_dict["x"]["value"] + var_dict["y"]["value"]

    constraint = ConstraintFunction(constraint_func, "<=", 8.0, name="sum_constraint")
    constraints = [constraint]

    # 4. Create problem
    problem = Problem(objective, constraints, variables)

    # 5. Solve with genetic algorithm
    solver = GeneticAlgorithmSolver()
    solution = solver.solve(problem)

    # 6. Display results
    best = solution.get_best_solution()
    print(f"Solution: x={best['variable_dict']['x']['value']:.3f}, "
          f"y={best['variable_dict']['y']['value']:.3f}")
    print(f"Objective: {best['objective_value']:.3f}")
    print(f"Feasible: {best['is_feasible']}")
    print(f"Iterations: {solution.get_convergence_data()['total_iterations']}")

if __name__ == "__main__":
    main()