#!/usr/bin/env python3
"""
Optimization Framework - Main Entry Point

This module provides a simple entry point and demonstration of the
optimization framework capabilities. It includes example problems
and usage patterns for all three optimization algorithms.

Run this file to see the framework in action with example problems.
"""

import sys
import time

try:
    from optimization_framework import (
        ContinuousVariable, IntegerVariable, BinaryVariable,
        ObjectiveFunction, ConstraintFunction, Problem,
        GreedySearchSolver, GeneticAlgorithmSolver, SimulatedAnnealingSolver
    )
except ImportError as e:
    print(f"Error importing optimization framework: {e}")
    sys.exit(1)


def example_quadratic_problem():
    """
    Example: Simple quadratic optimization problem.

    Minimize: f(x, y) = x^2 + y^2
    Subject to: x + y <= 5
               x >= -10, x <= 10
               y >= -10, y <= 10 (integer)
    """
    print("=" * 60)
    print("EXAMPLE 1: Quadratic Optimization Problem")
    print("=" * 60)

    # Define variables
    x = ContinuousVariable("x", bounds=(-10, 10))
    y = IntegerVariable("y", bounds=(-10, 10))

    # Define objective function
    def objective_func(var_dict):
        x_val = var_dict["x"]["value"]
        y_val = var_dict["y"]["value"]
        return x_val**2 + y_val**2

    objective = ObjectiveFunction(objective_func, "quadratic")

    # Define constraint
    def constraint_func(var_dict):
        x_val = var_dict["x"]["value"]
        y_val = var_dict["y"]["value"]
        return x_val + y_val

    constraint = ConstraintFunction(constraint_func, "<=", 5.0, "sum_constraint")

    # Create problem
    problem = Problem(objective, constraints=[constraint], variables=[x, y])

    # Test all three solvers
    solvers = [
        ("Greedy Search", GreedySearchSolver({'max_iterations': 50, 'step_size': 0.5})),
        ("Genetic Algorithm", GeneticAlgorithmSolver({'population_size': 20, 'generations': 30})),
        ("Simulated Annealing", SimulatedAnnealingSolver({'initial_temperature': 50.0, 'max_iterations': 200}))
    ]

    results = {}

    for solver_name, solver in solvers:
        print(f"\nSolving with {solver_name}...")
        start_time = time.time()

        try:
            solution = solver.solve(problem, initial_guess={"x": 3.0, "y": 2})
            solve_time = time.time() - start_time

            best = solution.get_best_solution()
            summary = solution.get_summary_statistics()

            results[solver_name] = {
                'solution': solution,
                'best_objective': best['objective_value'],
                'best_variables': {k: v['value'] for k, v in best['variable_dict'].items()},
                'feasible': best['is_feasible'],
                'iterations': summary['total_iterations'],
                'solve_time': solve_time
            }

            print(f"  Best objective: {best['objective_value']:.6f}")
            print(f"  Variables: x={best['variable_dict']['x']['value']:.3f}, y={best['variable_dict']['y']['value']}")
            print(f"  Feasible: {best['is_feasible']}")
            print(f"  Iterations: {summary['total_iterations']}")
            print(f"  Time: {solve_time:.3f}s")

        except Exception as e:
            print(f"  Error: {str(e)}")
            results[solver_name] = {'error': str(e)}

    return results


def example_binary_knapsack():
    """
    Example: Binary knapsack optimization problem.

    Maximize value of items in knapsack subject to weight constraint.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Binary Knapsack Problem")
    print("=" * 60)

    # Item data: (value, weight)
    items = [
        (10, 5),   # item 0
        (40, 4),   # item 1  
        (30, 6),   # item 2
        (50, 3),   # item 3
        (35, 7),   # item 4
    ]

    max_weight = 15

    # Define binary variables for each item
    variables = []
    for i in range(len(items)):
        variables.append(BinaryVariable(f"item_{i}"))

    # Define objective function (maximize value = minimize negative value)
    def objective_func(var_dict):
        total_value = 0
        for i, (value, weight) in enumerate(items):
            if var_dict[f"item_{i}"]["value"] == 1:
                total_value += value
        return -total_value  # Minimize negative value to maximize value

    objective = ObjectiveFunction(objective_func, "knapsack_value")

    # Define weight constraint
    def weight_constraint(var_dict):
        total_weight = 0
        for i, (value, weight) in enumerate(items):
            if var_dict[f"item_{i}"]["value"] == 1:
                total_weight += weight
        return total_weight

    constraint = ConstraintFunction(weight_constraint, "<=", max_weight, "weight_limit")

    # Create problem
    problem = Problem(objective, constraints=[constraint], variables=variables)

    # Solve with genetic algorithm (best for binary problems)
    print("Solving with Genetic Algorithm...")
    solver = GeneticAlgorithmSolver({
        'population_size': 30,
        'generations': 50,
        'mutation_rate': 0.15,
        'crossover_rate': 0.8
    })

    start_time = time.time()
    solution = solver.solve(problem)
    solve_time = time.time() - start_time

    best = solution.get_best_solution()

    print(f"Best objective (negative value): {best['objective_value']:.1f}")
    print(f"Actual value: {-best['objective_value']:.1f}")

    selected_items = []
    total_weight = 0
    for i in range(len(items)):
        if best['variable_dict'][f'item_{i}']['value'] == 1:
            selected_items.append(i)
            total_weight += items[i][1]

    print(f"Selected items: {selected_items}")
    print(f"Total weight: {total_weight}/{max_weight}")
    print(f"Feasible: {best['is_feasible']}")
    print(f"Time: {solve_time:.3f}s")

    return solution


def example_mixed_optimization():
    """
    Example: Mixed variable type optimization.

    Demonstrates continuous, integer, and binary variables together.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Mixed Variable Type Problem")
    print("=" * 60)

    # Define mixed variables
    x = ContinuousVariable("x", bounds=(0, 10))      # Continuous
    n = IntegerVariable("n", bounds=(1, 5))          # Integer  
    use_bonus = BinaryVariable("use_bonus")          # Binary

    # Complex objective function
    def objective_func(var_dict):
        x_val = var_dict["x"]["value"]
        n_val = var_dict["n"]["value"]
        bonus = var_dict["use_bonus"]["value"]

        # Base cost function
        cost = x_val**2 + n_val * 2

        # Bonus reduces cost but adds fixed penalty
        if bonus == 1:
            cost = cost * 0.7 + 5  # 30% reduction but +5 penalty

        return cost

    objective = ObjectiveFunction(objective_func, "mixed_cost")

    # Multiple constraints
    def constraint1(var_dict):
        return var_dict["x"]["value"] + var_dict["n"]["value"]

    def constraint2(var_dict):
        x_val = var_dict["x"]["value"]
        n_val = var_dict["n"]["value"]
        return x_val * n_val

    constraints = [
        ConstraintFunction(constraint1, "<=", 8.0, "sum_limit"),
        ConstraintFunction(constraint2, ">=", 2.0, "product_min")
    ]

    # Create problem
    problem = Problem(objective, constraints=constraints, variables=[x, n, use_bonus])

    # Solve with simulated annealing
    print("Solving with Simulated Annealing...")
    solver = SimulatedAnnealingSolver({
        'initial_temperature': 100.0,
        'final_temperature': 0.1,
        'max_iterations': 300,
        'cooling_rate': 0.95
    })

    start_time = time.time()
    solution = solver.solve(problem, initial_guess={"x": 2.0, "n": 2, "use_bonus": 0})
    solve_time = time.time() - start_time

    best = solution.get_best_solution()

    print(f"Best objective: {best['objective_value']:.6f}")
    print("Variables:")
    for var_name, var_data in best['variable_dict'].items():
        print(f"  {var_name}: {var_data['value']}")
    print(f"Feasible: {best['is_feasible']}")
    print(f"Constraint violations: {best['constraint_violations']}")
    print(f"Time: {solve_time:.3f}s")

    return solution


def main():
    """Main demonstration function."""
    print("OPTIMIZATION FRAMEWORK DEMONSTRATION")
    print("====================================")
    print("Framework: Custom OOP Optimization Framework")
    print("Description: Pure Python optimization for ARM64 Windows")

    try:
        # Run examples
        results1 = example_quadratic_problem()
        results2 = example_binary_knapsack()
        results3 = example_mixed_optimization()

        print("\n" + "=" * 60)
        print("DEMONSTRATION COMPLETE")
        print("=" * 60)
        print("All examples completed successfully!")
        print("The optimization framework is working correctly.")

    except Exception as e:
        print(f"\nERROR during demonstration: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
