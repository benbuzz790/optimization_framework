"""
Optimization Framework Documentation

This module contains comprehensive documentation for the optimization framework
including usage examples, API reference, and best practices.

OVERVIEW:
The Optimization Framework is a comprehensive, object-oriented optimization system 
designed specifically for ARM64 Windows environments where scipy is unavailable. 
It provides a clean, extensible interface for defining and solving optimization 
problems with multiple variable types and constraint handling.

FEATURES:
- Multiple Variable Types: Continuous, integer, and binary variables with bounds
- Function Wrappers: Objective and constraint functions with variable dictionary protocol
- Problem Definition: Comprehensive problem specification with validation
- Solution Tracking: Complete optimization history and analysis
- Multiple Algorithms: Greedy search, genetic algorithm, and simulated annealing
- NASA-Style Validation: Production-quality input validation throughout
- Modular Design: Clean separation of concerns with pluggable components

QUICK START EXAMPLE:

from optimization_framework import *

# Define variables
x = ContinuousVariable("x", bounds=(-10, 10))
y = IntegerVariable("y", bounds=(0, 20))

# Define objective function
def objective_func(var_dict):
    x_val = var_dict["x"]["value"]
    y_val = var_dict["y"]["value"]
    return x_val**2 + y_val**2

objective = ObjectiveFunction(objective_func, "quadratic")

# Define constraints
def constraint_func(var_dict):
    x_val = var_dict["x"]["value"]
    y_val = var_dict["y"]["value"]
    return x_val + y_val

constraint = ConstraintFunction(constraint_func, "<=", 15.0, "sum_limit")

# Create and solve problem
problem = Problem(objective, constraints=[constraint], variables=[x, y])
solver = GeneticAlgorithmSolver({'population_size': 30, 'generations': 50})
solution = solver.solve(problem, initial_guess={"x": 0.0, "y": 5})

# Get results
best = solution.get_best_solution()
print(f"Best solution: {best['objective_value']}")
print(f"Variables: {[(k, v['value']) for k, v in best['variable_dict'].items()]}")

ARCHITECTURE:

Core Components:

1. Variables (optimization_framework.variables)
   - Variable: Abstract base class
   - ContinuousVariable: Real-valued variables with bounds
   - IntegerVariable: Integer-valued variables with bounds  
   - BinaryVariable: Binary (0/1) variables

2. Functions (optimization_framework.functions)
   - ObjectiveFunction: Wrapper for objective functions
   - ConstraintFunction: Wrapper for constraint functions with violation tracking

3. Problems (optimization_framework.problems)
   - Problem: Complete problem specification with validation

4. Solutions (optimization_framework.solutions)
   - Solution: Complete optimization tracking with history and analysis

5. Solvers (optimization_framework.solvers)
   - BaseSolver: Abstract base class for algorithms
   - GreedySearchSolver: Local search optimization
   - GeneticAlgorithmSolver: Population-based evolutionary optimization
   - SimulatedAnnealingSolver: Temperature-based probabilistic search

VARIABLE DICTIONARY PROTOCOL:

All functions in the framework use a standardized variable dictionary format:

variable_dict = {
    "var_name": {
        "value": <current_value>,
        "type": "continuous" | "integer" | "binary",
        "bounds": (min_val, max_val),
        "variable_object": <Variable instance>
    }
}

This ensures type safety and consistent validation across all components.
"""


def variable_types_examples():
    """
    Examples of different variable types and their usage.

    Returns:
        dict: Dictionary of example variable definitions
    """
    from optimization_framework import ContinuousVariable, IntegerVariable, BinaryVariable

    examples = {
        'continuous': ContinuousVariable("x", bounds=(-5.0, 5.0)),
        'integer': IntegerVariable("n", bounds=(1, 100)),
        'binary': BinaryVariable("use_feature")
    }

    return examples


def objective_function_examples():
    """
    Examples of objective function definitions.

    Returns:
        dict: Dictionary of example objective functions
    """
    from optimization_framework import ObjectiveFunction
    import math

    # Simple quadratic function
    def quadratic_objective(var_dict):
        x = var_dict["x"]["value"]
        y = var_dict["y"]["value"]
        return x**2 + y**2

    # Complex multimodal function
    def complex_objective(var_dict):
        x = var_dict["x"]["value"]
        y = var_dict["y"]["value"]
        return x**2 + y**2 + math.sin(x*y) + 0.1*math.cos(5*x)

    # Mixed variable function
    def mixed_objective(var_dict):
        x = var_dict["x"]["value"]  # continuous
        n = var_dict["n"]["value"]  # integer
        flag = var_dict["flag"]["value"]  # binary

        base_cost = x**2 + n * 2
        if flag == 1:
            base_cost *= 0.8  # 20% discount if flag is set

        return base_cost

    examples = {
        'quadratic': ObjectiveFunction(quadratic_objective, "quadratic"),
        'complex': ObjectiveFunction(complex_objective, "complex_multimodal"),
        'mixed': ObjectiveFunction(mixed_objective, "mixed_variables")
    }

    return examples


def constraint_function_examples():
    """
    Examples of constraint function definitions.

    Returns:
        dict: Dictionary of example constraint functions
    """
    from optimization_framework import ConstraintFunction

    # Inequality constraint: x + y <= 10
    def sum_constraint(var_dict):
        return var_dict["x"]["value"] + var_dict["y"]["value"]

    # Equality constraint: x * y == 5
    def product_constraint(var_dict):
        return var_dict["x"]["value"] * var_dict["y"]["value"]

    # Greater-than constraint: x >= 0
    def positive_constraint(var_dict):
        return var_dict["x"]["value"]

    # Complex constraint with multiple variables
    def complex_constraint(var_dict):
        x = var_dict["x"]["value"]
        y = var_dict["y"]["value"]
        n = var_dict["n"]["value"]
        return x**2 + y**2 + n

    examples = {
        'sum_limit': ConstraintFunction(sum_constraint, "<=", 10.0, "sum_limit"),
        'product_equal': ConstraintFunction(product_constraint, "==", 5.0, "product_equal"),
        'positive_x': ConstraintFunction(positive_constraint, ">=", 0.0, "positive_x"),
        'complex_bound': ConstraintFunction(complex_constraint, "<=", 25.0, "complex_bound")
    }

    return examples


def solver_configuration_examples():
    """
    Examples of solver configurations for different algorithms.

    Returns:
        dict: Dictionary of configured solver examples
    """
    from optimization_framework import (
        GreedySearchSolver, GeneticAlgorithmSolver, SimulatedAnnealingSolver
    )

    # Greedy Search configurations
    greedy_fast = GreedySearchSolver({
        'max_iterations': 50,
        'step_size': 0.2,
        'tolerance': 1e-4
    })

    greedy_precise = GreedySearchSolver({
        'max_iterations': 200,
        'step_size': 0.05,
        'tolerance': 1e-8,
        'step_reduction_factor': 0.8,
        'min_step_size': 1e-10
    })

    # Genetic Algorithm configurations
    ga_small = GeneticAlgorithmSolver({
        'population_size': 20,
        'generations': 30,
        'mutation_rate': 0.15,
        'crossover_rate': 0.7
    })

    ga_large = GeneticAlgorithmSolver({
        'population_size': 100,
        'generations': 200,
        'mutation_rate': 0.05,
        'crossover_rate': 0.9,
        'selection_method': 'tournament',
        'tournament_size': 5,
        'elitism_count': 5
    })

    # Simulated Annealing configurations
    sa_fast = SimulatedAnnealingSolver({
        'initial_temperature': 50.0,
        'final_temperature': 0.1,
        'cooling_rate': 0.9,
        'max_iterations': 200
    })

    sa_thorough = SimulatedAnnealingSolver({
        'initial_temperature': 200.0,
        'final_temperature': 0.001,
        'cooling_rate': 0.99,
        'cooling_schedule': 'geometric',
        'max_iterations': 2000,
        'iterations_per_temperature': 20,
        'perturbation_magnitude': 0.05
    })

    examples = {
        'greedy_fast': greedy_fast,
        'greedy_precise': greedy_precise,
        'ga_small': ga_small,
        'ga_large': ga_large,
        'sa_fast': sa_fast,
        'sa_thorough': sa_thorough
    }

    return examples


def complete_optimization_example():
    """
    Complete example showing all framework components working together.

    Returns:
        dict: Results from the optimization example
    """
    from optimization_framework import *

    print("Complete Optimization Example")
    print("=" * 50)

    # 1. Define variables
    x = ContinuousVariable("x", bounds=(-5, 5))
    y = ContinuousVariable("y", bounds=(-5, 5))
    n = IntegerVariable("n", bounds=(1, 10))
    use_bonus = BinaryVariable("use_bonus")

    print(f"Variables defined: {[v.name for v in [x, y, n, use_bonus]]}")

    # 2. Define objective function
    def complex_objective(var_dict):
        x_val = var_dict["x"]["value"]
        y_val = var_dict["y"]["value"]
        n_val = var_dict["n"]["value"]
        bonus = var_dict["use_bonus"]["value"]

        # Base cost function
        cost = (x_val - 1)**2 + (y_val + 0.5)**2 + n_val * 0.5

        # Bonus reduces cost but adds penalty
        if bonus == 1:
            cost = cost * 0.7 + 2.0

        return cost

    objective = ObjectiveFunction(complex_objective, "complex_cost")
    print("Objective function defined")

    # 3. Define constraints
    def sum_constraint(var_dict):
        return var_dict["x"]["value"] + var_dict["y"]["value"]

    def product_constraint(var_dict):
        x_val = var_dict["x"]["value"]
        n_val = var_dict["n"]["value"]
        return abs(x_val * n_val)

    constraints = [
        ConstraintFunction(sum_constraint, "<=", 3.0, "sum_limit"),
        ConstraintFunction(product_constraint, "<=", 15.0, "product_limit")
    ]

    print(f"Constraints defined: {len(constraints)}")

    # 4. Create problem
    problem = Problem(
        objective=objective,
        constraints=constraints,
        variables=[x, y, n, use_bonus]
    )

    print("Problem created")

    # 5. Solve with multiple algorithms
    solvers = [
        ("Greedy Search", GreedySearchSolver({'max_iterations': 50, 'random_seed': 42})),
        ("Genetic Algorithm", GeneticAlgorithmSolver({
            'population_size': 30, 'generations': 40, 'random_seed': 42
        })),
        ("Simulated Annealing", SimulatedAnnealingSolver({
            'max_iterations': 100, 'initial_temperature': 50.0, 'random_seed': 42
        }))
    ]

    results = {}

    for solver_name, solver in solvers:
        print(f"\nSolving with {solver_name}...")

        solution = solver.solve(problem, initial_guess={
            "x": 2.0, "y": -1.0, "n": 3, "use_bonus": 0
        })

        best = solution.get_best_solution()
        summary = solution.get_summary_statistics()

        results[solver_name] = {
            'best_objective': best['objective_value'],
            'variables': {k: v['value'] for k, v in best['variable_dict'].items()},
            'feasible': best['is_feasible'],
            'iterations': summary['total_iterations'],
            'converged': summary['converged']
        }

        print(f"  Best objective: {best['objective_value']:.4f}")
        print(f"  Variables: {results[solver_name]['variables']}")
        print(f"  Feasible: {best['is_feasible']}")
        print(f"  Iterations: {summary['total_iterations']}")

    return results


def algorithm_selection_guide():
    """
    Guide for selecting the appropriate optimization algorithm.

    Returns:
        dict: Algorithm selection recommendations
    """
    guide = {
        'GreedySearchSolver': {
            'best_for': [
                'Smooth, unimodal functions',
                'Good initial guesses available',
                'Fast convergence needed',
                'Continuous variables primarily'
            ],
            'pros': [
                'Fast convergence',
                'Simple to understand',
                'Deterministic results',
                'Low memory usage'
            ],
            'cons': [
                'Gets stuck in local optima',
                'Requires good starting point',
                'Poor performance on multimodal functions',
                'Limited exploration capability'
            ],
            'recommended_config': {
                'max_iterations': 100,
                'step_size': 0.1,
                'tolerance': 1e-6
            }
        },

        'GeneticAlgorithmSolver': {
            'best_for': [
                'Discrete/mixed variable problems',
                'Multimodal functions',
                'Global optimization needed',
                'Robust solutions required'
            ],
            'pros': [
                'Global search capability',
                'Handles discrete variables well',
                'Robust to function characteristics',
                'Population-based diversity'
            ],
            'cons': [
                'Slower convergence',
                'Many parameters to tune',
                'Stochastic results',
                'Higher memory usage'
            ],
            'recommended_config': {
                'population_size': 50,
                'generations': 100,
                'mutation_rate': 0.1,
                'crossover_rate': 0.8
            }
        },

        'SimulatedAnnealingSolver': {
            'best_for': [
                'Continuous variables',
                'Functions with many local optima',
                'Single-solution evolution',
                'Temperature-based exploration'
            ],
            'pros': [
                'Can escape local optima',
                'Good for continuous optimization',
                'Theoretical convergence guarantees',
                'Single solution tracking'
            ],
            'cons': [
                'Sensitive to temperature schedule',
                'Slower than greedy search',
                'Parameter tuning required',
                'Cooling schedule selection critical'
            ],
            'recommended_config': {
                'initial_temperature': 100.0,
                'final_temperature': 0.01,
                'cooling_rate': 0.95,
                'max_iterations': 1000
            }
        }
    }

    return guide


def performance_tips():
    """
    Performance optimization tips for the framework.

    Returns:
        dict: Performance recommendations
    """
    tips = {
        'memory_management': [
            'Solution history can grow large - consider memory limits for long runs',
            'Use solution.get_best_solution() instead of storing full history',
            'Clear intermediate solutions if not needed for analysis'
        ],

        'algorithm_selection': [
            'Use GreedySearchSolver for smooth, unimodal functions',
            'Use GeneticAlgorithmSolver for discrete/mixed problems',
            'Use SimulatedAnnealingSolver for continuous multimodal functions'
        ],

        'configuration_tuning': [
            'Start with smaller population sizes and fewer iterations for testing',
            'Increase precision parameters only when needed',
            'Use random seeds for reproducible results during development'
        ],

        'function_optimization': [
            'Keep objective and constraint functions as simple as possible',
            'Avoid expensive operations in frequently called functions',
            'Cache expensive calculations when possible'
        ],

        'variable_bounds': [
            'Always specify reasonable bounds for variables',
            'Tighter bounds generally lead to faster convergence',
            'Use appropriate variable types (integer vs continuous)'
        ]
    }

    return tips


def testing_guide():
    """
    Guide for testing optimization problems and framework components.

    Returns:
        dict: Testing recommendations and examples
    """
    guide = {
        'unit_testing': {
            'description': 'Test individual components in isolation',
            'examples': [
                'Test variable validation with edge cases',
                'Test objective function evaluation with known inputs',
                'Test constraint satisfaction checking',
                'Test solver configuration validation'
            ],
            'command': 'python run_all_tests.py --module variables'
        },

        'integration_testing': {
            'description': 'Test complete optimization workflows',
            'examples': [
                'Test end-to-end optimization with known solutions',
                'Test all solvers on same problem for consistency',
                'Test mixed variable type problems',
                'Test constrained vs unconstrained problems'
            ],
            'command': 'python run_all_tests.py'
        },

        'problem_validation': {
            'description': 'Validate your optimization problem setup',
            'steps': [
                '1. Test objective function with manual inputs',
                '2. Verify constraint functions return expected values',
                '3. Check variable bounds are reasonable',
                '4. Test with simple solver first (GreedySearch)',
                '5. Compare results across different solvers'
            ]
        },

        'debugging_tips': [
            'Use small problem sizes during development',
            'Set random seeds for reproducible results',
            'Enable verbose output to see optimization progress',
            'Check solution feasibility and constraint violations',
            'Plot objective function if possible to understand landscape'
        ]
    }

    return guide


if __name__ == '__main__':
    """
    Run documentation examples when this module is executed directly.
    """
    print("OPTIMIZATION FRAMEWORK DOCUMENTATION")
    print("=" * 60)

    print("\n1. Variable Types Examples:")
    variables = variable_types_examples()
    for name, var in variables.items():
        print(f"  {name}: {var.name} with bounds {var.bounds}")

    print("\n2. Algorithm Selection Guide:")
    guide = algorithm_selection_guide()
    for algo, info in guide.items():
        print(f"  {algo}:")
        print(f"    Best for: {', '.join(info['best_for'][:2])}")
        print(f"    Key pros: {', '.join(info['pros'][:2])}")

    print("\n3. Performance Tips:")
    tips = performance_tips()
    print(f"  Memory: {tips['memory_management'][0]}")
    print(f"  Selection: {tips['algorithm_selection'][0]}")

    print("\n4. Running Complete Example:")
    try:
        results = complete_optimization_example()
        print(f"\nExample completed successfully!")
        print(f"Best result: {min(r['best_objective'] for r in results.values()):.4f}")
    except Exception as e:
        print(f"Example failed: {str(e)}")

    print("\nFor more examples, run: python main.py")
    print("For testing, run: python run_all_tests.py")