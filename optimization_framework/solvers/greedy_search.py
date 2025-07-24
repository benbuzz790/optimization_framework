"""
Greedy Search Solver Implementation

Local search optimization algorithm that performs iterative improvement
by making small steps in variable space and accepting improvements.
"""

import random
from typing import Dict, Any
from .base_solver import BaseSolver
from .solver_config import SolverConfig
from ..utils.assertions import production_assert
from ..variables import BinaryVariable, IntegerVariable, ContinuousVariable
from ..solutions import Solution


class GreedySearchSolver(BaseSolver):
    """
    Greedy local search optimization algorithm.

    Performs iterative improvement by making small steps in variable space
    and accepting improvements to the objective function. Uses configurable
    step sizes and search strategies.
    """

    def __init__(self, config: Dict = None):
        """Initialize GreedySearchSolver with configuration validation."""
        super().__init__(config)

    def _validate_config(self, config: Dict) -> Dict:
        """Validate and set default configuration for greedy search."""
        validated_config = config.copy()

        # Set defaults
        defaults = {
            'max_iterations': 100,
            'step_size': 0.1,
            'tolerance': 1e-6,
            'step_reduction_factor': 0.5,
            'min_step_size': 1e-8,
            'random_seed': None
        }

        for key, default_value in defaults.items():
            if key not in validated_config:
                validated_config[key] = default_value

        # Validate ranges
        production_assert(validated_config['max_iterations'] > 0,
                         "max_iterations must be positive")
        production_assert(validated_config['step_size'] > 0,
                         "step_size must be positive")
        production_assert(validated_config['tolerance'] >= 0,
                         "tolerance must be non-negative")
        production_assert(0 < validated_config['step_reduction_factor'] < 1,
                         "step_reduction_factor must be in (0, 1)")

        return validated_config

    def solve(self, problem, initial_guess: Dict[str, Any] = None):
        """Solve optimization problem using greedy local search."""
        # Call parent validation
        super().solve(problem, initial_guess)

        # Create solution tracking object
        solution = Solution(problem, "GreedySearchSolver")

        # Set random seed if specified
        if self.config['random_seed'] is not None:
            random.seed(self.config['random_seed'])

        # Initialize starting solution
        current_solution = self._create_initial_solution(problem, initial_guess)
        current_objective = problem.evaluate_objective(current_solution)
        current_violations = problem.get_constraint_violations(current_solution)

        # Track best solution
        best_solution = current_solution.copy()
        best_objective = current_objective

        # Initialize step size
        current_step_size = self.config['step_size']

        # Add initial solution to history
        solution.add_iteration(
            variable_dict=current_solution,
            objective_value=current_objective,
            constraint_violations=current_violations,
            metadata={'step_size': current_step_size, 'iteration_type': 'initial'}
        )

        # Main optimization loop
        iteration = 0
        consecutive_no_improvement = 0

        while (iteration < self.config['max_iterations'] and
               current_step_size >= self.config['min_step_size']):

            # Generate neighbor solution
            neighbor_solution = self._generate_neighbor(problem, current_solution, current_step_size)
            neighbor_objective = problem.evaluate_objective(neighbor_solution)
            neighbor_violations = problem.get_constraint_violations(neighbor_solution)

            # Check if neighbor is better
            if neighbor_objective < current_objective:
                # Accept improvement
                current_solution = neighbor_solution
                current_objective = neighbor_objective
                current_violations = neighbor_violations
                consecutive_no_improvement = 0

                # Update best if this is the best so far
                if neighbor_objective < best_objective:
                    best_solution = neighbor_solution.copy()
                    best_objective = neighbor_objective

                # Add accepted solution to history
                solution.add_iteration(
                    variable_dict=current_solution,
                    objective_value=current_objective,
                    constraint_violations=current_violations,
                    metadata={'step_size': current_step_size, 'iteration_type': 'improvement'}
                )
            else:
                consecutive_no_improvement += 1

                # Reduce step size if no improvement for several iterations
                if consecutive_no_improvement >= 5:
                    current_step_size *= self.config['step_reduction_factor']
                    consecutive_no_improvement = 0

                    # Add step reduction to history
                    solution.add_iteration(
                        variable_dict=current_solution,
                        objective_value=current_objective,
                        constraint_violations=current_violations,
                        metadata={'step_size': current_step_size, 'iteration_type': 'step_reduction'}
                    )

            iteration += 1

            # Check convergence
            if consecutive_no_improvement == 0 and iteration > 10:
                recent_history = solution.get_optimization_history()[-10:]
                objectives = [h['objective_value'] for h in recent_history]
                if max(objectives) - min(objectives) < self.config['tolerance']:
                    break

        return solution

    def _generate_neighbor(self, problem, current_solution: Dict, step_size: float) -> Dict:
        """Generate neighbor solution by perturbing current solution."""
        neighbor_values = {}

        for variable in problem.variables:
            var_name = variable.name
            current_value = current_solution[var_name]['value']

            if isinstance(variable, BinaryVariable):
                # Binary variables: flip with probability proportional to step size
                if random.random() < step_size:
                    neighbor_values[var_name] = 1 - current_value
                else:
                    neighbor_values[var_name] = current_value

            elif isinstance(variable, IntegerVariable):
                # Integer variables: add random integer step
                max_change = max(1, int(step_size * 10))
                change = random.randint(-max_change, max_change)
                new_value = current_value + change

                # Clip to bounds
                if variable.bounds:
                    new_value = variable.clip_to_bounds(new_value)

                neighbor_values[var_name] = int(new_value)

            else:  # ContinuousVariable
                # Continuous variables: add Gaussian noise
                noise = random.gauss(0, step_size)
                new_value = current_value + noise

                # Clip to bounds
                if variable.bounds:
                    new_value = variable.clip_to_bounds(new_value)

                neighbor_values[var_name] = new_value

        return problem.create_variable_dict(neighbor_values)