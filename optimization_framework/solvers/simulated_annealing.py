"""
Simulated Annealing Solver Implementation

Temperature-based probabilistic optimization algorithm that uses
cooling schedules to escape local optima and explore solution space.
"""

import random
import math
from typing import Dict, Any
from .base_solver import BaseSolver
from ..utils.assertions import production_assert
from ..variables import BinaryVariable, IntegerVariable, ContinuousVariable
from ..solutions import Solution


class SimulatedAnnealingSolver(BaseSolver):
    """
    Simulated annealing optimization solver.

    Uses temperature-based probabilistic acceptance of solutions to
    escape local optima and explore the solution space effectively.
    """

    def __init__(self, config: Dict = None):
        """Initialize SimulatedAnnealingSolver with configuration validation."""
        super().__init__(config)

    def _validate_config(self, config: Dict) -> Dict:
        """Validate and set default configuration for simulated annealing."""
        validated_config = config.copy()

        # Set defaults
        defaults = {
            'initial_temperature': 100.0,
            'final_temperature': 0.01,
            'cooling_rate': 0.95,
            'cooling_schedule': 'geometric',
            'max_iterations': 1000,
            'iterations_per_temperature': 10,
            'perturbation_magnitude': 0.1,
            'convergence_window': 50,
            'convergence_tolerance': 1e-6,
            'random_seed': None
        }

        for key, default_value in defaults.items():
            if key not in validated_config:
                validated_config[key] = default_value

        # Validate ranges
        production_assert(validated_config['initial_temperature'] > validated_config['final_temperature'],
                         "initial_temperature must be > final_temperature")
        production_assert(validated_config['final_temperature'] > 0,
                         "final_temperature must be positive")
        production_assert(0 < validated_config['cooling_rate'] < 1,
                         "cooling_rate must be in (0, 1)")
        production_assert(validated_config['cooling_schedule'] in ['geometric', 'linear', 'logarithmic'],
                         "cooling_schedule must be 'geometric', 'linear', or 'logarithmic'")
        production_assert(validated_config['max_iterations'] > 0,
                         "max_iterations must be positive")

        return validated_config

    def solve(self, problem, initial_guess: Dict[str, Any] = None):
        """Solve optimization problem using simulated annealing."""
        # Call parent validation
        super().solve(problem, initial_guess)

        # Create solution tracking object
        solution = Solution(problem, "SimulatedAnnealingSolver")

        # Set random seed if specified
        if self.config['random_seed'] is not None:
            random.seed(self.config['random_seed'])

        # Initialize starting solution
        current_solution = self._create_initial_solution(problem, initial_guess)
        current_objective = problem.evaluate_objective(current_solution)
        current_violations = problem.get_constraint_violations(current_solution)

        # Track best solution found so far
        best_solution = current_solution.copy()
        best_objective = current_objective
        best_violations = current_violations.copy()

        # Initialize temperature
        current_temperature = self.config['initial_temperature']
        final_temperature = self.config['final_temperature']

        # Add initial solution to history
        solution.add_iteration(
            variable_dict=current_solution,
            objective_value=current_objective,
            constraint_violations=current_violations,
            metadata={
                'temperature': current_temperature,
                'iteration_type': 'initial',
                'is_best_so_far': True
            }
        )

        # Main annealing loop
        total_iterations = 0
        max_iterations = self.config['max_iterations']
        iterations_per_temp = self.config['iterations_per_temperature']

        while (current_temperature > final_temperature and
               total_iterations < max_iterations):

            # Inner loop at current temperature
            for temp_iter in range(iterations_per_temp):
                if total_iterations >= max_iterations:
                    break

                # Generate neighbor solution
                neighbor_solution = self._generate_neighbor(problem, current_solution)
                neighbor_objective = problem.evaluate_objective(neighbor_solution)
                neighbor_violations = problem.get_constraint_violations(neighbor_solution)

                # Calculate acceptance probability
                accept_probability = self._calculate_acceptance_probability(
                    current_objective, neighbor_objective, current_temperature
                )

                # Decide whether to accept the neighbor
                accept_neighbor = random.random() < accept_probability

                # Update current solution if accepted
                if accept_neighbor:
                    current_solution = neighbor_solution
                    current_objective = neighbor_objective
                    current_violations = neighbor_violations

                    # Update best solution if this is better
                    is_new_best = neighbor_objective < best_objective
                    if is_new_best:
                        best_solution = neighbor_solution.copy()
                        best_objective = neighbor_objective
                        best_violations = neighbor_violations.copy()

                    # Add accepted solution to history
                    solution.add_iteration(
                        variable_dict=current_solution,
                        objective_value=current_objective,
                        constraint_violations=current_violations,
                        metadata={
                            'temperature': current_temperature,
                            'iteration_type': 'accepted',
                            'acceptance_probability': accept_probability,
                            'is_best_so_far': is_new_best,
                            'temperature_iteration': temp_iter
                        }
                    )

                total_iterations += 1

                # Check convergence
                if self._check_annealing_convergence(solution):
                    break

            # Cool down temperature
            current_temperature = self._update_temperature(
                current_temperature, total_iterations
            )

        # Add final best solution to history
        solution.add_iteration(
            variable_dict=best_solution,
            objective_value=best_objective,
            constraint_violations=best_violations,
            metadata={
                'temperature': current_temperature,
                'iteration_type': 'final_best',
                'total_iterations': total_iterations,
                'termination_reason': 'temperature_reached' if current_temperature <= final_temperature else 'max_iterations'
            }
        )

        return solution

    def _update_temperature(self, current_temp: float, iteration: int) -> float:
        """Update temperature according to cooling schedule."""
        cooling_schedule = self.config['cooling_schedule']
        cooling_rate = self.config['cooling_rate']
        initial_temp = self.config['initial_temperature']
        final_temp = self.config['final_temperature']

        if cooling_schedule == 'geometric':
            # Geometric cooling: T(k+1) = alpha * T(k)
            new_temp = current_temp * cooling_rate

        elif cooling_schedule == 'linear':
            # Linear cooling: T(k) = T0 - k * (T0 - Tf) / max_iterations
            max_iter = self.config['max_iterations']
            temp_decrease = (initial_temp - final_temp) / max_iter
            new_temp = initial_temp - iteration * temp_decrease

        elif cooling_schedule == 'logarithmic':
            # Logarithmic cooling: T(k) = T0 / log(1 + k)
            new_temp = initial_temp / math.log(1 + iteration + 1)

        else:
            production_assert(False, f"Unknown cooling schedule: {cooling_schedule}")

        # Ensure temperature doesn't go below final temperature
        new_temp = max(new_temp, final_temp)

        production_assert(new_temp > 0, f"Temperature must remain positive, got {new_temp}")

        return new_temp

    def _generate_neighbor(self, problem, current_solution: Dict) -> Dict:
        """Generate neighbor solution by perturbing current solution."""
        neighbor_values = {}
        perturbation_magnitude = self.config['perturbation_magnitude']

        for variable in problem.variables:
            var_name = variable.name
            current_value = current_solution[var_name]['value']

            if isinstance(variable, BinaryVariable):
                # Binary variables: flip with some probability
                if random.random() < perturbation_magnitude:
                    neighbor_values[var_name] = 1 - current_value
                else:
                    neighbor_values[var_name] = current_value

            elif isinstance(variable, IntegerVariable):
                # Integer variables: add random integer perturbation
                if variable.bounds:
                    range_size = variable.bounds[1] - variable.bounds[0]
                    max_change = max(1, int(range_size * perturbation_magnitude))
                else:
                    max_change = max(1, int(abs(current_value) * perturbation_magnitude + 1))

                change = random.randint(-max_change, max_change)
                new_value = current_value + change

                # Clip to bounds
                if variable.bounds:
                    new_value = variable.clip_to_bounds(new_value)

                neighbor_values[var_name] = int(new_value)

            else:  # ContinuousVariable
                # Continuous variables: add Gaussian noise
                if variable.bounds:
                    range_size = variable.bounds[1] - variable.bounds[0]
                    if range_size != float('inf'):
                        noise_std = range_size * perturbation_magnitude
                    else:
                        noise_std = abs(current_value) * perturbation_magnitude + 1.0
                else:
                    noise_std = abs(current_value) * perturbation_magnitude + 1.0

                noise = random.gauss(0, noise_std)
                new_value = current_value + noise

                # Clip to bounds
                if variable.bounds:
                    new_value = variable.clip_to_bounds(new_value)

                neighbor_values[var_name] = new_value

        # Create and validate neighbor solution
        neighbor_solution = problem.create_variable_dict(neighbor_values)

        return neighbor_solution

    def _calculate_acceptance_probability(self, current_obj: float, neighbor_obj: float,
                                        temperature: float) -> float:
        """Calculate probability of accepting neighbor solution."""
        production_assert(temperature > 0, f"Temperature must be positive, got {temperature}")

        # If neighbor is better, always accept
        if neighbor_obj < current_obj:
            return 1.0

        # If neighbor is worse, accept with probability exp(-ΔE/T)
        delta_energy = neighbor_obj - current_obj

        # Avoid overflow in exponential
        if delta_energy / temperature > 700:  # exp(700) is near float max
            acceptance_prob = 0.0
        else:
            acceptance_prob = math.exp(-delta_energy / temperature)

        production_assert(0 <= acceptance_prob <= 1,
                         f"Acceptance probability must be in [0,1], got {acceptance_prob}")

        return acceptance_prob

    def _check_annealing_convergence(self, solution) -> bool:
        """Check if annealing has converged."""
        convergence_window = self.config['convergence_window']
        tolerance = self.config['convergence_tolerance']

        history = solution.get_optimization_history()

        if len(history) < convergence_window:
            return False

        # Check if objective values have stabilized
        recent_objectives = [
            entry['objective_value'] for entry in history[-convergence_window:]
        ]

        objective_range = max(recent_objectives) - min(recent_objectives)

        return objective_range < tolerance