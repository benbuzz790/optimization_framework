"""
Solution class for optimization framework.

This module provides the Solution class for tracking complete optimization
solutions with full history, convergence analysis, and summary statistics.
"""

import sys
import os
import time
from typing import Dict, List, Any, Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.assertions import production_assert, validate_string


class Solution:
    """
    Complete optimization solution with full history.

    Tracks complete optimization process including all iterations,
    objective values, constraint violations, and metadata.
    Provides analysis and summary methods with memory-aware storage.

    Examples:
        # Create solution tracker
        solution = Solution(problem, "GeneticAlgorithm")

        # Add iterations during optimization
        solution.add_iteration(var_dict, obj_val, violations, metadata)

        # Get results
        best = solution.get_best_solution()
        history = solution.get_optimization_history()
        stats = solution.get_summary_statistics()
    """

    def __init__(self, problem, solver_name: str, max_history_size: Optional[int] = None):
        """
        Initialize solution tracking.

        Args:
            problem: Problem instance being solved
            solver_name: Name of solver algorithm
            max_history_size: Maximum number of iterations to store (None = unlimited)

        NASA Assert: problem is Problem instance, solver_name is non-empty string
        """
        # Use duck typing to avoid circular import issues
        production_assert(
            hasattr(problem, 'objective') and hasattr(problem, 'constraints') and 
            hasattr(problem, 'variables') and hasattr(problem, 'create_variable_dict'),
            f"Problem must have objective, constraints, variables, and create_variable_dict methods",
            TypeError
        )

        validate_string(solver_name, "Solver name", allow_empty=False)

        if max_history_size is not None:
            production_assert(
                isinstance(max_history_size, int) and max_history_size > 0,
                f"max_history_size must be positive integer, got {max_history_size}",
                ValueError
            )

        self.problem = problem
        self.solver_name = solver_name.strip()
        self.max_history_size = max_history_size
        self.history = []
        self.start_time = None
        self.end_time = None
        self._best_solution_cache = None
        self._cache_valid = False

    def start_timing(self):
        """Start timing the optimization process."""
        self.start_time = time.time()

    def end_timing(self):
        """End timing the optimization process."""
        self.end_time = time.time()

    def get_elapsed_time(self) -> Optional[float]:
        """
        Get elapsed optimization time in seconds.

        Returns:
            float: Elapsed time in seconds, or None if timing not used
        """
        if self.start_time is None:
            return None

        end_time = self.end_time if self.end_time is not None else time.time()
        return end_time - self.start_time

    def add_iteration(self, variable_dict: Dict, objective_value: float,
                     constraint_violations: Optional[Dict[str, float]] = None,
                     metadata: Optional[Dict] = None):
        """
        Add iteration to solution history.

        Args:
            variable_dict: Variable dictionary for this iteration
            objective_value: Objective function value
            constraint_violations: Optional constraint violation amounts
            metadata: Optional algorithm-specific metadata

        NASA Assert: variable_dict is valid, objective_value is numeric
        """
        # Validate variable dictionary using problem's objective function
        production_assert(
            isinstance(variable_dict, dict),
            f"Variable dict must be dict, got {type(variable_dict).__name__}",
            TypeError
        )

        # Use problem's validation
        try:
            self.problem.objective.evaluate(variable_dict)
        except Exception as e:
            production_assert(
                False,
                f"Invalid variable dictionary: {str(e)}",
                ValueError
            )

        production_assert(
            isinstance(objective_value, (int, float)),
            f"Objective value must be numeric, got {type(objective_value).__name__}",
            TypeError
        )

        production_assert(
            objective_value == objective_value,  # Check for NaN
            "Objective value cannot be NaN",
            ValueError
        )

        # Validate constraint violations if provided
        if constraint_violations is not None:
            production_assert(
                isinstance(constraint_violations, dict),
                f"Constraint violations must be dict, got {type(constraint_violations).__name__}",
                TypeError
            )

            for name, violation in constraint_violations.items():
                production_assert(
                    isinstance(violation, (int, float)) and violation >= 0,
                    f"Constraint violation for '{name}' must be non-negative numeric, got {violation}",
                    ValueError
                )

        # Calculate constraint violations if not provided
        if constraint_violations is None and self.problem.constraints:
            constraint_violations = self.problem.get_constraint_violations(variable_dict)

        # Create iteration data
        iteration_data = {
            'iteration': len(self.history),
            'variable_dict': self._deep_copy_variable_dict(variable_dict),
            'objective_value': float(objective_value),
            'constraint_violations': constraint_violations or {},
            'is_feasible': self.problem.is_feasible(variable_dict),
            'total_violation': sum((constraint_violations or {}).values()),
            'timestamp': time.time(),
            'metadata': metadata or {}
        }

        # Add to history with size management
        self.history.append(iteration_data)
        self._invalidate_cache()

        # Manage history size if limit is set
        if self.max_history_size is not None and len(self.history) > self.max_history_size:
            # Keep first and last iterations, compress middle
            self._compress_history()

    def _deep_copy_variable_dict(self, variable_dict: Dict) -> Dict:
        """Create a deep copy of variable dictionary for safe storage."""
        copied_dict = {}
        for var_name, var_data in variable_dict.items():
            copied_dict[var_name] = {
                'value': var_data['value'],
                'type': var_data['type'],
                'bounds': var_data['bounds'],
                'variable_object': var_data['variable_object']  # Reference is OK
            }
        return copied_dict

    def _compress_history(self):
        """Compress history by keeping key iterations and sampling others."""
        if len(self.history) <= self.max_history_size:
            return

        # Always keep first and last iterations
        keep_indices = {0, len(self.history) - 1}

        # Keep best solutions (feasible and infeasible)
        best_feasible_idx = None
        best_infeasible_idx = None
        best_feasible_obj = float('inf')
        best_infeasible_obj = float('inf')

        for i, iteration in enumerate(self.history):
            if iteration['is_feasible']:
                if iteration['objective_value'] < best_feasible_obj:
                    best_feasible_obj = iteration['objective_value']
                    best_feasible_idx = i
            else:
                if iteration['objective_value'] < best_infeasible_obj:
                    best_infeasible_obj = iteration['objective_value']
                    best_infeasible_idx = i

        if best_feasible_idx is not None:
            keep_indices.add(best_feasible_idx)
        if best_infeasible_idx is not None:
            keep_indices.add(best_infeasible_idx)

        # Sample remaining iterations evenly
        remaining_slots = self.max_history_size - len(keep_indices)
        if remaining_slots > 0:
            available_indices = set(range(len(self.history))) - keep_indices
            if available_indices:
                sample_step = max(1, len(available_indices) // remaining_slots)
                sampled = list(available_indices)[::sample_step][:remaining_slots]
                keep_indices.update(sampled)

        # Keep selected iterations
        self.history = [self.history[i] for i in sorted(keep_indices)]

        # Update iteration numbers
        for i, iteration in enumerate(self.history):
            iteration['compressed_index'] = i

    def _invalidate_cache(self):
        """Invalidate cached results."""
        self._cache_valid = False
        self._best_solution_cache = None

    def get_best_solution(self) -> Dict:
        """
        Get best feasible solution, or best infeasible if none feasible.

        Returns:
            Dict: Best solution data

        NASA Assert: history must not be empty
        """
        production_assert(
            len(self.history) > 0,
            "Cannot get best solution from empty history",
            ValueError
        )

        # Use cache if valid
        if self._cache_valid and self._best_solution_cache is not None:
            return self._best_solution_cache

        # Find best feasible solution
        feasible_solutions = [sol for sol in self.history if sol['is_feasible']]

        if feasible_solutions:
            # Return feasible solution with best objective value
            best_feasible = min(feasible_solutions, key=lambda x: x['objective_value'])
            self._best_solution_cache = best_feasible
        else:
            # Return infeasible solution with best objective value
            best_infeasible = min(self.history, key=lambda x: x['objective_value'])
            self._best_solution_cache = best_infeasible

        self._cache_valid = True
        return self._best_solution_cache

    def get_optimization_history(self) -> List[Dict]:
        """
        Get complete optimization history.

        Returns:
            List[Dict]: Copy of optimization history
        """
        return [iteration.copy() for iteration in self.history]

    def get_convergence_data(self) -> Dict:
        """
        Get convergence analysis data.

        Returns:
            Dict: Convergence analysis including objective history and statistics
        """
        if not self.history:
            return {
                'converged': False,
                'reason': 'No optimization history',
                'total_iterations': 0
            }

        objective_values = [sol['objective_value'] for sol in self.history]
        feasible_objectives = [sol['objective_value'] for sol in self.history if sol['is_feasible']]

        # Calculate improvement metrics
        initial_obj = objective_values[0]
        final_obj = objective_values[-1]
        best_obj = min(objective_values)

        # Calculate convergence metrics
        convergence_data = {
            'total_iterations': len(self.history),
            'initial_objective': initial_obj,
            'final_objective': final_obj,
            'best_objective': best_obj,
            'improvement': initial_obj - best_obj,
            'relative_improvement': (initial_obj - best_obj) / abs(initial_obj) if initial_obj != 0 else 0,
            'objective_history': objective_values,
            'feasible_objectives': feasible_objectives,
            'num_feasible': len(feasible_objectives),
            'feasibility_rate': len(feasible_objectives) / len(self.history),
            'converged': self.is_converged(),
            'stagnation_iterations': self._count_stagnation_iterations()
        }

        # Add timing information if available
        elapsed_time = self.get_elapsed_time()
        if elapsed_time is not None:
            convergence_data['elapsed_time'] = elapsed_time
            convergence_data['iterations_per_second'] = len(self.history) / elapsed_time

        return convergence_data

    def is_converged(self, tolerance: float = 1e-6, min_iterations: int = 10) -> bool:
        """
        Check if optimization has converged.

        Args:
            tolerance: Convergence tolerance for objective value changes
            min_iterations: Minimum iterations before checking convergence

        Returns:
            bool: True if optimization has converged

        NASA Assert: tolerance and min_iterations must be positive
        """
        production_assert(
            isinstance(tolerance, (int, float)) and tolerance > 0,
            f"Tolerance must be positive numeric, got {tolerance}",
            ValueError
        )
        production_assert(
            isinstance(min_iterations, int) and min_iterations > 0,
            f"min_iterations must be positive integer, got {min_iterations}",
            ValueError
        )

        if len(self.history) < min_iterations:
            return False

        # Check recent objective values for convergence
        recent_objectives = [sol['objective_value'] for sol in self.history[-min_iterations:]]
        objective_range = max(recent_objectives) - min(recent_objectives)

        return objective_range < tolerance

    def _count_stagnation_iterations(self, tolerance: float = 1e-8) -> int:
        """Count iterations since last significant improvement."""
        if len(self.history) < 2:
            return 0

        best_obj = min(sol['objective_value'] for sol in self.history)

        # Find last iteration with significant improvement
        for i in range(len(self.history) - 1, -1, -1):
            if abs(self.history[i]['objective_value'] - best_obj) < tolerance:
                return len(self.history) - 1 - i

        return len(self.history)

    def get_summary_statistics(self) -> Dict:
        """
        Get summary statistics of optimization run.

        Returns:
            Dict: Comprehensive summary statistics
        """
        if not self.history:
            return {
                'status': 'No optimization performed',
                'solver': self.solver_name
            }

        best_solution = self.get_best_solution()
        convergence_data = self.get_convergence_data()

        # Calculate constraint statistics
        constraint_stats = {}
        if self.problem.constraints:
            all_violations = {}
            for constraint in self.problem.constraints:
                violations = [sol['constraint_violations'].get(constraint.name, 0) 
                            for sol in self.history]
                all_violations[constraint.name] = {
                    'mean_violation': sum(violations) / len(violations),
                    'max_violation': max(violations),
                    'satisfaction_rate': sum(1 for v in violations if v == 0) / len(violations)
                }
            constraint_stats = all_violations

        # Calculate variable statistics
        variable_stats = {}
        for var_name in self.problem.get_variable_names():
            values = [sol['variable_dict'][var_name]['value'] for sol in self.history]
            variable_stats[var_name] = {
                'initial_value': values[0],
                'final_value': values[-1],
                'best_value': best_solution['variable_dict'][var_name]['value'],
                'mean_value': sum(values) / len(values),
                'min_value': min(values),
                'max_value': max(values)
            }

        summary = {
            'solver': self.solver_name,
            'status': 'completed',
            'total_iterations': len(self.history),
            'best_objective': best_solution['objective_value'],
            'is_feasible': best_solution['is_feasible'],
            'converged': convergence_data['converged'],
            'improvement': convergence_data['improvement'],
            'relative_improvement': convergence_data['relative_improvement'],
            'feasibility_rate': convergence_data['feasibility_rate'],
            'constraint_violations': best_solution['constraint_violations'],
            'total_violation': best_solution['total_violation'],
            'constraint_statistics': constraint_stats,
            'variable_statistics': variable_stats,
            'stagnation_iterations': convergence_data['stagnation_iterations']
        }

        # Add timing information if available
        elapsed_time = self.get_elapsed_time()
        if elapsed_time is not None:
            summary['elapsed_time'] = elapsed_time
            summary['iterations_per_second'] = convergence_data['iterations_per_second']

        return summary

    def get_feasible_solutions(self) -> List[Dict]:
        """
        Get all feasible solutions from history.

        Returns:
            List[Dict]: List of all feasible solutions
        """
        return [sol for sol in self.history if sol['is_feasible']]

    def get_pareto_front(self, secondary_objective_func=None) -> List[Dict]:
        """
        Get Pareto front for multi-objective analysis.

        Args:
            secondary_objective_func: Optional secondary objective function

        Returns:
            List[Dict]: Solutions on the Pareto front
        """
        if secondary_objective_func is None:
            # Single objective - just return best solution
            return [self.get_best_solution()]

        # Multi-objective Pareto front calculation
        feasible_solutions = self.get_feasible_solutions()
        if not feasible_solutions:
            return []

        pareto_solutions = []

        for candidate in feasible_solutions:
            is_dominated = False
            candidate_obj1 = candidate['objective_value']
            candidate_obj2 = secondary_objective_func(candidate['variable_dict'])

            for other in feasible_solutions:
                if other == candidate:
                    continue

                other_obj1 = other['objective_value']
                other_obj2 = secondary_objective_func(other['variable_dict'])

                # Check if candidate is dominated
                if (other_obj1 <= candidate_obj1 and other_obj2 <= candidate_obj2 and
                    (other_obj1 < candidate_obj1 or other_obj2 < candidate_obj2)):
                    is_dominated = True
                    break

            if not is_dominated:
                pareto_solutions.append(candidate)

        return pareto_solutions

    def export_history_csv(self, filename: str, include_variables: bool = True):
        """
        Export optimization history to CSV file.

        Args:
            filename: Output CSV filename
            include_variables: Whether to include variable values in export

        NASA Assert: filename must be string, history must not be empty
        """
        production_assert(
            isinstance(filename, str) and len(filename.strip()) > 0,
            f"Filename must be non-empty string, got '{filename}'",
            ValueError
        )
        production_assert(
            len(self.history) > 0,
            "Cannot export empty history",
            ValueError
        )

        import csv

        # Prepare headers
        headers = ['iteration', 'objective_value', 'is_feasible', 'total_violation']

        # Add constraint violation headers
        if self.problem.constraints:
            for constraint in self.problem.constraints:
                headers.append(f'violation_{constraint.name}')

        # Add variable headers if requested
        if include_variables:
            for var_name in self.problem.get_variable_names():
                headers.append(f'var_{var_name}')

        # Write CSV
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)

            for iteration in self.history:
                row = [
                    iteration['iteration'],
                    iteration['objective_value'],
                    iteration['is_feasible'],
                    iteration['total_violation']
                ]

                # Add constraint violations
                if self.problem.constraints:
                    for constraint in self.problem.constraints:
                        violation = iteration['constraint_violations'].get(constraint.name, 0)
                        row.append(violation)

                # Add variable values if requested
                if include_variables:
                    for var_name in self.problem.get_variable_names():
                        value = iteration['variable_dict'][var_name]['value']
                        row.append(value)

                writer.writerow(row)

    def __str__(self) -> str:
        """String representation of solution."""
        status = "empty" if not self.history else f"{len(self.history)} iterations"
        return f"Solution(solver='{self.solver_name}', {status})"

    def __repr__(self) -> str:
        """Detailed string representation of solution."""
        return self.__str__()