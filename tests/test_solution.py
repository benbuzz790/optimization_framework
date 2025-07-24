"""
Comprehensive unit tests for Solution class.

Tests solution tracking, history management, convergence analysis,
and summary statistics with NASA-style assert validation.
"""

import unittest
import sys
import os
import time
import tempfile

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimization_framework.solutions.solution import Solution
from optimization_framework.problems.problem import Problem, ObjectiveFunction, ConstraintFunction
from optimization_framework.variables import ContinuousVariable, IntegerVariable, BinaryVariable


class TestSolution(unittest.TestCase):
    """Test Solution class with comprehensive validation."""

    def setUp(self):
        """Set up test fixtures."""
        # Create test problem
        self.x_var = ContinuousVariable("x", bounds=(-10, 10))
        self.y_var = IntegerVariable("y", bounds=(0, 20))

        def quadratic_objective(var_dict):
            x = var_dict["x"]["value"]
            y = var_dict["y"]["value"]
            return x**2 + y**2

        self.objective = ObjectiveFunction(quadratic_objective, "quadratic")

        def sum_constraint(var_dict):
            x = var_dict["x"]["value"]
            y = var_dict["y"]["value"]
            return x + y

        self.constraint = ConstraintFunction(sum_constraint, "<=", 15.0, "sum_upper")

        self.problem = Problem(
            self.objective,
            constraints=[self.constraint],
            variables=[self.x_var, self.y_var]
        )

        # Create solution tracker
        self.solution = Solution(self.problem, "TestSolver")

        # Create test variable dictionaries
        self.feasible_dict = self.problem.create_variable_dict({"x": 3.0, "y": 4})
        self.infeasible_dict = self.problem.create_variable_dict({"x": 8.0, "y": 10})

    def test_initialization_valid(self):
        """Test valid solution initialization."""
        solution = Solution(self.problem, "GeneticAlgorithm")
        self.assertEqual(solution.solver_name, "GeneticAlgorithm")
        self.assertIs(solution.problem, self.problem)
        self.assertEqual(len(solution.history), 0)
        self.assertIsNone(solution.start_time)
        self.assertIsNone(solution.end_time)
        self.assertIsNone(solution.max_history_size)

        # With max history size
        solution_limited = Solution(self.problem, "TestSolver", max_history_size=100)
        self.assertEqual(solution_limited.max_history_size, 100)

    def test_initialization_invalid(self):
        """Test NASA assert failures during initialization."""
        # Invalid problem type
        with self.assertRaises(TypeError):
            Solution("not_problem", "TestSolver")

        # Invalid solver name
        with self.assertRaises(ValueError):
            Solution(self.problem, "")

        with self.assertRaises(ValueError):
            Solution(self.problem, "   ")

        # Invalid max_history_size
        with self.assertRaises(ValueError):
            Solution(self.problem, "TestSolver", max_history_size=0)

        with self.assertRaises(ValueError):
            Solution(self.problem, "TestSolver", max_history_size=-1)

    def test_timing_methods(self):
        """Test timing functionality."""
        # Initially no timing
        self.assertIsNone(self.solution.get_elapsed_time())

        # Start timing
        self.solution.start_timing()
        self.assertIsNotNone(self.solution.start_time)

        # Small delay
        time.sleep(0.01)

        # Check elapsed time (should be positive)
        elapsed = self.solution.get_elapsed_time()
        self.assertIsNotNone(elapsed)
        self.assertGreater(elapsed, 0)

        # End timing
        self.solution.end_timing()
        self.assertIsNotNone(self.solution.end_time)

        # Elapsed time should be stable now
        final_elapsed = self.solution.get_elapsed_time()
        self.assertAlmostEqual(elapsed, final_elapsed, delta=0.001)

    def test_add_iteration_valid(self):
        """Test valid iteration addition."""
        # Add feasible iteration
        self.solution.add_iteration(
            self.feasible_dict,
            25.0,  # 3^2 + 4^2
            {"sum_upper": 0.0},  # 3 + 4 = 7 <= 15
            {"algorithm_step": "initialization"}
        )

        self.assertEqual(len(self.solution.history), 1)
        iteration = self.solution.history[0]

        # Check iteration structure
        self.assertEqual(iteration['iteration'], 0)
        self.assertEqual(iteration['objective_value'], 25.0)
        self.assertTrue(iteration['is_feasible'])
        self.assertEqual(iteration['total_violation'], 0.0)
        self.assertEqual(iteration['constraint_violations']['sum_upper'], 0.0)
        self.assertEqual(iteration['metadata']['algorithm_step'], "initialization")

        # Add infeasible iteration
        self.solution.add_iteration(
            self.infeasible_dict,
            164.0,  # 8^2 + 10^2
            {"sum_upper": 3.0}  # 8 + 10 = 18 > 15, violation = 3
        )

        self.assertEqual(len(self.solution.history), 2)
        iteration2 = self.solution.history[1]

        self.assertEqual(iteration2['iteration'], 1)
        self.assertEqual(iteration2['objective_value'], 164.0)
        self.assertFalse(iteration2['is_feasible'])
        self.assertEqual(iteration2['total_violation'], 3.0)

    def test_add_iteration_auto_constraint_calculation(self):
        """Test automatic constraint violation calculation."""
        # Don't provide constraint violations - should be calculated automatically
        self.solution.add_iteration(self.infeasible_dict, 164.0)

        iteration = self.solution.history[0]
        self.assertIn('sum_upper', iteration['constraint_violations'])
        self.assertEqual(iteration['constraint_violations']['sum_upper'], 3.0)
        self.assertFalse(iteration['is_feasible'])

    def test_add_iteration_invalid(self):
        """Test NASA assert failures during iteration addition."""
        # Invalid variable dict
        with self.assertRaises(TypeError):
            self.solution.add_iteration("not_dict", 25.0)

        with self.assertRaises(ValueError):
            self.solution.add_iteration({}, 25.0)  # Empty dict

        # Invalid objective value
        with self.assertRaises(TypeError):
            self.solution.add_iteration(self.feasible_dict, "not_numeric")

        with self.assertRaises(ValueError):
            self.solution.add_iteration(self.feasible_dict, float('nan'))

        # Invalid constraint violations
        with self.assertRaises(TypeError):
            self.solution.add_iteration(self.feasible_dict, 25.0, "not_dict")

        with self.assertRaises(ValueError):
            self.solution.add_iteration(self.feasible_dict, 25.0, {"constraint": -1.0})

    def test_get_best_solution_empty(self):
        """Test getting best solution from empty history."""
        with self.assertRaises(ValueError):
            self.solution.get_best_solution()

    def test_get_best_solution_feasible(self):
        """Test getting best solution when feasible solutions exist."""
        # Add multiple iterations
        self.solution.add_iteration(self.feasible_dict, 25.0)  # feasible, obj=25
        self.solution.add_iteration(self.infeasible_dict, 10.0)  # infeasible, obj=10 (better)

        # Create better feasible solution
        better_feasible = self.problem.create_variable_dict({"x": 1.0, "y": 2})
        self.solution.add_iteration(better_feasible, 5.0)  # feasible, obj=5 (best)

        best = self.solution.get_best_solution()
        self.assertTrue(best['is_feasible'])
        self.assertEqual(best['objective_value'], 5.0)

    def test_get_best_solution_infeasible_only(self):
        """Test getting best solution when only infeasible solutions exist."""
        self.solution.add_iteration(self.infeasible_dict, 164.0)  # infeasible, obj=164

        # Add worse infeasible solution
        worse_infeasible = self.problem.create_variable_dict({"x": 9.0, "y": 15})
        self.solution.add_iteration(worse_infeasible, 306.0)  # infeasible, obj=306

        best = self.solution.get_best_solution()
        self.assertFalse(best['is_feasible'])
        self.assertEqual(best['objective_value'], 164.0)  # Better of the two infeasible

    def test_get_optimization_history(self):
        """Test optimization history retrieval."""
        # Empty history
        history = self.solution.get_optimization_history()
        self.assertEqual(len(history), 0)

        # Add iterations
        self.solution.add_iteration(self.feasible_dict, 25.0)
        self.solution.add_iteration(self.infeasible_dict, 164.0)

        history = self.solution.get_optimization_history()
        self.assertEqual(len(history), 2)

        # Should be a copy (modifications don't affect original)
        history[0]['modified'] = True
        self.assertNotIn('modified', self.solution.history[0])

    def test_get_convergence_data_empty(self):
        """Test convergence data for empty history."""
        convergence = self.solution.get_convergence_data()
        self.assertFalse(convergence['converged'])
        self.assertEqual(convergence['reason'], 'No optimization history')
        self.assertEqual(convergence['total_iterations'], 0)

    def test_get_convergence_data_with_history(self):
        """Test convergence data calculation."""
        # Add iterations with improving objective
        objectives = [100.0, 75.0, 50.0, 25.0, 25.0]  # Converging to 25

        for i, obj in enumerate(objectives):
            if i < 3:
                var_dict = self.infeasible_dict  # Use infeasible solution for first three
            else:
                var_dict = self.feasible_dict  # Use feasible solution for last two
            self.solution.add_iteration(var_dict, obj)

        convergence = self.solution.get_convergence_data()

        self.assertEqual(convergence['total_iterations'], 5)
        self.assertEqual(convergence['initial_objective'], 100.0)
        self.assertEqual(convergence['final_objective'], 25.0)
        self.assertEqual(convergence['best_objective'], 25.0)
        self.assertEqual(convergence['improvement'], 75.0)  # 100 - 25
        self.assertEqual(convergence['relative_improvement'], 0.75)  # 75/100
        self.assertEqual(len(convergence['objective_history']), 5)
        self.assertEqual(convergence['num_feasible'], 2)  # Last two iterations
        self.assertEqual(convergence['feasibility_rate'], 0.4)  # 2/5

    def test_is_converged(self):
        """Test convergence detection."""
        # Not enough iterations
        self.solution.add_iteration(self.feasible_dict, 25.0)
        self.assertFalse(self.solution.is_converged(min_iterations=10))

        # Add converged sequence
        for _ in range(15):
            self.solution.add_iteration(self.feasible_dict, 25.0 + 1e-8)  # Very small changes

        self.assertTrue(self.solution.is_converged(tolerance=1e-6))

        # Add large change - should not be converged
        self.solution.add_iteration(self.feasible_dict, 50.0)
        self.assertFalse(self.solution.is_converged(tolerance=1e-6))

    def test_is_converged_invalid_parameters(self):
        """Test convergence detection with invalid parameters."""
        self.solution.add_iteration(self.feasible_dict, 25.0)

        with self.assertRaises(ValueError):
            self.solution.is_converged(tolerance=-1.0)

        with self.assertRaises(ValueError):
            self.solution.is_converged(min_iterations=0)

    def test_get_summary_statistics_empty(self):
        """Test summary statistics for empty solution."""
        stats = self.solution.get_summary_statistics()
        self.assertEqual(stats['status'], 'No optimization performed')
        self.assertEqual(stats['solver'], 'TestSolver')

    def test_get_summary_statistics_with_history(self):
        """Test comprehensive summary statistics."""
        # Add timing
        self.solution.start_timing()
        time.sleep(0.01)

        # Add iterations
        self.solution.add_iteration(self.feasible_dict, 25.0)
        self.solution.add_iteration(self.infeasible_dict, 164.0)

        # Better feasible solution
        better_feasible = self.problem.create_variable_dict({"x": 1.0, "y": 2})
        self.solution.add_iteration(better_feasible, 5.0)

        self.solution.end_timing()

        stats = self.solution.get_summary_statistics()

        # Check basic stats
        self.assertEqual(stats['solver'], 'TestSolver')
        self.assertEqual(stats['status'], 'completed')
        self.assertEqual(stats['total_iterations'], 3)
        self.assertEqual(stats['best_objective'], 5.0)
        self.assertTrue(stats['is_feasible'])
        self.assertEqual(stats['improvement'], 20.0)  # 25 - 5
        self.assertEqual(stats['relative_improvement'], 0.8)  # 20/25
        self.assertAlmostEqual(stats['feasibility_rate'], 2/3, places=5)

        # Check timing info
        self.assertIn('elapsed_time', stats)
        self.assertIn('iterations_per_second', stats)
        self.assertGreater(stats['elapsed_time'], 0)

        # Check constraint statistics
        self.assertIn('constraint_statistics', stats)
        self.assertIn('sum_upper', stats['constraint_statistics'])

        # Check variable statistics
        self.assertIn('variable_statistics', stats)
        self.assertIn('x', stats['variable_statistics'])
        self.assertIn('y', stats['variable_statistics'])

        x_stats = stats['variable_statistics']['x']
        self.assertEqual(x_stats['initial_value'], 3.0)
        self.assertEqual(x_stats['final_value'], 1.0)
        self.assertEqual(x_stats['best_value'], 1.0)

    def test_get_feasible_solutions(self):
        """Test feasible solutions filtering."""
        # Add mixed solutions
        self.solution.add_iteration(self.feasible_dict, 25.0)      # feasible
        self.solution.add_iteration(self.infeasible_dict, 164.0)   # infeasible

        better_feasible = self.problem.create_variable_dict({"x": 1.0, "y": 2})
        self.solution.add_iteration(better_feasible, 5.0)          # feasible

        feasible_solutions = self.solution.get_feasible_solutions()
        self.assertEqual(len(feasible_solutions), 2)

        for solution in feasible_solutions:
            self.assertTrue(solution['is_feasible'])

    def test_get_pareto_front_single_objective(self):
        """Test Pareto front for single objective (should return best solution)."""
        self.solution.add_iteration(self.feasible_dict, 25.0)

        better_feasible = self.problem.create_variable_dict({"x": 1.0, "y": 2})
        self.solution.add_iteration(better_feasible, 5.0)

        pareto_front = self.solution.get_pareto_front()
        self.assertEqual(len(pareto_front), 1)
        self.assertEqual(pareto_front[0]['objective_value'], 5.0)

    def test_get_pareto_front_multi_objective(self):
        """Test Pareto front for multi-objective optimization."""
        def secondary_objective(var_dict):
            x = var_dict["x"]["value"]
            return abs(x)  # Minimize absolute value of x

        # Add solutions with trade-offs
        self.solution.add_iteration(self.feasible_dict, 25.0)  # x=3, primary=25, secondary=3

        solution2 = self.problem.create_variable_dict({"x": 1.0, "y": 2})
        self.solution.add_iteration(solution2, 5.0)  # x=1, primary=5, secondary=1

        solution3 = self.problem.create_variable_dict({"x": 0.0, "y": 3})
        self.solution.add_iteration(solution3, 9.0)  # x=0, primary=9, secondary=0

        pareto_front = self.solution.get_pareto_front(secondary_objective)

        # Should include solutions that are not dominated
        self.assertGreaterEqual(len(pareto_front), 1)

        # Check that all solutions in front are feasible
        for solution in pareto_front:
            self.assertTrue(solution['is_feasible'])

    def test_export_history_csv(self):
        """Test CSV export functionality."""
        # Add some history
        self.solution.add_iteration(self.feasible_dict, 25.0)
        self.solution.add_iteration(self.infeasible_dict, 164.0)

        # Export to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_filename = f.name

        try:
            self.solution.export_history_csv(temp_filename, include_variables=True)

            # Read and verify CSV content
            with open(temp_filename, 'r') as f:
                lines = f.readlines()

            # Should have header + 2 data rows
            self.assertEqual(len(lines), 3)

            # Check header
            header = lines[0].strip().split(',')
            expected_headers = ['iteration', 'objective_value', 'is_feasible', 'total_violation',
                              'violation_sum_upper', 'var_x', 'var_y']
            for expected in expected_headers:
                self.assertIn(expected, header)

            # Check first data row
            row1 = lines[1].strip().split(',')
            self.assertEqual(row1[0], '0')  # iteration
            self.assertEqual(row1[1], '25.0')  # objective_value
            self.assertEqual(row1[2], 'True')  # is_feasible

        finally:
            # Clean up
            os.unlink(temp_filename)

    def test_export_history_csv_invalid(self):
        """Test CSV export with invalid inputs."""
        # Empty filename
        with self.assertRaises(ValueError):
            self.solution.export_history_csv("")

        # Empty history
        with self.assertRaises(ValueError):
            self.solution.export_history_csv("test.csv")

    def test_history_compression(self):
        """Test history compression with size limits."""
        solution_limited = Solution(self.problem, "TestSolver", max_history_size=5)

        # Add more iterations than the limit
        for i in range(10):
            obj_value = 100.0 - i * 10  # Improving objective
            if i % 2 == 0:
                solution_limited.add_iteration(self.feasible_dict, obj_value)
            else:
                solution_limited.add_iteration(self.infeasible_dict, obj_value)

        # Should be compressed to max_history_size
        self.assertLessEqual(len(solution_limited.history), 5)

        # Should still have first and last iterations
        iterations = [iter_data['iteration'] for iter_data in solution_limited.history]
        self.assertIn(0, [iter_data.get('iteration', iter_data.get('compressed_index', -1)) 
                         for iter_data in solution_limited.history])

    def test_deep_copy_variable_dict(self):
        """Test deep copying of variable dictionaries."""
        original_dict = self.feasible_dict.copy()

        # Add iteration
        self.solution.add_iteration(original_dict, 25.0)

        # Modify original dict
        original_dict["x"]["value"] = 999.0

        # Stored dict should be unchanged
        stored_dict = self.solution.history[0]['variable_dict']
        self.assertEqual(stored_dict["x"]["value"], 3.0)

    def test_cache_invalidation(self):
        """Test that cache is properly invalidated."""
        # Add iteration and get best solution (populates cache)
        self.solution.add_iteration(self.feasible_dict, 25.0)
        best1 = self.solution.get_best_solution()

        # Add better solution
        better_feasible = self.problem.create_variable_dict({"x": 1.0, "y": 2})
        self.solution.add_iteration(better_feasible, 5.0)

        # Should get new best solution (cache should be invalidated)
        best2 = self.solution.get_best_solution()
        self.assertNotEqual(best1['objective_value'], best2['objective_value'])
        self.assertEqual(best2['objective_value'], 5.0)

    def test_string_representation(self):
        """Test string representation of solution."""
        # Empty solution
        empty_str = str(self.solution)
        self.assertIn("Solution", empty_str)
        self.assertIn("TestSolver", empty_str)
        self.assertIn("empty", empty_str)

        # Solution with history
        self.solution.add_iteration(self.feasible_dict, 25.0)
        history_str = str(self.solution)
        self.assertIn("1 iterations", history_str)


if __name__ == '__main__':
    unittest.main()
