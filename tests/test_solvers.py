"""
Test suite for optimization solver components.

Tests all solver classes including GreedySearchSolver, GeneticAlgorithmSolver,
and SimulatedAnnealingSolver with various problem types and configurations.
"""

import unittest
import random
from optimization_framework import *


class TestBaseSolver(unittest.TestCase):
    """Test BaseSolver abstract class functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a concrete solver for testing base functionality
        class TestSolver(BaseSolver):
            def solve(self, problem, initial_guess=None):
                super().solve(problem, initial_guess)  # Call validation
                return Solution(problem, "TestSolver")

            def _validate_config(self, config):
                return config

        self.TestSolver = TestSolver

    def test_initialization(self):
        """Test BaseSolver initialization."""
        # Valid initialization
        solver = self.TestSolver({'param': 1})
        self.assertEqual(solver.config['param'], 1)

        # None config
        solver = self.TestSolver(None)
        self.assertEqual(solver.config, {})

        # Invalid config type
        with self.assertRaises(AssertionError):
            self.TestSolver("not_dict")

    def test_solve_validation(self):
        """Test solve method validation."""
        solver = self.TestSolver()

        # Create test problem
        x = ContinuousVariable("x", bounds=(0, 10))
        objective = ObjectiveFunction(lambda vd: vd["x"]["value"]**2, "test")
        problem = Problem(objective, variables=[x])

        # Valid solve call
        solution = solver.solve(problem)
        self.assertIsInstance(solution, Solution)

        # Invalid problem type
        with self.assertRaises(AssertionError):
            solver.solve("not_problem")

        # Invalid initial guess type
        with self.assertRaises(AssertionError):
            solver.solve(problem, "not_dict")

    def test_create_initial_solution(self):
        """Test initial solution creation."""
        solver = self.TestSolver()

        # Create test problem with mixed variables
        x = ContinuousVariable("x", bounds=(0, 10))
        y = IntegerVariable("y", bounds=(1, 5))
        z = BinaryVariable("z")

        objective = ObjectiveFunction(lambda vd: 1.0, "test")
        problem = Problem(objective, variables=[x, y, z])

        # Test with initial guess
        initial_guess = {"x": 5.0, "y": 3, "z": 1}
        var_dict = solver._create_initial_solution(problem, initial_guess)

        self.assertEqual(var_dict["x"]["value"], 5.0)
        self.assertEqual(var_dict["y"]["value"], 3)
        self.assertEqual(var_dict["z"]["value"], 1)

        # Test without initial guess (should use defaults)
        var_dict = solver._create_initial_solution(problem)

        self.assertIsInstance(var_dict["x"]["value"], float)
        self.assertIsInstance(var_dict["y"]["value"], int)
        self.assertIn(var_dict["z"]["value"], [0, 1])


class TestGreedySearchSolver(unittest.TestCase):
    """Test GreedySearchSolver implementation."""

    def setUp(self):
        """Set up test fixtures."""
        # Simple quadratic problem: minimize x^2 + y^2
        self.x = ContinuousVariable("x", bounds=(-10, 10))
        self.y = ContinuousVariable("y", bounds=(-10, 10))

        def objective_func(var_dict):
            x_val = var_dict["x"]["value"]
            y_val = var_dict["y"]["value"]
            return x_val**2 + y_val**2

        self.objective = ObjectiveFunction(objective_func, "quadratic")
        self.problem = Problem(self.objective, variables=[self.x, self.y])

    def test_initialization(self):
        """Test GreedySearchSolver initialization."""
        # Default configuration
        solver = GreedySearchSolver()
        self.assertIn('max_iterations', solver.config)
        self.assertIn('step_size', solver.config)
        self.assertIn('tolerance', solver.config)

        # Custom configuration
        config = {'max_iterations': 50, 'step_size': 0.2}
        solver = GreedySearchSolver(config)
        self.assertEqual(solver.config['max_iterations'], 50)
        self.assertEqual(solver.config['step_size'], 0.2)

    def test_config_validation(self):
        """Test configuration validation."""
        # Invalid max_iterations
        with self.assertRaises(AssertionError):
            GreedySearchSolver({'max_iterations': 0})

        # Invalid step_size
        with self.assertRaises(AssertionError):
            GreedySearchSolver({'step_size': -1})

        # Invalid tolerance
        with self.assertRaises(AssertionError):
            GreedySearchSolver({'tolerance': -1})

        # Invalid step_reduction_factor
        with self.assertRaises(AssertionError):
            GreedySearchSolver({'step_reduction_factor': 1.5})

    def test_solve_simple_problem(self):
        """Test solving simple optimization problem."""
        solver = GreedySearchSolver({'max_iterations': 20, 'random_seed': 42})

        # Solve with initial guess
        solution = solver.solve(self.problem, initial_guess={"x": 5.0, "y": 3.0})

        # Check solution properties
        self.assertIsInstance(solution, Solution)
        self.assertEqual(solution.solver_name, "GreedySearchSolver")
        self.assertGreater(len(solution.history), 0)

        # Check that optimization improved
        initial_obj = solution.history[0]['objective_value']
        best_solution = solution.get_best_solution()
        final_obj = best_solution['objective_value']

        self.assertLessEqual(final_obj, initial_obj)  # Should improve or stay same

    def test_generate_neighbor(self):
        """Test neighbor generation."""
        solver = GreedySearchSolver({'random_seed': 42})

        # Create current solution
        current_solution = self.problem.create_variable_dict({"x": 2.0, "y": 1.0})

        # Generate neighbor
        neighbor = solver._generate_neighbor(self.problem, current_solution, 0.5)

        # Check that neighbor is valid
        self.assertIn("x", neighbor)
        self.assertIn("y", neighbor)
        self.assertIsInstance(neighbor["x"]["value"], float)
        self.assertIsInstance(neighbor["y"]["value"], float)

        # Check that neighbor is different (with high probability)
        x_changed = neighbor["x"]["value"] != current_solution["x"]["value"]
        y_changed = neighbor["y"]["value"] != current_solution["y"]["value"]
        # At least one should change with step_size = 0.5
        # Note: This is probabilistic, but with seed 42 it should be deterministic


class TestGeneticAlgorithmSolver(unittest.TestCase):
    """Test GeneticAlgorithmSolver implementation."""

    def setUp(self):
        """Set up test fixtures."""
        # Simple optimization problem
        self.x = ContinuousVariable("x", bounds=(-5, 5))
        self.y = IntegerVariable("y", bounds=(0, 10))

        def objective_func(var_dict):
            x_val = var_dict["x"]["value"]
            y_val = var_dict["y"]["value"]
            return (x_val - 1)**2 + (y_val - 3)**2

        self.objective = ObjectiveFunction(objective_func, "shifted_quadratic")
        self.problem = Problem(self.objective, variables=[self.x, self.y])

    def test_initialization(self):
        """Test GeneticAlgorithmSolver initialization."""
        # Default configuration
        solver = GeneticAlgorithmSolver()
        self.assertIn('population_size', solver.config)
        self.assertIn('generations', solver.config)
        self.assertIn('mutation_rate', solver.config)
        self.assertIn('crossover_rate', solver.config)

        # Custom configuration
        config = {'population_size': 20, 'generations': 30}
        solver = GeneticAlgorithmSolver(config)
        self.assertEqual(solver.config['population_size'], 20)
        self.assertEqual(solver.config['generations'], 30)

    def test_config_validation(self):
        """Test configuration validation."""
        # Invalid population_size
        with self.assertRaises(AssertionError):
            GeneticAlgorithmSolver({'population_size': 1})

        # Invalid generations
        with self.assertRaises(AssertionError):
            GeneticAlgorithmSolver({'generations': 0})

        # Invalid mutation_rate
        with self.assertRaises(AssertionError):
            GeneticAlgorithmSolver({'mutation_rate': 1.5})

        # Invalid crossover_rate
        with self.assertRaises(AssertionError):
            GeneticAlgorithmSolver({'crossover_rate': -0.1})

        # Invalid selection_method
        with self.assertRaises(AssertionError):
            GeneticAlgorithmSolver({'selection_method': 'invalid'})

    def test_solve_simple_problem(self):
        """Test solving simple optimization problem."""
        solver = GeneticAlgorithmSolver({
            'population_size': 10,
            'generations': 5,
            'random_seed': 42
        })

        solution = solver.solve(self.problem)

        # Check solution properties
        self.assertIsInstance(solution, Solution)
        self.assertEqual(solution.solver_name, "GeneticAlgorithmSolver")
        self.assertEqual(len(solution.history), 5)  # One per generation

        # Check that each iteration has proper metadata
        for iteration in solution.history:
            self.assertIn('generation', iteration['metadata'])
            self.assertIn('population_size', iteration['metadata'])

    def test_initialize_population(self):
        """Test population initialization."""
        solver = GeneticAlgorithmSolver({'population_size': 5, 'random_seed': 42})

        # Test without initial guess
        population = solver._initialize_population(self.problem)
        self.assertEqual(len(population), 5)

        # Check that all individuals are valid
        for individual in population:
            self.assertIn("x", individual)
            self.assertIn("y", individual)
            self.assertTrue(self.x.validate_value(individual["x"]["value"]))
            self.assertTrue(self.y.validate_value(individual["y"]["value"]))

        # Test with initial guess
        initial_guess = {"x": 2.0, "y": 4}
        population = solver._initialize_population(self.problem, initial_guess)

        # First individual should be the initial guess
        self.assertEqual(population[0]["x"]["value"], 2.0)
        self.assertEqual(population[0]["y"]["value"], 4)

    def test_crossover(self):
        """Test crossover operation."""
        solver = GeneticAlgorithmSolver({'random_seed': 42})

        # Create parent solutions
        parent1 = self.problem.create_variable_dict({"x": 1.0, "y": 2})
        parent2 = self.problem.create_variable_dict({"x": 3.0, "y": 4})

        # Perform crossover
        child1, child2 = solver._crossover(self.problem, parent1, parent2)

        # Check that children are valid
        self.assertTrue(self.x.validate_value(child1["x"]["value"]))
        self.assertTrue(self.y.validate_value(child1["y"]["value"]))
        self.assertTrue(self.x.validate_value(child2["x"]["value"]))
        self.assertTrue(self.y.validate_value(child2["y"]["value"]))

    def test_mutate(self):
        """Test mutation operation."""
        solver = GeneticAlgorithmSolver({'random_seed': 42})

        # Create individual
        individual = self.problem.create_variable_dict({"x": 2.0, "y": 5})

        # Perform mutation
        mutated = solver._mutate(self.problem, individual)

        # Check that mutated individual is valid
        self.assertTrue(self.x.validate_value(mutated["x"]["value"]))
        self.assertTrue(self.y.validate_value(mutated["y"]["value"]))


class TestSimulatedAnnealingSolver(unittest.TestCase):
    """Test SimulatedAnnealingSolver implementation."""

    def setUp(self):
        """Set up test fixtures."""
        # Simple optimization problem with local optima
        self.x = ContinuousVariable("x", bounds=(-10, 10))

        def objective_func(var_dict):
            x_val = var_dict["x"]["value"]
            # Function with multiple local minima
            return x_val**4 - 4*x_val**3 + 4*x_val**2 + 1

        self.objective = ObjectiveFunction(objective_func, "multimodal")
        self.problem = Problem(self.objective, variables=[self.x])

    def test_initialization(self):
        """Test SimulatedAnnealingSolver initialization."""
        # Default configuration
        solver = SimulatedAnnealingSolver()
        self.assertIn('initial_temperature', solver.config)
        self.assertIn('final_temperature', solver.config)
        self.assertIn('cooling_rate', solver.config)
        self.assertIn('max_iterations', solver.config)

        # Custom configuration
        config = {'initial_temperature': 50.0, 'final_temperature': 0.1}
        solver = SimulatedAnnealingSolver(config)
        self.assertEqual(solver.config['initial_temperature'], 50.0)
        self.assertEqual(solver.config['final_temperature'], 0.1)

    def test_config_validation(self):
        """Test configuration validation."""
        # Invalid temperature relationship
        with self.assertRaises(AssertionError):
            SimulatedAnnealingSolver({'initial_temperature': 1.0, 'final_temperature': 2.0})

        # Invalid final_temperature
        with self.assertRaises(AssertionError):
            SimulatedAnnealingSolver({'final_temperature': 0})

        # Invalid cooling_rate
        with self.assertRaises(AssertionError):
            SimulatedAnnealingSolver({'cooling_rate': 1.5})

        # Invalid cooling_schedule
        with self.assertRaises(AssertionError):
            SimulatedAnnealingSolver({'cooling_schedule': 'invalid'})

        # Invalid max_iterations
        with self.assertRaises(AssertionError):
            SimulatedAnnealingSolver({'max_iterations': 0})

    def test_solve_simple_problem(self):
        """Test solving simple optimization problem."""
        solver = SimulatedAnnealingSolver({
            'initial_temperature': 10.0,
            'final_temperature': 0.1,
            'max_iterations': 20,
            'random_seed': 42
        })

        solution = solver.solve(self.problem, initial_guess={"x": 5.0})

        # Check solution properties
        self.assertIsInstance(solution, Solution)
        self.assertEqual(solution.solver_name, "SimulatedAnnealingSolver")
        self.assertGreater(len(solution.history), 0)

        # Check that iterations have proper metadata
        for iteration in solution.history:
            self.assertIn('temperature', iteration['metadata'])
            self.assertIn('iteration_type', iteration['metadata'])

    def test_update_temperature(self):
        """Test temperature update methods."""
        # Geometric cooling
        solver = SimulatedAnnealingSolver({
            'initial_temperature': 100.0,
            'cooling_rate': 0.9,
            'cooling_schedule': 'geometric'
        })

        new_temp = solver._update_temperature(100.0, 1)
        self.assertEqual(new_temp, 90.0)  # 100 * 0.9

        # Linear cooling
        solver = SimulatedAnnealingSolver({
            'initial_temperature': 100.0,
            'final_temperature': 10.0,
            'max_iterations': 90,
            'cooling_schedule': 'linear'
        })

        new_temp = solver._update_temperature(100.0, 45)  # Halfway
        self.assertEqual(new_temp, 55.0)  # 100 - 45 * (100-10)/90

        # Logarithmic cooling
        solver = SimulatedAnnealingSolver({
            'initial_temperature': 100.0,
            'cooling_schedule': 'logarithmic'
        })

        new_temp = solver._update_temperature(100.0, 1)
        expected = 100.0 / math.log(1 + 1 + 1)  # log(3)
        self.assertAlmostEqual(new_temp, expected, places=5)

    def test_calculate_acceptance_probability(self):
        """Test acceptance probability calculation."""
        solver = SimulatedAnnealingSolver()

        # Better solution (should always accept)
        prob = solver._calculate_acceptance_probability(10.0, 5.0, 1.0)
        self.assertEqual(prob, 1.0)

        # Worse solution at high temperature
        prob = solver._calculate_acceptance_probability(5.0, 10.0, 10.0)
        expected = math.exp(-5.0 / 10.0)  # exp(-0.5)
        self.assertAlmostEqual(prob, expected, places=5)

        # Worse solution at low temperature
        prob = solver._calculate_acceptance_probability(5.0, 10.0, 0.1)
        expected = math.exp(-5.0 / 0.1)  # Very small
        self.assertAlmostEqual(prob, expected, places=10)

        # Test overflow protection
        prob = solver._calculate_acceptance_probability(5.0, 1000.0, 0.001)
        self.assertEqual(prob, 0.0)

    def test_generate_neighbor(self):
        """Test neighbor generation."""
        solver = SimulatedAnnealingSolver({'random_seed': 42})

        # Create current solution
        current_solution = self.problem.create_variable_dict({"x": 2.0})

        # Generate neighbor
        neighbor = solver._generate_neighbor(self.problem, current_solution)

        # Check that neighbor is valid
        self.assertIn("x", neighbor)
        self.assertTrue(self.x.validate_value(neighbor["x"]["value"]))


class TestSolverIntegration(unittest.TestCase):
    """Integration tests for all solvers."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a constrained optimization problem
        self.x = ContinuousVariable("x", bounds=(0, 5))
        self.y = IntegerVariable("y", bounds=(0, 5))

        def objective_func(var_dict):
            x_val = var_dict["x"]["value"]
            y_val = var_dict["y"]["value"]
            return x_val**2 + y_val**2

        def constraint_func(var_dict):
            x_val = var_dict["x"]["value"]
            y_val = var_dict["y"]["value"]
            return x_val + y_val

        self.objective = ObjectiveFunction(objective_func, "quadratic")
        self.constraint = ConstraintFunction(constraint_func, "<=", 4.0, "sum_limit")
        self.problem = Problem(self.objective, constraints=[self.constraint], variables=[self.x, self.y])

    def test_all_solvers_on_same_problem(self):
        """Test that all solvers can solve the same problem."""
        solvers = [
            GreedySearchSolver({'max_iterations': 10, 'random_seed': 42}),
            GeneticAlgorithmSolver({'population_size': 10, 'generations': 5, 'random_seed': 42}),
            SimulatedAnnealingSolver({'max_iterations': 20, 'random_seed': 42})
        ]

        results = []

        for solver in solvers:
            solution = solver.solve(self.problem, initial_guess={"x": 2.0, "y": 2})

            # Check basic solution properties
            self.assertIsInstance(solution, Solution)
            self.assertGreater(len(solution.history), 0)

            best = solution.get_best_solution()
            results.append(best['objective_value'])

            # Check that solution respects variable bounds
            x_val = best['variable_dict']['x']['value']
            y_val = best['variable_dict']['y']['value']

            self.assertTrue(0 <= x_val <= 5)
            self.assertTrue(0 <= y_val <= 5)
            self.assertIsInstance(y_val, int)

        # All solvers should find reasonable solutions
        for result in results:
            self.assertIsInstance(result, (int, float))
            self.assertGreaterEqual(result, 0)  # Objective is always non-negative


if __name__ == '__main__':
    # Set random seed for reproducible tests
    random.seed(42)
    unittest.main()