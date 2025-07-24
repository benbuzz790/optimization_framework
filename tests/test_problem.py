"""
Comprehensive unit tests for Problem class.

Tests problem definition, objective/constraint bundling, variable management,
and evaluation methods with NASA-style assert validation.
"""

import unittest
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimization_framework.problems.problem import Problem, ObjectiveFunction, ConstraintFunction
from optimization_framework.variables import ContinuousVariable, IntegerVariable, BinaryVariable


class TestObjectiveFunction(unittest.TestCase):
    """Test ObjectiveFunction wrapper class."""

    def setUp(self):
        """Set up test fixtures."""
        def simple_objective(var_dict):
            x = var_dict["x"]["value"]
            return x**2

        self.obj_func = ObjectiveFunction(simple_objective, "quadratic")

        # Create test variable dict
        x_var = ContinuousVariable("x", bounds=(-10, 10))
        self.test_var_dict = {"x": x_var.to_dict_entry(5.0)}

    def test_initialization_valid(self):
        """Test valid objective function initialization."""
        def test_func(var_dict):
            return 42.0

        obj = ObjectiveFunction(test_func, "test")
        self.assertEqual(obj.name, "test")
        self.assertEqual(obj.func, test_func)

        # Default name
        obj_default = ObjectiveFunction(test_func)
        self.assertEqual(obj_default.name, "objective")

    def test_initialization_invalid(self):
        """Test NASA assert failures during initialization."""
        # Non-callable function
        with self.assertRaises(TypeError):
            ObjectiveFunction("not_callable")

        with self.assertRaises(TypeError):
            ObjectiveFunction(123)

        # Invalid name
        with self.assertRaises(ValueError):
            ObjectiveFunction(lambda x: x, "")

        with self.assertRaises(ValueError):
            ObjectiveFunction(lambda x: x, "   ")

    def test_evaluate_valid(self):
        """Test valid objective function evaluation."""
        result = self.obj_func.evaluate(self.test_var_dict)
        self.assertEqual(result, 25.0)  # 5^2
        self.assertIsInstance(result, float)

    def test_evaluate_invalid_variable_dict(self):
        """Test evaluation with invalid variable dictionaries."""
        # Non-dict input
        with self.assertRaises(TypeError):
            self.obj_func.evaluate("not_dict")

        # Empty dict
        with self.assertRaises(ValueError):
            self.obj_func.evaluate({})

        # Missing required keys
        invalid_dict = {"x": {"value": 5.0}}  # Missing type, bounds, variable_object
        with self.assertRaises(KeyError):
            self.obj_func.evaluate(invalid_dict)

        # Invalid variable object
        invalid_dict = {
            "x": {
                "value": 5.0,
                "type": "continuous",
                "bounds": (-10, 10),
                "variable_object": "not_variable"
            }
        }
        with self.assertRaises(TypeError):
            self.obj_func.evaluate(invalid_dict)

    def test_evaluate_function_errors(self):
        """Test handling of function evaluation errors."""
        def error_func(var_dict):
            raise ValueError("Test error")

        obj = ObjectiveFunction(error_func)

        with self.assertRaises(RuntimeError):
            obj.evaluate(self.test_var_dict)

    def test_evaluate_invalid_return_types(self):
        """Test handling of invalid return types."""
        def string_return(var_dict):
            return "not_numeric"

        def nan_return(var_dict):
            return float('nan')

        obj_string = ObjectiveFunction(string_return)
        obj_nan = ObjectiveFunction(nan_return)

        with self.assertRaises(TypeError):
            obj_string.evaluate(self.test_var_dict)

        with self.assertRaises(ValueError):
            obj_nan.evaluate(self.test_var_dict)


class TestConstraintFunction(unittest.TestCase):
    """Test ConstraintFunction wrapper class."""

    def setUp(self):
        """Set up test fixtures."""
        def linear_constraint(var_dict):
            x = var_dict["x"]["value"]
            return x  # x <= 8, x >= 2, x == 5

        self.le_constraint = ConstraintFunction(linear_constraint, "<=", 8.0, "x_upper")
        self.ge_constraint = ConstraintFunction(linear_constraint, ">=", 2.0, "x_lower")
        self.eq_constraint = ConstraintFunction(linear_constraint, "==", 5.0, "x_equal")

        # Create test variable dict
        x_var = ContinuousVariable("x", bounds=(-10, 10))
        self.test_var_dict = {"x": x_var.to_dict_entry(5.0)}

    def test_initialization_valid(self):
        """Test valid constraint function initialization."""
        def test_func(var_dict):
            return 1.0

        constraint = ConstraintFunction(test_func, "<=", 5.0, "test")
        self.assertEqual(constraint.name, "test")
        self.assertEqual(constraint.constraint_type, "<=")
        self.assertEqual(constraint.bound, 5.0)

        # Default name
        constraint_default = ConstraintFunction(test_func, ">=", 0.0)
        self.assertEqual(constraint_default.name, "constraint")

    def test_initialization_invalid(self):
        """Test NASA assert failures during initialization."""
        def test_func(var_dict):
            return 1.0

        # Non-callable function
        with self.assertRaises(TypeError):
            ConstraintFunction("not_callable", "<=", 5.0)

        # Invalid constraint type
        with self.assertRaises(ValueError):
            ConstraintFunction(test_func, "!=", 5.0)

        with self.assertRaises(ValueError):
            ConstraintFunction(test_func, "<", 5.0)

        # Non-numeric bound
        with self.assertRaises(TypeError):
            ConstraintFunction(test_func, "<=", "five")

        # Invalid name
        with self.assertRaises(ValueError):
            ConstraintFunction(test_func, "<=", 5.0, "")

    def test_evaluate(self):
        """Test constraint function evaluation."""
        result = self.le_constraint.evaluate(self.test_var_dict)
        self.assertEqual(result, 5.0)

    def test_is_satisfied(self):
        """Test constraint satisfaction checking."""
        # x = 5.0
        self.assertTrue(self.le_constraint.is_satisfied(self.test_var_dict))  # 5 <= 8
        self.assertTrue(self.ge_constraint.is_satisfied(self.test_var_dict))  # 5 >= 2
        self.assertTrue(self.eq_constraint.is_satisfied(self.test_var_dict))  # 5 == 5

        # Test with different value
        x_var = ContinuousVariable("x", bounds=(-10, 10))
        test_dict_10 = {"x": x_var.to_dict_entry(10.0)}

        self.assertFalse(self.le_constraint.is_satisfied(test_dict_10))  # 10 <= 8
        self.assertTrue(self.ge_constraint.is_satisfied(test_dict_10))   # 10 >= 2
        self.assertFalse(self.eq_constraint.is_satisfied(test_dict_10))  # 10 == 5

    def test_violation_amount(self):
        """Test constraint violation calculation."""
        # x = 5.0 (satisfies all constraints)
        self.assertEqual(self.le_constraint.violation_amount(self.test_var_dict), 0.0)
        self.assertEqual(self.ge_constraint.violation_amount(self.test_var_dict), 0.0)
        self.assertEqual(self.eq_constraint.violation_amount(self.test_var_dict), 0.0)

        # x = 10.0
        x_var = ContinuousVariable("x", bounds=(-10, 10))
        test_dict_10 = {"x": x_var.to_dict_entry(10.0)}

        self.assertEqual(self.le_constraint.violation_amount(test_dict_10), 2.0)  # 10 - 8
        self.assertEqual(self.ge_constraint.violation_amount(test_dict_10), 0.0)  # satisfied
        self.assertEqual(self.eq_constraint.violation_amount(test_dict_10), 5.0)  # |10 - 5|

        # x = 0.0
        test_dict_0 = {"x": x_var.to_dict_entry(0.0)}

        self.assertEqual(self.le_constraint.violation_amount(test_dict_0), 0.0)   # satisfied
        self.assertEqual(self.ge_constraint.violation_amount(test_dict_0), 2.0)   # 2 - 0
        self.assertEqual(self.eq_constraint.violation_amount(test_dict_0), 5.0)   # |0 - 5|


class TestProblem(unittest.TestCase):
    """Test Problem class with comprehensive validation."""

    def setUp(self):
        """Set up test fixtures."""
        # Create variables
        self.x_var = ContinuousVariable("x", bounds=(-10, 10))
        self.y_var = IntegerVariable("y", bounds=(0, 20))
        self.flag_var = BinaryVariable("flag")

        # Create objective function
        def quadratic_objective(var_dict):
            x = var_dict["x"]["value"]
            y = var_dict["y"]["value"]
            return x**2 + y**2

        self.objective = ObjectiveFunction(quadratic_objective, "quadratic")

        # Create constraints
        def sum_constraint(var_dict):
            x = var_dict["x"]["value"]
            y = var_dict["y"]["value"]
            return x + y

        self.constraint1 = ConstraintFunction(sum_constraint, "<=", 15.0, "sum_upper")
        self.constraint2 = ConstraintFunction(sum_constraint, ">=", -5.0, "sum_lower")

        # Create problem
        self.problem = Problem(
            self.objective,
            constraints=[self.constraint1, self.constraint2],
            variables=[self.x_var, self.y_var, self.flag_var]
        )

    def test_initialization_valid(self):
        """Test valid problem initialization."""
        # Minimal problem (objective only)
        minimal_problem = Problem(self.objective)
        self.assertEqual(minimal_problem.objective, self.objective)
        self.assertEqual(len(minimal_problem.constraints), 0)
        self.assertEqual(len(minimal_problem.variables), 0)

        # Full problem
        self.assertEqual(self.problem.objective, self.objective)
        self.assertEqual(len(self.problem.constraints), 2)
        self.assertEqual(len(self.problem.variables), 3)

    def test_initialization_invalid(self):
        """Test NASA assert failures during initialization."""
        # Invalid objective
        with self.assertRaises(TypeError):
            Problem("not_objective")

        # Invalid constraints list
        with self.assertRaises(TypeError):
            Problem(self.objective, constraints="not_list")

        with self.assertRaises(TypeError):
            Problem(self.objective, constraints=["not_constraint"])

        # Invalid variables list
        with self.assertRaises(TypeError):
            Problem(self.objective, variables="not_list")

        with self.assertRaises(TypeError):
            Problem(self.objective, variables=["not_variable"])

        # Duplicate variable names
        duplicate_vars = [
            ContinuousVariable("x"),
            ContinuousVariable("x")  # Duplicate name
        ]
        with self.assertRaises(ValueError):
            Problem(self.objective, variables=duplicate_vars)

    def test_create_variable_dict(self):
        """Test variable dictionary creation."""
        values = {"x": 5.0, "y": 10, "flag": 1}
        var_dict = self.problem.create_variable_dict(values)

        # Check structure
        self.assertIsInstance(var_dict, dict)
        self.assertEqual(len(var_dict), 3)

        # Check contents
        self.assertEqual(var_dict["x"]["value"], 5.0)
        self.assertEqual(var_dict["y"]["value"], 10)
        self.assertEqual(var_dict["flag"]["value"], 1)

        # Check types
        self.assertEqual(var_dict["x"]["type"], "continuous")
        self.assertEqual(var_dict["y"]["type"], "integer")
        self.assertEqual(var_dict["flag"]["type"], "binary")

    def test_create_variable_dict_invalid(self):
        """Test variable dictionary creation with invalid inputs."""
        # Non-dict input
        with self.assertRaises(TypeError):
            self.problem.create_variable_dict("not_dict")

        # Unknown variable
        with self.assertRaises(KeyError):
            self.problem.create_variable_dict({"unknown": 5.0})

        # Invalid value for variable type
        with self.assertRaises(ValueError):
            self.problem.create_variable_dict({"x": 5.0, "y": 10.5, "flag": 1})  # y should be int

        # Missing variables
        with self.assertRaises(ValueError):
            self.problem.create_variable_dict({"x": 5.0})  # Missing y and flag

    def test_evaluate_objective(self):
        """Test objective function evaluation."""
        values = {"x": 3.0, "y": 4, "flag": 0}
        var_dict = self.problem.create_variable_dict(values)

        result = self.problem.evaluate_objective(var_dict)
        self.assertEqual(result, 25.0)  # 3^2 + 4^2

    def test_evaluate_constraints(self):
        """Test constraint evaluation."""
        values = {"x": 5.0, "y": 8, "flag": 1}
        var_dict = self.problem.create_variable_dict(values)

        constraint_values = self.problem.evaluate_constraints(var_dict)
        self.assertEqual(len(constraint_values), 2)
        self.assertEqual(constraint_values[0], 13.0)  # x + y = 5 + 8
        self.assertEqual(constraint_values[1], 13.0)  # Same function, different bounds

    def test_is_feasible(self):
        """Test feasibility checking."""
        # Feasible solution
        feasible_values = {"x": 5.0, "y": 8, "flag": 1}  # sum = 13, within [-5, 15]
        feasible_dict = self.problem.create_variable_dict(feasible_values)
        self.assertTrue(self.problem.is_feasible(feasible_dict))

        # Infeasible solution (violates upper bound)
        infeasible_values = {"x": 10.0, "y": 10, "flag": 1}  # sum = 20 > 15
        infeasible_dict = self.problem.create_variable_dict(infeasible_values)
        self.assertFalse(self.problem.is_feasible(infeasible_dict))

        # Infeasible solution (violates lower bound)
        infeasible_values2 = {"x": -8.0, "y": 2, "flag": 0}  # sum = -6 < -5
        infeasible_dict2 = self.problem.create_variable_dict(infeasible_values2)
        self.assertFalse(self.problem.is_feasible(infeasible_dict2))

    def test_get_constraint_violations(self):
        """Test detailed constraint violation reporting."""
        # Violating solution
        values = {"x": 10.0, "y": 10, "flag": 1}  # sum = 20
        var_dict = self.problem.create_variable_dict(values)

        violations = self.problem.get_constraint_violations(var_dict)
        self.assertEqual(len(violations), 2)
        self.assertEqual(violations["sum_upper"], 5.0)  # 20 - 15
        self.assertEqual(violations["sum_lower"], 0.0)  # satisfied

    def test_get_total_violation(self):
        """Test total violation calculation."""
        values = {"x": 10.0, "y": 10, "flag": 1}  # sum = 20
        var_dict = self.problem.create_variable_dict(values)

        total_violation = self.problem.get_total_violation(var_dict)
        self.assertEqual(total_violation, 5.0)  # Only upper bound violated

    def test_get_variable_names(self):
        """Test variable name retrieval."""
        names = self.problem.get_variable_names()
        self.assertEqual(set(names), {"x", "y", "flag"})

    def test_get_variable_by_name(self):
        """Test variable retrieval by name."""
        x_retrieved = self.problem.get_variable_by_name("x")
        self.assertIs(x_retrieved, self.x_var)

        # Unknown variable
        with self.assertRaises(KeyError):
            self.problem.get_variable_by_name("unknown")

    def test_get_variable_bounds(self):
        """Test variable bounds retrieval."""
        bounds = self.problem.get_variable_bounds()
        expected_bounds = {
            "x": (-10, 10),
            "y": (0, 20),
            "flag": (0, 1)
        }
        self.assertEqual(bounds, expected_bounds)

    def test_get_variable_types(self):
        """Test variable types retrieval."""
        types = self.problem.get_variable_types()
        expected_types = {
            "x": "continuous",
            "y": "integer",
            "flag": "binary"
        }
        self.assertEqual(types, expected_types)

    def test_validate_solution(self):
        """Test comprehensive solution validation."""
        # Valid feasible solution
        values = {"x": 5.0, "y": 8, "flag": 1}
        var_dict = self.problem.create_variable_dict(values)

        validation = self.problem.validate_solution(var_dict)

        self.assertTrue(validation['objective_valid'])
        self.assertEqual(validation['objective_value'], 89.0)  # 5^2 + 8^2
        self.assertTrue(validation['is_feasible'])
        self.assertEqual(validation['total_violation'], 0.0)
        self.assertEqual(validation['num_constraints'], 2)
        self.assertEqual(validation['num_variables'], 3)
        self.assertEqual(len(validation['constraint_errors']), 0)

    def test_problem_with_no_constraints(self):
        """Test problem with no constraints."""
        unconstrained = Problem(self.objective, variables=[self.x_var, self.y_var])

        values = {"x": 5.0, "y": 10}
        var_dict = unconstrained.create_variable_dict(values)

        self.assertTrue(unconstrained.is_feasible(var_dict))  # Always feasible
        self.assertEqual(len(unconstrained.get_constraint_violations(var_dict)), 0)
        self.assertEqual(unconstrained.get_total_violation(var_dict), 0.0)

    def test_problem_with_no_variables(self):
        """Test problem with no variables defined."""
        no_vars_problem = Problem(self.objective)

        # Should still work with manually created variable dict
        x_var = ContinuousVariable("x")
        manual_dict = {"x": x_var.to_dict_entry(5.0)}

        # This should work for evaluation even without variables in problem
        # (though create_variable_dict would fail)
        with self.assertRaises(KeyError):
            no_vars_problem.create_variable_dict({"x": 5.0})

    def test_string_representation(self):
        """Test string representation of problem."""
        problem_str = str(self.problem)
        self.assertIn("Problem", problem_str)
        self.assertIn("quadratic", problem_str)
        self.assertIn("variables=3", problem_str)
        self.assertIn("constraints=2", problem_str)


if __name__ == '__main__':
    unittest.main()
