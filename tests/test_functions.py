"""
Test suite for function framework components.

Tests ObjectiveFunction and ConstraintFunction classes with comprehensive
validation and edge case coverage.
"""

import unittest
import math
from optimization_framework import *


class TestObjectiveFunction(unittest.TestCase):
    """Test ObjectiveFunction wrapper class."""

    def setUp(self):
        """Set up test fixtures."""
        self.x = ContinuousVariable("x", bounds=(-10, 10))
        self.y = IntegerVariable("y", bounds=(0, 20))

        def simple_func(var_dict):
            return var_dict["x"]["value"]**2 + var_dict["y"]["value"]

        self.objective = ObjectiveFunction(simple_func, "test_objective")

        # Create valid variable dictionary
        self.var_dict = {
            "x": {"value": 2.0, "type": "continuous", "bounds": (-10, 10), "variable_object": self.x},
            "y": {"value": 3, "type": "integer", "bounds": (0, 20), "variable_object": self.y}
        }

    def test_initialization(self):
        """Test ObjectiveFunction initialization."""
        # Valid initialization
        def func(var_dict): return 1.0
        obj = ObjectiveFunction(func, "test")
        self.assertEqual(obj.name, "test")
        self.assertEqual(obj.func, func)

        # Invalid function
        with self.assertRaises(AssertionError):
            ObjectiveFunction("not_callable", "test")

        # Invalid name
        with self.assertRaises(AssertionError):
            ObjectiveFunction(func, 123)

        # Empty name
        with self.assertRaises(AssertionError):
            ObjectiveFunction(func, "")

    def test_evaluate_valid(self):
        """Test evaluation with valid inputs."""
        result = self.objective.evaluate(self.var_dict)
        expected = 2.0**2 + 3  # x^2 + y = 4 + 3 = 7
        self.assertEqual(result, expected)
        self.assertIsInstance(result, float)

    def test_evaluate_invalid_dict(self):
        """Test evaluation with invalid variable dictionary."""
        # Not a dictionary
        with self.assertRaises(AssertionError):
            self.objective.evaluate("not_dict")

        # Empty dictionary
        with self.assertRaises(AssertionError):
            self.objective.evaluate({})

        # Missing required keys
        invalid_dict = {"x": {"value": 1.0}}  # Missing required keys
        with self.assertRaises(AssertionError):
            self.objective.evaluate(invalid_dict)

    def test_evaluate_function_error(self):
        """Test evaluation when function raises exception."""
        def error_func(var_dict):
            raise ValueError("Test error")

        error_obj = ObjectiveFunction(error_func, "error_test")

        with self.assertRaises(Exception):
            error_obj.evaluate(self.var_dict)

    def test_evaluate_non_numeric_return(self):
        """Test evaluation when function returns non-numeric value."""
        def string_func(var_dict):
            return "not_numeric"

        string_obj = ObjectiveFunction(string_func, "string_test")

        with self.assertRaises(AssertionError):
            string_obj.evaluate(self.var_dict)

    def test_evaluate_nan_return(self):
        """Test evaluation when function returns NaN."""
        def nan_func(var_dict):
            return float('nan')

        nan_obj = ObjectiveFunction(nan_func, "nan_test")

        with self.assertRaises(AssertionError):
            nan_obj.evaluate(self.var_dict)

    def test_validate_variable_dict(self):
        """Test variable dictionary validation."""
        # Valid dictionary
        self.assertTrue(self.objective.validate_variable_dict(self.var_dict))

        # Invalid variable name type
        invalid_dict = {123: {"value": 1.0, "type": "continuous", "bounds": None, "variable_object": self.x}}
        with self.assertRaises(AssertionError):
            self.objective.validate_variable_dict(invalid_dict)

        # Invalid variable data type
        invalid_dict = {"x": "not_dict"}
        with self.assertRaises(AssertionError):
            self.objective.validate_variable_dict(invalid_dict)


class TestConstraintFunction(unittest.TestCase):
    """Test ConstraintFunction wrapper class."""

    def setUp(self):
        """Set up test fixtures."""
        self.x = ContinuousVariable("x", bounds=(-10, 10))
        self.y = IntegerVariable("y", bounds=(0, 20))

        def constraint_func(var_dict):
            return var_dict["x"]["value"] + var_dict["y"]["value"]

        self.constraint = ConstraintFunction(constraint_func, "<=", 10.0, "sum_constraint")

        # Create valid variable dictionary
        self.var_dict = {
            "x": {"value": 3.0, "type": "continuous", "bounds": (-10, 10), "variable_object": self.x},
            "y": {"value": 4, "type": "integer", "bounds": (0, 20), "variable_object": self.y}
        }

    def test_initialization(self):
        """Test ConstraintFunction initialization."""
        def func(var_dict): return 1.0

        # Valid initialization
        constraint = ConstraintFunction(func, "<=", 5.0, "test")
        self.assertEqual(constraint.constraint_type, "<=")
        self.assertEqual(constraint.bound, 5.0)
        self.assertEqual(constraint.name, "test")

        # Invalid function
        with self.assertRaises(AssertionError):
            ConstraintFunction("not_callable", "<=", 5.0, "test")

        # Invalid constraint type
        with self.assertRaises(AssertionError):
            ConstraintFunction(func, "invalid", 5.0, "test")

        # Invalid bound type
        with self.assertRaises(AssertionError):
            ConstraintFunction(func, "<=", "not_numeric", "test")

        # Valid constraint types
        for constraint_type in ["<=", ">=", "=="]:
            constraint = ConstraintFunction(func, constraint_type, 5.0, "test")
            self.assertEqual(constraint.constraint_type, constraint_type)

    def test_evaluate(self):
        """Test constraint function evaluation."""
        result = self.constraint.evaluate(self.var_dict)
        expected = 3.0 + 4  # x + y = 7
        self.assertEqual(result, expected)

    def test_is_satisfied_less_equal(self):
        """Test constraint satisfaction for <= constraints."""
        # Satisfied case (7 <= 10)
        self.assertTrue(self.constraint.is_satisfied(self.var_dict))

        # Boundary case (exactly equal)
        boundary_dict = self.var_dict.copy()
        boundary_dict["x"]["value"] = 6.0  # 6 + 4 = 10
        self.assertTrue(self.constraint.is_satisfied(boundary_dict))

        # Violated case
        violated_dict = self.var_dict.copy()
        violated_dict["x"]["value"] = 7.0  # 7 + 4 = 11 > 10
        self.assertFalse(self.constraint.is_satisfied(violated_dict))

    def test_is_satisfied_greater_equal(self):
        """Test constraint satisfaction for >= constraints."""
        def constraint_func(var_dict):
            return var_dict["x"]["value"] + var_dict["y"]["value"]

        constraint = ConstraintFunction(constraint_func, ">=", 5.0, "min_constraint")

        # Satisfied case (7 >= 5)
        self.assertTrue(constraint.is_satisfied(self.var_dict))

        # Boundary case
        boundary_dict = self.var_dict.copy()
        boundary_dict["x"]["value"] = 1.0  # 1 + 4 = 5
        self.assertTrue(constraint.is_satisfied(boundary_dict))

        # Violated case
        violated_dict = self.var_dict.copy()
        violated_dict["x"]["value"] = 0.0  # 0 + 4 = 4 < 5
        self.assertFalse(constraint.is_satisfied(violated_dict))

    def test_is_satisfied_equal(self):
        """Test constraint satisfaction for == constraints."""
        def constraint_func(var_dict):
            return var_dict["x"]["value"] + var_dict["y"]["value"]

        constraint = ConstraintFunction(constraint_func, "==", 7.0, "equal_constraint")

        # Satisfied case (exactly equal)
        self.assertTrue(constraint.is_satisfied(self.var_dict))

        # Nearly equal (within tolerance)
        near_dict = self.var_dict.copy()
        near_dict["x"]["value"] = 3.0000000001  # Very close to 7
        self.assertTrue(constraint.is_satisfied(near_dict))

        # Not equal
        not_equal_dict = self.var_dict.copy()
        not_equal_dict["x"]["value"] = 4.0  # 4 + 4 = 8 != 7
        self.assertFalse(constraint.is_satisfied(not_equal_dict))

    def test_violation_amount(self):
        """Test constraint violation amount calculation."""
        # No violation case
        violation = self.constraint.violation_amount(self.var_dict)
        self.assertEqual(violation, 0.0)

        # Violation case
        violated_dict = self.var_dict.copy()
        violated_dict["x"]["value"] = 8.0  # 8 + 4 = 12, violation = 12 - 10 = 2
        violation = self.constraint.violation_amount(violated_dict)
        self.assertEqual(violation, 2.0)

    def test_violation_amount_greater_equal(self):
        """Test violation amount for >= constraints."""
        def constraint_func(var_dict):
            return var_dict["x"]["value"] + var_dict["y"]["value"]

        constraint = ConstraintFunction(constraint_func, ">=", 10.0, "min_constraint")

        # No violation (7 >= 10 is false, violation = 10 - 7 = 3)
        violation = constraint.violation_amount(self.var_dict)
        self.assertEqual(violation, 3.0)

        # Satisfied case
        satisfied_dict = self.var_dict.copy()
        satisfied_dict["x"]["value"] = 7.0  # 7 + 4 = 11 >= 10
        violation = constraint.violation_amount(satisfied_dict)
        self.assertEqual(violation, 0.0)

    def test_violation_amount_equal(self):
        """Test violation amount for == constraints."""
        def constraint_func(var_dict):
            return var_dict["x"]["value"] + var_dict["y"]["value"]

        constraint = ConstraintFunction(constraint_func, "==", 10.0, "equal_constraint")

        # Violation case (|7 - 10| = 3)
        violation = constraint.violation_amount(self.var_dict)
        self.assertEqual(violation, 3.0)

        # No violation case
        satisfied_dict = self.var_dict.copy()
        satisfied_dict["x"]["value"] = 6.0  # 6 + 4 = 10
        violation = constraint.violation_amount(satisfied_dict)
        self.assertEqual(violation, 0.0)


if __name__ == '__main__':
    unittest.main()