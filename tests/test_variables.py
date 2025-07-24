"""
Comprehensive unit tests for Variable classes.

Tests all variable types with NASA-style assert validation,
bounds handling, and variable dictionary protocol support.
"""

import unittest
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimization_framework.variables import Variable, ContinuousVariable, IntegerVariable, BinaryVariable


class TestContinuousVariable(unittest.TestCase):
    """Test ContinuousVariable class with comprehensive validation."""

    def setUp(self):
        """Set up test fixtures."""
        self.unbounded_var = ContinuousVariable("x")
        self.bounded_var = ContinuousVariable("y", bounds=(-5.0, 10.0))

    def test_initialization_valid(self):
        """Test valid initialization cases."""
        # Unbounded variable
        var1 = ContinuousVariable("test")
        self.assertEqual(var1.name, "test")
        self.assertEqual(var1.bounds, (-float('inf'), float('inf')))

        # Bounded variable
        var2 = ContinuousVariable("bounded", bounds=(0.0, 100.0))
        self.assertEqual(var2.name, "bounded")
        self.assertEqual(var2.bounds, (0.0, 100.0))

    def test_initialization_invalid(self):
        """Test NASA assert failures during initialization."""
        # Invalid name types
        with self.assertRaises(TypeError):
            ContinuousVariable(123)

        with self.assertRaises(ValueError):
            ContinuousVariable("")

        with self.assertRaises(ValueError):
            ContinuousVariable("   ")

        # Invalid bounds
        with self.assertRaises(ValueError):
            ContinuousVariable("test", bounds="invalid")

        with self.assertRaises(ValueError):
            ContinuousVariable("test", bounds=(1, 2, 3))

        with self.assertRaises(ValueError):
            ContinuousVariable("test", bounds=(10.0, 5.0))  # min > max

        with self.assertRaises(TypeError):
            ContinuousVariable("test", bounds=("a", "b"))

    def test_validate_value(self):
        """Test value validation logic."""
        # Valid values
        self.assertTrue(self.unbounded_var.validate_value(0.0))
        self.assertTrue(self.unbounded_var.validate_value(100))
        self.assertTrue(self.unbounded_var.validate_value(-50.5))

        # Valid bounded values
        self.assertTrue(self.bounded_var.validate_value(0.0))
        self.assertTrue(self.bounded_var.validate_value(5.5))
        self.assertTrue(self.bounded_var.validate_value(-5.0))
        self.assertTrue(self.bounded_var.validate_value(10.0))

        # Invalid types
        self.assertFalse(self.unbounded_var.validate_value("string"))
        self.assertFalse(self.unbounded_var.validate_value([1, 2, 3]))
        self.assertFalse(self.unbounded_var.validate_value(None))

        # Invalid bounded values
        self.assertFalse(self.bounded_var.validate_value(-6.0))
        self.assertFalse(self.bounded_var.validate_value(11.0))

        # NaN values
        self.assertFalse(self.unbounded_var.validate_value(float('nan')))

    def test_clip_to_bounds(self):
        """Test bounds clipping functionality."""
        # Test clipping with bounded variable
        self.assertEqual(self.bounded_var.clip_to_bounds(-10.0), -5.0)
        self.assertEqual(self.bounded_var.clip_to_bounds(15.0), 10.0)
        self.assertEqual(self.bounded_var.clip_to_bounds(5.0), 5.0)

        # Test NASA asserts
        # Unbounded variables have infinite bounds, so clipping should work
        clipped = self.unbounded_var.clip_to_bounds(5.0)
        self.assertEqual(clipped, 5.0)

        with self.assertRaises(TypeError):
            self.bounded_var.clip_to_bounds("invalid")  # Invalid type

    def test_to_dict_entry(self):
        """Test variable dictionary protocol compliance."""
        # Valid conversion
        entry = self.bounded_var.to_dict_entry(5.0)
        expected_keys = ["value", "type", "bounds", "variable_object"]

        self.assertIsInstance(entry, dict)
        for key in expected_keys:
            self.assertIn(key, entry)

        self.assertEqual(entry["value"], 5.0)
        self.assertEqual(entry["type"], "continuous")
        self.assertEqual(entry["bounds"], (-5.0, 10.0))
        self.assertIs(entry["variable_object"], self.bounded_var)

        # Invalid value
        with self.assertRaises(TypeError):
            self.bounded_var.to_dict_entry("invalid")

        # Out of bounds value
        with self.assertRaises(ValueError):
            self.bounded_var.to_dict_entry(15.0)

    def test_get_type_name(self):
        """Test type name identification."""
        self.assertEqual(self.unbounded_var.get_type_name(), "continuous")
        self.assertEqual(self.bounded_var.get_type_name(), "continuous")

    def test_get_random_value(self):
        """Test random value generation."""
        # Bounded variable
        for _ in range(100):
            value = self.bounded_var.get_random_value()
            self.assertTrue(self.bounded_var.validate_value(value))
            self.assertGreaterEqual(value, -5.0)
            self.assertLessEqual(value, 10.0)

        # Unbounded variable should fail
        with self.assertRaises(ValueError):
            self.unbounded_var.get_random_value()

    def test_get_midpoint_value(self):
        """Test midpoint calculation."""
        midpoint = self.bounded_var.get_midpoint_value()
        self.assertEqual(midpoint, 2.5)  # (-5 + 10) / 2

        # Unbounded variable should fail
        with self.assertRaises(ValueError):
            self.unbounded_var.get_midpoint_value()

    def test_discretize(self):
        """Test discretization functionality."""
        # Test with different numbers of points
        points_5 = self.bounded_var.discretize(5)
        self.assertEqual(len(points_5), 5)
        self.assertEqual(points_5[0], -5.0)
        self.assertEqual(points_5[-1], 10.0)

        points_1 = self.bounded_var.discretize(1)
        self.assertEqual(len(points_1), 1)
        self.assertEqual(points_1[0], 2.5)  # midpoint

        # Invalid inputs
        with self.assertRaises(ValueError):
            self.bounded_var.discretize(0)

        with self.assertRaises(ValueError):
            self.bounded_var.discretize(-1)

        with self.assertRaises(ValueError):
            self.unbounded_var.discretize(5)

    def test_get_neighbor_value(self):
        """Test neighbor value generation."""
        current = 5.0
        step_size = 1.0

        for _ in range(100):
            neighbor = self.bounded_var.get_neighbor_value(current, step_size)
            self.assertTrue(self.bounded_var.validate_value(neighbor))
            self.assertGreaterEqual(neighbor, -5.0)
            self.assertLessEqual(neighbor, 10.0)

        # Invalid inputs
        with self.assertRaises(ValueError):
            self.bounded_var.get_neighbor_value(15.0, 1.0)  # Invalid current value

        with self.assertRaises(ValueError):
            self.bounded_var.get_neighbor_value(5.0, -1.0)  # Invalid step size

    def test_equality_and_hashing(self):
        """Test variable equality and hashing."""
        var1 = ContinuousVariable("test", bounds=(0.0, 10.0))
        var2 = ContinuousVariable("test", bounds=(0.0, 10.0))
        var3 = ContinuousVariable("different", bounds=(0.0, 10.0))

        self.assertEqual(var1, var2)
        self.assertNotEqual(var1, var3)
        self.assertEqual(hash(var1), hash(var2))
        self.assertNotEqual(hash(var1), hash(var3))


class TestIntegerVariable(unittest.TestCase):
    """Test IntegerVariable class with comprehensive validation."""

    def setUp(self):
        """Set up test fixtures."""
        self.default_var = IntegerVariable("x")
        self.custom_var = IntegerVariable("y", bounds=(-10, 50))

    def test_initialization_valid(self):
        """Test valid initialization cases."""
        # Default bounds
        var1 = IntegerVariable("test")
        self.assertEqual(var1.name, "test")
        self.assertEqual(var1._int_bounds, (0, 1000))

        # Custom bounds
        var2 = IntegerVariable("custom", bounds=(-5, 25))
        self.assertEqual(var2.name, "custom")
        self.assertEqual(var2._int_bounds, (-5, 25))

    def test_initialization_invalid(self):
        """Test NASA assert failures during initialization."""
        # Non-integer bounds
        with self.assertRaises(TypeError):
            IntegerVariable("test", bounds=(0.5, 10.5))

        with self.assertRaises(TypeError):
            IntegerVariable("test", bounds=(0, 10.0))

        # Invalid bounds order
        with self.assertRaises(ValueError):
            IntegerVariable("test", bounds=(10, 5))

    def test_validate_value(self):
        """Test integer value validation."""
        # Valid values
        self.assertTrue(self.custom_var.validate_value(0))
        self.assertTrue(self.custom_var.validate_value(25))
        self.assertTrue(self.custom_var.validate_value(-10))

        # Invalid types (even if mathematically equivalent)
        self.assertFalse(self.custom_var.validate_value(5.0))
        self.assertFalse(self.custom_var.validate_value(5.5))
        self.assertFalse(self.custom_var.validate_value("5"))

        # Out of bounds
        self.assertFalse(self.custom_var.validate_value(-11))
        self.assertFalse(self.custom_var.validate_value(51))

    def test_clip_to_bounds(self):
        """Test integer bounds clipping."""
        self.assertEqual(self.custom_var.clip_to_bounds(-15), -10)
        self.assertEqual(self.custom_var.clip_to_bounds(60), 50)
        self.assertEqual(self.custom_var.clip_to_bounds(25), 25)

        # Invalid type
        with self.assertRaises(TypeError):
            self.custom_var.clip_to_bounds(5.5)

    def test_to_dict_entry(self):
        """Test variable dictionary protocol for integers."""
        entry = self.custom_var.to_dict_entry(25)

        self.assertEqual(entry["value"], 25)
        self.assertEqual(entry["type"], "integer")
        self.assertEqual(entry["bounds"], (-10, 50))  # Original integer bounds
        self.assertIs(entry["variable_object"], self.custom_var)

        # Invalid value
        with self.assertRaises(TypeError):
            self.custom_var.to_dict_entry(25.5)

    def test_get_type_name(self):
        """Test type name identification."""
        self.assertEqual(self.default_var.get_type_name(), "integer")
        self.assertEqual(self.custom_var.get_type_name(), "integer")

    def test_get_random_value(self):
        """Test random integer generation."""
        for _ in range(100):
            value = self.custom_var.get_random_value()
            self.assertTrue(self.custom_var.validate_value(value))
            self.assertGreaterEqual(value, -10)
            self.assertLessEqual(value, 50)
            self.assertIsInstance(value, int)

    def test_get_midpoint_value(self):
        """Test integer midpoint calculation."""
        midpoint = self.custom_var.get_midpoint_value()
        self.assertEqual(midpoint, 20)  # (-10 + 50) // 2
        self.assertIsInstance(midpoint, int)

    def test_get_range_size(self):
        """Test range size calculation."""
        size = self.custom_var.get_range_size()
        self.assertEqual(size, 61)  # 50 - (-10) + 1

    def test_enumerate_all_values(self):
        """Test value enumeration."""
        small_var = IntegerVariable("small", bounds=(0, 5))
        values = small_var.enumerate_all_values()

        self.assertEqual(values, [0, 1, 2, 3, 4, 5])
        self.assertEqual(len(values), 6)

        # Large range should fail
        with self.assertRaises(ValueError):
            self.default_var.enumerate_all_values()  # Range too large

    def test_get_neighbor_value(self):
        """Test integer neighbor generation."""
        current = 25

        for _ in range(100):
            neighbor = self.custom_var.get_neighbor_value(current, max_step=5)
            self.assertTrue(self.custom_var.validate_value(neighbor))
            self.assertGreaterEqual(neighbor, -10)
            self.assertLessEqual(neighbor, 50)
            self.assertIsInstance(neighbor, int)

        # Invalid inputs
        with self.assertRaises(ValueError):
            self.custom_var.get_neighbor_value(60, 1)  # Invalid current

        with self.assertRaises(ValueError):
            self.custom_var.get_neighbor_value(25, 0)  # Invalid step


class TestBinaryVariable(unittest.TestCase):
    """Test BinaryVariable class with comprehensive validation."""

    def setUp(self):
        """Set up test fixtures."""
        self.binary_var = BinaryVariable("flag")

    def test_initialization(self):
        """Test binary variable initialization."""
        var = BinaryVariable("test")
        self.assertEqual(var.name, "test")
        self.assertEqual(var.bounds, (0, 1))

        # Invalid name
        with self.assertRaises(ValueError):
            BinaryVariable("")

    def test_validate_value(self):
        """Test binary value validation."""
        # Valid values
        self.assertTrue(self.binary_var.validate_value(0))
        self.assertTrue(self.binary_var.validate_value(1))

        # Invalid values
        self.assertFalse(self.binary_var.validate_value(0.0))  # Float not allowed
        self.assertFalse(self.binary_var.validate_value(1.0))  # Float not allowed
        self.assertFalse(self.binary_var.validate_value(2))
        self.assertFalse(self.binary_var.validate_value(-1))
        self.assertFalse(self.binary_var.validate_value("0"))
        self.assertFalse(self.binary_var.validate_value(True))

    def test_get_type_name(self):
        """Test type name identification."""
        self.assertEqual(self.binary_var.get_type_name(), "binary")

    def test_get_random_value(self):
        """Test random binary generation."""
        values = set()
        for _ in range(100):
            value = self.binary_var.get_random_value()
            self.assertTrue(self.binary_var.validate_value(value))
            values.add(value)

        # Should generate both 0 and 1 over many trials
        self.assertEqual(values, {0, 1})

    def test_flip_value(self):
        """Test binary value flipping."""
        self.assertEqual(self.binary_var.flip_value(0), 1)
        self.assertEqual(self.binary_var.flip_value(1), 0)

        # Invalid input
        with self.assertRaises(ValueError):
            self.binary_var.flip_value(2)

    def test_get_neighbor_value(self):
        """Test binary neighbor generation."""
        self.assertEqual(self.binary_var.get_neighbor_value(0), 1)
        self.assertEqual(self.binary_var.get_neighbor_value(1), 0)

    def test_get_all_values(self):
        """Test getting all possible binary values."""
        values = self.binary_var.get_all_values()
        self.assertEqual(values, [0, 1])

    def test_hamming_distance(self):
        """Test Hamming distance calculation."""
        self.assertEqual(self.binary_var.hamming_distance(0, 0), 0)
        self.assertEqual(self.binary_var.hamming_distance(1, 1), 0)
        self.assertEqual(self.binary_var.hamming_distance(0, 1), 1)
        self.assertEqual(self.binary_var.hamming_distance(1, 0), 1)

        # Invalid inputs
        with self.assertRaises(ValueError):
            self.binary_var.hamming_distance(2, 0)

    def test_boolean_conversion(self):
        """Test boolean conversion methods."""
        # to_boolean
        self.assertEqual(self.binary_var.to_boolean(0), False)
        self.assertEqual(self.binary_var.to_boolean(1), True)

        # from_boolean
        self.assertEqual(self.binary_var.from_boolean(False), 0)
        self.assertEqual(self.binary_var.from_boolean(True), 1)

        # Invalid inputs
        with self.assertRaises(ValueError):
            self.binary_var.to_boolean(2)

        with self.assertRaises(TypeError):
            self.binary_var.from_boolean("true")

    def test_clip_to_bounds(self):
        """Test binary clipping logic."""
        self.assertEqual(self.binary_var.clip_to_bounds(-5), 0)
        self.assertEqual(self.binary_var.clip_to_bounds(0), 0)
        self.assertEqual(self.binary_var.clip_to_bounds(0.1), 1)
        self.assertEqual(self.binary_var.clip_to_bounds(10), 1)

        # Invalid type
        with self.assertRaises(TypeError):
            self.binary_var.clip_to_bounds("invalid")

    def test_probability_methods(self):
        """Test probability-based methods."""
        # get_probability_from_continuous
        prob_neg = self.binary_var.get_probability_from_continuous(-2.0)
        prob_zero = self.binary_var.get_probability_from_continuous(0.0)
        prob_pos = self.binary_var.get_probability_from_continuous(2.0)

        self.assertLess(prob_neg, 0.5)
        self.assertAlmostEqual(prob_zero, 0.5, places=5)
        self.assertGreater(prob_pos, 0.5)

        # sample_from_probability
        samples_0 = [self.binary_var.sample_from_probability(0.0) for _ in range(100)]
        samples_1 = [self.binary_var.sample_from_probability(1.0) for _ in range(100)]

        self.assertTrue(all(s == 0 for s in samples_0))
        self.assertTrue(all(s == 1 for s in samples_1))

        # Invalid probability
        with self.assertRaises(ValueError):
            self.binary_var.sample_from_probability(-0.1)

        with self.assertRaises(ValueError):
            self.binary_var.sample_from_probability(1.1)


if __name__ == '__main__':
    unittest.main()
