"""
Test Suite for Optimization Framework

This package contains comprehensive tests for all framework components:
- test_variables.py: Variable classes and validation
- test_functions.py: ObjectiveFunction and ConstraintFunction
- test_problem.py: Problem class and integration
- test_solution.py: Solution tracking and analysis
- test_solvers.py: All optimization algorithms

Run all tests with: python -m pytest tests/
Or run individual test files: python -m unittest tests.test_variables
"""

# Import all test modules for easy access
from . import test_variables
from . import test_functions
from . import test_problem
from . import test_solution
from . import test_solvers

__all__ = [
    'test_variables',
    'test_functions', 
    'test_problem',
    'test_solution',
    'test_solvers'
]