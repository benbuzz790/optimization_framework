"""
Function Framework - Objective and Constraint Function Wrappers

This module provides wrapper classes for objective and constraint functions
that enforce the variable dictionary protocol and provide comprehensive validation.

Classes:
    ObjectiveFunction: Wrapper for objective functions with validation
    ConstraintFunction: Wrapper for constraint functions with violation tracking
"""

from .objective_function import ObjectiveFunction
from .constraint_function import ConstraintFunction

__all__ = ['ObjectiveFunction', 'ConstraintFunction']