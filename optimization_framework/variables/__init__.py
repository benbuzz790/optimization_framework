"""
Variable classes for optimization framework.

This module provides the complete variable system including base Variable class
and all specialized subclasses (Continuous, Integer, Binary) with full NASA-style
validation and variable dictionary protocol support.
"""

from .variable import Variable
from .continuous_variable import ContinuousVariable
from .integer_variable import IntegerVariable
from .binary_variable import BinaryVariable

__all__ = ['Variable', 'ContinuousVariable', 'IntegerVariable', 'BinaryVariable']