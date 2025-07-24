"""
Base Solver Implementation

Abstract base class for optimization algorithms with common functionality
and interface standardization.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional
from ..utils.assertions import production_assert
from ..variables import BinaryVariable, IntegerVariable, ContinuousVariable


class BaseSolver(ABC):
    """
    Abstract base class for optimization algorithms.

    Defines common interface and validation patterns for all optimization
    algorithms. Subclasses implement specific optimization strategies.
    """

    def __init__(self, config: Dict = None):
        """
        Initialize solver with configuration.

        Args:
            config: Algorithm-specific configuration dictionary

        NASA Assert: config is dict if provided
        """
        if config is not None:
            production_assert(isinstance(config, dict), f"Config must be dict, got {type(config)}")

        self.config = self._validate_config(config or {})

    @abstractmethod
    def solve(self, problem, initial_guess: Dict[str, Any] = None):
        """
        Solve optimization problem.

        Args:
            problem: Problem instance to solve
            initial_guess: Optional initial variable values

        Returns:
            Solution: Complete optimization solution with history

        NASA Assert: problem is Problem instance, initial_guess matches problem variables
        """
        # Import here to avoid circular imports
        from ..problems import Problem

        production_assert(isinstance(problem, Problem), 
                         f"Problem must be Problem instance, got {type(problem)}")

        if initial_guess is not None:
            production_assert(isinstance(initial_guess, dict), 
                             f"Initial guess must be dict, got {type(initial_guess)}")
            # Validate initial guess creates valid variable dict
            problem.create_variable_dict(initial_guess)

    @abstractmethod
    def _validate_config(self, config: Dict) -> Dict:
        """
        Validate solver-specific configuration.

        Args:
            config: Configuration dictionary to validate

        Returns:
            Dict: Validated configuration with defaults filled in
        """
        pass

    def _create_initial_solution(self, problem, initial_guess: Dict[str, Any] = None) -> Dict:
        """
        Create initial variable dictionary.

        Args:
            problem: Problem instance
            initial_guess: Optional initial values

        Returns:
            Dict: Valid variable dictionary for starting optimization

        NASA Assert: generated values are valid for all variables
        """
        if initial_guess is not None:
            return problem.create_variable_dict(initial_guess)

        # Generate default initial values
        initial_values = {}
        for variable in problem.variables:
            if isinstance(variable, BinaryVariable):
                initial_values[variable.name] = 0
            elif isinstance(variable, IntegerVariable):
                if variable.bounds:
                    initial_values[variable.name] = variable.bounds[0]
                else:
                    initial_values[variable.name] = 0
            else:  # ContinuousVariable
                if variable.bounds and variable.bounds[0] != -float('inf'):
                    initial_values[variable.name] = variable.bounds[0]
                else:
                    initial_values[variable.name] = 0.0

        return problem.create_variable_dict(initial_values)