"""
Problem class for optimization framework.

This module provides the Problem class that bundles objective functions,
constraint functions, and variables into a complete optimization problem
specification with validation and evaluation methods.
"""
from typing import Dict, List, Any, Optional
from ..utils.assertions import production_assert
from ..variables import Variable
from ..functions import ObjectiveFunction, ConstraintFunction

class Problem:
    """
    Optimization problem definition with objective and constraints.

    Bundles objective function, constraint functions, and variables into
    a complete problem specification with validation and evaluation methods.
    """

    def __init__(self, objective: ObjectiveFunction, 
                 constraints: List[ConstraintFunction] = None,
                 variables: List[Variable] = None):
        """
        Initialize optimization problem.

        Args:
            objective: ObjectiveFunction instance
            constraints: List of ConstraintFunction instances
            variables: List of Variable instances

        NASA Assert: objective is ObjectiveFunction, constraints/variables are proper lists
        """
        production_assert(isinstance(objective, ObjectiveFunction), 
                         f"Objective must be ObjectiveFunction, got {type(objective).__name__}")

        if constraints is not None:
            production_assert(isinstance(constraints, list), 
                             f"Constraints must be list, got {type(constraints)}")
            for i, constraint in enumerate(constraints):
                production_assert(isinstance(constraint, ConstraintFunction), 
                                 f"Constraint {i} must be ConstraintFunction, got {type(constraint)}")

        if variables is not None:
            production_assert(isinstance(variables, list), 
                             f"Variables must be list, got {type(variables)}")
            for i, variable in enumerate(variables):
                production_assert(isinstance(variable, Variable), 
                                 f"Variable {i} must be Variable, got {type(variable)}")

            # Check for duplicate variable names
            var_names = [var.name for var in variables]
            production_assert(len(var_names) == len(set(var_names)), 
                             f"Duplicate variable names found: {var_names}")

        self.objective = objective
        self.constraints = constraints or []
        self.variables = variables or []

    def create_variable_dict(self, values: Dict[str, Any]) -> Dict:
        """
        Create properly formatted variable dictionary.

        Args:
            values: Dictionary mapping variable names to values

        Returns:
            Dict: Properly formatted variable dictionary

        NASA Assert: all variable names exist, all values validate
        """
        production_assert(isinstance(values, dict), f"Values must be dict, got {type(values)}")

        var_dict = {}
        var_lookup = {var.name: var for var in self.variables}

        for var_name, value in values.items():
            production_assert(var_name in var_lookup, 
                             f"Unknown variable '{var_name}'. Available: {list(var_lookup.keys())}")

            variable = var_lookup[var_name]
            production_assert(variable.validate_value(value), 
                             f"Invalid value for variable '{var_name}': {value}")

            var_dict[var_name] = variable.to_dict_entry(value)

        # Ensure all variables are present
        for variable in self.variables:
            production_assert(variable.name in var_dict, 
                             f"Missing value for variable '{variable.name}'")

        return var_dict

    def evaluate_objective(self, variable_dict: Dict) -> float:
        """Evaluate objective function."""
        return self.objective.evaluate(variable_dict)

    def evaluate_constraints(self, variable_dict: Dict) -> List[float]:
        """Evaluate all constraint functions."""
        return [constraint.evaluate(variable_dict) for constraint in self.constraints]

    def is_feasible(self, variable_dict: Dict) -> bool:
        """Check if solution satisfies all constraints."""
        return all(constraint.is_satisfied(variable_dict) for constraint in self.constraints)

    def get_constraint_violations(self, variable_dict: Dict) -> Dict[str, float]:
        """Get detailed constraint violation information."""
        violations = {}
        for constraint in self.constraints:
            violation = constraint.violation_amount(variable_dict)
            violations[constraint.name] = violation
        return violations
from typing import Dict, List, Any, Optional
from ..utils.assertions import production_assert
from ..variables import Variable
from ..functions import ObjectiveFunction, ConstraintFunction