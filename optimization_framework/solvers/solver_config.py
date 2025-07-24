"""
Solver Configuration Management

Provides centralized configuration handling, validation, and default
value management for all optimization algorithms.
"""

from typing import Dict, Any, Optional, Union
from ..utils.assertions import production_assert


class SolverConfig:
    """
    Centralized solver configuration management with validation.

    Provides standardized configuration handling, validation, and default
    value management for all optimization algorithms.
    """

    def __init__(self, config_dict: Dict = None):
        """
        Initialize solver configuration.

        Args:
            config_dict: Dictionary of configuration parameters

        NASA Assert: config_dict is dict if provided
        """
        if config_dict is not None:
            production_assert(isinstance(config_dict, dict), 
                             f"Config must be dict, got {type(config_dict)}")

        self._config = config_dict.copy() if config_dict else {}
        self._validated_keys = set()

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value with optional default.

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        production_assert(isinstance(key, str), f"Config key must be string, got {type(key)}")
        return self._config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value.

        Args:
            key: Configuration key
            value: Configuration value
        """
        production_assert(isinstance(key, str), f"Config key must be string, got {type(key)}")
        self._config[key] = value

    def validate_numeric_range(self, key: str, min_val: float = None, 
                              max_val: float = None, required: bool = True) -> float:
        """
        Validate numeric configuration parameter within range.

        Args:
            key: Configuration key
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            required: Whether parameter is required

        Returns:
            Validated numeric value

        NASA Assert: Value is numeric and within range
        """
        if key not in self._config:
            production_assert(not required, f"Required config parameter '{key}' not found")
            return None

        value = self._config[key]
        production_assert(isinstance(value, (int, float)), 
                         f"Config '{key}' must be numeric, got {type(value)}")

        if min_val is not None:
            production_assert(value >= min_val, 
                             f"Config '{key}' must be >= {min_val}, got {value}")

        if max_val is not None:
            production_assert(value <= max_val, 
                             f"Config '{key}' must be <= {max_val}, got {value}")

        self._validated_keys.add(key)
        return float(value)

    def validate_integer_range(self, key: str, min_val: int = None, 
                              max_val: int = None, required: bool = True) -> int:
        """
        Validate integer configuration parameter within range.

        Args:
            key: Configuration key
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            required: Whether parameter is required

        Returns:
            Validated integer value

        NASA Assert: Value is integer and within range
        """
        if key not in self._config:
            production_assert(not required, f"Required config parameter '{key}' not found")
            return None

        value = self._config[key]
        production_assert(isinstance(value, int), 
                         f"Config '{key}' must be integer, got {type(value)}")

        if min_val is not None:
            production_assert(value >= min_val, 
                             f"Config '{key}' must be >= {min_val}, got {value}")

        if max_val is not None:
            production_assert(value <= max_val, 
                             f"Config '{key}' must be <= {max_val}, got {value}")

        self._validated_keys.add(key)
        return value

    def validate_probability(self, key: str, required: bool = True) -> float:
        """
        Validate probability configuration parameter (0 <= value <= 1).

        Args:
            key: Configuration key
            required: Whether parameter is required

        Returns:
            Validated probability value
        """
        return self.validate_numeric_range(key, min_val=0.0, max_val=1.0, required=required)

    def validate_positive(self, key: str, required: bool = True) -> float:
        """
        Validate positive numeric configuration parameter.

        Args:
            key: Configuration key
            required: Whether parameter is required

        Returns:
            Validated positive value
        """
        value = self.validate_numeric_range(key, min_val=0.0, required=required)
        if value is not None:
            production_assert(value > 0, f"Config '{key}' must be positive, got {value}")
        return value

    def validate_choice(self, key: str, choices: list, required: bool = True) -> Any:
        """
        Validate configuration parameter is one of allowed choices.

        Args:
            key: Configuration key
            choices: List of allowed values
            required: Whether parameter is required

        Returns:
            Validated choice value
        """
        if key not in self._config:
            production_assert(not required, f"Required config parameter '{key}' not found")
            return None

        value = self._config[key]
        production_assert(value in choices, 
                         f"Config '{key}' must be one of {choices}, got {value}")

        self._validated_keys.add(key)
        return value

    def get_unvalidated_keys(self) -> set:
        """
        Get set of configuration keys that haven't been validated.

        Returns:
            Set of unvalidated keys
        """
        return set(self._config.keys()) - self._validated_keys

    def to_dict(self) -> Dict:
        """
        Get configuration as dictionary.

        Returns:
            Configuration dictionary
        """
        return self._config.copy()