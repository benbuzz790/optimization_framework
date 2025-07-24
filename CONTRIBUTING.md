# Contributing to Optimization Framework

Welcome to the Optimization Framework project! We're excited that you're interested in contributing. This guide will help you get started and ensure that your contributions align with our project standards.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style Guidelines](#code-style-guidelines)
- [Testing Requirements](#testing-requirements)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)
- [Documentation Requirements](#documentation-requirements)
- [Community Guidelines](#community-guidelines)
- [Architecture Overview](#architecture-overview)

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- A text editor or IDE of your choice

### First Time Contributors

If you're new to open source or this project:

1. **Read the Architecture**: Start with `optimization_framework_architecture.py` to understand the system design
2. **Look for "good first issue" labels**: These are beginner-friendly issues
3. **Join our discussions**: Feel free to ask questions in issues or discussions
4. **Start small**: Consider documentation improvements or small bug fixes first

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/optimization-framework.git
cd optimization-framework

# Add the original repository as upstream
git remote add upstream https://github.com/ORIGINAL_OWNER/optimization-framework.git
```

### 2. Create Development Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install development dependencies
pip install -r requirements-dev.txt
```

### 3. Verify Setup

```bash
# Run tests to ensure everything works
python -m pytest tests/

# Run code quality checks
python -m flake8 optimization_framework/
python -m black --check optimization_framework/
```

## Code Style Guidelines

### General Python Style

We follow **PEP 8** with some specific additions:

- **Line length**: 88 characters (Black formatter default)
- **Imports**: Use absolute imports, group by standard library, third-party, local
- **Naming**: 
  - Classes: `PascalCase`
  - Functions/methods: `snake_case`
  - Constants: `UPPER_SNAKE_CASE`
  - Private methods: `_leading_underscore`

### NASA-Style Production Asserts

**Critical**: All code must implement NASA-style production asserts for input validation.

#### Assert Pattern
```python
from optimization_framework.utils.assertions import production_assert

def my_function(param1, param2):
    # Always validate inputs first
    production_assert(isinstance(param1, str), f"param1 must be string, got {type(param1)}")
    production_assert(len(param2) > 0, "param2 cannot be empty")
    production_assert(param2 >= 0, f"param2 must be non-negative, got {param2}")
    
    # Your implementation here
    return result
```

#### Assert Categories
1. **Type Validation**: Check parameter types
2. **Range Validation**: Check numeric bounds and constraints
3. **State Validation**: Check object state consistency
4. **Business Logic**: Check domain-specific rules

#### Assert Guidelines
- **Always assert first**: Validate all inputs before processing
- **Descriptive messages**: Include expected vs actual values
- **Fail fast**: Don't continue with invalid data
- **Use production_assert()**: Never use bare `assert` statements

### Variable Dictionary Protocol

All functions that work with optimization variables must follow the **Variable Dictionary Protocol**:

```python
def objective_function(variable_dict: Dict) -> float:
    """
    Example objective function following variable dict protocol.
    
    Args:
        variable_dict: Dictionary with format:
        {
            "var_name": {
                "value": <current_value>,
                "type": "continuous" | "integer" | "binary",
                "bounds": (min_val, max_val),
                "variable_object": <Variable instance>
            }
        }
    """
    # Always validate variable dict format
    production_assert(isinstance(variable_dict, dict), "variable_dict must be dict")
    production_assert(len(variable_dict) > 0, "variable_dict cannot be empty")
    
    for var_name, var_data in variable_dict.items():
        production_assert("value" in var_data, f"Missing 'value' for {var_name}")
        production_assert("variable_object" in var_data, f"Missing 'variable_object' for {var_name}")
        
        # Validate value against variable object
        var_obj = var_data["variable_object"]
        production_assert(var_obj.validate_value(var_data["value"]), 
                         f"Invalid value for {var_name}: {var_data['value']}")
    
    # Your function implementation
    return some_calculation(variable_dict)
```

### Code Formatting

We use **Black** for automatic code formatting:

```bash
# Format your code before committing
black optimization_framework/
black tests/

# Check formatting
black --check optimization_framework/
```

### Import Organization

```python
# Standard library imports
import os
import sys
from typing import Dict, List, Optional

# Third-party imports
import numpy as np  # (if we add it later)

# Local imports
from optimization_framework.variables import Variable
from optimization_framework.utils.assertions import production_assert
```

## Testing Requirements

### Test Coverage

- **Minimum 90% code coverage** for all new code
- **100% coverage** for critical paths (optimization algorithms, validation)
- All NASA asserts must be tested for both success and failure cases

### Test Structure

```python
import pytest
from optimization_framework.variables import ContinuousVariable
from optimization_framework.utils.assertions import production_assert

class TestContinuousVariable:
    """Test suite for ContinuousVariable class."""
    
    def test_valid_initialization(self):
        """Test valid variable initialization."""
        var = ContinuousVariable("x", bounds=(0.0, 10.0))
        assert var.name == "x"
        assert var.bounds == (0.0, 10.0)
    
    def test_invalid_bounds_assertion(self):
        """Test that invalid bounds trigger production assert."""
        with pytest.raises(AssertionError, match="Invalid bounds"):
            ContinuousVariable("x", bounds=(10.0, 0.0))
    
    def test_value_validation(self):
        """Test value validation logic."""
        var = ContinuousVariable("x", bounds=(0.0, 10.0))
        
        # Valid values
        assert var.validate_value(5.0) is True
        assert var.validate_value(0.0) is True
        assert var.validate_value(10.0) is True
        
        # Invalid values
        assert var.validate_value(-1.0) is False
        assert var.validate_value(11.0) is False
        assert var.validate_value("invalid") is False
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test with infinite bounds
        var = ContinuousVariable("x")
        assert var.validate_value(float('inf')) is True
        assert var.validate_value(-float('inf')) is True
        
        # Test with very small bounds
        var = ContinuousVariable("x", bounds=(1e-10, 1e-9))
        assert var.validate_value(5e-10) is True
```

### Test Categories

1. **Unit Tests**: Test individual methods and classes
2. **Integration Tests**: Test component interactions
3. **End-to-End Tests**: Test complete optimization workflows
4. **Performance Tests**: Test algorithm performance and memory usage
5. **Assert Tests**: Test all NASA-style assertions

### Running Tests

```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=optimization_framework --cov-report=html

# Run specific test file
python -m pytest tests/test_variables.py

# Run tests with verbose output
python -m pytest -v

# Run only failed tests
python -m pytest --lf
```

## Pull Request Process

### Before Submitting

1. **Create feature branch**: `git checkout -b feature/your-feature-name`
2. **Write tests**: Ensure new code has comprehensive tests
3. **Run full test suite**: `python -m pytest`
4. **Check code quality**: `black . && flake8`
5. **Update documentation**: Add/update docstrings and README if needed
6. **Test edge cases**: Verify your code handles invalid inputs gracefully

### Pull Request Template

When creating a PR, please include:

```markdown
## Description
Brief description of changes and motivation.

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] I have tested edge cases and error conditions

## NASA-Style Asserts
- [ ] All new functions include appropriate production_assert statements
- [ ] Input validation is comprehensive and descriptive
- [ ] Assert failure cases are tested

## Variable Dictionary Protocol
- [ ] All functions accepting variables follow the variable dict protocol
- [ ] Variable dict validation is implemented where required
- [ ] Protocol compliance is tested

## Checklist
- [ ] My code follows the project's style guidelines
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] Any dependent changes have been merged and published
```

### Review Process

1. **Automated checks**: All CI checks must pass
2. **Code review**: At least one maintainer review required
3. **Testing**: Reviewer will test functionality
4. **Documentation**: Ensure documentation is updated
5. **Merge**: Squash and merge after approval

## Issue Reporting

### Bug Reports

Use the bug report template and include:

- **Environment**: Python version, OS, relevant package versions
- **Steps to reproduce**: Minimal code example
- **Expected behavior**: What should happen
- **Actual behavior**: What actually happens
- **Error messages**: Full stack traces
- **NASA Assert failures**: Include full assert message if applicable

### Feature Requests

- **Use case**: Describe the problem you're trying to solve
- **Proposed solution**: How you think it should work
- **Alternatives**: Other approaches you've considered
- **Implementation**: Are you willing to implement it?

### Performance Issues

- **Benchmark**: Include timing information
- **Test case**: Provide reproducible example
- **Environment**: Hardware specs, Python version
- **Profiling**: Include profiling data if available

## Documentation Requirements

### Docstring Standards

We use **Google-style docstrings**:

```python
def solve_optimization_problem(problem: Problem, solver_config: Dict) -> Solution:
    """
    Solve an optimization problem using specified algorithm.
    
    This function provides a high-level interface for solving optimization
    problems with comprehensive validation and error handling.
    
    Args:
        problem: Problem instance containing objective, constraints, and variables.
            Must be fully initialized with valid objective function and variables.
        solver_config: Configuration dictionary for the optimization algorithm.
            Required keys depend on the solver type. See solver documentation
            for specific requirements.
    
    Returns:
        Solution: Complete optimization solution with history and statistics.
            Includes best solution found, convergence information, and full
            optimization trajectory.
    
    Raises:
        AssertionError: If inputs fail NASA-style validation checks.
        ValueError: If problem is infeasible or solver configuration is invalid.
        RuntimeError: If optimization algorithm encounters unrecoverable error.
    
    Example:
        >>> from optimization_framework import *
        >>> 
        >>> # Define variables
        >>> x = ContinuousVariable("x", bounds=(0, 10))
        >>> y = ContinuousVariable("y", bounds=(0, 10))
        >>> 
        >>> # Define objective
        >>> def objective(var_dict):
        >>>     return var_dict["x"]["value"]**2 + var_dict["y"]["value"]**2
        >>> 
        >>> obj_func = ObjectiveFunction(objective)
        >>> problem = Problem(obj_func, variables=[x, y])
        >>> 
        >>> # Solve
        >>> config = {"max_iterations": 1000, "tolerance": 1e-6}
        >>> solution = solve_optimization_problem(problem, config)
        >>> print(f"Best objective: {solution.get_best_solution()['objective_value']}")
    
    Note:
        This function implements NASA-style production asserts for all inputs.
        All validation failures will provide detailed error messages including
        expected vs actual values and suggestions for correction.
    """
```

### README Updates

When adding new features:

1. **Update feature list**: Add to main features section
2. **Add examples**: Include usage examples
3. **Update installation**: If new dependencies are added
4. **Performance notes**: Include performance characteristics

### API Documentation

- **Keep current**: Update docs with code changes
- **Include examples**: Every public method needs examples
- **Error handling**: Document all possible exceptions
- **Performance**: Include time/space complexity where relevant

## Community Guidelines

### Code of Conduct

We are committed to providing a welcoming and inclusive environment:

- **Be respectful**: Treat all contributors with respect
- **Be constructive**: Provide helpful feedback and suggestions
- **Be patient**: Remember that everyone has different experience levels
- **Be collaborative**: Work together to improve the project

### Communication

- **Issues**: Use GitHub issues for bug reports and feature requests
- **Discussions**: Use GitHub discussions for questions and general discussion
- **Code review**: Be thorough but kind in code reviews
- **Documentation**: Help improve documentation for all users

### Recognition

We value all contributions:

- **Contributors file**: All contributors are listed
- **Release notes**: Significant contributions are highlighted
- **Mentorship**: Experienced contributors help newcomers
- **Learning**: We encourage learning and skill development

## Architecture Overview

### Core Principles

1. **Variable Dictionary Protocol**: Ensures type safety and consistency
2. **NASA-Style Asserts**: Production-quality input validation
3. **Modular Design**: Components can be developed and tested independently
4. **Full History Tracking**: Complete optimization process recording
5. **Algorithm Agnostic**: Easy to add new optimization algorithms

### Component Structure

```
optimization_framework/
├── variables/          # Variable types (continuous, integer, binary)
├── functions/          # Objective and constraint function wrappers
├── problems/           # Problem definition and validation
├── solvers/            # Optimization algorithms
├── solutions/          # Solution tracking and analysis
└── utils/              # Shared utilities (assertions, validation)
```

### Key Interfaces

- **Variable**: Base class for all optimization variables
- **ObjectiveFunction**: Wrapper ensuring variable dict protocol
- **ConstraintFunction**: Constraint handling with violation tracking
- **Problem**: Complete problem specification
- **BaseSolver**: Abstract base for all optimization algorithms
- **Solution**: Complete optimization results with history

## Getting Help

### Resources

- **Architecture Document**: `optimization_framework_architecture.py`
- **Examples**: `examples/` directory
- **Tests**: `tests/` directory for usage examples
- **Issues**: GitHub issues for questions and problems

### Contact

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Code Review**: Tag maintainers in PRs for review

---

Thank you for contributing to the Optimization Framework! Your contributions help make this project better for everyone. 🚀
