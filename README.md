# Optimization Framework

[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)](https://github.com/optimization-framework/optimization-framework)
[![ARM64 Compatible](https://img.shields.io/badge/ARM64-compatible-green.svg)](https://github.com/optimization-framework/optimization-framework)

A comprehensive Object-Oriented Programming (OOP) optimization framework designed specifically for environments where scipy is unavailable, such as ARM64 Windows systems. This framework provides a clean, extensible interface for defining and solving optimization problems with multiple variable types and sophisticated constraint handling.

## 🚀 Key Features

- **Multiple Variable Types**: Continuous, integer, and binary variables with flexible bounds
- **Flexible Constraint Handling**: Support for ≤, ≥, and = constraints with violation tracking
- **Multiple Optimization Algorithms**: 
  - Greedy Search (local optimization)
  - Genetic Algorithm (global optimization)
  - Simulated Annealing (global optimization with cooling)
- **NASA-Style Production Asserts**: Comprehensive input validation throughout the codebase
- **Complete Solution Tracking**: Full optimization history with convergence analysis
- **Clean OOP Architecture**: Modular, extensible design following SOLID principles
- **ARM64 Windows Compatible**: No scipy dependency, works on all architectures

## 📦 Installation

### From PyPI (Recommended)
```bash
pip install optimization-framework
```

### From Source
```bash
git clone https://github.com/optimization-framework/optimization-framework.git
cd optimization-framework
pip install -e .
```

### Development Installation
```bash
git clone https://github.com/optimization-framework/optimization-framework.git
cd optimization-framework
pip install -e ".[dev]"
```

## 🏃‍♂️ Quick Start

### Basic Usage

```python
from optimization_framework import *

# Define variables
x = ContinuousVariable("x", bounds=(-10, 10))
y = ContinuousVariable("y", bounds=(-10, 10))

# Define objective function (minimize x² + y²)
def objective_func(var_dict):
    x_val = var_dict["x"]["value"]
    y_val = var_dict["y"]["value"]
    return x_val**2 + y_val**2

objective = ObjectiveFunction(objective_func, "minimize_sum_squares")

# Create and solve problem
problem = Problem(objective, variables=[x, y])
solver = GeneticAlgorithmSolver()
solution = solver.solve(problem)

print(f"Best solution: {solution.get_best_solution()}")
print(f"Optimization summary: {solution.get_summary_statistics()}")
```

### Advanced Example with Constraints

```python
from optimization_framework import *

# Define variables
x = ContinuousVariable("x", bounds=(0, 5))
y = ContinuousVariable("y", bounds=(0, 5))
z = IntegerVariable("z", bounds=(0, 10))

# Define objective function
def objective_func(var_dict):
    x_val = var_dict["x"]["value"]
    y_val = var_dict["y"]["value"] 
    z_val = var_dict["z"]["value"]
    return -(x_val * y_val + z_val)  # Maximize x*y + z

objective = ObjectiveFunction(objective_func, "maximize_profit")

# Define constraints
def constraint1(var_dict):
    x_val = var_dict["x"]["value"]
    y_val = var_dict["y"]["value"]
    return x_val + y_val  # x + y <= 4

def constraint2(var_dict):
    x_val = var_dict["x"]["value"]
    z_val = var_dict["z"]["value"]
    return x_val * z_val  # x * z >= 2

constraints = [
    ConstraintFunction(constraint1, "<=", 4.0, "resource_limit"),
    ConstraintFunction(constraint2, ">=", 2.0, "minimum_production")
]

# Create and solve problem
problem = Problem(objective, constraints, variables=[x, y, z])

# Try different solvers
solvers = [
    GreedySearchSolver({"max_iterations": 1000}),
    GeneticAlgorithmSolver({"population_size": 50, "generations": 100}),
    SimulatedAnnealingSolver({"initial_temp": 100, "cooling_rate": 0.95})
]

for solver in solvers:
    solution = solver.solve(problem)
    print(f"{solver.__class__.__name__}: {solution.get_summary_statistics()}")
```

## 🏗️ Architecture Overview

### Core Components

```
OptimizationFramework/
├── variables/          # Variable type definitions
│   ├── Variable        # Base variable class
│   ├── ContinuousVariable
│   ├── IntegerVariable
│   └── BinaryVariable
├── functions/          # Function wrappers
│   ├── ObjectiveFunction
│   └── ConstraintFunction
├── problems/           # Problem definition
│   └── Problem
├── solvers/           # Optimization algorithms
│   ├── BaseSolver
│   ├── GreedySearchSolver
│   ├── GeneticAlgorithmSolver
│   └── SimulatedAnnealingSolver
├── solutions/         # Solution tracking
│   └── Solution
└── utils/            # Utilities
    └── assertions.py
```

### Variable Dictionary Protocol

All functions in the framework use a consistent variable dictionary format:

```python
variable_dict = {
    "var_name": {
        "value": <current_value>,
        "type": "continuous" | "integer" | "binary",
        "bounds": (min_val, max_val),
        "variable_object": <Variable instance>
    }
}
```

This ensures type safety and consistent validation across all components.

## 🔧 Solver Configuration

### Greedy Search Solver
```python
config = {
    "max_iterations": 1000,
    "step_size": 0.1,
    "tolerance": 1e-6
}
solver = GreedySearchSolver(config)
```

### Genetic Algorithm Solver
```python
config = {
    "population_size": 100,
    "generations": 200,
    "mutation_rate": 0.1,
    "crossover_rate": 0.8
}
solver = GeneticAlgorithmSolver(config)
```

### Simulated Annealing Solver
```python
config = {
    "initial_temp": 1000,
    "final_temp": 0.1,
    "cooling_rate": 0.95,
    "max_iterations": 10000
}
solver = SimulatedAnnealingSolver(config)
```

## 📊 Solution Analysis

The framework provides comprehensive solution analysis:

```python
solution = solver.solve(problem)

# Get best solution
best = solution.get_best_solution()
print(f"Best objective value: {best['objective_value']}")
print(f"Is feasible: {best['is_feasible']}")

# Analyze convergence
convergence = solution.get_convergence_data()
print(f"Total iterations: {convergence['total_iterations']}")
print(f"Improvement: {convergence['improvement']}")

# Get complete history
history = solution.get_optimization_history()
for iteration in history[-5:]:  # Last 5 iterations
    print(f"Iter {iteration['iteration']}: {iteration['objective_value']}")

# Summary statistics
summary = solution.get_summary_statistics()
print(f"Solver: {summary['solver']}")
print(f"Converged: {summary['converged']}")
```

## 🛡️ NASA-Style Validation

The framework implements comprehensive production-level validation:

```python
from optimization_framework.utils.assertions import production_assert

# All inputs are validated
production_assert(isinstance(bounds, tuple), "Bounds must be tuple")
production_assert(len(bounds) == 2, "Bounds must have 2 elements")
production_assert(bounds[0] <= bounds[1], f"Invalid bounds: {bounds}")

# Type safety throughout
production_assert(isinstance(value, (int, float)), "Value must be numeric")
production_assert(variable.validate_value(value), f"Invalid value: {value}")
```

## 🧪 Testing

Run the test suite:

```bash
# Install test dependencies
pip install -e ".[test]"

# Run tests
pytest

# Run with coverage
pytest --cov=optimization_framework --cov-report=html
```

## 📈 Performance Considerations

- **Memory Management**: Solution history with optional memory limits
- **Efficient Updates**: Optimized variable dictionary operations
- **Vectorization Ready**: Constraint evaluation supports batch processing
- **Caching**: Variable validation results cached where beneficial

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
git clone https://github.com/optimization-framework/optimization-framework.git
cd optimization-framework
pip install -e ".[dev]"
pre-commit install
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Designed for ARM64 Windows compatibility where scipy is unavailable
- Inspired by production-grade optimization frameworks
- Built with NASA-style validation principles
- Anthropic's Claude via 'bots'

## 📚 Documentation

- [API Reference](https://optimization-framework.readthedocs.io/en/latest/api/)
- [User Guide](https://optimization-framework.readthedocs.io/en/latest/guide/)
- [Examples](https://optimization-framework.readthedocs.io/en/latest/examples/)

---

**Made with ❤️ for the optimization community**
