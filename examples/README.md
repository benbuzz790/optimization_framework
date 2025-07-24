# Optimization Framework Examples

Welcome to the comprehensive examples collection for the custom optimization framework! This folder contains carefully crafted examples that demonstrate all aspects of the framework, from basic usage to advanced optimization scenarios.

## 🚀 Quick Start

If you're new to the framework, start with these examples in order:

1. **`basic_optimization.py`** - Learn the fundamental concepts with a simple quadratic optimization
2. **`constrained_optimization.py`** - Understand constraint handling with a production planning problem  
3. **`portfolio_optimization.py`** - See real-world application with financial portfolio optimization

## 📚 Learning Progression

### 🟢 **Current Examples (Available Now)**

#### `basic_optimization.py`
- **Level**: Beginner
- **Concepts**: Variable definition, objective functions, problem setup, solution analysis
- **Problem**: Minimize quadratic function with linear constraint
- **Variables**: Continuous and integer variables with bounds
- **Best for**: First-time users learning framework basics

#### `constrained_optimization.py` 
- **Level**: Intermediate
- **Concepts**: Multiple constraint types, mixed variables, constraint violation analysis
- **Problem**: Production planning with resource constraints
- **Variables**: Continuous production, integer batches, binary decisions
- **Best for**: Understanding complex constraint handling

#### `portfolio_optimization.py`
- **Level**: Advanced  
- **Concepts**: Real-world modeling, correlation matrices, risk-return tradeoffs
- **Problem**: Financial portfolio optimization with diversification requirements
- **Variables**: Continuous weights, binary selections, integer lot sizes
- **Best for**: Practical application to finance and risk management

## 🔧 How to Run Examples

### Prerequisites
- Python 3.7+
- Optimization framework installed (see main README.md)
- No additional dependencies required

### Running Examples
```bash
# From the project root directory
cd examples

# Run basic example
python basic_optimization.py

# Run constrained example  
python constrained_optimization.py

# Run portfolio example
python portfolio_optimization.py
```

## 🎯 When to Use Each Optimizer

### Greedy Search Solver
- **Best for**: Simple problems, local optimization, quick solutions
- **Problem types**: Continuous variables, smooth objectives
- **Configuration**: Adjust `step_size` and `max_iterations`
- **Example usage**: See `basic_optimization.py`

### Genetic Algorithm Solver  
- **Best for**: Complex landscapes, discrete variables, global optimization
- **Problem types**: Mixed variables, combinatorial problems, multi-modal functions
- **Configuration**: Tune `population_size`, `generations`, `mutation_rate`
- **Example usage**: See `constrained_optimization.py` and `portfolio_optimization.py`

### Simulated Annealing Solver
- **Best for**: Escaping local optima, noisy objectives, exploration-heavy problems
- **Problem types**: Discrete optimization, complex constraints, large search spaces
- **Configuration**: Set `initial_temperature`, `cooling_rate`, `max_iterations`
- **Example usage**: See `constrained_optimization.py`

## 🛠️ Troubleshooting

### Common Issues

**Import Errors**
```python
# If you get import errors, ensure the framework is installed:
pip install -e .  # From project root
```

**Constraint Violations**
```python
# Check constraint setup in your problem:
problem = Problem(objective, constraints=[constraint], variables=[x, y])
solution = solver.solve(problem)
print(solution.get_best_solution()['constraint_violations'])
```

**Poor Convergence**
```python
# Try different algorithm configurations:
# For Genetic Algorithm - increase population or generations
solver = GeneticAlgorithmSolver({'population_size': 50, 'generations': 100})

# For Simulated Annealing - adjust temperature schedule  
solver = SimulatedAnnealingSolver({'initial_temperature': 100, 'cooling_rate': 0.95})
```

**Variable Bounds Issues**
```python
# Ensure bounds are properly defined:
x = ContinuousVariable("x", bounds=(-10, 10))  # Both bounds required
y = IntegerVariable("y", bounds=(0, 100))      # Integer bounds
z = BinaryVariable("z")                        # No bounds needed
```

## ⚡ Performance Tips

### For Large Problems
- Use Genetic Algorithm with smaller populations initially
- Implement early stopping based on convergence
- Consider constraint penalty methods for heavily constrained problems

### For Real-Time Applications  
- Start with Greedy Search for quick approximate solutions
- Use warm starts with good initial guesses
- Limit iterations based on time constraints

### Memory Management
- Monitor solution history size for long optimizations
- Use solution tracking selectively for large problems
- Clear intermediate results when not needed

## 🔍 Example Details

| Example | Variables | Constraints | Algorithms | Difficulty |
|---------|-----------|-------------|------------|------------|
| `basic_optimization.py` | 2 (continuous, integer) | 1 linear | All three | Beginner |
| `constrained_optimization.py` | 6 mixed types | 8 various types | GA, SA | Intermediate |
| `portfolio_optimization.py` | 30+ mixed | 10+ complex | GA primary | Advanced |

## 📖 Further Reading

- **Main Documentation**: See project README.md for installation and API reference
- **Architecture Guide**: `optimization_framework_architecture.py` for system design
- **Contributing**: `CONTRIBUTING.md` for adding new examples
- **Algorithm Details**: Individual solver documentation in framework modules

## 🤝 Contributing New Examples

We welcome new examples! Please follow these guidelines:

1. **Clear problem statement** with real-world context
2. **Step-by-step comments** explaining each framework concept
3. **Multiple solution approaches** when applicable  
4. **Comprehensive output analysis** showing result interpretation
5. **Error handling** with helpful debugging information

See `CONTRIBUTING.md` for detailed contribution guidelines.

## 📞 Support

- **Issues**: Report bugs or request examples via GitHub issues
- **Questions**: Use GitHub discussions for usage questions
- **Documentation**: Refer to main project documentation for API details

---

**Happy Optimizing!** 🎯

These examples demonstrate the power and flexibility of the optimization framework. Start with the basics and work your way up to complex real-world problems. The framework is designed to grow with your optimization needs!
