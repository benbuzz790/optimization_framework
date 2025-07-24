"""
Optimization Framework Examples README

Welcome to the comprehensive examples collection for the custom optimization framework! 
This folder contains carefully crafted examples that demonstrate all aspects of the 
framework, from basic usage to advanced optimization scenarios.

TABLE OF CONTENTS:
1. Quick Start
2. Learning Progression  
3. Example Categories
4. Prerequisites
5. How to Run Examples
6. Example Descriptions
7. When to Use Each Optimizer
8. Troubleshooting
9. Performance Tips
10. Further Reading

QUICK START:
If you're new to the framework, start with these three examples in order:

1. 01_basic_continuous.py - Learn the fundamental concepts
2. 02_mixed_variables.py - Understand different variable types
3. 03_with_constraints.py - Add constraints to your problems

Then explore the algorithm-specific examples to understand when to use each optimizer.

LEARNING PROGRESSION:

🟢 Beginner Level (Start here if you're new to optimization or this framework):
- 01_basic_continuous.py - Framework basics with simple quadratic function
- 02_mixed_variables.py - Working with different variable types
- 03_with_constraints.py - Adding constraints to problems
- 04_greedy_search_intro.py - Your first optimization algorithm

🟡 Intermediate Level (Once comfortable with basics):
- 05_genetic_algorithm_basics.py - Population-based optimization
- 06_simulated_annealing_intro.py - Probabilistic optimization
- 07_comparing_algorithms.py - Algorithm selection strategies
- 08_custom_objective_functions.py - Advanced function design
- 09_constraint_handling.py - Complex constraint scenarios

🔴 Advanced Level (For experienced users tackling complex problems):
- 10_portfolio_optimization.py - Real-world financial optimization
- 11_engineering_design.py - Multi-objective engineering problems
- 12_scheduling_optimization.py - Combinatorial optimization
- 13_hyperparameter_tuning.py - Algorithm configuration optimization
- 14_large_scale_problems.py - Performance and memory considerations
- 15_custom_algorithms.py - Extending the framework

EXAMPLE CATEGORIES:

📊 Problem Types:
- Continuous Optimization: Smooth, differentiable functions
- Integer Programming: Discrete decision variables
- Mixed-Integer: Combination of continuous and discrete variables
- Combinatorial: Permutations, selections, scheduling
- Multi-objective: Trading off multiple competing objectives

🔧 Algorithm Demonstrations:
- Greedy Search: Local optimization, hill climbing
- Genetic Algorithms: Population evolution, global search
- Simulated Annealing: Probabilistic acceptance, escaping local optima

🏭 Application Domains:
- Finance: Portfolio optimization, risk management
- Engineering: Design optimization, resource allocation
- Operations Research: Scheduling, routing, assignment
- Machine Learning: Hyperparameter tuning, feature selection

PREREQUISITES:

Software Requirements:
- Python 3.8 or higher
- The optimization framework installed (see main README)
- Standard libraries: math, random, time, json

Mathematical Background:
- Basic: Understanding of functions, variables, and constraints
- Intermediate: Familiarity with optimization concepts (local/global optima)
- Advanced: Knowledge of specific algorithms and their trade-offs

Programming Skills:
- Basic: Python syntax, functions, classes
- Intermediate: Object-oriented programming, error handling
- Advanced: Algorithm implementation, performance optimization

HOW TO RUN EXAMPLES:

Individual Examples:
# Navigate to examples folder
cd examples

# Run a specific example
python 01_basic_continuous.py

# Run with verbose output
python 01_basic_continuous.py --verbose

# Run with custom parameters (where supported)
python 05_genetic_algorithm_basics.py --population_size 100 --generations 50

Batch Execution:
# Run all beginner examples
python run_examples.py --level beginner

# Run algorithm comparison
python run_examples.py --category algorithms

# Run performance benchmarks
python run_examples.py --benchmark

Interactive Mode:
# Launch interactive example explorer
python interactive_examples.py

EXAMPLE DESCRIPTIONS:

🟢 BEGINNER EXAMPLES:

01_basic_continuous.py
Concept: Framework fundamentals with continuous variables
Problem: Minimize f(x, y) = (x-2)² + (y-3)² 
Learning Goals:
- Creating variables with bounds
- Defining objective functions
- Setting up and solving problems
- Interpreting results

Expected Output:
Optimal solution found: x=2.00, y=3.00
Objective value: 0.000001
Iterations: 23
Convergence: True

02_mixed_variables.py
Concept: Working with different variable types
Problem: Resource allocation with continuous budget and integer quantities
Learning Goals:
- ContinuousVariable, IntegerVariable, BinaryVariable
- Variable bounds and validation
- Mixed-variable problem setup

Expected Output:
Optimal allocation:
- Budget: $1000.00 (continuous)
- Quantity: 15 units (integer)  
- Use premium: Yes (binary)
Total cost: $950.00

03_with_constraints.py
Concept: Adding constraints to optimization problems
Problem: Production planning with resource constraints
Learning Goals:
- ConstraintFunction creation
- Feasible vs infeasible solutions
- Constraint violation handling

Expected Output:
Feasible solution found:
- Production A: 12 units
- Production B: 8 units
Constraints satisfied:
- Resource constraint: 18.5 <= 20.0 ✓
- Demand constraint: 20 >= 15 ✓

04_greedy_search_intro.py
Concept: Local search optimization
Problem: Function minimization with multiple local optima
Learning Goals:
- GreedySearchSolver configuration
- Local vs global optima
- Algorithm limitations

Expected Output:
Greedy Search Results:
Starting point: x=0.5, y=0.5
Final solution: x=1.02, y=1.98
Objective: 0.0008
Iterations: 45
Note: Found local optimum, may not be global

🟡 INTERMEDIATE EXAMPLES:

05_genetic_algorithm_basics.py
Concept: Population-based evolutionary optimization
Problem: Multi-modal function with many local optima
Learning Goals:
- Population initialization and evolution
- Selection, crossover, mutation
- Global search capabilities

Expected Output:
Genetic Algorithm Results:
Population size: 50
Generations: 100
Best solution: x=3.14, y=2.72
Objective: -15.42
Diversity maintained: Yes
Global optimum likely found: Yes

06_simulated_annealing_intro.py
Concept: Probabilistic optimization with cooling
Problem: Traveling salesman-like discrete optimization
Learning Goals:
- Temperature schedules
- Acceptance probability
- Balancing exploration vs exploitation

Expected Output:
Simulated Annealing Results:
Initial temperature: 100.0
Final temperature: 0.01
Best tour length: 245.7
Accepted moves: 1,247 / 10,000
Temperature schedule: Exponential cooling

07_comparing_algorithms.py
Concept: Algorithm selection and performance comparison
Problem: Same problem solved with all three algorithms
Learning Goals:
- Algorithm strengths and weaknesses
- Performance metrics comparison
- When to use each algorithm

Expected Output:
Algorithm Comparison Results:
Problem: Rosenbrock function (2D)

Greedy Search:    Best: 0.045  Time: 0.12s  Success: No
Genetic Algorithm: Best: 0.001  Time: 2.34s  Success: Yes  
Simulated Annealing: Best: 0.008  Time: 0.89s  Success: Partial

Recommendation: Genetic Algorithm (best solution quality)

🔴 ADVANCED EXAMPLES:

10_portfolio_optimization.py
Concept: Real-world financial optimization
Problem: Optimize portfolio allocation with risk constraints
Learning Goals:
- Multi-objective optimization
- Risk-return trade-offs
- Real-world constraint modeling

Expected Output:
Portfolio Optimization Results:
Expected Return: 12.5%
Portfolio Risk (σ): 8.2%
Sharpe Ratio: 1.52

Asset Allocation:
- Stocks: 60.0%
- Bonds: 30.0%  
- Cash: 10.0%

Risk Constraints: All satisfied ✓

11_engineering_design.py
Concept: Multi-constraint engineering optimization
Problem: Structural design with safety and cost constraints
Learning Goals:
- Complex constraint systems
- Engineering trade-offs
- Safety factor considerations

Expected Output:
Structural Design Results:
Beam dimensions: 0.3m × 0.5m
Material: Steel Grade A
Total cost: $2,847
Safety factor: 2.1 (required: 2.0) ✓
Weight: 145 kg
Deflection: 2.3mm (limit: 5.0mm) ✓

WHEN TO USE EACH OPTIMIZER:

🔍 Greedy Search (Local Optimization)
Best For:
- Smooth, unimodal functions
- Quick local improvements
- Fine-tuning existing solutions
- Real-time applications requiring fast results

Avoid When:
- Multiple local optima exist
- Global optimum is required
- Function is highly discontinuous

Configuration Tips:
config = {
    'max_iterations': 1000,    # Increase for thorough search
    'step_size': 0.1,         # Smaller for precision, larger for speed
    'tolerance': 1e-6         # Convergence criteria
}

🧬 Genetic Algorithm (Global Search)
Best For:
- Multi-modal optimization problems
- Combinatorial optimization
- When solution quality is more important than speed
- Problems with complex constraint landscapes

Avoid When:
- Simple unimodal problems
- Real-time applications
- Very high-dimensional problems (>100 variables)

Configuration Tips:
config = {
    'population_size': 50,     # Larger for complex problems
    'generations': 200,        # More for difficult problems
    'mutation_rate': 0.1,      # Higher for exploration
    'crossover_rate': 0.8      # Lower for exploitation
}

🌡️ Simulated Annealing (Balanced Approach)
Best For:
- Discrete optimization problems
- When you need better than greedy but faster than GA
- Problems with moderate number of local optima
- Combinatorial problems like scheduling

Avoid When:
- Continuous smooth functions (use greedy)
- Highly multi-modal problems (use GA)
- When you need population diversity

Configuration Tips:
config = {
    'initial_temp': 100.0,     # Higher for more exploration
    'final_temp': 0.01,        # Lower for precision
    'cooling_rate': 0.95,      # Slower cooling = better quality
    'max_iterations': 10000    # Enough for cooling schedule
}

TROUBLESHOOTING:

Common Issues and Solutions:

🚫 "Variable validation failed"
Cause: Variable values outside bounds or wrong type
Solution:
# Check variable bounds
print(f"Variable bounds: {variable.bounds}")
print(f"Current value: {value}")

# Ensure value is within bounds
if variable.bounds:
    value = max(variable.bounds[0], min(variable.bounds[1], value))

🚫 "Constraint violation detected"
Cause: Solution doesn't satisfy constraints
Solution:
# Check constraint violations
violations = problem.get_constraint_violations(variable_dict)
print("Constraint violations:", violations)

# Adjust solver parameters for better constraint handling
config['penalty_factor'] = 10.0  # Higher penalty for violations

🚫 "Optimization not converging"
Cause: Poor algorithm configuration or difficult problem
Solution:
# Increase iteration limits
config['max_iterations'] = 10000

# For GA: increase population size
config['population_size'] = 100

# For SA: slower cooling
config['cooling_rate'] = 0.99

# Check convergence criteria
solution.is_converged(tolerance=1e-4)  # Relax tolerance

🚫 "Memory usage too high"
Cause: Large optimization history storage
Solution:
# Limit history storage (if implemented)
config['max_history_size'] = 1000

# Or clear history periodically
if len(solution.history) > 1000:
    solution.history = solution.history[-500:]  # Keep recent history

Performance Issues:

Slow Convergence:
1. Check problem scaling: Normalize variables to similar ranges
2. Adjust step sizes: Smaller steps for precision, larger for speed
3. Verify gradients: Ensure objective function is well-behaved
4. Consider algorithm: Switch to more appropriate optimizer

Memory Problems:
1. Reduce history tracking: Limit stored iterations
2. Use efficient data structures: Consider numpy arrays for large problems
3. Batch processing: Solve smaller sub-problems
4. Profile memory usage: Identify memory leaks

Accuracy Issues:
1. Increase precision: Tighter convergence tolerances
2. More iterations: Allow longer optimization runs
3. Better initialization: Start closer to optimal solution
4. Algorithm tuning: Adjust algorithm-specific parameters

PERFORMANCE TIPS:

General Optimization:
- Scale variables: Keep all variables in similar ranges (0-1 or -1 to 1)
- Good initialization: Start optimization near expected solution
- Constraint formulation: Use soft constraints when possible
- Function evaluation: Cache expensive function evaluations

Algorithm-Specific Tips:

Greedy Search:
# Use adaptive step sizes
config = {
    'initial_step_size': 1.0,
    'step_reduction_factor': 0.8,
    'min_step_size': 1e-6
}

Genetic Algorithm:
# Balance exploration vs exploitation
config = {
    'population_size': min(50, 10 * num_variables),  # Scale with problem size
    'elite_size': max(2, population_size // 10),     # Keep best solutions
    'tournament_size': 3                             # Selection pressure
}

Simulated Annealing:
# Design cooling schedule for problem
total_evaluations = 10000
config = {
    'initial_temp': problem_scale * 10,              # Based on objective range
    'cooling_rate': (final_temp/initial_temp)**(1/total_evaluations)
}

FURTHER READING:

Framework Documentation:
- Main README: Framework installation and basic usage
- API Reference: Complete class and method documentation
- Architecture Guide: Framework design and extension points
- Developer Guide: Contributing and customizing the framework

Optimization Theory:
Books:
- "Numerical Optimization" by Nocedal & Wright
- "Introduction to Genetic Algorithms" by Melanie Mitchell
- "Simulated Annealing: Theory and Applications" by van Laarhoven & Aarts

Online Resources:
- Optimization Online (http://www.optimization-online.org/)
- NEOS Guide (https://neos-guide.org/)
- OR-Tools Documentation (https://developers.google.com/optimization)

Algorithm Implementation:
- Research Papers: Check docs/references.md for algorithm-specific papers
- Benchmarks: Standard test problems for algorithm validation
- Communities: Optimization forums and discussion groups

Related Tools:
- SciPy: If available on your platform, for comparison
- DEAP: Distributed Evolutionary Algorithms in Python
- Optuna: Hyperparameter optimization framework
- PuLP: Linear programming in Python

CONTRIBUTING EXAMPLES:

Have an interesting optimization problem or use case? We welcome contributions!

Guidelines for New Examples:
1. Clear learning objective: What concept does it teach?
2. Complete documentation: Explain the problem and solution
3. Expected outputs: Show what users should see
4. Error handling: Robust code that handles edge cases
5. Performance notes: Mention computational requirements

Example Template:
'''
Example: [Brief Description]

Learning Objectives:
- Objective 1
- Objective 2

Problem Description:
[Detailed problem explanation]

Expected Runtime: [X seconds/minutes]
Difficulty Level: [Beginner/Intermediate/Advanced]
'''

# Your example code here

Submission Process:
1. Fork the repository
2. Add your example to the appropriate difficulty folder
3. Update this README with your example description
4. Submit a pull request with clear description

Happy Optimizing! 🚀

For questions, issues, or suggestions, please check the main repository's 
issue tracker or discussion forum.
"""

# This file serves as the comprehensive README for the examples folder
# It can be imported to access the documentation programmatically

def print_readme():
    """Print the complete README documentation."""
    print(__doc__)

def get_example_info(example_name):
    """
    Get information about a specific example.

    Args:
        example_name: Name of the example file (e.g., '01_basic_continuous.py')

    Returns:
        dict: Example information including concept, problem, learning goals
    """
    examples_info = {
        '01_basic_continuous.py': {
            'concept': 'Framework fundamentals with continuous variables',
            'problem': 'Minimize f(x, y) = (x-2)² + (y-3)²',
            'level': 'Beginner',
            'learning_goals': [
                'Creating variables with bounds',
                'Defining objective functions', 
                'Setting up and solving problems',
                'Interpreting results'
            ]
        },
        '02_mixed_variables.py': {
            'concept': 'Working with different variable types',
            'problem': 'Resource allocation with continuous budget and integer quantities',
            'level': 'Beginner',
            'learning_goals': [
                'ContinuousVariable, IntegerVariable, BinaryVariable',
                'Variable bounds and validation',
                'Mixed-variable problem setup'
            ]
        },
        '03_with_constraints.py': {
            'concept': 'Adding constraints to optimization problems',
            'problem': 'Production planning with resource constraints',
            'level': 'Beginner',
            'learning_goals': [
                'ConstraintFunction creation',
                'Feasible vs infeasible solutions',
                'Constraint violation handling'
            ]
        },
        '04_greedy_search_intro.py': {
            'concept': 'Local search optimization',
            'problem': 'Function minimization with multiple local optima',
            'level': 'Beginner',
            'learning_goals': [
                'GreedySearchSolver configuration',
                'Local vs global optima',
                'Algorithm limitations'
            ]
        },
        '05_genetic_algorithm_basics.py': {
            'concept': 'Population-based evolutionary optimization',
            'problem': 'Multi-modal function with many local optima',
            'level': 'Intermediate',
            'learning_goals': [
                'Population initialization and evolution',
                'Selection, crossover, mutation',
                'Global search capabilities'
            ]
        },
        '06_simulated_annealing_intro.py': {
            'concept': 'Probabilistic optimization with cooling',
            'problem': 'Traveling salesman-like discrete optimization',
            'level': 'Intermediate',
            'learning_goals': [
                'Temperature schedules',
                'Acceptance probability',
                'Balancing exploration vs exploitation'
            ]
        },
        '07_comparing_algorithms.py': {
            'concept': 'Algorithm selection and performance comparison',
            'problem': 'Same problem solved with all three algorithms',
            'level': 'Intermediate',
            'learning_goals': [
                'Algorithm strengths and weaknesses',
                'Performance metrics comparison',
                'When to use each algorithm'
            ]
        },
        '10_portfolio_optimization.py': {
            'concept': 'Real-world financial optimization',
            'problem': 'Optimize portfolio allocation with risk constraints',
            'level': 'Advanced',
            'learning_goals': [
                'Multi-objective optimization',
                'Risk-return trade-offs',
                'Real-world constraint modeling'
            ]
        },
        '11_engineering_design.py': {
            'concept': 'Multi-constraint engineering optimization',
            'problem': 'Structural design with safety and cost constraints',
            'level': 'Advanced',
            'learning_goals': [
                'Complex constraint systems',
                'Engineering trade-offs',
                'Safety factor considerations'
            ]
        }
    }

    return examples_info.get(example_name, {'error': 'Example not found'})

def get_algorithm_recommendations(problem_type):
    """
    Get algorithm recommendations based on problem type.

    Args:
        problem_type: Type of optimization problem

    Returns:
        dict: Recommended algorithms and configurations
    """
    recommendations = {
        'smooth_unimodal': {
            'primary': 'GreedySearchSolver',
            'config': {'max_iterations': 1000, 'step_size': 0.1, 'tolerance': 1e-6},
            'reason': 'Efficient for smooth functions with single optimum'
        },
        'multimodal': {
            'primary': 'GeneticAlgorithmSolver',
            'config': {'population_size': 50, 'generations': 200, 'mutation_rate': 0.1},
            'reason': 'Best for finding global optimum in complex landscapes'
        },
        'discrete_combinatorial': {
            'primary': 'SimulatedAnnealingSolver',
            'config': {'initial_temp': 100.0, 'cooling_rate': 0.95, 'max_iterations': 10000},
            'reason': 'Effective for discrete problems with moderate complexity'
        },
        'mixed_integer': {
            'primary': 'GeneticAlgorithmSolver',
            'secondary': 'SimulatedAnnealingSolver',
            'reason': 'Handle mixed variable types effectively'
        }
    }

    return recommendations.get(problem_type, {'error': 'Problem type not recognized'})

if __name__ == "__main__":
    print_readme()