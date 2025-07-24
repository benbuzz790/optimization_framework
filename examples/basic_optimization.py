"""
Basic Optimization Framework Example

This example demonstrates how to use the optimization framework for simple problems.
We'll solve a quadratic optimization problem step-by-step with detailed explanations.

Problem: Minimize f(x, y) = (x - 2)² + (y - 3)² + 5
Subject to: x + y <= 8
           x >= 0, y >= 0

The optimal solution should be at x=2, y=3 with objective value = 5.
"""

import sys
import os

# Add the parent directory to the path to import our framework
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the optimization framework components
from optimization_framework import (
    ContinuousVariable, ObjectiveFunction, ConstraintFunction, 
    Problem, Solution, IntegerVariable, BinaryVariable
)

def main():
    """
    Main function demonstrating basic optimization framework usage.

    This example walks through each step of setting up and solving
    an optimization problem using our custom framework.
    """

    print("=" * 60)
    print("OPTIMIZATION FRAMEWORK - BASIC EXAMPLE")
    print("=" * 60)
    print()

    # =================================================================
    # STEP 1: DEFINE OPTIMIZATION VARIABLES
    # =================================================================
    print("STEP 1: Defining Optimization Variables")
    print("-" * 40)

    # Create continuous variables with bounds
    # Variable bounds help the optimizer stay in reasonable ranges
    x_var = ContinuousVariable(name="x", bounds=(0.0, 10.0))
    y_var = ContinuousVariable(name="y", bounds=(0.0, 10.0))

    print(f"Created variable 'x': type={x_var.get_type_name()}, bounds={x_var.bounds}")
    print(f"Created variable 'y': type={y_var.get_type_name()}, bounds={y_var.bounds}")
    print()

    # Collect all variables for the problem
    variables = [x_var, y_var]

    # =================================================================
    # STEP 2: DEFINE THE OBJECTIVE FUNCTION
    # =================================================================
    print("STEP 2: Defining the Objective Function")
    print("-" * 40)

    def quadratic_objective(variable_dict):
        """
        Quadratic objective function: f(x, y) = (x - 2)² + (y - 3)² + 5

        Args:
            variable_dict: Dictionary following the variable dict protocol
                          Format: {"var_name": {"value": val, "type": type, ...}}

        Returns:
            float: Objective function value

        Note: This function has a global minimum at (x=2, y=3) with value=5
        """
        # Extract variable values from the variable dictionary
        # The framework ensures type safety and bounds checking
        x = variable_dict["x"]["value"]
        y = variable_dict["y"]["value"]

        # Calculate the quadratic function
        result = (x - 2.0)**2 + (y - 3.0)**2 + 5.0

        return result

    # Wrap the function in an ObjectiveFunction object
    # This provides validation and consistent interface
    objective = ObjectiveFunction(
        func=quadratic_objective,
        name="quadratic_minimization"
    )

    print("Objective function: f(x, y) = (x - 2)² + (y - 3)² + 5")
    print("Global minimum: x=2, y=3, f(2,3)=5")
    print()

    # =================================================================
    # STEP 3: DEFINE CONSTRAINT FUNCTIONS
    # =================================================================
    print("STEP 3: Defining Constraint Functions")
    print("-" * 40)

    def linear_constraint(variable_dict):
        """
        Linear constraint function: g(x, y) = x + y

        This will be used with constraint type "<=" and bound 8.0
        So the constraint becomes: x + y <= 8

        Args:
            variable_dict: Dictionary following the variable dict protocol

        Returns:
            float: Constraint function value
        """
        x = variable_dict["x"]["value"]
        y = variable_dict["y"]["value"]

        return x + y

    # Create constraint function object
    # The constraint "x + y <= 8" means linear_constraint(vars) <= 8.0
    constraint = ConstraintFunction(
        func=linear_constraint,
        constraint_type="<=",  # Less than or equal
        bound=8.0,
        name="sum_constraint"
    )

    print("Constraint: x + y <= 8")
    print("This constraint is active when x + y = 8")
    print()

    # Collect all constraints
    constraints = [constraint]

    # =================================================================
    # STEP 4: CREATE THE OPTIMIZATION PROBLEM
    # =================================================================
    print("STEP 4: Creating the Optimization Problem")
    print("-" * 40)

    # Bundle everything into a Problem object
    # This validates that all components work together
    problem = Problem(
        objective=objective,
        constraints=constraints,
        variables=variables
    )

    print("Problem created successfully!")
    print(f"Variables: {[var.name for var in problem.variables]}")
    print(f"Constraints: {[const.name for const in problem.constraints]}")
    print()

    # =================================================================
    # STEP 5: TEST THE PROBLEM SETUP
    # =================================================================
    print("STEP 5: Testing the Problem Setup")
    print("-" * 40)

    # Create a test point to verify our setup
    test_values = {"x": 1.0, "y": 2.0}
    test_var_dict = problem.create_variable_dict(test_values)

    print(f"Test point: x={test_values['x']}, y={test_values['y']}")

    # Evaluate objective function
    obj_value = problem.evaluate_objective(test_var_dict)
    print(f"Objective value: f(1, 2) = {obj_value}")
    print(f"Expected: (1-2)² + (2-3)² + 5 = 1 + 1 + 5 = 7 ✓")

    # Check constraint satisfaction
    is_feasible = problem.is_feasible(test_var_dict)
    constraint_violations = problem.get_constraint_violations(test_var_dict)

    print(f"Constraint x + y = {test_values['x'] + test_values['y']} <= 8: {is_feasible}")
    print(f"Constraint violations: {constraint_violations}")
    print()

    # =================================================================
    # STEP 6: DEMONSTRATE SOLUTION TRACKING
    # =================================================================
    print("STEP 6: Demonstrating Solution Tracking")
    print("-" * 40)

    # Create a solution object to track optimization progress
    # In real usage, the solver would create and manage this
    solution = Solution(problem=problem, solver_name="manual_demo")

    # Simulate adding some optimization iterations
    test_points = [
        {"x": 0.0, "y": 0.0},  # Starting point
        {"x": 1.0, "y": 2.0},  # Intermediate point
        {"x": 2.0, "y": 3.0},  # Optimal point
        {"x": 2.5, "y": 2.5},  # Another test point
    ]

    print("Simulating optimization iterations:")
    for i, values in enumerate(test_points):
        var_dict = problem.create_variable_dict(values)
        obj_val = problem.evaluate_objective(var_dict)
        violations = problem.get_constraint_violations(var_dict)

        solution.add_iteration(
            variable_dict=var_dict,
            objective_value=obj_val,
            constraint_violations=violations,
            metadata={"iteration_type": "demo"}
        )

        feasible = "✓" if problem.is_feasible(var_dict) else "✗"
        print(f"  Iteration {i}: x={values['x']}, y={values['y']}, "
              f"f={obj_val:.3f}, feasible={feasible}")

    print()

    # =================================================================
    # STEP 7: ANALYZE THE SOLUTION
    # =================================================================
    print("STEP 7: Solution Analysis")
    print("-" * 40)

    # Get the best solution found
    best_solution = solution.get_best_solution()
    best_vars = best_solution['variable_dict']

    print("BEST SOLUTION FOUND:")
    print(f"  x = {best_vars['x']['value']}")
    print(f"  y = {best_vars['y']['value']}")
    print(f"  Objective value = {best_solution['objective_value']}")
    print(f"  Feasible = {best_solution['is_feasible']}")
    print(f"  Constraint violations = {best_solution['constraint_violations']}")
    print()

    # Get optimization statistics
    stats = solution.get_summary_statistics()
    print("OPTIMIZATION STATISTICS:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()

    # Get convergence information
    convergence = solution.get_convergence_data()
    print("CONVERGENCE ANALYSIS:")
    print(f"  Total iterations: {convergence['total_iterations']}")
    print(f"  Initial objective: {convergence['initial_objective']:.3f}")
    print(f"  Final objective: {convergence['final_objective']:.3f}")
    print(f"  Best objective: {convergence['best_objective']:.3f}")
    print(f"  Total improvement: {convergence['improvement']:.3f}")
    print()

    # =================================================================
    # STEP 8: BEST PRACTICES AND TIPS
    # =================================================================
    print("STEP 8: Best Practices and Tips")
    print("-" * 40)

    print("FRAMEWORK USAGE BEST PRACTICES:")
    print()

    print("1. VARIABLE DEFINITION:")
    print("   - Always set reasonable bounds for continuous variables")
    print("   - Use descriptive variable names")
    print("   - Choose appropriate variable types (continuous/integer/binary)")
    print()

    print("2. OBJECTIVE FUNCTIONS:")
    print("   - Keep functions simple and well-documented")
    print("   - Handle edge cases (division by zero, etc.)")
    print("   - Use the variable_dict protocol consistently")
    print("   - Test functions independently before optimization")
    print()

    print("3. CONSTRAINTS:")
    print("   - Define constraints clearly with appropriate bounds")
    print("   - Test constraint satisfaction manually")
    print("   - Consider constraint scaling for numerical stability")
    print()

    print("4. PROBLEM SETUP:")
    print("   - Validate your problem setup with known test points")
    print("   - Check that the optimal solution (if known) satisfies constraints")
    print("   - Start with simple problems before adding complexity")
    print()

    print("5. SOLUTION ANALYSIS:")
    print("   - Always check if the solution is feasible")
    print("   - Examine constraint violations for infeasible solutions")
    print("   - Use the optimization history to understand convergence")
    print("   - Compare results with analytical solutions when available")
    print()

    # =================================================================
    # STEP 9: COMMON TROUBLESHOOTING
    # =================================================================
    print("STEP 9: Common Issues and Solutions")
    print("-" * 40)

    print("COMMON ISSUES:")
    print()

    print("1. 'Invalid variable dictionary' errors:")
    print("   - Check that all variables are included in your values dict")
    print("   - Ensure variable names match exactly (case-sensitive)")
    print("   - Verify that values respect variable bounds and types")
    print()

    print("2. Constraint violations:")
    print("   - Check constraint function implementation")
    print("   - Verify constraint type (<=, >=, ==) and bound values")
    print("   - Test constraints with known feasible/infeasible points")
    print()

    print("3. Poor optimization performance:")
    print("   - Check variable bounds (too wide or too narrow)")
    print("   - Consider problem scaling (normalize variables/objectives)")
    print("   - Try different initial guesses")
    print("   - Verify that the problem is well-posed")
    print()

    print("4. Numerical issues:")
    print("   - Avoid functions that can return NaN or infinity")
    print("   - Use appropriate tolerances for equality constraints")
    print("   - Consider the numerical precision of your problem")
    print()

    print("=" * 60)
    print("EXAMPLE COMPLETED SUCCESSFULLY!")
    print("=" * 60)


def demonstrate_variable_types():
    """
    Additional demonstration of different variable types.

    This function shows how to work with integer and binary variables
    in addition to continuous variables.
    """
    print("\nBONUS: Different Variable Types Demo")
    print("-" * 40)

    # Create different types of variables
    continuous_var = ContinuousVariable("height", bounds=(0.0, 10.0))
    integer_var = IntegerVariable("count", bounds=(1, 100))  # Note: IntegerVariable expects int bounds
    binary_var = BinaryVariable("switch")

    print("Variable Types:")
    print(f"  Continuous: {continuous_var.name} ∈ {continuous_var.bounds}")
    print(f"  Integer: {integer_var.name} ∈ {integer_var.bounds}")
    print(f"  Binary: {binary_var.name} ∈ {binary_var.bounds}")
    print()

    # Test variable validation
    test_values = [
        (continuous_var, 5.5, True),   # Valid continuous
        (continuous_var, 15.0, False), # Out of bounds
        (integer_var, 50, True),       # Valid integer
        (integer_var, 5.5, False),     # Not an integer
        (binary_var, 1, True),         # Valid binary
        (binary_var, 2, False),        # Invalid binary
    ]

    print("Variable Validation Tests:")
    for var, value, expected in test_values:
        result = var.validate_value(value)
        status = "✓" if result == expected else "✗"
        print(f"  {var.name}={value}: {result} {status}")

    print()


if __name__ == "__main__":
    """
    Run the basic optimization example.

    This demonstrates the complete workflow of using the optimization
    framework from problem setup to solution analysis.
    """
    try:
        main()
        demonstrate_variable_types()

    except Exception as e:
        print(f"\nERROR: {str(e)}")
        print("\nThis example requires the optimization framework to be implemented.")
        print("Please ensure all framework components are available before running.")

        # Show what the error might indicate
        if "No module named" in str(e):
            print("\nTroubleshooting:")
            print("- Check that the optimization framework files are in the correct location")
            print("- Verify that all required classes are implemented")
            print("- Make sure the import paths are correct")

        import traceback
        print(f"\nFull traceback:\n{traceback.format_exc()}")
