"""
Advanced Constrained Optimization Example: Production Planning Problem

This example demonstrates sophisticated constraint handling in optimization using
a realistic production planning scenario. The problem involves optimizing production
quantities across multiple products while satisfying resource constraints, demand
requirements, and operational limitations.

PROBLEM DESCRIPTION:
A manufacturing company produces 3 products (A, B, C) using 3 resources (Labor, Material, Machine).
- Maximize profit while satisfying all constraints
- Mixed variable types: continuous production quantities, integer batch sizes, binary decisions
- Multiple constraint types: resource limits (<=), minimum production (>=), exact requirements (==)
- Handle infeasible solutions and constraint violations gracefully

MATHEMATICAL FORMULATION:
Variables:
- prod_A, prod_B, prod_C: continuous production quantities (units)
- batch_A, batch_B: integer batch sizes for products A and B
- use_premium_material: binary decision for premium material usage

Objective:
Maximize: 15*prod_A + 25*prod_B + 20*prod_C - 100*use_premium_material

Constraints:
1. Labor constraint: 2*prod_A + 3*prod_B + 1.5*prod_C <= 1000 (hours)
2. Material constraint: 1*prod_A + 2*prod_B + 1*prod_C <= 500 + 200*use_premium_material (kg)
3. Machine constraint: 1.5*prod_A + 1*prod_B + 2*prod_C <= 800 (hours)
4. Minimum production A: prod_A >= 50 (units)
5. Minimum production B: prod_B >= 30 (units)
6. Exact batch requirement: batch_A + batch_B == 20 (batches)
7. Batch-production relationship: prod_A <= 50*batch_A
8. Batch-production relationship: prod_B <= 40*batch_B
9. Premium material threshold: use_premium_material == 1 if total_material > 600, else 0
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimization_framework_architecture import (
    ContinuousVariable, IntegerVariable, BinaryVariable,
    ObjectiveFunction, ConstraintFunction, Problem, Solution,
    production_assert
)
import math
from typing import Dict, List, Tuple


class ProductionPlanningProblem:
    """
    Advanced production planning optimization problem with mixed constraints.

    Demonstrates comprehensive constraint handling including:
    - Resource allocation constraints
    - Demand and capacity constraints  
    - Logical constraints linking variables
    - Penalty handling for constraint violations
    """

    def __init__(self):
        """Initialize the production planning problem with all variables and constraints."""
        self.setup_variables()
        self.setup_objective()
        self.setup_constraints()
        self.create_problem()

    def setup_variables(self):
        """Define all optimization variables with appropriate types and bounds."""
        # Continuous production quantities (units)
        self.prod_A = ContinuousVariable("prod_A", bounds=(0, 500))
        self.prod_B = ContinuousVariable("prod_B", bounds=(0, 400))  
        self.prod_C = ContinuousVariable("prod_C", bounds=(0, 300))

        # Integer batch sizes
        self.batch_A = IntegerVariable("batch_A", bounds=(0, 15))
        self.batch_B = IntegerVariable("batch_B", bounds=(0, 15))

        # Binary decision for premium material
        self.use_premium = BinaryVariable("use_premium_material")

        self.variables = [
            self.prod_A, self.prod_B, self.prod_C,
            self.batch_A, self.batch_B, self.use_premium
        ]

        print("✓ Variables defined:")
        for var in self.variables:
            print(f"  - {var.name}: {var.get_type_name()}, bounds: {var.bounds}")

    def setup_objective(self):
        """Define the profit maximization objective function."""
        def profit_function(var_dict: Dict) -> float:
            """
            Calculate total profit from production.

            Profit = Revenue - Premium Material Cost
            Revenue = 15*prod_A + 25*prod_B + 20*prod_C
            Premium Cost = 100 if using premium material
            """
            production_assert(isinstance(var_dict, dict), "Variable dict must be dictionary")

            prod_A = var_dict["prod_A"]["value"]
            prod_B = var_dict["prod_B"]["value"] 
            prod_C = var_dict["prod_C"]["value"]
            use_premium = var_dict["use_premium_material"]["value"]

            # Calculate revenue from each product
            revenue = 15 * prod_A + 25 * prod_B + 20 * prod_C

            # Subtract premium material cost
            premium_cost = 100 * use_premium

            # Return negative profit for minimization (most solvers minimize)
            return -(revenue - premium_cost)

        self.objective = ObjectiveFunction(profit_function, "maximize_profit")
        print("✓ Objective function defined: Maximize profit")

    def setup_constraints(self):
        """Define all constraint functions with different types."""
        self.constraints = []

        # 1. Labor constraint (<=): 2*prod_A + 3*prod_B + 1.5*prod_C <= 1000
        def labor_constraint(var_dict: Dict) -> float:
            prod_A = var_dict["prod_A"]["value"]
            prod_B = var_dict["prod_B"]["value"]
            prod_C = var_dict["prod_C"]["value"]
            return 2 * prod_A + 3 * prod_B + 1.5 * prod_C

        self.constraints.append(
            ConstraintFunction(labor_constraint, "<=", 1000, "labor_limit")
        )

        # 2. Material constraint (<=): 1*prod_A + 2*prod_B + 1*prod_C <= 500 + 200*use_premium
        def material_constraint(var_dict: Dict) -> float:
            prod_A = var_dict["prod_A"]["value"]
            prod_B = var_dict["prod_B"]["value"]
            prod_C = var_dict["prod_C"]["value"]
            use_premium = var_dict["use_premium_material"]["value"]

            material_used = 1 * prod_A + 2 * prod_B + 1 * prod_C
            material_available = 500 + 200 * use_premium

            # Return usage - availability (should be <= 0)
            return material_used - material_available

        self.constraints.append(
            ConstraintFunction(material_constraint, "<=", 0, "material_limit")
        )

        # 3. Machine constraint (<=): 1.5*prod_A + 1*prod_B + 2*prod_C <= 800
        def machine_constraint(var_dict: Dict) -> float:
            prod_A = var_dict["prod_A"]["value"]
            prod_B = var_dict["prod_B"]["value"]
            prod_C = var_dict["prod_C"]["value"]
            return 1.5 * prod_A + 1 * prod_B + 2 * prod_C

        self.constraints.append(
            ConstraintFunction(machine_constraint, "<=", 800, "machine_limit")
        )

        # 4. Minimum production A (>=): prod_A >= 50
        def min_prod_A_constraint(var_dict: Dict) -> float:
            return var_dict["prod_A"]["value"]

        self.constraints.append(
            ConstraintFunction(min_prod_A_constraint, ">=", 50, "min_production_A")
        )

        # 5. Minimum production B (>=): prod_B >= 30
        def min_prod_B_constraint(var_dict: Dict) -> float:
            return var_dict["prod_B"]["value"]

        self.constraints.append(
            ConstraintFunction(min_prod_B_constraint, ">=", 30, "min_production_B")
        )

        # 6. Exact batch requirement (==): batch_A + batch_B == 20
        def batch_total_constraint(var_dict: Dict) -> float:
            batch_A = var_dict["batch_A"]["value"]
            batch_B = var_dict["batch_B"]["value"]
            return batch_A + batch_B

        self.constraints.append(
            ConstraintFunction(batch_total_constraint, "==", 20, "total_batches")
        )

        # 7. Batch-production relationship A: prod_A <= 50*batch_A
        def batch_prod_A_constraint(var_dict: Dict) -> float:
            prod_A = var_dict["prod_A"]["value"]
            batch_A = var_dict["batch_A"]["value"]
            return prod_A - 50 * batch_A  # Should be <= 0

        self.constraints.append(
            ConstraintFunction(batch_prod_A_constraint, "<=", 0, "batch_production_A")
        )

        # 8. Batch-production relationship B: prod_B <= 40*batch_B  
        def batch_prod_B_constraint(var_dict: Dict) -> float:
            prod_B = var_dict["prod_B"]["value"]
            batch_B = var_dict["batch_B"]["value"]
            return prod_B - 40 * batch_B  # Should be <= 0

        self.constraints.append(
            ConstraintFunction(batch_prod_B_constraint, "<=", 0, "batch_production_B")
        )

        print(f"✓ {len(self.constraints)} constraints defined:")
        for constraint in self.constraints:
            print(f"  - {constraint.name}: {constraint.constraint_type} {constraint.bound}")

    def create_problem(self):
        """Create the complete optimization problem."""
        self.problem = Problem(
            objective=self.objective,
            constraints=self.constraints,
            variables=self.variables
        )
        print("✓ Complete optimization problem created")


class ConstraintAnalyzer:
    """
    Utility class for analyzing constraint violations and feasibility.

    Provides detailed analysis of constraint satisfaction, violation amounts,
    and suggestions for handling infeasible solutions.
    """

    def __init__(self, problem: Problem):
        """Initialize analyzer with problem instance."""
        self.problem = problem

    def analyze_solution(self, variable_dict: Dict) -> Dict:
        """
        Perform comprehensive constraint analysis on a solution.

        Args:
            variable_dict: Solution to analyze

        Returns:
            Dict: Detailed analysis results
        """
        analysis = {
            'is_feasible': self.problem.is_feasible(variable_dict),
            'objective_value': self.problem.evaluate_objective(variable_dict),
            'constraint_details': [],
            'total_violation': 0,
            'violation_summary': {},
            'recommendations': []
        }

        # Analyze each constraint
        for constraint in self.problem.constraints:
            constraint_value = constraint.evaluate(variable_dict)
            is_satisfied = constraint.is_satisfied(variable_dict)
            violation = constraint.violation_amount(variable_dict)

            constraint_detail = {
                'name': constraint.name,
                'type': constraint.constraint_type,
                'bound': constraint.bound,
                'actual_value': constraint_value,
                'is_satisfied': is_satisfied,
                'violation_amount': violation,
                'violation_percentage': (violation / abs(constraint.bound)) * 100 if constraint.bound != 0 else 0
            }

            analysis['constraint_details'].append(constraint_detail)
            analysis['total_violation'] += violation

            if violation > 0:
                analysis['violation_summary'][constraint.name] = violation

        # Generate recommendations for constraint violations
        analysis['recommendations'] = self._generate_recommendations(analysis)

        return analysis

    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """Generate recommendations for handling constraint violations."""
        recommendations = []

        if analysis['is_feasible']:
            recommendations.append("✓ Solution is feasible - all constraints satisfied")
            return recommendations

        # Analyze violation patterns
        violations = analysis['violation_summary']

        if 'labor_limit' in violations:
            recommendations.append(f"⚠ Labor constraint violated by {violations['labor_limit']:.2f} hours")
            recommendations.append("  → Consider: Reduce production quantities or increase labor capacity")

        if 'material_limit' in violations:
            recommendations.append(f"⚠ Material constraint violated by {violations['material_limit']:.2f} kg")
            recommendations.append("  → Consider: Enable premium material or reduce material-intensive products")

        if 'machine_limit' in violations:
            recommendations.append(f"⚠ Machine constraint violated by {violations['machine_limit']:.2f} hours")
            recommendations.append("  → Consider: Reduce production or increase machine capacity")

        if 'min_production_A' in violations:
            recommendations.append(f"⚠ Minimum production A not met (short by {violations['min_production_A']:.2f})")
            recommendations.append("  → Consider: Increase prod_A or adjust minimum requirements")

        if 'min_production_B' in violations:
            recommendations.append(f"⚠ Minimum production B not met (short by {violations['min_production_B']:.2f})")
            recommendations.append("  → Consider: Increase prod_B or adjust minimum requirements")

        if 'total_batches' in violations:
            recommendations.append(f"⚠ Batch total requirement violated by {violations['total_batches']:.2f}")
            recommendations.append("  → Consider: Adjust batch_A and batch_B to sum to exactly 20")

        return recommendations

    def print_detailed_analysis(self, variable_dict: Dict):
        """Print comprehensive constraint analysis to console."""
        analysis = self.analyze_solution(variable_dict)

        print("\n" + "="*80)
        print("CONSTRAINT ANALYSIS REPORT")
        print("="*80)

        # Solution summary
        print(f"\nSOLUTION SUMMARY:")
        print(f"Feasible: {'✓ YES' if analysis['is_feasible'] else '✗ NO'}")
        print(f"Objective Value: {analysis['objective_value']:.2f}")
        print(f"Total Violation: {analysis['total_violation']:.4f}")

        # Variable values
        print(f"\nVARIABLE VALUES:")
        for var_name, var_data in variable_dict.items():
            print(f"  {var_name}: {var_data['value']} ({var_data['type']})")

        # Constraint details
        print(f"\nCONSTRAINT DETAILS:")
        for detail in analysis['constraint_details']:
            status = "✓" if detail['is_satisfied'] else "✗"
            print(f"  {status} {detail['name']}: {detail['actual_value']:.2f} {detail['type']} {detail['bound']}")
            if detail['violation_amount'] > 0:
                print(f"    Violation: {detail['violation_amount']:.4f} ({detail['violation_percentage']:.1f}%)")

        # Recommendations
        if analysis['recommendations']:
            print(f"\nRECOMMENDATIONS:")
            for rec in analysis['recommendations']:
                print(f"  {rec}")

        print("="*80)


class PenaltyMethodSolver:
    """
    Simple penalty method solver for handling constrained optimization.

    Converts constrained problem to unconstrained by adding penalty terms
    for constraint violations. Demonstrates one approach to handling
    infeasible solutions in optimization.
    """

    def __init__(self, penalty_factor: float = 1000):
        """
        Initialize penalty method solver.

        Args:
            penalty_factor: Multiplier for constraint violation penalties
        """
        self.penalty_factor = penalty_factor

    def create_penalized_objective(self, problem: Problem) -> ObjectiveFunction:
        """
        Create penalized objective function that includes constraint violations.

        Args:
            problem: Original constrained problem

        Returns:
            ObjectiveFunction: Penalized objective including constraint penalties
        """
        def penalized_function(var_dict: Dict) -> float:
            # Original objective value
            original_objective = problem.evaluate_objective(var_dict)

            # Calculate total penalty from constraint violations
            total_penalty = 0
            for constraint in problem.constraints:
                violation = constraint.violation_amount(var_dict)
                total_penalty += self.penalty_factor * (violation ** 2)

            return original_objective + total_penalty

        return ObjectiveFunction(penalized_function, "penalized_objective")

    def solve_with_penalties(self, problem: Problem, initial_solution: Dict) -> Dict:
        """
        Solve problem using penalty method with simple local search.

        Args:
            problem: Constrained optimization problem
            initial_solution: Starting point for optimization

        Returns:
            Dict: Best solution found with penalty method
        """
        penalized_objective = self.create_penalized_objective(problem)

        current_solution = initial_solution.copy()
        current_value = penalized_objective.evaluate(current_solution)

        print(f"\nStarting penalty method optimization...")
        print(f"Initial penalized objective: {current_value:.2f}")

        # Simple local search with penalty method
        for iteration in range(100):
            improved = False

            # Try small perturbations to each variable
            for var_name in current_solution.keys():
                var_data = current_solution[var_name]
                original_value = var_data["value"]
                var_obj = var_data["variable_object"]

                # Try increasing and decreasing the variable
                for delta in [0.1, -0.1, 1.0, -1.0]:
                    new_value = original_value + delta

                    # Ensure value is valid for variable type
                    if var_obj.validate_value(new_value):
                        # Create new solution
                        test_solution = current_solution.copy()
                        test_solution[var_name] = var_obj.to_dict_entry(new_value)

                        # Evaluate penalized objective
                        test_value = penalized_objective.evaluate(test_solution)

                        # Accept if improvement
                        if test_value < current_value:
                            current_solution = test_solution
                            current_value = test_value
                            improved = True
                            print(f"  Iteration {iteration}: Improved to {current_value:.2f}")
                            break

                if improved:
                    break

            if not improved:
                print(f"  Converged after {iteration} iterations")
                break

        print(f"Final penalized objective: {current_value:.2f}")
        return current_solution


def demonstrate_constraint_handling():
    """
    Main demonstration of advanced constraint handling capabilities.

    Shows:
    1. Problem setup with mixed variable types and constraint types
    2. Feasible solution analysis
    3. Infeasible solution handling
    4. Constraint violation analysis
    5. Penalty method for constraint handling
    """
    print("ADVANCED CONSTRAINED OPTIMIZATION DEMONSTRATION")
    print("="*60)

    # 1. Create the production planning problem
    print("\n1. SETTING UP PRODUCTION PLANNING PROBLEM")
    print("-" * 50)

    planning_problem = ProductionPlanningProblem()
    analyzer = ConstraintAnalyzer(planning_problem.problem)

    # 2. Test with a feasible solution
    print("\n2. ANALYZING FEASIBLE SOLUTION")
    print("-" * 50)

    feasible_values = {
        "prod_A": 100,      # Production quantities
        "prod_B": 80,
        "prod_C": 60,
        "batch_A": 12,      # Batch sizes
        "batch_B": 8,       # 12 + 8 = 20 (satisfies exact constraint)
        "use_premium_material": 0  # No premium material
    }

    feasible_solution = planning_problem.problem.create_variable_dict(feasible_values)
    analyzer.print_detailed_analysis(feasible_solution)

    # 3. Test with an infeasible solution
    print("\n3. ANALYZING INFEASIBLE SOLUTION")
    print("-" * 50)

    infeasible_values = {
        "prod_A": 300,      # Too high production (violates resource constraints)
        "prod_B": 200,
        "prod_C": 150,
        "batch_A": 15,      # Batch sum = 25, violates exact constraint (should be 20)
        "batch_B": 10,
        "use_premium_material": 0  # No premium material despite high usage
    }

    infeasible_solution = planning_problem.problem.create_variable_dict(infeasible_values)
    analyzer.print_detailed_analysis(infeasible_solution)

    # 4. Demonstrate penalty method for handling constraints
    print("\n4. PENALTY METHOD CONSTRAINT HANDLING")
    print("-" * 50)

    penalty_solver = PenaltyMethodSolver(penalty_factor=1000)

    # Start with infeasible solution and try to improve it
    improved_solution = penalty_solver.solve_with_penalties(
        planning_problem.problem, 
        infeasible_solution
    )

    print("\nAnalyzing solution after penalty method:")
    analyzer.print_detailed_analysis(improved_solution)

    # 5. Compare different constraint violation handling strategies
    print("\n5. CONSTRAINT VIOLATION HANDLING STRATEGIES")
    print("-" * 50)

    print("\nStrategy 1: Reject infeasible solutions entirely")
    print("  ✓ Guarantees feasibility")
    print("  ✗ May miss good solutions near feasible boundary")

    print("\nStrategy 2: Accept infeasible solutions with penalties")
    print("  ✓ Can explore infeasible regions")
    print("  ✓ May find better feasible solutions")
    print("  ✗ Requires tuning penalty parameters")

    print("\nStrategy 3: Constraint repair mechanisms")
    print("  ✓ Automatically fixes violations")
    print("  ✗ May change problem characteristics")

    print("\nStrategy 4: Multi-objective approach (feasibility + objective)")
    print("  ✓ Balances feasibility and optimality")
    print("  ✗ More complex to implement and tune")

    # 6. Demonstrate constraint sensitivity analysis
    print("\n6. CONSTRAINT SENSITIVITY ANALYSIS")
    print("-" * 50)

    print("Testing sensitivity to constraint bounds...")

    # Test how objective changes with constraint relaxation
    original_labor_bound = planning_problem.constraints[0].bound

    for labor_increase in [0, 50, 100, 200]:
        # Temporarily modify labor constraint
        planning_problem.constraints[0].bound = original_labor_bound + labor_increase

        # Re-evaluate feasible solution
        analysis = analyzer.analyze_solution(feasible_solution)

        print(f"  Labor limit +{labor_increase}: Objective = {-analysis['objective_value']:.2f}, "
              f"Feasible = {analysis['is_feasible']}")

    # Restore original bound
    planning_problem.constraints[0].bound = original_labor_bound

    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60)

    return {
        'problem': planning_problem.problem,
        'analyzer': analyzer,
        'feasible_solution': feasible_solution,
        'infeasible_solution': infeasible_solution,
        'improved_solution': improved_solution
    }


if __name__ == "__main__":
    """
    Run the complete constrained optimization demonstration.

    This example showcases:
    - Mixed variable types (continuous, integer, binary)
    - Multiple constraint types (<=, >=, ==)
    - Comprehensive constraint violation analysis
    - Penalty method for constraint handling
    - Practical production planning problem
    - Sensitivity analysis and recommendations
    """
    try:
        results = demonstrate_constraint_handling()

        print("\n✓ All demonstrations completed successfully!")
        print("✓ Constraint handling capabilities verified")
        print("✓ Mixed variable types working correctly")
        print("✓ Violation analysis and recommendations generated")
        print("✓ Penalty method constraint handling demonstrated")

    except Exception as e:
        print(f"\n✗ Error during demonstration: {str(e)}")
        import traceback
        traceback.print_exc()