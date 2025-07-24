#!/usr/bin/env python3
"""
Comprehensive Test Runner for Optimization Framework

This script runs all tests for the optimization framework and provides
detailed reporting on test results, coverage, and performance.

Usage:
    python run_all_tests.py              # Run all tests
    python run_all_tests.py --verbose    # Verbose output
    python run_all_tests.py --module variables  # Run specific module tests
"""

import sys
import unittest
import time
import argparse
from io import StringIO

try:
    from optimization_framework import (
        ContinuousVariable, IntegerVariable, BinaryVariable,
        ObjectiveFunction, ConstraintFunction, Problem,
        GreedySearchSolver, GeneticAlgorithmSolver, SimulatedAnnealingSolver
    )
    FRAMEWORK_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import optimization framework: {e}")
    FRAMEWORK_AVAILABLE = False


def run_test_module(module_name, verbose=False):
    """
    Run tests for a specific module.

    Args:
        module_name: Name of test module (e.g., 'variables', 'solvers')
        verbose: Whether to show verbose output

    Returns:
        tuple: (success_count, failure_count, error_count, total_time)
    """
    print(f"\n{'='*60}")
    print(f"TESTING MODULE: {module_name.upper()}")
    print(f"{'='*60}")

    # Import the test module
    try:
        test_module = __import__(f'tests.test_{module_name}', fromlist=[''])
    except ImportError as e:
        print(f"ERROR: Could not import test module 'test_{module_name}': {e}")
        return 0, 0, 1, 0

    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(test_module)

    # Run tests with custom result handler
    stream = StringIO() if not verbose else sys.stdout
    runner = unittest.TextTestRunner(
        stream=stream,
        verbosity=2 if verbose else 1,
        buffer=True
    )

    start_time = time.time()
    result = runner.run(suite)
    end_time = time.time()

    # Print results
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    successes = total_tests - failures - errors
    test_time = end_time - start_time

    print(f"\nResults for {module_name}:")
    print(f"  Tests run: {total_tests}")
    print(f"  Successes: {successes}")
    print(f"  Failures: {failures}")
    print(f"  Errors: {errors}")
    print(f"  Time: {test_time:.3f}s")

    # Show failure details if not verbose
    if not verbose and (failures > 0 or errors > 0):
        print(f"\nFailure/Error Details:")
        for test, traceback in result.failures + result.errors:
            print(f"  FAIL: {test}")
            print(f"    {traceback.split('AssertionError:')[-1].strip() if 'AssertionError:' in traceback else 'See full output for details'}")

    return successes, failures, errors, test_time


def run_integration_tests(verbose=False):
    """
    Run integration tests that test framework components together.

    Args:
        verbose: Whether to show verbose output

    Returns:
        tuple: (success_count, failure_count, error_count, total_time)
    """
    print(f"\n{'='*60}")
    print("INTEGRATION TESTS")
    print(f"{'='*60}")

    if not FRAMEWORK_AVAILABLE:
        print("Framework not available for integration tests")
        return 0, 0, 1, 0

    start_time = time.time()

    try:
        # Test 1: Simple end-to-end optimization
        print("Test 1: End-to-end optimization...")

        x = ContinuousVariable("x", bounds=(-5, 5))
        y = IntegerVariable("y", bounds=(0, 10))

        def obj_func(var_dict):
            return var_dict["x"]["value"]**2 + var_dict["y"]["value"]**2

        objective = ObjectiveFunction(obj_func, "quadratic")
        problem = Problem(objective, variables=[x, y])

        # Test all solvers
        solvers = [
            ("Greedy", GreedySearchSolver({'max_iterations': 10})),
            ("Genetic", GeneticAlgorithmSolver({'population_size': 10, 'generations': 5})),
            ("Annealing", SimulatedAnnealingSolver({'max_iterations': 20}))
        ]

        solver_results = []
        for name, solver in solvers:
            try:
                solution = solver.solve(problem, initial_guess={"x": 2.0, "y": 3})
                best = solution.get_best_solution()
                solver_results.append((name, best['objective_value'], "SUCCESS"))
                print(f"  {name}: {best['objective_value']:.3f} - SUCCESS")
            except Exception as e:
                solver_results.append((name, None, f"ERROR: {str(e)}"))
                print(f"  {name}: ERROR - {str(e)}")

        # Test 2: Constrained optimization
        print("\nTest 2: Constrained optimization...")

        def constraint_func(var_dict):
            return var_dict["x"]["value"] + var_dict["y"]["value"]

        constraint = ConstraintFunction(constraint_func, "<=", 5.0, "sum_limit")
        constrained_problem = Problem(objective, constraints=[constraint], variables=[x, y])

        solver = GeneticAlgorithmSolver({'population_size': 15, 'generations': 10})
        solution = solver.solve(constrained_problem)
        best = solution.get_best_solution()

        print(f"  Best objective: {best['objective_value']:.3f}")
        print(f"  Feasible: {best['is_feasible']}")
        print(f"  Constraint violations: {best['constraint_violations']}")

        # Test 3: Mixed variable types
        print("\nTest 3: Mixed variable types...")

        z = BinaryVariable("z")

        def mixed_obj(var_dict):
            x_val = var_dict["x"]["value"]
            y_val = var_dict["y"]["value"]
            z_val = var_dict["z"]["value"]
            return x_val**2 + y_val**2 + z_val * 10

        mixed_objective = ObjectiveFunction(mixed_obj, "mixed")
        mixed_problem = Problem(mixed_objective, variables=[x, y, z])

        solver = SimulatedAnnealingSolver({'max_iterations': 30})
        solution = solver.solve(mixed_problem)
        best = solution.get_best_solution()

        print(f"  Best objective: {best['objective_value']:.3f}")
        print(f"  Variables: x={best['variable_dict']['x']['value']:.2f}, y={best['variable_dict']['y']['value']}, z={best['variable_dict']['z']['value']}")

        end_time = time.time()
        test_time = end_time - start_time

        print(f"\nIntegration tests completed successfully in {test_time:.3f}s")
        return 3, 0, 0, test_time

    except Exception as e:
        end_time = time.time()
        test_time = end_time - start_time
        print(f"Integration test failed: {str(e)}")
        if verbose:
            import traceback
            traceback.print_exc()
        return 0, 0, 1, test_time


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description='Run optimization framework tests')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show verbose test output')
    parser.add_argument('--module', '-m', type=str,
                       help='Run tests for specific module only')
    parser.add_argument('--no-integration', action='store_true',
                       help='Skip integration tests')

    args = parser.parse_args()

    print("OPTIMIZATION FRAMEWORK TEST SUITE")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print(f"Test mode: {'Verbose' if args.verbose else 'Standard'}")

    # Test modules to run
    test_modules = ['variables', 'functions', 'problem', 'solution', 'solvers']

    if args.module:
        if args.module in test_modules:
            test_modules = [args.module]
        else:
            print(f"ERROR: Unknown module '{args.module}'. Available: {', '.join(test_modules)}")
            return 1

    # Run unit tests
    total_successes = 0
    total_failures = 0
    total_errors = 0
    total_time = 0

    for module in test_modules:
        successes, failures, errors, test_time = run_test_module(module, args.verbose)
        total_successes += successes
        total_failures += failures
        total_errors += errors
        total_time += test_time

    # Run integration tests
    if not args.no_integration and not args.module:
        int_successes, int_failures, int_errors, int_time = run_integration_tests(args.verbose)
        total_successes += int_successes
        total_failures += int_failures
        total_errors += int_errors
        total_time += int_time

    # Print final summary
    print(f"\n{'='*60}")
    print("FINAL TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Total tests run: {total_successes + total_failures + total_errors}")
    print(f"Successes: {total_successes}")
    print(f"Failures: {total_failures}")
    print(f"Errors: {total_errors}")
    print(f"Total time: {total_time:.3f}s")

    if total_failures == 0 and total_errors == 0:
        print("\n🎉 ALL TESTS PASSED! 🎉")
        print("The optimization framework is working correctly.")
        return 0
    else:
        print(f"\n❌ TESTS FAILED: {total_failures} failures, {total_errors} errors")
        print("Please review the test output above for details.")
        return 1


if __name__ == '__main__':
    sys.exit(main())