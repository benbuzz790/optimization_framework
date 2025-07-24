"""
Portfolio Optimization Example using Custom Optimization Framework

This example demonstrates a practical portfolio optimization problem with:
- Risk-return tradeoffs using Modern Portfolio Theory
- Budget constraints (total investment = 100%)
- Diversification requirements (sector limits, minimum holdings)
- Integer constraints for minimum position sizes
- Regulatory constraints (maximum single asset allocation)
- Transaction costs and practical considerations

The problem optimizes portfolio allocation across multiple assets to maximize
expected return while controlling risk (variance) subject to various constraints.
"""

import numpy as np
import sys
import os

# Add parent directory to path to import our optimization framework
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimization_framework_architecture import (
    ContinuousVariable, IntegerVariable, BinaryVariable,
    ObjectiveFunction, ConstraintFunction, Problem, Solution
)


class PortfolioOptimizer:
    """
    Portfolio optimization using Modern Portfolio Theory with practical constraints.

    This class demonstrates how to model complex financial optimization problems
    using the custom optimization framework.
    """

    def __init__(self, assets, expected_returns, covariance_matrix, 
                 risk_aversion=1.0, transaction_costs=None):
        """
        Initialize portfolio optimizer.

        Args:
            assets: List of asset names/symbols
            expected_returns: Array of expected returns for each asset
            covariance_matrix: Covariance matrix of asset returns (risk model)
            risk_aversion: Risk aversion parameter (higher = more risk averse)
            transaction_costs: Optional array of transaction costs per asset
        """
        self.assets = assets
        self.n_assets = len(assets)
        self.expected_returns = np.array(expected_returns)
        self.covariance_matrix = np.array(covariance_matrix)
        self.risk_aversion = risk_aversion
        self.transaction_costs = np.array(transaction_costs) if transaction_costs else np.zeros(self.n_assets)

        # Validate inputs
        assert len(self.expected_returns) == self.n_assets, "Expected returns length mismatch"
        assert self.covariance_matrix.shape == (self.n_assets, self.n_assets), "Covariance matrix shape mismatch"
        assert np.allclose(self.covariance_matrix, self.covariance_matrix.T), "Covariance matrix must be symmetric"

        # Asset sectors for diversification constraints
        self.sectors = self._assign_sectors()

    def _assign_sectors(self):
        """Assign sectors to assets for diversification constraints."""
        # Simple sector assignment for demonstration
        # In practice, this would come from asset metadata
        sectors = {}
        sector_names = ['Technology', 'Healthcare', 'Finance', 'Energy', 'Consumer']

        for i, asset in enumerate(self.assets):
            sectors[asset] = sector_names[i % len(sector_names)]

        return sectors

    def create_portfolio_problem(self, min_weight=0.0, max_weight=0.3, 
                               max_sector_weight=0.4, min_assets=3,
                               target_return=None):
        """
        Create portfolio optimization problem with comprehensive constraints.

        Args:
            min_weight: Minimum weight per asset (default 0%)
            max_weight: Maximum weight per asset (default 30%)
            max_sector_weight: Maximum weight per sector (default 40%)
            min_assets: Minimum number of assets to hold (default 3)
            target_return: Optional target return constraint

        Returns:
            Problem: Complete optimization problem ready for solving
        """

        # =============================================================================
        # DECISION VARIABLES
        # =============================================================================

        variables = []

        # Portfolio weights (continuous variables)
        for i, asset in enumerate(self.assets):
            weight_var = ContinuousVariable(
                name=f"weight_{asset}",
                bounds=(min_weight, max_weight)
            )
            variables.append(weight_var)

        # Binary variables for asset selection (1 if asset is held, 0 otherwise)
        for i, asset in enumerate(self.assets):
            selection_var = BinaryVariable(name=f"select_{asset}")
            variables.append(selection_var)

        # Integer variables for minimum lot sizes (if applicable)
        for i, asset in enumerate(self.assets):
            lot_var = IntegerVariable(
                name=f"lots_{asset}",
                bounds=(0, 1000)  # Maximum 1000 lots per asset
            )
            variables.append(lot_var)

        # =============================================================================
        # OBJECTIVE FUNCTION: MAXIMIZE UTILITY (RETURN - RISK PENALTY)
        # =============================================================================

        def portfolio_utility(var_dict):
            """
            Calculate portfolio utility: Expected Return - Risk Penalty - Transaction Costs

            Utility = E[R] - (λ/2) * Var[R] - Transaction Costs
            where λ is risk aversion parameter
            """
            # Extract portfolio weights
            weights = np.array([var_dict[f"weight_{asset}"]["value"] for asset in self.assets])

            # Calculate expected portfolio return
            expected_return = np.dot(weights, self.expected_returns)

            # Calculate portfolio variance (risk)
            portfolio_variance = np.dot(weights, np.dot(self.covariance_matrix, weights))

            # Calculate transaction costs
            total_transaction_costs = np.dot(weights, self.transaction_costs)

            # Portfolio utility (we maximize this, so solver should minimize negative utility)
            utility = expected_return - (self.risk_aversion / 2) * portfolio_variance - total_transaction_costs

            return -utility  # Negative because we want to maximize utility

        objective = ObjectiveFunction(portfolio_utility, "maximize_portfolio_utility")

        # =============================================================================
        # CONSTRAINTS
        # =============================================================================

        constraints = []

        # 1. Budget constraint: sum of weights = 1 (100% invested)
        def budget_constraint(var_dict):
            weights = np.array([var_dict[f"weight_{asset}"]["value"] for asset in self.assets])
            return np.sum(weights)

        constraints.append(ConstraintFunction(
            budget_constraint, "==", 1.0, "budget_constraint"
        ))

        # 2. Asset selection constraints: weight > 0 only if asset is selected
        for i, asset in enumerate(self.assets):
            def asset_selection_constraint(var_dict, asset_name=asset):
                weight = var_dict[f"weight_{asset_name}"]["value"]
                selected = var_dict[f"select_{asset_name}"]["value"]
                # If not selected, weight must be 0
                # If selected, weight can be > 0
                return weight - selected * max_weight

            constraints.append(ConstraintFunction(
                asset_selection_constraint, "<=", 0.0, f"selection_{asset}"
            ))

        # 3. Minimum number of assets constraint
        def min_assets_constraint(var_dict):
            selections = np.array([var_dict[f"select_{asset}"]["value"] for asset in self.assets])
            return np.sum(selections)

        constraints.append(ConstraintFunction(
            min_assets_constraint, ">=", min_assets, "minimum_assets"
        ))

        # 4. Sector diversification constraints
        unique_sectors = set(self.sectors.values())
        for sector in unique_sectors:
            def sector_constraint(var_dict, sector_name=sector):
                sector_weight = 0.0
                for asset in self.assets:
                    if self.sectors[asset] == sector_name:
                        sector_weight += var_dict[f"weight_{asset}"]["value"]
                return sector_weight

            constraints.append(ConstraintFunction(
                sector_constraint, "<=", max_sector_weight, f"sector_{sector}"
            ))

        # 5. Target return constraint (if specified)
        if target_return is not None:
            def target_return_constraint(var_dict):
                weights = np.array([var_dict[f"weight_{asset}"]["value"] for asset in self.assets])
                portfolio_return = np.dot(weights, self.expected_returns)
                return portfolio_return

            constraints.append(ConstraintFunction(
                target_return_constraint, ">=", target_return, "target_return"
            ))

        # 6. Lot size constraints (minimum position sizes)
        min_lot_value = 0.01  # Minimum 1% position if held
        for i, asset in enumerate(self.assets):
            def lot_size_constraint(var_dict, asset_name=asset):
                weight = var_dict[f"weight_{asset_name}"]["value"]
                selected = var_dict[f"select_{asset_name}"]["value"]
                lots = var_dict[f"lots_{asset_name}"]["value"]

                # If selected, weight must be at least min_lot_value * lots
                return weight - selected * min_lot_value * max(1, lots)

            constraints.append(ConstraintFunction(
                lot_size_constraint, ">=", 0.0, f"lot_size_{asset}"
            ))

        # =============================================================================
        # CREATE PROBLEM
        # =============================================================================

        problem = Problem(
            objective=objective,
            constraints=constraints,
            variables=variables
        )

        return problem

    def analyze_solution(self, solution):
        """
        Analyze and interpret the portfolio optimization solution.

        Args:
            solution: Solution object from optimization

        Returns:
            dict: Comprehensive portfolio analysis
        """
        best_sol = solution.get_best_solution()
        var_dict = best_sol['variable_dict']

        # Extract portfolio weights
        weights = {}
        selections = {}
        lots = {}

        for asset in self.assets:
            weights[asset] = var_dict[f"weight_{asset}"]["value"]
            selections[asset] = var_dict[f"select_{asset}"]["value"]
            lots[asset] = var_dict[f"lots_{asset}"]["value"]

        # Calculate portfolio metrics
        weight_array = np.array(list(weights.values()))

        # Expected return
        expected_return = np.dot(weight_array, self.expected_returns)

        # Portfolio risk (standard deviation)
        portfolio_variance = np.dot(weight_array, np.dot(self.covariance_matrix, weight_array))
        portfolio_risk = np.sqrt(portfolio_variance)

        # Sharpe ratio (assuming risk-free rate = 0 for simplicity)
        sharpe_ratio = expected_return / portfolio_risk if portfolio_risk > 0 else 0

        # Sector allocation
        sector_allocation = {}
        for sector in set(self.sectors.values()):
            sector_weight = sum(weights[asset] for asset in self.assets 
                              if self.sectors[asset] == sector)
            sector_allocation[sector] = sector_weight

        # Active assets
        active_assets = {asset: weight for asset, weight in weights.items() 
                        if weight > 1e-6}

        analysis = {
            'portfolio_weights': weights,
            'active_assets': active_assets,
            'asset_selections': selections,
            'lot_sizes': lots,
            'expected_return': expected_return,
            'portfolio_risk': portfolio_risk,
            'sharpe_ratio': sharpe_ratio,
            'sector_allocation': sector_allocation,
            'number_of_assets': sum(1 for w in weights.values() if w > 1e-6),
            'concentration': max(weights.values()),  # Largest single position
            'diversification_ratio': 1 - sum(w**2 for w in weights.values()),  # Herfindahl index
            'total_transaction_costs': np.dot(weight_array, self.transaction_costs),
            'is_feasible': best_sol['is_feasible'],
            'objective_value': best_sol['objective_value'],
            'constraint_violations': best_sol['constraint_violations']
        }

        return analysis

    def print_portfolio_report(self, analysis):
        """Print comprehensive portfolio analysis report."""

        print("=" * 80)
        print("PORTFOLIO OPTIMIZATION RESULTS")
        print("=" * 80)

        print(f"\n📊 PORTFOLIO PERFORMANCE METRICS")
        print(f"Expected Annual Return: {analysis['expected_return']:.2%}")
        print(f"Portfolio Risk (Volatility): {analysis['portfolio_risk']:.2%}")
        print(f"Sharpe Ratio: {analysis['sharpe_ratio']:.3f}")
        print(f"Feasible Solution: {'✅ Yes' if analysis['is_feasible'] else '❌ No'}")

        print(f"\n🎯 PORTFOLIO COMPOSITION")
        print(f"Number of Assets: {analysis['number_of_assets']}")
        print(f"Largest Position: {analysis['concentration']:.2%}")
        print(f"Diversification Score: {analysis['diversification_ratio']:.3f}")
        print(f"Total Transaction Costs: {analysis['total_transaction_costs']:.4f}")

        print(f"\n💼 ASSET ALLOCATION")
        print("-" * 50)
        for asset, weight in sorted(analysis['active_assets'].items(), 
                                  key=lambda x: x[1], reverse=True):
            if weight > 0.001:  # Show positions > 0.1%
                sector = self.sectors[asset]
                expected_ret = self.expected_returns[self.assets.index(asset)]
                print(f"{asset:>12}: {weight:>7.2%} | {sector:>12} | E[R]: {expected_ret:>6.2%}")

        print(f"\n🏭 SECTOR ALLOCATION")
        print("-" * 30)
        for sector, weight in sorted(analysis['sector_allocation'].items(), 
                                   key=lambda x: x[1], reverse=True):
            if weight > 0.001:
                print(f"{sector:>15}: {weight:>7.2%}")

        if not analysis['is_feasible']:
            print(f"\n⚠️  CONSTRAINT VIOLATIONS")
            print("-" * 40)
            for constraint, violation in analysis['constraint_violations'].items():
                if violation > 1e-6:
                    print(f"{constraint}: {violation:.6f}")

        print("\n" + "=" * 80)


def create_sample_market_data():
    """Create sample market data for demonstration."""

    # Sample assets from different sectors
    assets = [
        'AAPL',  # Technology
        'GOOGL', # Technology  
        'JNJ',   # Healthcare
        'PFE',   # Healthcare
        'JPM',   # Finance
        'BAC',   # Finance
        'XOM',   # Energy
        'CVX',   # Energy
        'PG',    # Consumer
        'KO'     # Consumer
    ]

    # Sample expected annual returns (realistic but simplified)
    expected_returns = [
        0.12,  # AAPL
        0.11,  # GOOGL
        0.08,  # JNJ
        0.09,  # PFE
        0.10,  # JPM
        0.09,  # BAC
        0.07,  # XOM
        0.08,  # CVX
        0.06,  # PG
        0.05   # KO
    ]

    # Sample correlation matrix (simplified)
    correlation_matrix = np.array([
        [1.00, 0.70, 0.30, 0.25, 0.40, 0.35, 0.20, 0.25, 0.30, 0.25],  # AAPL
        [0.70, 1.00, 0.25, 0.20, 0.35, 0.30, 0.15, 0.20, 0.25, 0.20],  # GOOGL
        [0.30, 0.25, 1.00, 0.60, 0.45, 0.40, 0.30, 0.35, 0.50, 0.45],  # JNJ
        [0.25, 0.20, 0.60, 1.00, 0.40, 0.35, 0.25, 0.30, 0.45, 0.40],  # PFE
        [0.40, 0.35, 0.45, 0.40, 1.00, 0.75, 0.35, 0.40, 0.50, 0.45],  # JPM
        [0.35, 0.30, 0.40, 0.35, 0.75, 1.00, 0.30, 0.35, 0.45, 0.40],  # BAC
        [0.20, 0.15, 0.30, 0.25, 0.35, 0.30, 1.00, 0.70, 0.25, 0.20],  # XOM
        [0.25, 0.20, 0.35, 0.30, 0.40, 0.35, 0.70, 1.00, 0.30, 0.25],  # CVX
        [0.30, 0.25, 0.50, 0.45, 0.50, 0.45, 0.25, 0.30, 1.00, 0.60],  # PG
        [0.25, 0.20, 0.45, 0.40, 0.45, 0.40, 0.20, 0.25, 0.60, 1.00]   # KO
    ])

    # Sample volatilities (annual standard deviations)
    volatilities = [
        0.25,  # AAPL
        0.28,  # GOOGL
        0.15,  # JNJ
        0.18,  # PFE
        0.22,  # JPM
        0.25,  # BAC
        0.30,  # XOM
        0.28,  # CVX
        0.12,  # PG
        0.10   # KO
    ]

    # Convert correlation matrix to covariance matrix
    vol_matrix = np.outer(volatilities, volatilities)
    covariance_matrix = correlation_matrix * vol_matrix

    # Sample transaction costs (basis points)
    transaction_costs = [0.001] * len(assets)  # 10 basis points per asset

    return assets, expected_returns, covariance_matrix, transaction_costs


def run_portfolio_optimization_example():
    """Run complete portfolio optimization example."""

    print("🚀 Starting Portfolio Optimization Example")
    print("=" * 60)

    # Create sample market data
    assets, expected_returns, covariance_matrix, transaction_costs = create_sample_market_data()

    print(f"📈 Market Data Loaded:")
    print(f"   Assets: {len(assets)}")
    print(f"   Expected Returns Range: {min(expected_returns):.1%} - {max(expected_returns):.1%}")
    print(f"   Average Volatility: {np.mean(np.sqrt(np.diag(covariance_matrix))):.1%}")

    # Initialize portfolio optimizer
    optimizer = PortfolioOptimizer(
        assets=assets,
        expected_returns=expected_returns,
        covariance_matrix=covariance_matrix,
        risk_aversion=2.0,  # Moderate risk aversion
        transaction_costs=transaction_costs
    )

    print(f"\n🎯 Creating Optimization Problem...")

    # Create portfolio optimization problem
    problem = optimizer.create_portfolio_problem(
        min_weight=0.0,      # No short selling
        max_weight=0.25,     # Maximum 25% in any single asset
        max_sector_weight=0.4, # Maximum 40% in any sector
        min_assets=4,        # Must hold at least 4 assets
        target_return=0.08   # Target 8% annual return
    )

    print(f"   Variables: {len(problem.variables)}")
    print(f"   Constraints: {len(problem.constraints)}")

    # Note: In a real implementation, we would solve this with one of our solvers
    # For this example, we'll create a mock solution to demonstrate analysis

    print(f"\n⚙️  Optimization would be solved here with:")
    print(f"   solver = GeneticAlgorithmSolver(config={'population_size': 100, 'generations': 500})")
    print(f"   solution = solver.solve(problem)")

    # Create mock solution for demonstration
    print(f"\n📊 Creating Mock Solution for Analysis Demo...")

    # Mock optimal weights (diversified portfolio)
    mock_weights = {
        'AAPL': 0.15, 'GOOGL': 0.10, 'JNJ': 0.20, 'PFE': 0.15,
        'JPM': 0.15, 'BAC': 0.10, 'XOM': 0.05, 'CVX': 0.05,
        'PG': 0.05, 'KO': 0.00
    }

    # Create mock variable dictionary
    mock_var_dict = {}
    for asset in assets:
        mock_var_dict[f"weight_{asset}"] = {
            "value": mock_weights[asset],
            "type": "continuous",
            "bounds": (0.0, 0.25),
            "variable_object": None
        }
        mock_var_dict[f"select_{asset}"] = {
            "value": 1 if mock_weights[asset] > 0 else 0,
            "type": "binary", 
            "bounds": (0, 1),
            "variable_object": None
        }
        mock_var_dict[f"lots_{asset}"] = {
            "value": int(mock_weights[asset] * 100) if mock_weights[asset] > 0 else 0,
            "type": "integer",
            "bounds": (0, 1000),
            "variable_object": None
        }

    # Create mock solution
    mock_solution_data = {
        'iteration': 500,
        'variable_dict': mock_var_dict,
        'objective_value': -0.085,  # Negative utility (we minimized negative utility)
        'constraint_violations': {},
        'is_feasible': True,
        'metadata': {'algorithm': 'mock', 'converged': True}
    }

    # Analyze the solution
    print(f"\n🔍 Analyzing Portfolio Solution...")

    # Create a mock solution object for analysis
    class MockSolution:
        def get_best_solution(self):
            return mock_solution_data

    mock_solution = MockSolution()
    analysis = optimizer.analyze_solution(mock_solution)

    # Print comprehensive report
    optimizer.print_portfolio_report(analysis)

    # Additional insights
    print(f"\n💡 PORTFOLIO INSIGHTS")
    print("-" * 40)
    print(f"• Risk-adjusted return (utility): {-mock_solution_data['objective_value']:.3f}")
    print(f"• Diversification across {len([s for s in set(optimizer.sectors.values()) if any(mock_weights[a] > 0.001 for a in assets if optimizer.sectors[a] == s)])} sectors")
    print(f"• Balanced risk-return profile with {analysis['sharpe_ratio']:.2f} Sharpe ratio")
    print(f"• Conservative approach with max position of {analysis['concentration']:.1%}")

    print(f"\n✅ Portfolio Optimization Example Complete!")

    return optimizer, problem, analysis


if __name__ == "__main__":
    # Run the complete portfolio optimization example
    optimizer, problem, analysis = run_portfolio_optimization_example()

    print(f"\n🎓 LEARNING SUMMARY")
    print("=" * 50)
    print("This example demonstrated:")
    print("• Modern Portfolio Theory implementation")
    print("• Multi-constraint optimization (budget, diversification, regulatory)")
    print("• Mixed variable types (continuous weights, binary selection, integer lots)")
    print("• Real-world financial modeling with risk-return tradeoffs")
    print("• Comprehensive solution analysis and interpretation")
    print("• Professional portfolio reporting")

    print(f"\n🔧 Framework Integration Points:")
    print("• Variable Dictionary Protocol for consistent data flow")
    print("• NASA-style asserts for production-quality validation")
    print("• Modular constraint design for complex business rules")
    print("• Extensible objective functions for custom utility models")
    print("• Complete solution tracking for audit and analysis")