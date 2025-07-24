"""
Genetic Algorithm Solver Implementation

Population-based evolutionary optimization algorithm using selection,
crossover, and mutation operations to explore the solution space.
"""

import random
from typing import Dict, Any, List, Tuple
from .base_solver import BaseSolver
from ..utils.assertions import production_assert
from ..variables import BinaryVariable, IntegerVariable, ContinuousVariable
from ..solutions import Solution


class GeneticAlgorithmSolver(BaseSolver):
    """
    Genetic algorithm optimization solver.

    Uses population-based evolutionary search with selection, crossover,
    and mutation operations to explore the solution space.
    """

    def __init__(self, config: Dict = None):
        """Initialize GeneticAlgorithmSolver with configuration validation."""
        super().__init__(config)

    def _validate_config(self, config: Dict) -> Dict:
        """Validate and set default configuration for genetic algorithm."""
        validated_config = config.copy()

        # Set defaults
        defaults = {
            'population_size': 50,
            'generations': 100,
            'mutation_rate': 0.1,
            'crossover_rate': 0.8,
            'selection_method': 'tournament',
            'tournament_size': 3,
            'elitism_count': 2,
            'random_seed': None
        }

        for key, default_value in defaults.items():
            if key not in validated_config:
                validated_config[key] = default_value

        # Validate ranges
        production_assert(validated_config['population_size'] > 1,
                         "population_size must be > 1")
        production_assert(validated_config['generations'] > 0,
                         "generations must be positive")
        production_assert(0 <= validated_config['mutation_rate'] <= 1,
                         "mutation_rate must be in [0, 1]")
        production_assert(0 <= validated_config['crossover_rate'] <= 1,
                         "crossover_rate must be in [0, 1]")
        production_assert(validated_config['selection_method'] in ['tournament', 'roulette'],
                         "selection_method must be 'tournament' or 'roulette'")

        return validated_config

    def solve(self, problem, initial_guess: Dict[str, Any] = None):
        """Solve optimization problem using genetic algorithm."""
        # Call parent validation
        super().solve(problem, initial_guess)

        # Create solution tracking object
        solution = Solution(problem, "GeneticAlgorithmSolver")

        # Set random seed if specified
        if self.config['random_seed'] is not None:
            random.seed(self.config['random_seed'])

        # Initialize population
        population = self._initialize_population(problem, initial_guess)

        # Main evolutionary loop
        for generation in range(self.config['generations']):
            # Evaluate population fitness
            fitness_scores = []
            for individual in population:
                objective = problem.evaluate_objective(individual)
                violations = problem.get_constraint_violations(individual)
                total_violation = sum(violations.values())

                # Simple penalty method for constraints
                penalized_objective = objective + 1000 * total_violation

                fitness_scores.append({
                    'objective': objective,
                    'penalized': penalized_objective,
                    'violations': violations,
                    'feasible': total_violation == 0
                })

            # Find best individual in current generation
            best_idx = min(range(len(fitness_scores)),
                          key=lambda i: fitness_scores[i]['penalized'])
            best_individual = population[best_idx]
            best_fitness = fitness_scores[best_idx]

            # Add best individual to solution history
            solution.add_iteration(
                variable_dict=best_individual,
                objective_value=best_fitness['objective'],
                constraint_violations=best_fitness['violations'],
                metadata={
                    'generation': generation,
                    'population_size': len(population),
                    'best_fitness': best_fitness['penalized'],
                    'feasible_count': sum(1 for f in fitness_scores if f['feasible'])
                }
            )

            # Create next generation
            new_population = []

            # Elitism: keep best individuals
            elite_indices = sorted(range(len(fitness_scores)),
                                 key=lambda i: fitness_scores[i]['penalized'])[:self.config['elitism_count']]
            for idx in elite_indices:
                new_population.append(population[idx].copy())

            # Generate offspring
            while len(new_population) < self.config['population_size']:
                # Selection
                parent1 = self._select_parent(population, fitness_scores)
                parent2 = self._select_parent(population, fitness_scores)

                # Crossover
                if random.random() < self.config['crossover_rate']:
                    child1, child2 = self._crossover(problem, parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()

                # Mutation
                if random.random() < self.config['mutation_rate']:
                    child1 = self._mutate(problem, child1)
                if random.random() < self.config['mutation_rate']:
                    child2 = self._mutate(problem, child2)

                new_population.extend([child1, child2])

            # Trim to exact population size
            population = new_population[:self.config['population_size']]

        return solution

    def _initialize_population(self, problem, initial_guess: Dict[str, Any] = None) -> List[Dict]:
        """Initialize population of individuals."""
        population = []

        # Add initial guess if provided
        if initial_guess is not None:
            population.append(problem.create_variable_dict(initial_guess))

        # Generate random individuals
        while len(population) < self.config['population_size']:
            individual_values = {}

            for variable in problem.variables:
                if isinstance(variable, BinaryVariable):
                    individual_values[variable.name] = random.choice([0, 1])
                elif isinstance(variable, IntegerVariable):
                    if variable.bounds:
                        individual_values[variable.name] = random.randint(
                            variable.bounds[0], variable.bounds[1])
                    else:
                        individual_values[variable.name] = random.randint(0, 100)
                else:  # ContinuousVariable
                    if variable.bounds:
                        individual_values[variable.name] = random.uniform(
                            variable.bounds[0], variable.bounds[1])
                    else:
                        individual_values[variable.name] = random.uniform(-10, 10)

            population.append(problem.create_variable_dict(individual_values))

        return population

    def _select_parent(self, population: List[Dict], fitness_scores: List[Dict]) -> Dict:
        """Select parent for reproduction."""
        if self.config['selection_method'] == 'tournament':
            # Tournament selection
            tournament_size = min(self.config['tournament_size'], len(population))
            tournament_indices = random.sample(range(len(population)), tournament_size)
            best_idx = min(tournament_indices,
                          key=lambda i: fitness_scores[i]['penalized'])
            return population[best_idx].copy()

        else:  # roulette selection
            # Roulette wheel selection (fitness proportionate)
            # Convert minimization to maximization
            max_fitness = max(f['penalized'] for f in fitness_scores)
            weights = [max_fitness - f['penalized'] + 1 for f in fitness_scores]

            total_weight = sum(weights)
            if total_weight == 0:
                return random.choice(population).copy()

            pick = random.uniform(0, total_weight)
            current = 0
            for i, weight in enumerate(weights):
                current += weight
                if current >= pick:
                    return population[i].copy()

            return population[-1].copy()

    def _crossover(self, problem, parent1: Dict, parent2: Dict) -> Tuple[Dict, Dict]:
        """Perform crossover between two parents."""
        child1_values = {}
        child2_values = {}

        for variable in problem.variables:
            var_name = variable.name
            val1 = parent1[var_name]['value']
            val2 = parent2[var_name]['value']

            if isinstance(variable, BinaryVariable):
                # Single point crossover for binary
                if random.random() < 0.5:
                    child1_values[var_name] = val1
                    child2_values[var_name] = val2
                else:
                    child1_values[var_name] = val2
                    child2_values[var_name] = val1

            elif isinstance(variable, IntegerVariable):
                # Arithmetic crossover for integers
                alpha = random.random()
                new_val1 = int(alpha * val1 + (1 - alpha) * val2)
                new_val2 = int(alpha * val2 + (1 - alpha) * val1)

                if variable.bounds:
                    new_val1 = variable.clip_to_bounds(new_val1)
                    new_val2 = variable.clip_to_bounds(new_val2)

                child1_values[var_name] = new_val1
                child2_values[var_name] = new_val2

            else:  # ContinuousVariable
                # Arithmetic crossover for continuous
                alpha = random.random()
                new_val1 = alpha * val1 + (1 - alpha) * val2
                new_val2 = alpha * val2 + (1 - alpha) * val1

                if variable.bounds:
                    new_val1 = variable.clip_to_bounds(new_val1)
                    new_val2 = variable.clip_to_bounds(new_val2)

                child1_values[var_name] = new_val1
                child2_values[var_name] = new_val2

        child1 = problem.create_variable_dict(child1_values)
        child2 = problem.create_variable_dict(child2_values)

        return child1, child2

    def _mutate(self, problem, individual: Dict) -> Dict:
        """Mutate an individual."""
        mutated_values = {}

        for variable in problem.variables:
            var_name = variable.name
            current_value = individual[var_name]['value']

            if isinstance(variable, BinaryVariable):
                # Bit flip mutation
                mutated_values[var_name] = 1 - current_value

            elif isinstance(variable, IntegerVariable):
                # Gaussian mutation for integers
                if variable.bounds:
                    range_size = variable.bounds[1] - variable.bounds[0]
                    mutation_strength = max(1, int(range_size * 0.1))
                else:
                    mutation_strength = max(1, int(abs(current_value) * 0.1))

                change = random.randint(-mutation_strength, mutation_strength)
                new_value = current_value + change

                if variable.bounds:
                    new_value = variable.clip_to_bounds(new_value)

                mutated_values[var_name] = int(new_value)

            else:  # ContinuousVariable
                # Gaussian mutation for continuous
                if variable.bounds:
                    range_size = variable.bounds[1] - variable.bounds[0]
                    if range_size != float('inf'):
                        mutation_strength = range_size * 0.1
                    else:
                        mutation_strength = abs(current_value) * 0.1 + 1.0
                else:
                    mutation_strength = abs(current_value) * 0.1 + 1.0

                change = random.gauss(0, mutation_strength)
                new_value = current_value + change

                if variable.bounds:
                    new_value = variable.clip_to_bounds(new_value)

                mutated_values[var_name] = new_value

        return problem.create_variable_dict(mutated_values)