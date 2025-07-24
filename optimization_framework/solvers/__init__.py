"""
Solvers Module - Optimization Algorithms

This module provides the complete solver infrastructure including:
- BaseSolver: Abstract base class for all optimization algorithms
- GreedySearchSolver: Local search optimization
- GeneticAlgorithmSolver: Population-based evolutionary optimization  
- SimulatedAnnealingSolver: Temperature-based probabilistic search
- SolverConfig: Configuration management utilities

All solvers follow the variable dictionary protocol and provide
comprehensive solution tracking with NASA-style validation.
"""

from .base_solver import BaseSolver
from .greedy_search import GreedySearchSolver
from .genetic_algorithm import GeneticAlgorithmSolver
from .simulated_annealing import SimulatedAnnealingSolver
from .solver_config import SolverConfig

__all__ = [
    'BaseSolver',
    'GreedySearchSolver', 
    'GeneticAlgorithmSolver',
    'SimulatedAnnealingSolver',
    'SolverConfig'
]