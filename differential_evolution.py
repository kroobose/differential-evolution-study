"""
Differential Evolution Algorithm Implementation.

This module provides a clean, well-documented implementation of the
Differential Evolution (DE) optimization algorithm.

References:
    Storn, R., & Price, K. (1997). Differential Evolution – A Simple and
    Efficient Heuristic for Global Optimization over Continuous Spaces.
    Journal of Global Optimization, 11(4), 341-359.
"""

import numpy as np
from typing import Callable, List, Tuple, Optional


class DifferentialEvolution:
    """
    Differential Evolution optimizer for continuous optimization problems.

    This implementation uses the DE/rand/1/bin strategy:
    - rand: Base vector is randomly selected
    - 1: One difference vector is used
    - bin: Binomial crossover

    Attributes:
        bounds: List of (min, max) tuples for each dimension.
        population_size: Number of candidate solutions in the population.
        mutation_factor: Scaling factor F for mutation (typically 0.5-1.0).
        crossover_rate: Crossover probability CR (typically 0.7-0.9).
        seed: Random seed for reproducibility.
    """

    MIN_POPULATION_SIZE = 4  # Minimum required for DE/rand/1 mutation

    def __init__(
        self,
        bounds: List[Tuple[float, float]],
        population_size: int = 50,
        mutation_factor: float = 0.8,
        crossover_rate: float = 0.7,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize the Differential Evolution optimizer.

        Args:
            bounds: List of (min, max) tuples defining the search space
                    for each dimension.
            population_size: Number of individuals in the population.
                            Must be at least 4 for DE/rand/1 mutation.
            mutation_factor: Scaling factor F for differential mutation.
            crossover_rate: Probability of crossover CR for each dimension.
            seed: Random seed for reproducibility. If None, results will vary.

        Raises:
            ValueError: If population_size < 4 or bounds are invalid.
        """
        # Validate population size
        if population_size < self.MIN_POPULATION_SIZE:
            raise ValueError(
                f"population_size must be at least {self.MIN_POPULATION_SIZE} "
                f"for DE/rand/1 mutation, got {population_size}"
            )

        self.bounds = np.array(bounds)
        self.dimensions = len(bounds)
        self.population_size = population_size
        self.mutation_factor = mutation_factor
        self.crossover_rate = crossover_rate

        # Create random number generator for reproducibility
        # Using Generator instead of global np.random.seed() for thread-safety
        self._rng = np.random.default_rng(seed)

        # Initialize population and fitness tracking
        self.population: Optional[np.ndarray] = None
        self.fitness: Optional[np.ndarray] = None
        self.best_solution: Optional[np.ndarray] = None
        self.best_fitness: float = np.inf
        self.history: List[np.ndarray] = []

    def _initialize_population(self) -> np.ndarray:
        """
        Initialize population with random solutions within bounds.

        Returns:
            Array of shape (population_size, dimensions) with random
            solutions uniformly distributed within the specified bounds.
        """
        lower_bounds = self.bounds[:, 0]
        upper_bounds = self.bounds[:, 1]

        # Generate random values in [0, 1] and scale to bounds
        population = self._rng.random((self.population_size, self.dimensions))
        population = lower_bounds + population * (upper_bounds - lower_bounds)

        return population

    def _mutate(self, target_idx: int) -> np.ndarray:
        """
        Create mutant vector using DE/rand/1 strategy.

        The mutation formula is: v = x_r1 + F * (x_r2 - x_r3)
        where r1, r2, r3 are distinct random indices different from target_idx.

        Args:
            target_idx: Index of the target vector (excluded from selection).

        Returns:
            Mutant vector of shape (dimensions,).
        """
        # Select 3 distinct random indices, all different from target_idx
        indices = list(range(self.population_size))
        indices.remove(target_idx)
        r1, r2, r3 = self._rng.choice(indices, size=3, replace=False)

        # DE/rand/1 mutation
        mutant = (
            self.population[r1]
            + self.mutation_factor * (self.population[r2] - self.population[r3])
        )

        return mutant

    def _crossover(self, target: np.ndarray, mutant: np.ndarray) -> np.ndarray:
        """
        Perform binomial crossover between target and mutant vectors.

        At least one dimension is guaranteed to come from the mutant vector
        (the j_rand dimension).

        Args:
            target: Original target vector.
            mutant: Mutant vector created by mutation.

        Returns:
            Trial vector combining elements from target and mutant.
        """
        trial = np.copy(target)

        # Ensure at least one dimension comes from mutant (j_rand)
        j_rand = self._rng.integers(0, self.dimensions)

        for j in range(self.dimensions):
            if self._rng.random() < self.crossover_rate or j == j_rand:
                trial[j] = mutant[j]

        return trial

    def _enforce_bounds(self, vector: np.ndarray) -> np.ndarray:
        """
        Clip vector values to stay within bounds.

        Args:
            vector: Solution vector to constrain.

        Returns:
            Vector with values clipped to the specified bounds.
        """
        lower_bounds = self.bounds[:, 0]
        upper_bounds = self.bounds[:, 1]
        return np.clip(vector, lower_bounds, upper_bounds)

    def _select(
        self,
        target: np.ndarray,
        trial: np.ndarray,
        target_fitness: float,
        trial_fitness: float,
    ) -> Tuple[np.ndarray, float]:
        """
        Greedy selection between target and trial vectors.

        Args:
            target: Original target vector.
            trial: Trial vector (candidate replacement).
            target_fitness: Fitness value of target.
            trial_fitness: Fitness value of trial.

        Returns:
            Tuple of (selected_vector, selected_fitness).
        """
        if trial_fitness <= target_fitness:
            return trial, trial_fitness
        else:
            return target, target_fitness

    def optimize(
        self,
        objective_func: Callable[[np.ndarray], float],
        max_iterations: int = 1000,
        tolerance: float = 1e-8,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, float]:
        """
        Run the Differential Evolution optimization.

        Args:
            objective_func: Function to minimize. Takes a 1D numpy array
                           and returns a scalar fitness value.
            max_iterations: Maximum number of generations.
            tolerance: Convergence tolerance. Stops if best fitness
                       improvement is below this threshold.
            verbose: If True, print progress information.

        Returns:
            Tuple of (best_solution, best_fitness).
        """
        # Initialize population
        self.population = self._initialize_population()
        self.fitness = np.array(
            [objective_func(ind) for ind in self.population]
        )

        # Track best solution
        best_idx = np.argmin(self.fitness)
        self.best_solution = self.population[best_idx].copy()
        self.best_fitness = self.fitness[best_idx]
        self.history = [self.population.copy()]

        stagnation_count = 0
        stagnation_limit = 20  # Number of iterations without improvement to trigger convergence

        for iteration in range(max_iterations):
            new_population = np.zeros_like(self.population)
            new_fitness = np.zeros(self.population_size)

            for i in range(self.population_size):
                # Mutation
                mutant = self._mutate(i)

                # Crossover
                trial = self._crossover(self.population[i], mutant)

                # Enforce bounds
                trial = self._enforce_bounds(trial)

                # Evaluate trial
                trial_fitness = objective_func(trial)

                # Selection
                new_population[i], new_fitness[i] = self._select(
                    self.population[i],
                    trial,
                    self.fitness[i],
                    trial_fitness,
                )

            # Update population
            self.population = new_population
            self.fitness = new_fitness
            self.history.append(self.population.copy())

            # Update best solution
            best_idx = np.argmin(self.fitness)
            prev_best = self.best_fitness
            if self.fitness[best_idx] < self.best_fitness:
                self.best_solution = self.population[best_idx].copy()
                self.best_fitness = self.fitness[best_idx]

            # Check convergence (stagnation-based)
            improvement = prev_best - self.best_fitness
            if improvement < tolerance:
                stagnation_count += 1
            else:
                stagnation_count = 0

            if stagnation_count >= stagnation_limit:
                if verbose:
                    print(f"Converged at iteration {iteration} (no improvement for {stagnation_limit} iterations)")
                break

            if verbose and iteration % 100 == 0:
                print(
                    f"Iteration {iteration}: Best fitness = {self.best_fitness:.6e}"
                )

        return self.best_solution, self.best_fitness

    def get_history(self) -> List[np.ndarray]:
        """
        Get the optimization history.

        Returns:
            List of population arrays at each iteration.
        """
        return self.history
