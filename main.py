"""
Differential Evolution Example.

This script demonstrates the usage of the Differential Evolution optimizer
on standard benchmark functions.
"""

from differential_evolution import DifferentialEvolution
from funcs import ackley, sphere, rastrigin, ackley_2d
from visualization import plot_optimization_result, create_animation


def main():
    """Run Differential Evolution on the Ackley function."""
    # Define search bounds for 2D optimization
    bounds = [(-5, 5), (-5, 5)]

    # Create optimizer
    de = DifferentialEvolution(
        bounds=bounds,
        population_size=50,
        mutation_factor=0.8,
        crossover_rate=0.7,
        seed=42,  # For reproducibility
    )

    # Run optimization on Ackley function
    print("Optimizing Ackley function...")
    print("-" * 50)

    best_solution, best_fitness = de.optimize(
        objective_func=ackley,
        max_iterations=200,
        tolerance=1e-8,
        verbose=True,
    )

    print("-" * 50)
    print(f"Best solution found: {best_solution}")
    print(f"Best fitness: {best_fitness:.6e}")
    print(f"Expected optimum: [0, 0] with fitness ≈ 0")

    # Visualize results
    print("\nGenerating visualization...")
    fig = plot_optimization_result(
        func=ackley_2d,
        history=de.get_history(),
        bounds=(-5, 5),
        title="Ackley Function Optimization",
        save_path="optimization_result.png",
    )
    print("Saved: optimization_result.png")

    # Create animation (optional - takes longer)
    print("\nCreating animation...")
    create_animation(
        func=ackley_2d,
        history=de.get_history(),
        bounds=(-5, 5),
        save_path="optimization_animation.gif",
        fps=10,
    )


def benchmark_all():
    """Run optimization on multiple benchmark functions."""
    functions = [
        ("Sphere", sphere, [(-5.12, 5.12), (-5.12, 5.12)]),
        ("Ackley", ackley, [(-5, 5), (-5, 5)]),
        ("Rastrigin", rastrigin, [(-5.12, 5.12), (-5.12, 5.12)]),
    ]

    print("=" * 60)
    print("Differential Evolution Benchmark Results")
    print("=" * 60)

    for name, func, bounds in functions:
        de = DifferentialEvolution(
            bounds=bounds,
            population_size=50,
            mutation_factor=0.8,
            crossover_rate=0.7,
            seed=42,
        )

        best_solution, best_fitness = de.optimize(
            objective_func=func,
            max_iterations=500,
            tolerance=1e-10,
            verbose=False,
        )

        print(f"\n{name} Function:")
        print(f"  Best solution: [{best_solution[0]:.6f}, {best_solution[1]:.6f}]")
        print(f"  Best fitness:  {best_fitness:.6e}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
    print("\n")
    benchmark_all()
