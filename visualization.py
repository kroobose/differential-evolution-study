"""
Visualization utilities for Differential Evolution optimization.

This module provides functions to visualize optimization progress including
2D contour plots, 3D surface plots, and animations.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from typing import Callable, List, Optional, Tuple


def plot_contour_2d(
    func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    bounds: Tuple[float, float] = (-5, 5),
    resolution: int = 400,
    levels: int = 50,
    ax: Optional[plt.Axes] = None,
    cmap: str = "viridis",
) -> plt.Axes:
    """
    Plot 2D contour of an objective function.

    Args:
        func: 2D function to plot (accepts meshgrid arrays).
        bounds: (min, max) bounds for both x and y axes.
        resolution: Number of points along each axis.
        levels: Number of contour levels.
        ax: Matplotlib axes to plot on. Creates new figure if None.
        cmap: Colormap name.

    Returns:
        Matplotlib axes object.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    x = np.linspace(bounds[0], bounds[1], resolution)
    y = np.linspace(bounds[0], bounds[1], resolution)
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)

    ax.contour(X, Y, Z, levels=levels, cmap=cmap)
    ax.scatter(0, 0, color="red", s=100, marker="*", label="Global Minimum", zorder=5)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.legend()

    return ax


def plot_surface_3d(
    func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    bounds: Tuple[float, float] = (-5, 5),
    resolution: int = 100,
    ax: Optional[plt.Axes] = None,
    cmap: str = "viridis",
    alpha: float = 0.7,
) -> plt.Axes:
    """
    Plot 3D surface of an objective function.

    Args:
        func: 2D function to plot (accepts meshgrid arrays).
        bounds: (min, max) bounds for both x and y axes.
        resolution: Number of points along each axis.
        ax: Matplotlib 3D axes to plot on. Creates new figure if None.
        cmap: Colormap name.
        alpha: Surface transparency.

    Returns:
        Matplotlib 3D axes object.
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

    x = np.linspace(bounds[0], bounds[1], resolution)
    y = np.linspace(bounds[0], bounds[1], resolution)
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)

    ax.plot_surface(X, Y, Z, cmap=cmap, alpha=alpha, edgecolor="none")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("$f(x)$")

    return ax


def plot_optimization_result(
    func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    history: List[np.ndarray],
    bounds: Tuple[float, float] = (-5, 5),
    title: str = "Differential Evolution Optimization",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot the optimization result with both 3D surface and 2D contour views.

    Args:
        func: 2D objective function for plotting.
        history: List of population arrays from optimization.
        bounds: (min, max) bounds for plotting.
        title: Plot title.
        save_path: Path to save the figure. If None, displays interactively.

    Returns:
        Matplotlib figure object.
    """
    fig = plt.figure(figsize=(14, 6))

    # 3D surface plot
    ax1 = fig.add_subplot(121, projection="3d")
    plot_surface_3d(func, bounds=bounds, ax=ax1, alpha=0.3)
    ax1.set_title(f"{title} - 3D View")

    # Plot final population
    final_pop = history[-1]
    x_final = final_pop[:, 0]
    y_final = final_pop[:, 1]
    z_final = func(x_final, y_final)
    ax1.scatter(x_final, y_final, z_final, c="red", s=20, label="Final Population")

    # 2D contour plot
    ax2 = fig.add_subplot(122)
    plot_contour_2d(func, bounds=bounds, ax=ax2)
    ax2.set_title(f"{title} - 2D Contour")

    # Plot optimization trajectory (best individual from each generation)
    if len(history) > 1:
        trajectory = []
        for pop in history:
            # Find best individual in population
            fitness = [func(pop[i, 0], pop[i, 1]) for i in range(len(pop))]
            best_idx = np.argmin(fitness)
            trajectory.append(pop[best_idx])

        trajectory = np.array(trajectory)
        ax2.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            "r.-",
            linewidth=1,
            markersize=3,
            label="Best Solution Trajectory",
        )

    # Plot final population
    ax2.scatter(x_final, y_final, c="blue", s=10, alpha=0.5, label="Final Population")
    ax2.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def create_animation(
    func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    history: List[np.ndarray],
    bounds: Tuple[float, float] = (-5, 5),
    save_path: str = "optimization.gif",
    fps: int = 10,
    interval: int = 200,
) -> None:
    """
    Create an animation of the optimization process.

    Args:
        func: 2D objective function for plotting.
        history: List of population arrays from optimization.
        bounds: (min, max) bounds for plotting.
        save_path: Path to save the GIF animation.
        fps: Frames per second for the animation.
        interval: Delay between frames in milliseconds.
    """
    # Create figure
    fig = plt.figure(figsize=(14, 6))

    # Create meshgrid for surface/contour
    resolution = 100
    x = np.linspace(bounds[0], bounds[1], resolution)
    y = np.linspace(bounds[0], bounds[1], resolution)
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)

    # 3D subplot
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.plot_surface(X, Y, Z, cmap="viridis", alpha=0.3, edgecolor="none")
    ax1.set_xlabel("$x_1$")
    ax1.set_ylabel("$x_2$")
    ax1.set_zlabel("$f(x)$")
    ax1.set_title("3D View")

    # 2D subplot
    ax2 = fig.add_subplot(122)
    ax2.contour(X, Y, Z, levels=50, cmap="viridis")
    ax2.scatter(0, 0, color="red", s=100, marker="*", label="Global Minimum", zorder=5)
    ax2.set_xlabel("$x_1$")
    ax2.set_ylabel("$x_2$")
    ax2.set_title("2D Contour")

    # Initialize scatter plots
    (points_3d,) = ax1.plot([], [], [], "ro", markersize=4)
    (points_2d,) = ax2.plot([], [], "ro", markersize=4, label="Population")

    # Title for iteration number
    title = fig.suptitle("Iteration: 0")

    def init():
        points_3d.set_data([], [])
        points_3d.set_3d_properties([])
        points_2d.set_data([], [])
        return points_3d, points_2d

    def update(frame):
        pop = history[frame]
        x_pop = pop[:, 0]
        y_pop = pop[:, 1]
        z_pop = func(x_pop, y_pop)

        # Update 3D scatter
        points_3d.set_data(x_pop, y_pop)
        points_3d.set_3d_properties(z_pop)

        # Update 2D scatter
        points_2d.set_data(x_pop, y_pop)

        # Update title
        title.set_text(f"Iteration: {frame}")

        return points_3d, points_2d

    ani = FuncAnimation(
        fig,
        update,
        frames=len(history),
        init_func=init,
        interval=interval,
        blit=True,
    )

    # Save animation
    ani.save(save_path, writer=PillowWriter(fps=fps))
    print(f"Animation saved to {save_path}")

    plt.close(fig)
