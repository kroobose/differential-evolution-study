"""
Benchmark Functions for Optimization.

This module provides standard benchmark functions commonly used to evaluate
optimization algorithms. All functions are designed to work with n-dimensional
inputs.
"""

import numpy as np
from typing import Union

# Type alias for array-like inputs
ArrayLike = Union[np.ndarray, float]


def sphere(x: np.ndarray) -> float:
    """
    Sphere function (De Jong's function 1).

    A simple unimodal convex function. The global minimum is at the origin.

    f(x) = sum(x_i^2)

    Args:
        x: Input vector of any dimension.

    Returns:
        Function value at x.

    Properties:
        - Global minimum: f(0, 0, ..., 0) = 0
        - Search domain: Usually [-5.12, 5.12]^n
        - Unimodal, convex, separable
    """
    return np.sum(x**2)


def ackley(x: np.ndarray, a: float = 20, b: float = 0.2, c: float = 2 * np.pi) -> float:
    """
    Ackley function.

    A widely used multimodal test function with many local minima but one
    global minimum at the origin.

    Args:
        x: Input vector of any dimension.
        a: Depth parameter (default: 20).
        b: Width parameter (default: 0.2).
        c: Frequency parameter (default: 2π).

    Returns:
        Function value at x.

    Properties:
        - Global minimum: f(0, 0, ..., 0) ≈ 0
        - Search domain: Usually [-5, 5]^n
        - Multimodal with many local minima
    """
    n = len(x)
    sum_sq = np.sum(x**2)
    sum_cos = np.sum(np.cos(c * x))

    term1 = -a * np.exp(-b * np.sqrt(sum_sq / n))
    term2 = -np.exp(sum_cos / n)

    return term1 + term2 + a + np.exp(1)


def rastrigin(x: np.ndarray, A: float = 10) -> float:
    """
    Rastrigin function.

    A highly multimodal function with regularly distributed local minima.
    It is a fairly difficult problem due to its large search space and
    large number of local minima.

    Args:
        x: Input vector of any dimension.
        A: Amplitude parameter (default: 10).

    Returns:
        Function value at x.

    Properties:
        - Global minimum: f(0, 0, ..., 0) = 0
        - Search domain: Usually [-5.12, 5.12]^n
        - Highly multimodal with 10^n local minima
    """
    n = len(x)
    return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))


def rosenbrock(x: np.ndarray) -> float:
    """
    Rosenbrock function (Banana function).

    A classic optimization test function. The global minimum lies inside
    a long, narrow, parabolic-shaped flat valley.

    Args:
        x: Input vector of dimension >= 2.

    Returns:
        Function value at x.

    Properties:
        - Global minimum: f(1, 1, ..., 1) = 0
        - Search domain: Usually [-5, 10]^n
        - Unimodal for n <= 3, multimodal for n >= 4
    """
    return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


def griewank(x: np.ndarray) -> float:
    """
    Griewank function.

    A function with many widespread local minima regularly distributed.

    Args:
        x: Input vector of any dimension.

    Returns:
        Function value at x.

    Properties:
        - Global minimum: f(0, 0, ..., 0) = 0
        - Search domain: Usually [-600, 600]^n
        - Multimodal
    """
    n = len(x)
    sum_sq = np.sum(x**2) / 4000
    prod_cos = np.prod(np.cos(x / np.sqrt(np.arange(1, n + 1))))

    return sum_sq - prod_cos + 1


# For backward compatibility with 2D plotting functions
def ackley_2d(x: ArrayLike, y: ArrayLike, a: float = 20, b: float = 0.2, c: float = 2 * np.pi) -> ArrayLike:
    """
    Ackley function for 2D inputs (for visualization purposes).

    Args:
        x: First coordinate (scalar or array).
        y: Second coordinate (scalar or array).
        a: Depth parameter (default: 20).
        b: Width parameter (default: 0.2).
        c: Frequency parameter (default: 2π).

    Returns:
        Function value(s) at (x, y).
    """
    term1 = -a * np.exp(-b * np.sqrt((x**2 + y**2) / 2))
    term2 = -np.exp((np.cos(c * x) + np.cos(c * y)) / 2)
    return term1 + term2 + a + np.exp(1)


def sphere_2d(x: ArrayLike, y: ArrayLike) -> ArrayLike:
    """
    Sphere function for 2D inputs (for visualization purposes).

    Args:
        x: First coordinate (scalar or array).
        y: Second coordinate (scalar or array).

    Returns:
        Function value(s) at (x, y).
    """
    return x**2 + y**2
