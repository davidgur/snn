# gen_probabilities.py
# David Gurevich
# March 26, 2024

import sys
import numpy as np
import sympy as sp

from scipy.stats import multivariate_normal
from scipy.integrate import dblquad
from tqdm import tqdm

def a(t, x, y):
    return np.array([1, 1])

def b(t, x, y):
    return np.array([0, 0])

def calculate_dt(dx):
    sp.init_printing(wrap_line=False)

    x = sp.Symbol('x')
    dt = sp.Symbol('dt')

    # Evaluate the integral 
    integral_expr = sp.integrate(sp.exp((-1/2) * (x / dt)**2), x)
    def_integral = integral_expr.subs(x, 3 * dx / 2) - integral_expr.subs(x, -3 * dx / 2)
    normalized = def_integral / (dt * sp.sqrt(2 * sp.pi))
    sol_dt = sp.solve(normalized - 0.95, dt)

    return sol_dt[0]

def p_ij(xi, yi, xj, yj, dx, dt):
    # The probability of a particle moving from state (xi, yi) to (xj, yj)
    # in time dt is given by the following formula:
    # P_ij = P[N(xi + b(t, xi, yi)dt, a^2(t, xi, yi)dt) \in (xj - dx/2, xj + dx/2) \cap (yj - dx/2, yj + dx/2)]

    mu = np.array([xi, yi]) + b(0, xi, yi) * dt
    co = np.diag(a(0, xi, yi)**2 * dt)

    rv = multivariate_normal(mu, co)
    x_range = [xj - dx/2, xj + dx/2]
    y_range = [yj - dx/2, yj + dx/2]

    # Perform Monte Carlo integration
    num_samples = 10000
    samples = rv.rvs(num_samples)
    p = 0
    for i in range(num_samples):
        if x_range[0] <= samples[i][0] <= x_range[1] and y_range[0] <= samples[i][1] <= y_range[1]:
            p += 1

    p /= num_samples

    return p

if __name__ == "__main__":
    # Length of the domain
    L = float(sys.argv[1])

    # Number of states in the length
    n = int(sys.argv[2])

    # Step size (dx)
    dx = L / n

    # We need to calculate dt such that the probability of a
    # particle moving beyond any of it's neighbouring states is
    # less than 0.05.
    dt = calculate_dt(dx)

    # Calculate the probabilities
    P = np.zeros((n, n, 5))
    for i in tqdm(range(n)):
        for j in range(n):
            if i == 0 or i == n - 1 or j == 0 or j == n - 1:
                # If we are at the boundary, we can't move out of the domain
                P[i, j, 0] = 0
                P[i, j, 1] = 0
                P[i, j, 2] = 0
                P[i, j, 3] = 0
                P[i, j, 4] = 1
                continue

            xi = i * dx
            yi = j * dx

            # Left
            P[i, j, 0] = p_ij(xi, yi, xi - dx, yi, dx, dt)

            # Right
            P[i, j, 1] = p_ij(xi, yi, xi + dx, yi, dx, dt)

            # Up
            P[i, j, 2] = p_ij(xi, yi, xi, yi + dx, dx, dt)

            # Down
            P[i, j, 3] = p_ij(xi, yi, xi, yi - dx, dx, dt)

            # Stay
            P[i, j, 4] = 1 - np.sum(P[i, j, :4])

    # Save the parameters to a file
    print(L)
    print(n)
    print(dx)
    print(dt)

    # Save the probabilities to a file
    # Each row corresponds to a state (i, j)
    # Each column corresponds to a action (left, right, up, down, stay)
    for i in range(n):
        for j in range(n):
            for k in range(5):
                print(P[i, j, k], end=' ')
            print()
