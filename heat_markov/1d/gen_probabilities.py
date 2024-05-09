# gen_probabilities.py
# David Gurevich
# March 27, 2024

# Generate the probabilities for the 1D heat equation

import sys
import numpy as np
import sympy as sp

from scipy.stats import norm
from tqdm import tqdm

def a(t, x):
    return 1

def b(t, x):
    return 0

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

def p_ij(xi, xj, dx, dt):
    # The probability of a particle moving from state xi to xj
    # in time dt is given by the following formula:
    # P_ij = P[N(xi + b(t, xi)dt, a^2(t, xi)dt) \in (xj - dx/2, xj + dx/2)]

    mu = xi + b(0, xi) * dt
    co = a(0, xi)**2 * dt

    rv = norm(mu, co)
    x_range = [xj - dx/2, xj + dx/2]

    # Here, we can use the cdf of the normal distribution to calculate the probability
    p = rv.cdf(x_range[1]) - rv.cdf(x_range[0])

    return p

if __name__ == "__main__":
    # Length of the domain
    L = float(sys.argv[1])

    # Number of states in the length
    n = int(sys.argv[2])

    dx = L / n
    dt = calculate_dt(dx)

    # Generate the probabilities
    P = np.zeros((n, 3))
    for i in tqdm(range(n)):
        if i == 0 or i == n - 1:
            # If we are at the boundary, we can't move out of the domain
            P[i] = [0, 0, 1]
            continue
    
        xi = i * dx

        # Left
        P[i, 0] = p_ij(xi, xi - dx, dx, dt)

        # Right
        P[i, 1] = p_ij(xi, xi + dx, dx, dt)

        # Stay
        P[i, 2] = 1 - P[i, 0] - P[i, 1]

    # Save the paramters to a file
    print(L)
    print(n)
    print(dx)
    print(dt)

    # Save the probabilities to a file
    # Each row corresponds to a state i, and each column corresponds
    # to the probability of moving to state j
    # The columns correspond to (left, right, stay)
    for i in range(n):
        print(P[i, 0], P[i, 1], P[i, 2])
