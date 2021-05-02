import numpy as np
import cvxpy as cp
from numpy import sin, cos

from dynamics import *

def create_cost_function(x_var, u_var, Q, R, N):
    cost = 0
    for i in range(N):
        cost += (cp.quad_form(x_var[i], Q) + cp.quad_form(u_var[i], R))
    cost += cp.quad_form(x_var[N], Q)
    return cost

def create_dynamics_constraint_mip(x_curr, x_prev, u_prev, z_prev, mode, x_nom, u_nom, constants, h=0.01, M=100):
    # print(x_nom, u_nom)
    A = [A1, A2, A3][mode](x_nom, u_nom, constants)
    B = [B1, B2, B3][mode](x_nom, u_nom, constants)
    leq = (x_curr <= ((np.eye(4) + h * A) @ x_prev + h * B @ u_prev + M * (1 - z_prev)))
    geq = (-x_curr <= (-(np.eye(4) + h * A) @ x_prev - h * B @ u_prev + M * (1 - z_prev)))
    return [leq, geq]

# def create_dynamics_constraint_time0_mip(x_curr, x_prev, u_prev, z_prev, mode, x_nom, u_nom, constants, h=0.01, M=100):
#     f0 = 

def create_mc_constraint_mip(x, u, z, mode, x_nom, u_nom, constants, h=0.01, M=100):
    E = [E1, E2, E3][mode](x_nom, u_nom, constants)
    D = [D1, D2, D3][mode](x_nom, u_nom, constants)
    g = [g1, g2, g3][mode](x_nom, u_nom, constants)

    return [E @ x + D @ u <= (g + M * (1 - z))]


def solve():
    h = 0.03
    N = 20
    M = 1000 # Big M technique for MIPs

    box_half_width = 1
    f_max = 1
    m_max = 1
    mu = 1
    c = f_max / m_max
    constants = [mu, c, box_half_width]

    x_var = cp.Variable((N + 1, 4))
    u_var = cp.Variable((N, 2))
    z_var = cp.Variable((N, 3), boolean=True)

    Q = np.diag([1, 1, 0.1, 1])
    R = np.diag([1.0, 1.0])

    def create_constraints(x_nom, u_nom, x0, u0):
        constraints = []
        # Add constraints for the first timestep. This is handled separately
        # since we know the starting configuration of the system, so x_var = (x - x_nom) = 0.

        # Add constraints for all subsequent timesteps.
        for i in range(N):
            for mode in range(3):
                constraints += create_dynamics_constraint_mip(x_var[i+1], x_var[i], u_var[i], z_var[i, mode],
                                                              mode, x_nom[i], u_nom[i], constants, h, M)
                constraints += create_mc_constraint_mip(x_var[i], u_var[i], z_var[i, mode], 
                                                        mode, x_nom[i], u_nom[i], constants)
            constraints += [z_var[i, 0] + z_var[i, 1] + z_var[i, 2] == 1]
        constraints += [x_var[0] == (x0 - x_nom[0])]
        return constraints

    x_nom = np.array([[0.05 * h * i, 0, 0, 0] for i in range(N + 1)])
    u_nom = np.array([[0.05, 0] for i in range(N)])
    constraints = create_constraints(x_nom, u_nom, x_nom[0] + np.array([0, 0.05, 0, 0]), u_nom[0])
    cost = create_cost_function(x_var, u_var, Q, R, N)
    objective = cp.Minimize(cost)
    prob = cp.Problem(objective, constraints)
    print("Solving...")
    prob.solve(solver=cp.GUROBI, verbose=True)
    print(z_var.value)
    print("------")
    print(u_var.value)


if __name__ == '__main__':
    solve()
    # x = cp.Variable(2)
    # obj = cp.Minimize(x[0] + cp.norm(x, 1))
    # constraints = [x >= 2]
    # prob = cp.Problem(obj, constraints)
    # prob.solve(solver=cp.GUROBI)
    # print("optimal value with GUROBI:", prob.value)

