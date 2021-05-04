import numpy as np
import cvxpy as cp
from numpy import sin, cos
import matplotlib.pyplot as plt 
from matplotlib.patches import Rectangle, Circle
from matplotlib.transforms import Affine2D

from dynamics import *

make_mode = lambda i, j: 3*i + j

def create_cost_function(x_var, u_var, Q, R, N):
    cost = 0
    for i in range(N):
        cost += (cp.quad_form(x_var[i], Q) + cp.quad_form(u_var[i], R))
    cost += cp.quad_form(x_var[N], Q)
    return cost

# def create_dynamics_constraint_mip(x_curr, x_prev, u_prev, z_prev, mode, x_nom, u_nom, constants, h=0.01, M=100):
#     # print(x_nom, u_nom)
#     A = [A1, A2, A3][mode](x_nom, u_nom, constants)
#     B = [B1, B2, B3][mode](x_nom, u_nom, constants)
#     leq = (x_curr <= ((np.eye(4) + h * A) @ x_prev + h * B @ u_prev + M * (1 - z_prev)))
#     geq = (-x_curr <= (-(np.eye(4) + h * A) @ x_prev - h * B @ u_prev + M * (1 - z_prev)))
#     return [leq, geq]

# # def create_dynamics_constraint_time0_mip(x_curr, x_prev, u_prev, z_prev, mode, x_nom, u_nom, constants, h=0.01, M=100):
# #     f0 = 

# def create_mc_constraint_mip(x, u, z, mode, x_nom, u_nom, constants, h=0.01, M=100):
#     E = [E1, E2, E3][mode](x_nom, u_nom, constants)
#     D = [D1, D2, D3][mode](x_nom, u_nom, constants)
#     g = [g1, g2, g3][mode](x_nom, u_nom, constants)

#     return [E @ x + D @ u <= (g + M * (1 - z))]

def create_dynamics_constraint_same_side(x_curr, x_prev, u_prev, z_prev, mode, x_nom, u_nom, constants, h=0.01, M=100):
    # print(x_nom, u_nom)
    A = A_same_side(x_nom, u_nom, constants, mode)
    B = B_same_side(x_nom, u_nom, constants, mode)
    leq = (x_curr <= ((np.eye(5) + h * A) @ x_prev + h * B @ u_prev + M * (1 - z_prev)))
    geq = (-x_curr <= (-(np.eye(5) + h * A) @ x_prev - h * B @ u_prev + M * (1 - z_prev)))
    return [leq, geq]

# def create_dynamics_constraint_time0_mip(x_curr, x_prev, u_prev, z_prev, mode, x_nom, u_nom, constants, h=0.01, M=100):
#     f0 = 

def create_mc_constraint_same_side(x, u, z, mode, x_nom, u_nom, constants, h=0.01, M=100):
    x1 = cp.hstack([x[0], x[1], x[2], x[3]])
    x2 = cp.hstack([x[0], x[1], x[2], x[4]])
    u1 = cp.hstack([u[0], u[1]])
    u2 = cp.hstack([u[2], u[3]])

    E_1 = [E1, E2, E3][mode[0]]([x_nom[0], x_nom[1], x_nom[2], x_nom[3]], u_nom[:2], constants)
    D_1 = [D1, D2, D3][mode[0]]([x_nom[0], x_nom[1], x_nom[2], x_nom[3]], u_nom[:2], constants)
    g_1 = [g1, g2, g3][mode[0]]([x_nom[0], x_nom[1], x_nom[2], x_nom[3]], u_nom[:2], constants)

    E_2 = [E1, E2, E3][mode[0]]([x_nom[0], x_nom[1], x_nom[2], x_nom[4]], u_nom[2:], constants)
    D_2 = [D1, D2, D3][mode[0]]([x_nom[0], x_nom[1], x_nom[2], x_nom[4]], u_nom[2:], constants)
    g_2 = [g1, g2, g3][mode[0]]([x_nom[0], x_nom[1], x_nom[2], x_nom[4]], u_nom[2:], constants)

    return [E_1 @ x1 + D_1 @ u1 <= (g_1 + M * (1 - z)), E_2 @ x2 + D_2 @ u2 <= (g_2 + M * (1 - z))]

def create_dynamics_constraint_opp_side(x_curr, x_prev, u_prev, z_prev, mode, x_nom, u_nom, constants, h=0.01, M=100):
    # print(x_nom, u_nom)
    A = A_opp_side(x_nom, u_nom, constants, mode)
    B = B_opp_side(x_nom, u_nom, constants, mode)
    leq = (x_curr <= ((np.eye(5) + h * A) @ x_prev + h * B @ u_prev + M * (1 - z_prev)))
    geq = (-x_curr <= (-(np.eye(5) + h * A) @ x_prev - h * B @ u_prev + M * (1 - z_prev)))
    return [leq, geq]

# def create_dynamics_constraint_time0_mip(x_curr, x_prev, u_prev, z_prev, mode, x_nom, u_nom, constants, h=0.01, M=100):
#     f0 = 

def create_mc_constraint_opp_side(x, u, z, mode, x_nom, u_nom, constants, h=0.01, M=100):
    x1 = cp.hstack([x[0], x[1], x[2], x[3]])
    x2 = cp.hstack([x[0], x[1], x[2], x[4]])
    u1 = cp.hstack([u[0], u[1]])
    u2 = cp.hstack([u[2], u[3]])

    E_1 = [E1, E2, E3][mode[0]]([x_nom[0], x_nom[1], x_nom[2], x_nom[3]], u_nom[:2], constants)
    D_1 = [D1, D2, D3][mode[0]]([x_nom[0], x_nom[1], x_nom[2], x_nom[3]], u_nom[:2], constants)
    g_1 = [g1, g2, g3][mode[0]]([x_nom[0], x_nom[1], x_nom[2], x_nom[3]], u_nom[:2], constants)

    E_2 = [E1, E2, E3][mode[0]]([x_nom[0], x_nom[1], x_nom[2], x_nom[4]], u_nom[2:], constants)
    D_2 = [D1, D2, D3][mode[0]]([x_nom[0], x_nom[1], x_nom[2], x_nom[4]], u_nom[2:], constants)
    g_2 = [g1, g2, g3][mode[0]]([x_nom[0], x_nom[1], x_nom[2], x_nom[4]], u_nom[2:], constants)

    return [E_1 @ x1 + D_1 @ u1 <= (g_1 + M * (1 - z)), E_2 @ x2 + D_2 @ u2 <= (g_2 + M * (1 - z))]

def plot_trajectory(times, states, box_width, same=True, skip=10):
    # corner_body = np.array([-0.5 * box_width, -0.5 * box_width])
    fig, ax = plt.subplots()
    for (x, y, theta, py1, py2) in states[0:-1:skip]:
        # print(x, y, theta, py)
        rect = Rectangle((x - 0.5*box_width, y - 0.5*box_width), box_width, box_width, fill=False)
        R = Affine2D().rotate_deg_around(x, y, theta * 180.0/np.pi) + ax.transData
        rect.set_transform(R)
        ax.add_patch(rect)

        contact1_body = np.array([-0.5*box_width, py1])
        if same:
            contact2_body = np.array([-0.5*box_width, py2])
        else:
            contact2_body = np.array([0.5*box_width, -py2])
        contact1 = rot(theta) @ contact1_body + np.array([x, y])
        contact2 = rot(theta) @ contact2_body + np.array([x, y])

        circ1 = Circle(contact1, 0.003, color='r')
        ax.add_patch(circ1)
        circ2 = Circle(contact2, 0.003, color='b')
        ax.add_patch(circ2)

    ax.plot(states[:, 0], states[:, 1]) # plot (x, y)
    ax.set_aspect('equal')
    plt.show()

    plt.plot(times, states[:, 0], label='x')
    plt.plot(times, states[:, 1], label='y')
    plt.plot(times, states[:, 2], label='theta')
    plt.legend()
    plt.show()

def solve():
    h = 0.03
    N = 1500
    M = 1000 # Big M technique for MIPs

    box_half_width = 0.05
    f_max = 1
    m_max = 1
    mu = 0.1
    c = f_max / m_max
    constants = [mu, c, box_half_width]

    x_var = cp.Variable((N + 1, 4))
    u_var = cp.Variable((N, 2))
    # z_var = cp.Variable((N, 3), boolean=True)

    z_var = np.zeros((N, 3))
    z_var[:, 0] = 1
    # z_var[int(N/3):, 0] = 1
    # z_var[:int(N/3), 1] = 1

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
            # constraints += [z_var[i, 0] + z_var[i, 1] + z_var[i, 2] == 1]
        constraints += [x_var[0] == (x0 - x_nom[0])]
        return constraints

    x_nom = np.array([[0.05 * h * i, 0, 0, 0] for i in range(N + 1)])
    u_nom = np.array([[0.05, 0] for i in range(N)])
    constraints = create_constraints(x_nom, u_nom, x_nom[0] + np.array([0, 0.15, 0, 0]), u_nom[0])
    cost = create_cost_function(x_var, u_var, Q, R, N)
    objective = cp.Minimize(cost)
    prob = cp.Problem(objective, constraints)
    print("Solving...")
    prob.solve(solver=cp.GUROBI, verbose=True)
    # print(z_var.value)
    print("------")
    # print(u_var.value)
    x = x_nom + x_var.value
    time = np.array([h * i for i in range(N + 1)])
    # plt.plot(x[:, 0], x[:, 1]) # plot (x, y)
    # plt.show()
    # plt.plot(time, x[:, 2]) # plot theta vs time
    # plt.plot(time, x[:, 3]) # plot py vs time
    # plt.show()
    plot_trajectory(time, x, box_half_width * 2)

def solve_same_side():
    h = 0.05
    N = 500
    M = 1000 # Big M technique for MIPs

    box_half_width = 0.05
    f_max = 1
    m_max = 1
    mu = 0.3
    c = f_max / m_max
    constants = [mu, c, box_half_width]

    x_var = cp.Variable((N + 1, 5))
    u_var = cp.Variable((N, 4))
    # z_var = cp.Variable((N, 3), boolean=True)

    z_var = np.zeros((N, 9))
    z_var[:, 0] = 1
    # z_var[int(N/3):, 0] = 1
    # z_var[:int(N/3), 1] = 1

    Q = np.diag([1, 1, 0.1, 1, 1])
    R = np.diag([1.0, 1.0, 1, 1])

    def create_constraints(x_nom, u_nom, x0, u0):
        constraints = []
        # Add constraints for the first timestep. This is handled separately
        # since we know the starting configuration of the system, so x_var = (x - x_nom) = 0.

        # Add constraints for all subsequent timesteps.
        for i in range(N):
            for mode1 in range(3):
                for mode2 in range(3):
                    mode = make_mode(mode1, mode2)
                    constraints += create_dynamics_constraint_same_side(x_var[i+1], x_var[i], u_var[i], z_var[i, mode],
                                                                  (mode1, mode2), x_nom[i], u_nom[i], constants, h, M)
                    constraints += create_mc_constraint_same_side(x_var[i], u_var[i], z_var[i, mode], 
                                                            (mode1, mode2), x_nom[i], u_nom[i], constants)
            # constraints += [z_var[i, 0] + z_var[i, 1] + z_var[i, 2] == 1]
        constraints += [x_var[0] == (x0 - x_nom[0])]
        return constraints

    x_nom = np.array([[0.05 * h * i, 0, 0, 0.015, -0.015] for i in range(N + 1)])
    u_nom = np.array([[0.025, 0, 0.025, 0] for i in range(N)])
    constraints = create_constraints(x_nom, u_nom, x_nom[0] + np.array([0, 0.15, 0, 0, 0]), u_nom[0])
    cost = create_cost_function(x_var, u_var, Q, R, N)
    objective = cp.Minimize(cost)
    prob = cp.Problem(objective, constraints)
    print("Solving...")
    prob.solve(solver=cp.GUROBI, verbose=True)
    # print(z_var.value)
    print("------")
    # print(u_var.value)
    x = x_nom + x_var.value
    time = np.array([h * i for i in range(N + 1)])
    # plt.plot(x[:, 0], x[:, 1]) # plot (x, y)
    # plt.show()
    # plt.plot(time, x[:, 2]) # plot theta vs time
    # plt.plot(time, x[:, 3]) # plot py vs time
    # plt.show()
    plot_trajectory(time, x, box_half_width * 2, same=True, skip=10)

def solve_opp_side():
    h = 0.05
    N = 500
    M = 1000 # Big M technique for MIPs

    box_half_width = 0.05
    f_max = 1
    m_max = 1
    mu = 0.3
    c = f_max / m_max
    constants = [mu, c, box_half_width]

    x_var = cp.Variable((N + 1, 5))
    u_var = cp.Variable((N, 4))
    # z_var = cp.Variable((N, 3), boolean=True)

    z_var = np.zeros((N, 9))
    z_var[:, 0] = 1
    # z_var[int(N/3):, 0] = 1
    # z_var[:int(N/3), 1] = 1

    Q = np.diag([1, 1, 0.1, 1, 1])
    R = np.diag([1.0, 1.0, 1, 1])

    def create_constraints(x_nom, u_nom, x0, u0):
        constraints = []
        # Add constraints for the first timestep. This is handled separately
        # since we know the starting configuration of the system, so x_var = (x - x_nom) = 0.

        # Add constraints for all subsequent timesteps.
        for i in range(N):
            for mode1 in range(3):
                for mode2 in range(3):
                    mode = make_mode(mode1, mode2)
                    constraints += create_dynamics_constraint_opp_side(x_var[i+1], x_var[i], u_var[i], z_var[i, mode],
                                                                  (mode1, mode2), x_nom[i], u_nom[i], constants, h, M)
                    constraints += create_mc_constraint_opp_side(x_var[i], u_var[i], z_var[i, mode], 
                                                            (mode1, mode2), x_nom[i], u_nom[i], constants)
            # constraints += [z_var[i, 0] + z_var[i, 1] + z_var[i, 2] == 1]
        constraints += [x_var[0] == (x0 - x_nom[0])]
        return constraints

    x_nom = np.array([[0, 0, (np.pi / 4) * (i / (N + 1)), 0.015, 0.015] for i in range(N + 1)])
    u_nom = np.array([[0.025, 0, 0.025, 0] for i in range(N)])
    constraints = create_constraints(x_nom, u_nom, x_nom[0] + np.array([0, 0, 0, 0, 0]), u_nom[0])
    cost = create_cost_function(x_var, u_var, Q, R, N)
    objective = cp.Minimize(cost)
    prob = cp.Problem(objective, constraints)
    print("Solving...")
    prob.solve(solver=cp.GUROBI, verbose=True)
    # print(z_var.value)
    print("------")
    # print(u_var.value)
    x = x_nom + x_var.value
    time = np.array([h * i for i in range(N + 1)])
    plot_trajectory(time, x, box_half_width * 2, same=False, skip=50)




if __name__ == '__main__':
    solve_same_side()
    solve_opp_side()

