import numpy as np
import cvxpy as cp
from numpy import sin, cos

def rot(theta):
    return np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])

def gamma_t(mu, c, px, py):
    return (mu*c*c - px*py + mu*px*px) / (c*c + py*py - mu*px*py)

def gamma_b(mu, c, px, py):
    return (-mu*c*c - px*py - mu*px*px) / (c*c + py*py + mu*px*py)

def f_same_side(state, inp, constants, mode):
    """
    Mode needs to be a 2-tuple (i, j) specifying mode of each contact.
    """
    x, y, theta, py1, py2 = state
    mu, c, px1, px2 = constants

def A_same_side(state, u, constants, mode):
    x, y, theta, py1, py2 = state
    mu, c, px = constants

    A_1 = [A1, A2, A3][mode[0]]([x, y, theta, py1], u[:2], constants)
    A_2 = [A1, A2, A3][mode[1]]([x, y, theta, py2], u[2:], constants)

    Dg1 = np.zeros((5, 5))
    Dg1[:4, :4] = A_1

    Dg2 = np.zeros((5, 5))
    Dg2[:3, :3] = A_2[:3, :3]
    Dg2[4, :3] = A_2[3, :3]
    Dg2[:3, 4] = A_2[:3, 3]
    Dg2[4, 4] = A_2[3, 3]

    return Dg1 + Dg2

def B_same_side(state, u, constants, mode):
    x, y, theta, py1, py2 = state
    mu, c, px = constants
    B_1 = [B1, B2, B3][mode[0]]([x, y, theta, py1], u[:2], constants)
    B_2 = [B1, B2, B3][mode[1]]([x, y, theta, py2], u[2:], constants)

    g1 = np.zeros((5, 2))
    g1[:4, :] = B_1

    g2 = np.zeros((5, 2))
    g2[:3, :] = B_2[:3, :]
    g2[4, :] = B_2[3, :]

    B = np.zeros((5, 4))
    B[:, :2] = g1
    B[:, 2:] = g2
    return B

def A_opp_side(state, u, constants, mode):
    x, y, theta, py1, py2 = state
    mu, c, px = constants

    A_1 = [A1, A2, A3][mode[0]]([x, y, theta, py1], u[:2], constants)
    A_2 = [A1, A2, A3][mode[1]]([x, y, theta, py2], u[2:], constants)

    Dg1 = np.zeros((5, 5))
    Dg1[:4, :4] = A_1

    Dg2 = np.zeros((5, 5))
    Dg2[:3, :3] = A_2[:3, :3]
    Dg2[4, :3] = A_2[3, :3]
    Dg2[:3, 4] = A_2[:3, 3]
    Dg2[4, 4] = A_2[3, 3]
    Dg2[2:, :2] *= -1

    return Dg1 + Dg2

def B_opp_side(state, u, constants, mode):
    x, y, theta, py1, py2 = state
    mu, c, px = constants
    B_1 = [B1, B2, B3][mode[0]]([x, y, theta, py1], u[:2], constants)
    B_2 = [B1, B2, B3][mode[1]]([x, y, theta, py2], u[2:], constants)

    g1 = np.zeros((5, 2))
    g1[:4, :] = B_1

    g2 = np.zeros((5, 2))
    g2[:3, :] = B_2[:3, :]
    g2[4, :] = B_2[3, :]
    g2[:2, :] *= -1

    B = np.zeros((5, 4))
    B[:, :2] = g1
    B[:, 2:] = g2
    return B

# def Ki(state, inp, constants, mode):
#     x, y, theta, py = state
#     mu, c, px = constants
#     C = rot(theta)
#     Q = (1 / (c*c + px*px + py*py)) * np.array([c*c + px*px, px*py], [px*py, c*c + py*py])

#     if mode == 0:
#         Pi = np.eye(2)
#     elif mode == 1:
#         gt = gamma_t(mu, c, px, py)
#         Pi = np.array([[1, 0], [gt, 0]])
#     else:
#         gb = gamma_b(mu, c, px, py)
#         Pi = np.array([[1, 0], [gb, 0]])
#     return C.T @ Q @ Pi

def fi(state, inp, constants, Pi, bi, ci, return_b=False):
    x, y, theta, py = state
    mu, c, px = constants

    C = rot(theta)
    Q = (1 / (c*c + px*px + py*py)) * np.array([c*c + px*px, px*py], [px*py, c*c + py*py])

    B = np.zeros((4, 2))
    B[:2, :] = C.T @ Q @ Pi
    B[2,  :] = bi
    B[3,  :] = ci
    if return_b:
        return B
    return B @ inp

def f1(state, u, constants, return_b=False):
    x, y, theta, py = state
    mu, c, px = constants

    P1 = np.eye(2)
    b1 = np.array([-py / (c*c + px*px + py*py), px])
    c1 = np.array([0.0, 0.0])
    return fi(state, u, constants, P1, b1, c1, return_b)

def f2(state, u, constants, return_b=False):
    x, y, theta, py = state
    mu, c, px = constants

    gt = gamma_t(mu, c, px, py)

    P2 = np.array([[1, 0], [gt, 0]])
    b2 = np.array([(-py + gt*px) / (c*c + px*px + py*py), 0])
    c2 = np.array([-gt, 0])
    return fi(state, u, constants, P2, b2, c2, return_b)

def f3(state, u, constants, return_b=False):
    x, y, theta, py = state
    mu, c, px = constants

    gb = gamma_b(mu, c, px, py)

    P3 = np.array([[1, 0], [gb, 0]])
    b3 = np.array([(-py + gb*px) / (c*c + px*px + py*py), 0])
    c3 = np.array([-gb, 0])
    return fi(state, u, constants, P3, b3, c3, return_b)

def A1(state, u, constants):
    x, y, theta, py = state
    mu, c, px = constants
    vn, vt = u

    T = np.zeros((4, 4))

    T[0][2] = -((c*c)*vt*cos(theta)+(py*py)*vt*cos(theta)+(c*c)*vn*sin(theta)+(px*px)*vn*sin(theta)+px*py*vn*cos(theta)+px*py*vt*sin(theta))/(c*c+px*px+py*py)
    T[0][3] = -1.0/pow(c*c+px*px+py*py,2.0)*(-(px*px*px)*vt*cos(theta)+(px*px*px)*vn*sin(theta)+(c*c)*py*vn*cos(theta)*2.0-(c*c)*px*vt*cos(theta)+(px*px)*py*vn*cos(theta)*2.0+px*(py*py)*vt*cos(theta)+(c*c)*px*vn*sin(theta)-px*(py*py)*vn*sin(theta)+(px*px)*py*vt*sin(theta)*2.0)
    T[1][2] = ((c*c)*vn*cos(theta)+(px*px)*vn*cos(theta)-(c*c)*vt*sin(theta)-(py*py)*vt*sin(theta)+px*py*vt*cos(theta)-px*py*vn*sin(theta))/(c*c+px*px+py*py)
    T[1][3] = 1.0/pow(c*c+px*px+py*py,2.0)*((px*px*px)*vn*cos(theta)+(px*px*px)*vt*sin(theta)+(c*c)*px*vn*cos(theta)-px*(py*py)*vn*cos(theta)+(px*px)*py*vt*cos(theta)*2.0-(c*c)*py*vn*sin(theta)*2.0+(c*c)*px*vt*sin(theta)-(px*px)*py*vn*sin(theta)*2.0-px*(py*py)*vt*sin(theta))
    T[2][3] = -vn*(c*c+px*px-py*py)*1.0/pow(c*c+px*px+py*py,2.0)
    return T

def A2(state, u, constants):
    x, y, theta, py = state
    mu, c, px = constants
    vn, vt = u

    T = np.zeros((4, 4))

    T[0][2] = -((c*c)*vn*(sin(theta)+mu*cos(theta)))/(c*c+py*py-mu*px*py)
    T[0][3] = -(c*c)*vn*(py*2.0-mu*px)*(cos(theta)-mu*sin(theta))*1.0/pow(c*c+py*py-mu*px*py,2.0)
    T[1][2] = ((c*c)*vn*(cos(theta)-mu*sin(theta)))/(c*c+py*py-mu*px*py)
    T[1][3] = -(c*c)*vn*(py*2.0-mu*px)*(sin(theta)+mu*cos(theta))*1.0/pow(c*c+py*py-mu*px*py,2.0)
    T[2][3] = -vn*1.0/pow(c*c+py*py-mu*px*py,2.0)*(c*c-py*py-(mu*mu)*(px*px)+mu*px*py*2.0)
    T[3][3] = vn*1.0/pow(c*c+py*py-mu*px*py,2.0)*((c*c)*px-px*(py*py)-(mu*mu)*(px*px*px)+mu*(px*px)*py*2.0-(c*c)*(mu*mu)*px+(c*c)*mu*py*2.0);

    return T 

def A3(state, u, constants):
    x, y, theta, py = state
    mu, c, px = constants
    vn, vt = u

    T = np.zeros((4, 4))
    T[0][2] = -((c*c)*vn*(sin(theta)-mu*cos(theta)))/(c*c+py*py+mu*px*py)
    T[0][3] = -(c*c)*vn*(py*2.0+mu*px)*(cos(theta)+mu*sin(theta))*1.0/pow(c*c+py*py+mu*px*py,2.0)
    T[1][2] = ((c*c)*vn*(cos(theta)+mu*sin(theta)))/(c*c+py*py+mu*px*py)
    T[1][3] = -(c*c)*vn*(py*2.0+mu*px)*(sin(theta)-mu*cos(theta))*1.0/pow(c*c+py*py+mu*px*py,2.0)
    T[2][3] = vn*1.0/pow(c*c+py*py+mu*px*py,2.0)*(-c*c+py*py+(mu*mu)*(px*px)+mu*px*py*2.0)
    T[3][3] = -vn*1.0/pow(c*c+py*py+mu*px*py,2.0)*(-(c*c)*px+px*(py*py)+(mu*mu)*(px*px*px)+mu*(px*px)*py*2.0+(c*c)*(mu*mu)*px+(c*c)*mu*py*2.0)

    return T

def B1(state, u, constants):
    x, y, theta, py = state
    mu, c, px = constants
    vn, vt = u

    T = np.zeros((4, 2))

    T[0][0] = ((px*px)*cos(theta)+(c*c)*cos(theta)-px*py*sin(theta))/(c*c+px*px+py*py)
    T[0][1] = -((c*c)*sin(theta)+(py*py)*sin(theta)-px*py*cos(theta))/(c*c+px*px+py*py)
    T[1][0] = ((c*c)*sin(theta)+(px*px)*sin(theta)+px*py*cos(theta))/(c*c+px*px+py*py)
    T[1][1] = ((py*py)*cos(theta)+(c*c)*cos(theta)+px*py*sin(theta))/(c*c+px*px+py*py)
    T[2][0] = -py/(c*c+px*px+py*py)
    T[2][1] = px

    return T

def B2(state, u, constants):
    x, y, theta, py = state
    mu, c, px = constants
    vn, vt = u

    T = np.zeros((4, 2))

    T[0][0] = ((c*c)*(cos(theta)-mu*sin(theta)))/(c*c+py*py-mu*px*py)
    T[1][0] = ((c*c)*(sin(theta)+mu*cos(theta)))/(c*c+py*py-mu*px*py)
    T[2][0] = -(py-mu*px)/(c*c+py*py-mu*px*py)
    T[3][0] = -(-px*py+(c*c)*mu+mu*(px*px))/(c*c+py*py-mu*px*py)

    return T

def B3(state, u, constants):
    x, y, theta, py = state
    mu, c, px = constants
    vn, vt = u

    T = np.zeros((4, 2))

    T[0][0] = ((c*c)*(cos(theta)+mu*sin(theta)))/(c*c+py*py+mu*px*py)
    T[1][0] = ((c*c)*(sin(theta)-mu*cos(theta)))/(c*c+py*py+mu*px*py)
    T[2][0] = -(py+mu*px)/(c*c+py*py+mu*px*py)
    T[3][0] = (px*py+(c*c)*mu+mu*(px*px))/(c*c+py*py+mu*px*py)

    return T

def Ct(state, u, constants):
    x, y, theta, py = state
    mu, c, px = constants
    vn, vt = u

    T = np.zeros(4)
    T[3] = -1.0/pow(c*c+py*py-mu*px*py,2.0)*((c*c)*px-px*(py*py)-(mu*mu)*(px*px*px)+mu*(px*px)*py*2.0-(c*c)*(mu*mu)*px+(c*c)*mu*py*2.0)
    return T

def Cb(state, u, constants):
    x, y, theta, py = state
    mu, c, px = constants
    vn, vt = u

    T = np.zeros(4)
    T[3] = 1.0/pow(c*c+py*py+mu*px*py,2.0)*(-(c*c)*px+px*(py*py)+(mu*mu)*(px*px*px)+mu*(px*px)*py*2.0+(c*c)*(mu*mu)*px+(c*c)*mu*py*2.0)
    return T

def E1(state, u, constants):
    x, y, theta, py = state
    mu, c, px = constants
    vn, vt = u

    E = np.zeros((2, 4))
    E[0, :] = -Ct(state, u, constants)
    E[1, :] = Cb(state, u, constants)
    return vn * E

def E2(state, u, constants):
    x, y, theta, py = state
    mu, c, px = constants
    vn, vt = u
    return vn * Ct(state, u, constants)

def E3(state, u, constants):
    x, y, theta, py = state
    mu, c, px = constants
    vn, vt = u
    return -vn * Cb(state, u, constants)

def D1(state, u, constants):
    x, y, theta, py = state
    mu, c, px = constants
    vn, vt = u
    D = np.array([[-gamma_t(mu, c, px, py), 1.0], [gamma_b(mu, c, px, py), -1.0]])
    return D

def D2(state, u, constants):
    x, y, theta, py = state
    mu, c, px = constants
    vn, vt = u
    D = np.array([gamma_t(mu, c, px, py), -1.0])
    return D

def D3(state, u, constants):
    x, y, theta, py = state
    mu, c, px = constants
    vn, vt = u
    D = np.array([-gamma_b(mu, c, px, py), 1.0])
    return D

def g1(state, u, constants):
    x, y, theta, py = state
    mu, c, px = constants
    vn, vt = u
    return np.array([-vt + gamma_t(mu, c, px, py) * vn, vt - gamma_b(mu, c, px, py) * vn])

def g2(state, u, constants, eps=0.01):
    x, y, theta, py = state
    mu, c, px = constants
    vn, vt = u
    return vt - gamma_t(mu, c, px, py) * vn - eps

def g3(state, u, constants, eps=0.01):
    x, y, theta, py = state
    mu, c, px = constants
    vn, vt = u
    return -vt + gamma_b(mu, c, px, py) * vn - eps

