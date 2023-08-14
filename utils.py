from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from sympy.vector import CoordSys3D

from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


def get_U_equation(direction="z"):
    t1, p1, t2, p2 = sp.symbols("\\theta_1 \\phi_1 \\theta_2 \\phi_2")  # For latex
    x1, y1, z1 = sp.cos(p1) * sp.sin(t1), sp.sin(p1) * sp.sin(t1), sp.cos(t1)
    x2, y2, z2 = sp.cos(p2) * sp.sin(t2), sp.sin(p2) * sp.sin(t2), sp.cos(t2)

    N = CoordSys3D("N")
    d1 = x1 * N.i + y1 * N.j + z1 * N.k
    d2 = x2 * N.i + y2 * N.j + z2 * N.k
    Ux_symbolic = (d1.dot(d2) - 3 * d1.dot(N.i) * d2.dot(N.i)).simplify()
    Uy_symbolic = (d1.dot(d2) - 3 * d1.dot(N.j) * d2.dot(N.j)).simplify()
    Uz_symbolic = (d1.dot(d2) - 3 * d1.dot(N.k) * d2.dot(N.k)).simplify()
    if direction == "z":
        return Uz_symbolic, [t1, p1, t2, p2]
    elif direction == "y":
        return Uy_symbolic, [t1, p1, t2, p2]
    elif direction == "x":
        return Ux_symbolic, [t1, p1, t2, p2]
    else:
        raise ValueError("direction must be x, y or z")


def create_spehircal_mesh_points(n, start_from_zero=True):
    t = np.linspace(0, np.rad2deg(2 * np.pi), n)
    p = np.linspace(0, np.rad2deg(np.pi), n)
    if not start_from_zero:
        t = np.linspace(0, np.rad2deg(2 * np.pi), n)
        p = np.linspace(-np.rad2deg(np.pi / 2), np.rad2deg(np.pi / 2), n)
    return t, p

