import numba as nb
import numpy as np
import math
import numba as nb


def energy():
    pass


@nb.jit(nopython=False)
def electro_static_potential(atom_number, cart_coords, charges, cell, alpha=5, nmax=3, mmax=9):
    """
    Args:
        atom_number (int): Index of the particle for which the total potential is to be calculated.
        cart_coords (numpy.ndarray): Array of shape (N, 3) containing the Cartesian coordinates of N particles.
        charges (numpy.ndarray): Array of shape (N,) containing the charges of N particles.
        cell (numpy.ndarray): Array of shape (3, 3) containing the lattice vectors of the simulation cell.
        alpha (float): Constant used in the calculation of the potential. Default is 5.
        nmax (int): Maximum value for the indices n0, n1, and n2 in the real space sum. Default is 3.
        mmax (int): Maximum value for the indices m0, m1, and m2 in the reciprocal space sum. Default is 9.

    Returns:
        float: The total potential for the particle with index i.
    """
    invcell = np.linalg.inv(cell).T
    area = abs(np.linalg.det(cell))
    Vr = potential_realsum(atom_number, cart_coords, charges, cell, alpha, nmax)
    Vf = potential_recipsum(atom_number, cart_coords, charges, invcell, alpha, mmax, area)
    Vs = potential_selfsum(atom_number, cart_coords, charges, cell, alpha)
    return Vr + Vf + Vs


@nb.jit(nopython=False)
def potential_realsum(atom_number, cart_coords, charges, cell, alpha, nmax):
    Vr = 0
    for j in range(len(charges)):
        rij = cart_coords[atom_number, :] - cart_coords[j, :]
        Vrloc = 0
        for n2 in range(-nmax, nmax + 1):
            for n1 in range(-nmax, nmax + 1):
                for n0 in range(-nmax, nmax + 1):
                    if max([abs(n0), abs(n1), abs(n2)]):
                        n = n0 * cell[0, :] + n1 * cell[1, :] + n2 * cell[2, :]
                        rn = np.linalg.norm(rij - n)
                        Vrloc += math.erfc(alpha * rn) / rn
        Vr += charges[j] * Vrloc
    return Vr


@nb.jit(nopython=False)
def potential_recipsum(atom_number, cart_coords, charges, cell, alpha, mmax, area):
    Vf = 0
    for j in range(len(charges)):
        rij = cart_coords[atom_number, :] - cart_coords[j, :]
        Vfloc = 0
        for n2 in range(-mmax, mmax + 1):
            for n1 in range(-mmax, mmax + 1):
                for n0 in range(-mmax, mmax + 1):
                    if max([abs(n0), abs(n1), abs(n2)]):
                        m = 2 * np.pi * (n0 * cell[0, :] + n1 * cell[1, :] + n2 * cell[2, :])
                        Vfloc += np.exp(
                            1.0j * np.dot(m, rij) - np.dot(m, m) / (alpha ** 2 * 4)
                        ) / (np.dot(m, m) / (4 * np.pi ** 2))

        Vf += charges[j] / (np.pi * area) * Vfloc.real
    return Vf


@nb.jit(nopython=False)
def potential_selfsum(atom_number, cart_coords, charges, cell, alpha):
    Vs = 0
    for j in range(len(charges)):
        if atom_number == j:
            Vs -= 2 * charges[j] * alpha / np.sqrt(np.pi)
        else:
            rn = np.linalg.norm(cart_coords[atom_number, :] - cart_coords[j, :])
            Vs += charges[j] * math.erfc(alpha * rn) / rn
    return Vs
