from pymatgen.io.cif import CifParser, CifWriter
import numpy as np
from pymatgen.core import Lattice
from pymatgen.analysis.ewald import EwaldSummation


class RotateStructure:
    def __init__(self, original_structure=None, a_reduce=1, move=1) -> None:
        if original_structure is None:
            parser = CifParser("../data/dipoles1.cif")
            original_structure = parser.get_structures()[0]
        self.original_structure = original_structure
        self.a_reduce = a_reduce
        self.move = move

    def get_rotated_structure(self, phi, theta, remove_Rb=False, add_charge=True):
        rotated_structure = self.original_structure.copy()
        coord = (
            rotated_structure.cart_coords[3]
            + rotated_structure.cart_coords[5]
            + rotated_structure.cart_coords[6]
            + rotated_structure.cart_coords[8]
        )
        coord = coord / 4
        coord[2] = coord[2] * self.move
        new_a = rotated_structure.lattice.a * self.a_reduce

        rotated_structure.rotate_sites([3, 5, 6, 8], np.deg2rad(theta), [0, 1, 0], coord)

        rotated_structure.rotate_sites([3, 5, 6, 8], np.deg2rad(phi), [1, 0, 0], coord)
        rotated_structure

        lattice = Lattice.from_parameters(
            a=new_a,
            b=rotated_structure.lattice.b,
            c=rotated_structure.lattice.c,
            alpha=90,
            beta=rotated_structure.lattice.beta,
            gamma=90,
        )
        rotated_structure.lattice = lattice
        if remove_Rb:
            rotated_structure.remove_species(["Rb"])
        if add_charge:
            rotated_structure.add_oxidation_state_by_element({"Ge": -1, "Cl": 0.3, "Rb": 0})
        return rotated_structure


def get_electrostatic_energy(structure):
    return EwaldSummation(structure).total_energy


def get_U_mean_distance(structure):
    k = 6
    dist1 = [
        i[1] for i in structure.get_neighbors(structure[2], 10) if "Cl" in i[0].species_string
    ]
    val1 = [np.sort(dist1)[:k]]
    dist2 = [
        i[1] for i in structure.get_neighbors(structure[3], 10) if "Cl" in i[0].species_string
    ]
    val2 = [np.sort(dist2)[:k]]
    return np.mean([val1, val2])

