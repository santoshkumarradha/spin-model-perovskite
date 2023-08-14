import random
import numpy as np
from pymatgen.transformations.standard_transformations import SupercellTransformation
from pymatgen.io.ase import AseAtomsAdaptor
from ase.geometry.analysis import Analysis
from chgnet.model.model import CHGNet
from chgnet.model.dynamics import MolecularDynamics
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import multiprocessing
from pathlib import Path
import sys
from dask import delayed
from dask.distributed import Client, LocalCluster
import dask
from chgnet.model.model import CHGNet
from chgnet.model.dynamics import MolecularDynamics
from pymatgen.core import Structure
from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor
import warnings
import matplotlib.pyplot as plt
import dask
from dask.distributed import Client
import numpy as np

warnings.filterwarnings("ignore", module="pymatgen")
warnings.filterwarnings("ignore", module="ase")
sys.path.append("../")
from core import RotateStructure
from utils import create_spehircal_mesh_points


def get_structure(theta):
    rs = RotateStructure()
    return rs.get_rotated_structure(phi=0, theta=theta, remove_Rb=False, add_charge=False)


from pymatgen.analysis.local_env import VoronoiNN, LocalStructOrderParams


def get_ge_octahedral_order(structure):
    """
    Compute the octahedral order parameter for Ge atoms in the given structure.
    """
    vnn = VoronoiNN(cutoff=5)  # Using VoronoiNN to get neighbors
    lsop = LocalStructOrderParams(["oct"])  # Using "oct" for octahedral order parameter
    structure.remove_species(["Rb"])
    octahedral_orders = []

    for idx, site in enumerate(structure):
        if site.species_string == "Ge":
            neighbors_info = vnn.get_nn_info(structure, idx)
            neighbor_indices = [info["site_index"] for info in neighbors_info]
            op = lsop.get_order_parameters(structure, idx, indices_neighs=neighbor_indices)

            if op[0] is not None:  # Ensure the order parameter was computed
                octahedral_orders.append(op[0])

    return np.mean(octahedral_orders)  # Return average octahedral order for all Ge atoms


@dask.delayed
def run_md_and_get_energy(
    temp, structure, chgnet, nudge_amount, runs=10, calculate_coordination_number=True
):
    energies = []
    coordination_numbers = []

    for r in range(runs):
        # Nudge the atomic positions
        for atom in structure:
            atom.coords += np.random.uniform(-nudge_amount, nudge_amount, 3)
        traj_path = str(Path(f"../data/md_files/md_{temp}{r}K.traj").resolve())
        log_path = str(Path(f"../data/md_files/md_{temp}{r}K.log").resolve())

        # Run MD simulation at the current temperature
        md = MolecularDynamics(
            atoms=structure,
            model=chgnet,
            ensemble="nvt",
            temperature=temp,
            timestep=2,
            trajectory=traj_path,
            logfile=log_path,
            loginterval=100,
            use_device="cpu",
        )
        md.run(101)  # Increased simulation time

        # Update the structure with the final atomic positions after the MD run
        ase_atoms = read(traj_path, index=-1)
        structure = AseAtomsAdaptor.get_structure(ase_atoms)

        # Predict the energy of the system after the MD run
        prediction = chgnet.predict_structure(structure)
        energy = prediction["e"]
        if calculate_coordination_number:
            try:
                coordination_number = get_ge_octahedral_order(structure)
            except:
                coordination_number = 0
        else:
            coordination_number = 0

        energies.append(energy)
        coordination_numbers.append(coordination_number)

    # return np.mean(energies), np.mean(coordination_numbers)
    return energies, coordination_numbers


def detect_phase_transition(structure1_future, structure2_future, temperature_range):

    args_list1 = [(temp, structure1_future, chgnet, 0.008, 20, True) for temp in temperature_range]
    args_list2 = [(temp, structure2_future, chgnet, 0.008, 20, True) for temp in temperature_range]

    results1 = [run_md_and_get_energy(*args) for args in args_list1]
    results2 = [run_md_and_get_energy(*args) for args in args_list2]

    results1, results2 = dask.compute(results1, results2)

    energy_diffs = [np.mean(res1[0]) - np.mean(res2[0]) for res1, res2 in zip(results1, results2)]
    coordination_diffs = [
        np.mean(res1[1]) - np.mean(res2[1]) for res1, res2 in zip(results1, results2)
    ]

    fig, ax1 = plt.subplots()
    ax1.plot(temperature_range, energy_diffs, "-o", label="Energy Difference")
    ax1.set_xlabel("Temperature (K)")
    ax1.set_ylabel("Energy Difference (eV/atom)")
    ax1.set_title("Energy and Coordination Number Differences vs Temperature")

    ax2 = ax1.twinx()
    ax2.plot(temperature_range, coordination_diffs, "-og", label="Coordination Number Difference")
    ax2.set_ylabel("Coordination Number Difference (Ge-Cl)")

    fig.tight_layout()
    plt.show()

    return energy_diffs, coordination_diffs


if __name__ == "__main__":
    np.random.seed(5)
    chgnet = CHGNet.load()
    cluster = LocalCluster(n_workers=8, memory_limit="auto")
    client = Client(cluster)
    structure1 = get_structure(0)
    structure2 = get_structure(180)
    structure1 = structure1 * (2, 2, 2)
    structure2 = structure2 * (2, 2, 2)

    structure1_future = client.scatter(structure1)
    structure2_future = client.scatter(structure2)

    temperature_range = [int(t) for t in np.linspace(100, 2000, 19)]
    energy_diffs, coordination_diffs = detect_phase_transition(
        structure1_future, structure2_future, temperature_range
    )
    client.shutdown()
    print("energy", energy_diffs)
    print("coord", coordination_diffs)

