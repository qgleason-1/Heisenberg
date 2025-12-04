from Heisenberg_Library import *
import numpy as np

if __name__ == "__main__":

    lattice_size = 20
    temperature = 1.3
    num_sweeps = 1000
    thermalization_sweeps = 1000
    J = 0.5

    final_lattice, energies, H1_list, H2_list = simulate_heisenberg(
        lattice_size, temperature, num_sweeps, thermalization_sweeps, J=J
    )

    energies = np.array(energies)
    H1_list = np.array(H1_list)
    H2_list = np.array(H2_list)

    sweeps = np.arange(len(energies))

    output = np.column_stack((sweeps, energies, H1_list, H2_list))

    filename = f"heisenberg_L{lattice_size}_T{temperature}_J{J}.csv"

    np.savetxt(
        filename,
        output,
        delimiter=",",
        header="sweep,energy_per_site,H1_per_site,H2_per_site",
        comments=""
    )

