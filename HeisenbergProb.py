from Heisenberg_Library import *
import numpy as np

if __name__ == "__main__":

    L = 16
    T = 0.1
    num_sweeps = 1000
    thermal_sweeps = 1000
    J = 1.0

    # Chosen imaginary-time step (since we are NOT doing Trotterization)
    delta_tau = 0.1   # you control this manually now

    final_lattice, energies, H1_list, H2_list = simulate_heisenberg(
        L,
        T,
        num_sweeps,
        thermal_sweeps,
        J=J
    )

    energies = np.array(energies)
    H1_list = np.array(H1_list)
    H2_list = np.array(H2_list)

    N = L * L

    Pc, Z = compute_Pc_from_snapshots(
        H1_list,
        H2_list,
        delta_tau,
        N
    )

    sweeps = np.arange(len(energies))

    output = np.column_stack((
        sweeps,
        energies,
        H1_list,
        H2_list,
        Pc
    ))

    filename = f"heisenbergprob_L{L}_T{T}_dt{delta_tau}.csv"

    np.savetxt(
        filename,
        output,
        delimiter=",",
        header="sweep,energy_per_site,H1_per_site,H2_per_site,Pc",
        comments=""
    )
