from Heisenberg_Library import *
import numpy as np
if __name__ == "__main__":
    L = 16
    T = 0.1
    num_sweeps = 1000
    thermal_sweeps = 1000
    J = 1.0

    lattice, energies, H1_list, H2_list = simulate_heisenberg(
        size=L,
        temperature=T,
        num_sweeps=num_sweeps,
        thermalization_sweeps=thermal_sweeps,
        J=J
    )

    N = L * L
    delta_tau = 0.1  

    Pc, Z = compute_Pc_from_snapshots(H1_list, H2_list, delta_tau, N)
