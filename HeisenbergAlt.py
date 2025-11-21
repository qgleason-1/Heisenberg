from Heisenberg_Library import *
if __name__ == "__main__":
    lattice_size = 4
    temperature = 2.8
    num_sweeps = 100
    thermalization_sweeps = 100
    J = 1.0

    final_lattice, energies = simulate_heisenberg(
        lattice_size, temperature, num_sweeps, thermalization_sweeps, J=1.0

    ## export data to file (.csv)
    )
