import numpy as np
import random


sigma_x = np.array([[0, 1],
                    [1, 0]], dtype=complex)

sigma_y = np.array([[0, -1j],
                    [1j,  0]], dtype=complex)

sigma_z = np.array([[1,  0],
                    [0, -1]], dtype=complex)


def initialize_lattice(size):
   
    lattice = np.empty((size, size, 3), dtype=object)
    for i in range(size):
        for j in range(size):
            lattice[i, j, 0] = sigma_x
            lattice[i, j, 1] = sigma_y
            lattice[i, j, 2] = sigma_z
    return lattice


def pauli_dot(Si, Sj, Jx=1.0, Jy=1.0, Jz=1.0):
    
    return (
        Jx * (np.trace(Si[0] @ Sj[0]).real / 2.0) +
        Jy * (np.trace(Si[1] @ Sj[1]).real / 2.0) +
        Jz * (np.trace(Si[2] @ Sj[2]).real / 2.0)
    )


def local_energy(lattice, i, j, Jx=1.0, Jy=1.0, Jz=1.0):
   
    L = lattice.shape[0]
    Si = lattice[i, j]

    nn = [
        lattice[(i + 1) % L, j],       # down
        lattice[(i - 1) % L, j],       # up
        lattice[i, (j + 1) % L],       # right
        lattice[i, (j - 1) % L],       # left
    ]

    return -sum(pauli_dot(Si, Sn, Jx=Jx, Jy=Jy, Jz=Jz) for Sn in nn)


def total_energy(lattice, Jx=1.0, Jy=1.0, Jz=1.0):
    L = lattice.shape[0]
    E = 0.0
    for i in range(L):
        for j in range(L):
            E += local_energy(lattice, i, j, Jx=Jx, Jy=Jy, Jz=Jz)
    return 0.5 * E


def bond_energy(lattice, i, j, ip, jp, Jx=1.0, Jy=1.0, Jz=1.0):
    
    Si = lattice[i, j]
    Sj = lattice[ip, jp]
    return -pauli_dot(Si, Sj, Jx=Jx, Jy=Jy, Jz=Jz)


def energies_H1_H2(lattice, Jx=1.0, Jy=1.0, Jz=1.0):
    
    L = lattice.shape[0]
    H1 = 0.0
    H2 = 0.0

    for i in range(L):
        for j in range(L):
            # bond to the right: (i,j) - (i, j+1)
            ip, jp = i, (j + 1) % L
            e = bond_energy(lattice, i, j, ip, jp, Jx=Jx, Jy=Jy, Jz=Jz)
            if (i + j) % 2 == 0:
                H1 += e
            else:
                H2 += e

            # bond downward: (i,j) - (i+1, j)
            ip, jp = (i + 1) % L, j
            e = bond_energy(lattice, i, j, ip, jp, Jx=Jx, Jy=Jy, Jz=Jz)
            if (i + j) % 2 == 0:
                H1 += e
            else:
                H2 += e

    return H1, H2
    # total_energy(lattice) should ≈ H1 + H2.


def new_pauli_set():
   
    out = np.empty(3, dtype=object)
    out[0] = sigma_x if random.random() < 0.5 else -sigma_x
    out[1] = sigma_y if random.random() < 0.5 else -sigma_y
    out[2] = sigma_z if random.random() < 0.5 else -sigma_z
    return out


def metropolis_step(lattice, T, Jx=1.0, Jy=1.0, Jz=1.0):
    
    L = lattice.shape[0]
    i, j = random.randint(0, L - 1), random.randint(0, L - 1)
    T = max(T, 1e-6)

    E_old = local_energy(lattice, i, j, Jx=Jx, Jy=Jy, Jz=Jz)
    old_spin = lattice[i, j].copy()

    proposal = new_pauli_set()
    lattice[i, j, 0] = proposal[0]
    lattice[i, j, 1] = proposal[1]
    lattice[i, j, 2] = proposal[2]

    E_new = local_energy(lattice, i, j, Jx=Jx, Jy=Jy, Jz=Jz)
    dE = E_new - E_old

    if not (dE <= 0.0 or random.random() < np.exp(-dE / T)):
        # reject: restore old spin
        lattice[i, j, 0] = old_spin[0]
        lattice[i, j, 1] = old_spin[1]
        lattice[i, j, 2] = old_spin[2]


def metropolis_sweep_all_sites(lattice, T, Jx=1.0, Jy=1.0, Jz=1.0):
   
    L = lattice.shape[0]
    idx = [(i, j) for i in range(L) for j in range(L)]
    random.shuffle(idx)
    T = max(T, 1e-6)

    for (i, j) in idx:
        E_old = local_energy(lattice, i, j, Jx=Jx, Jy=Jy, Jz=Jz)
        old_spin = lattice[i, j].copy()

        proposal = new_pauli_set()
        lattice[i, j, 0] = proposal[0]
        lattice[i, j, 1] = proposal[1]
        lattice[i, j, 2] = proposal[2]

        E_new = local_energy(lattice, i, j, Jx=Jx, Jy=Jy, Jz=Jz)
        dE = E_new - E_old

        if not (dE <= 0.0 or random.random() < np.exp(-dE / T)):
            # reject: restore old spin
            lattice[i, j, 0] = old_spin[0]
            lattice[i, j, 1] = old_spin[1]
            lattice[i, j, 2] = old_spin[2]

def simulate_heisenberg(size, temperature, num_sweeps,
                        thermalization_sweeps, J=1.0):
  
    lattice = initialize_lattice(size)

    energies = []
    H1_list = []
    H2_list = []

    N = size * size

    for sweep in range(num_sweeps + thermalization_sweeps):

        # Metropolis sweep
        metropolis_sweep_all_sites(lattice, temperature, Jx=J, Jy=J, Jz=J)

        # Measurements after thermalization
        if sweep >= thermalization_sweeps:

            # Total energy per site
            E_tot = total_energy(lattice, Jx=J, Jy=J, Jz=J) / N

            # Checkerboard energies per site
            H1, H2 = energies_H1_H2(lattice, Jx=J, Jy=J, Jz=J)
            H1 /= N
            H2 /= N

            energies.append(E_tot)
            H1_list.append(H1)
            H2_list.append(H2)

    return lattice, energies, H1_list, H2_list

def compute_Pc_from_snapshots(H1_list, H2_list, delta_tau, N):
    
    H1_arr = np.array(H1_list)
    H2_arr = np.array(H2_list)

    # total energy for each configuration (not per site anymore)
    H_tot = N * (H1_arr + H2_arr)

    # numerical stability: shift by min(H_tot)
    exponent = -delta_tau * (H_tot - H_tot.min())
    weights = np.exp(exponent)

    Z = weights.sum()
    Pc = weights / Z
    return Pc, Z
