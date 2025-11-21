import numpy as np
import random

# Pauli matrices 
sigma_x = np.array([[0, 1],
                    [1, 0]], dtype=complex)

sigma_y = np.array([[0, -1j],
                    [1j,  0]], dtype=complex)

sigma_z = np.array([[1,  0],
                    [0, -1]], dtype=complex)

def random_pauli_spin():
    """Return a random Pauli matrix (sigx, sigy, or sigz)"""
    return random.choice([sigma_x, sigma_y, sigma_z])


def initialize_lattice(size):
    """
    Initialize a size×size lattice where each site contains
    the 3 Pauli matrices [sigx, sigy, sigz]
    """
    lattice = np.empty((size, size, 3), dtype=object)
    for i in range(size):
        for j in range(size):
            lattice[i, j, 0] = sigma_x
            lattice[i, j, 1] = sigma_y
            lattice[i, j, 2] = sigma_z
    return lattice
    
def pauli_dot(Si, Sn, Jx=1.0, Jy=1.0, Jz=1.0):
    """
    Compute Jx * sigixsignx + Jy * sigiysigny + Jz * sigizsignz
    as scalars via (1/2) Tr(). Si, Sn are arrays [sigx,sigy,sigz].
    """
    return (
        Jx * (np.trace(Si[0] @ Sn[0]).real / 2.0) +
        Jy * (np.trace(Si[1] @ Sn[1]).real / 2.0) +
        Jz * (np.trace(Si[2] @ Sn[2]).real / 2.0)
    )


# Energies
def local_energy(lattice, i, j, Jx=1.0, Jy=1.0, Jz=1.0):
    """
    Energy contribution of site (i,j) with its 4 nearest neighbors:
      H = - sum_{<ij>} (Jx sigixsigjx + Jy sigiysigjy + Jz sigizsigjz)
    Periodic boundary conditions.
    """
    L = lattice.shape[0]
    Si = lattice[i, j]

    nn = [
        lattice[(i + 1) % L, j],
        lattice[(i - 1) % L, j],
        lattice[i, (j + 1) % L],
        lattice[i, (j - 1) % L],
    ]

    return -sum(pauli_dot(Si, Sn, Jx, Jy, Jz) for Sn in nn)


def total_energy(lattice, Jx=1.0, Jy=1.0, Jz=1.0):
    """
    Total Heisenberg XYZ nearest-neighbor energy.
    Divide by 2 to correct for double-counting bonds.
    """
    L = lattice.shape[0]
    E = 0.0
    for i in range(L):
        for j in range(L):
            E += local_energy(lattice, i, j, Jx=Jx, Jy=Jy, Jz=Jz)
    return 0.5 * E

#Metropolis
def new_pauli_set():
    # Return length-3 OBJECT array: [Sx, Sy, Sz]
    out = np.empty(3, dtype=object)
    out[0] = sigma_x if random.random() < 0.5 else -sigma_x
    out[1] = sigma_y if random.random() < 0.5 else -sigma_y
    out[2] = sigma_z if random.random() < 0.5 else -sigma_z
    return out

def metropolis_step(lattice, T, Jx=1.0, Jy=1.0, Jz=1.0):
    L = lattice.shape[0]
    i, j = random.randint(0, L - 1), random.randint(0, L - 1)
    E_old = local_energy(lattice, i, j, Jx=Jx, Jy=Jy, Jz=Jz)
    old_spin = lattice[i, j].copy()

    proposal = new_pauli_set()
    lattice[i, j, 0] = proposal[0]
    lattice[i, j, 1] = proposal[1]
    lattice[i, j, 2] = proposal[2]

    E_new = local_energy(lattice, i, j, Jx=Jx, Jy=Jy, Jz=Jz)
    dE = E_new - E_old
    T = max(T, 1e-6)
    if not (dE <= 0.0 or random.random() < np.exp(-dE / T)):
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
            lattice[i, j, 0] = old_spin[0]
            lattice[i, j, 1] = old_spin[1]
            lattice[i, j, 2] = old_spin[2]

def simulate_heisenberg(size, temperature, num_sweeps, thermalization_sweeps, J=1.0):
    """
    Run the Metropolis simulation using full sweeps (each site visited once).
    Returns:
        lattice    – final lattice configuration
        energies   – list of energy per spin measured after thermalization
    """
    lattice = initialize_lattice(size)
    energies = []
    N = size * size

    for sweep in range(num_sweeps + thermalization_sweeps):
        # One full sweep over ALL sites (random order)
        metropolis_sweep_all_sites(lattice, temperature, Jx=J, Jy=J, Jz=J)

        # Start recording after thermalization
        if sweep >= thermalization_sweeps:
            E = total_energy(lattice, Jx=J, Jy=J, Jz=J) / N
            energies.append(E)

    return lattice, energies

