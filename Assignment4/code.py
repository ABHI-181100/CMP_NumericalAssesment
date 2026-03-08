# ---------------- Q1 : Band structure (1st & 2nd BZ) ----------------

import numpy as np
import matplotlib.pyplot as plt

A = 3
a = 1
levels = 5
G = 2*np.pi/a

VG = {1:A/2, 2:A/4, 3:A/6}

def band_structure(bz):

    k_vals = np.linspace(-bz*np.pi/a, bz*np.pi/a, 1000)

    g = np.arange(-levels//2+1, levels//2+1)*G
    bands = np.zeros((levels,len(k_vals)))

    for i,k in enumerate(k_vals):

        H = np.diag(0.5*(k-g)**2)

        for n,v in VG.items():
            H += np.diag([v]*(levels-n), n)
            H += np.diag([v]*(levels-n), -n)

        bands[:,i] = np.linalg.eigvalsh(H)

    return k_vals,bands


def plot_bands(bz):

    k,b = band_structure(bz)

    plt.figure(figsize=(8,6))

    for i in range(levels):
        plt.plot(k,b[i],lw=1)

    for x in np.arange(-bz,bz+1)*np.pi/a:
        plt.axvline(x,color='gray',ls='--',lw=0.7)

    plt.xlabel("k")
    plt.ylabel("Energy")
    plt.title(f"Band Structure ({bz} BZ)")
    plt.ylim(-1,100)
    plt.show()


# 1st BZ
plot_bands(1)

# 2nd BZ
plot_bands(2)


# ---------------- Q2 : Density of States ----------------

energies = bands.flatten()

# finer energy grid
E = np.linspace(energies.min(), energies.max(), 2000)

# larger smearing for smoother DOS
eta = 0.2

DOS = np.zeros_like(E)

for e in energies:
    DOS += np.exp(-(E - e)**2 / (2*eta**2))

DOS /= (np.sqrt(2*np.pi) * eta)
DOS /= len(k_vals)

plt.figure(figsize=(10,6))

plt.plot(E, DOS, linewidth=2)

plt.xlabel("Energy")
plt.ylabel("Density of States")
plt.title("Density of States (Nearly Free Electron Model)")

plt.grid(True, alpha=0.3)
plt.xlim(energies.min(), energies.max())

plt.show()


# ---------------- Q3 : Band structure convergence with basis size ----------------

def compute_bands(M, k_vals, A):

    G = np.pi/a
    V1 = A/2
    V2 = A/4
    V3 = A/6

    basis_m = np.arange(-M, M+1)
    bands = []

    for k in k_vals:

        size = len(basis_m)
        H = np.zeros((size,size))

        for i,m in enumerate(basis_m):

            H[i,i] = (k + m*G)**2

            if i+1 < size:
                H[i,i+1] = V1
                H[i+1,i] = V1

            if i+2 < size:
                H[i,i+2] = V2
                H[i+2,i] = V2

            if i+3 < size:
                H[i,i+3] = V3
                H[i+3,i] = V3

        eigvals = np.linalg.eigvalsh(H)
        bands.append(eigvals)

    return np.array(bands)


# different Fourier basis sizes
M_list = [1,2,3,4]

k_vals = np.linspace(-np.pi/a, np.pi/a, 800)

plt.figure(figsize=(10,7))

for M in M_list:

    bands = compute_bands(M, k_vals, A)

    # plot first band for comparison
    plt.plot(k_vals, bands[:,0], label=f"M = {M}")

plt.axvline(np.pi/(2*a), linestyle='--', color='black')
plt.axvline(-np.pi/(2*a), linestyle='--', color='black')

plt.xlabel("k")
plt.ylabel("Energy")
plt.title("Convergence of Band Structure with Fourier Basis Size")
plt.legend()
plt.grid(True)

plt.show()


# ---------------- Q4 (Part 1) : Fermi energy and band structure ----------------

Nk = 1000
k_vals = np.linspace(-np.pi/a, np.pi/a, Nk)

bands = []

for k in k_vals:

    N = len(basis)
    H = np.zeros((N,N))

    for i,m in enumerate(basis):

        H[i,i] = (k + m*G)**2

        if i+1 < N:
            H[i,i+1] = V1
            H[i+1,i] = V1

        if i+2 < N:
            H[i,i+2] = V2
            H[i+2,i] = V2

        if i+3 < N:
            H[i,i+3] = V3
            H[i+3,i] = V3

    bands.append(eigvalsh(H))

bands = np.array(bands)

# --- Compute Fermi energy ---
allE = bands.flatten()
allE.sort()

Ne = Nk // 2          # 1 electron per unit cell (spin degeneracy = 2)
EF = allE[Ne]

print("Fermi Energy =", EF)

# --- Plot band structure with Fermi level ---
plt.figure(figsize=(8,6))

for n in range(5):
    plt.plot(k_vals, bands[:,n])

plt.axhline(EF, color='red', linestyle='--', label='Fermi Energy')

plt.xlabel("k")
plt.ylabel("Energy")
plt.title("Band Structure with Fermi Energy")
plt.legend()
plt.grid()
plt.show()


# ---------------- Q4 (Part 2) : Fermi energy convergence ----------------

Nk_list = [200, 400, 800, 1200, 2000, 4000]
EF_list = []

for Nk in Nk_list:

    k_vals = np.linspace(-np.pi/a, np.pi/a, Nk)

    bands = []

    for k in k_vals:

        N = len(basis)
        H = np.zeros((N,N))

        for i,m in enumerate(basis):

            H[i,i] = (k + m*G)**2

            if i+1 < N:
                H[i,i+1] = V1
                H[i+1,i] = V1

            if i+2 < N:
                H[i,i+2] = V2
                H[i+2,i] = V2

            if i+3 < N:
                H[i,i+3] = V3
                H[i+3,i] = V3

        bands.append(eigvalsh(H))

    bands = np.array(bands)

    allE = bands.flatten()
    allE.sort()

    Ne = Nk // 2
    EF = allE[Ne]

    EF_list.append(EF)

# ---- Convergence plot ----

plt.figure(figsize=(7,5))
plt.plot(Nk_list, EF_list, marker='o')

plt.xlabel("Number of k-points (Nk)")
plt.ylabel("Fermi Energy")
plt.title("Fermi Energy Convergence")
plt.grid()
plt.show()


# ---------------- Q5 : Modified potential ----------------

V1 = A
V2 = A/2
V3 = A/4
V4 = A/6

bands2 = []

for k in k_vals:

    N = len(basis)
    H = np.zeros((N,N))

    for i,m in enumerate(basis):

        # kinetic term
        H[i,i] = (k + m*G/2)**2

        # potential Fourier terms
        if i+1 < N:
            H[i,i+1] = V1
            H[i+1,i] = V1

        if i+2 < N:
            H[i,i+2] = V2
            H[i+2,i] = V2

        if i+3 < N:
            H[i,i+3] = V3
            H[i+3,i] = V3

        if i+4 < N:
            H[i,i+4] = V4
            H[i+4,i] = V4

    bands2.append(eigvalsh(H))

bands2 = np.array(bands2)

# ---- Band structure plot ----

plt.figure(figsize=(8,6))

for n in range(5):
    plt.plot(k_vals, bands2[:,n])

plt.xlabel("k")
plt.ylabel("Energy")
plt.title("Band Structure (Modified Potential)")
plt.grid()
plt.show()


# ---- Density of States ----

energies = bands2.flatten()

E_grid = np.linspace(np.min(energies), np.max(energies), 2000)

eta = 0.2
DOS = np.zeros_like(E_grid)

for E in energies:
    DOS += np.exp(-(E_grid - E)**2/(2*eta**2))

DOS /= (np.sqrt(2*np.pi)*eta)
DOS /= len(k_vals)

plt.figure(figsize=(8,6))
plt.plot(E_grid, DOS)

plt.xlabel("Energy")
plt.ylabel("Density of States")
plt.title("DOS (Modified Potential)")
plt.grid()
plt.show()


# ---------------- Q6 : Total energy per unit cell ----------------

def total_energy(A):

    V1=A
    V2=A/2
    V3=A/4

    bands=[]

    for k in k_vals:

        H=np.zeros((len(basis),len(basis)))

        for i,m in enumerate(basis):

            H[i,i]=(k+m*G)**2

            if i+1<len(basis):
                H[i,i+1]=V1
                H[i+1,i]=V1

        bands.append(eigvalsh(H))

    bands=np.array(bands)

    E=bands.flatten()
    E.sort()

    occ=E[:Nk//2]

    return np.sum(occ)/Nk


A_vals=np.linspace(0,6,20)
Etot=[total_energy(A) for A in A_vals]

plt.figure()
plt.plot(A_vals,Etot,'o-')
plt.xlabel("A")
plt.ylabel("Energy per unit cell")
plt.title("Total energy vs A")
plt.grid()
plt.show()
