import numpy as np
import matplotlib.pyplot as plt

def create_matrix(E0,b,n):
    A = np.zeros((n,n))
    for i in range(n):
        A[i,i] = E0
        if i < n-1:
            A[i,i+1] = b
            A[i+1,i] = b
    
    A[0, n-1] = b     # periodic boundary condition
    A[n-1, 0] = b       # interaction between first and last atom

    return A

def diagonalization(A):
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    Dia_matrix = np.diag(eigenvalues)
    C_matrix = eigenvectors
    return eigenvalues, C_matrix, Dia_matrix



a=4.9e-10  # lattice constant in meters for Na BCC
E0 = -5  # on-site energy in eV
b=-1   # hopping parameter in eV
n=100  # number of atoms

M1 = create_matrix(E0,b,n)
eigenvalues, C_matrix, Dia_matrix = diagonalization(M1)
print("Eigenvalues:\n", eigenvalues)

for E in eigenvalues:
    plt.hlines(E, -0.1, 0.1, alpha=0.6)

plt.ylabel("Eigenvalues (Energy eV)")
plt.xticks([])
plt.title("Eigenvalue Spectrum")
plt.show()



m = np.arange(n)
k = 2 * np.pi * m / (n * a)
k = (k + np.pi/a) % (2*np.pi/a) - np.pi/a       # fold into first Brillouin zone
idx = np.argsort(k)
k = k[idx]
E = eigenvalues[idx]


k_cont = np.linspace(-(np.pi/a), (np.pi/a), 1000)
E_cont = E0 - 2*b*np.cos(k_cont * a)

plt.plot(k_cont, E_cont, '-', label="Analytical")
plt.xlabel("k (1/m)")
plt.ylabel("Energy (eV)")
plt.legend()
plt.title("Tight-binding dispersion")
plt.show()



k = np.linspace(-np.pi/a, np.pi/a, 1000) # in 1st Brillouin zone
E = E0 - 2*b*np.cos(k*a)
dos, energy = np.histogram(E, bins=200, density=True)
E_mid = 0.5*(energy[1:] + energy[:-1])

plt.plot(E_mid, dos)
plt.xlabel("Energy (eV)")
plt.ylabel("DOS ")
plt.show()






