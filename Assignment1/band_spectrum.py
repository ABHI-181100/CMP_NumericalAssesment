
import numpy as np
import matplotlib.pyplot as plt

def create_matrix(E0,b,n):
    A = np.zeros((n,n))
    for i in range(n):
        A[i,i] = E0
        if i < n-1:
            A[i,i+1] = b
            A[i+1,i] = b

    # A[0, n-1] = b     # periodic boundary condition
    # A[n-1, 0] = b       # interaction between first and last atom

    return A

def diagonalization(A):
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    Dia_matrix = np.diag(eigenvalues)
    C_matrix = eigenvectors
    return eigenvalues, C_matrix, Dia_matrix


n  = int(input("Number of atoms n: "))   # number of atoms
dE = float(input("Energy step binwidth ΔE  (eV): "))

a=4.9e-10  # lattice constant in meters for Na BCC
E0 = -5  # on-site energy in eV
b=-1   # hopping parameter in eV 

M1 = create_matrix(E0,b,n)
eigenvalues, C_matrix, Dia_matrix = diagonalization(M1)
# print("Eigenvalues:\n", eigenvalues)
eigenvalues = np.sort(eigenvalues)

for i in range(len(eigenvalues)//2):
    plt.hlines(eigenvalues[i], 0, 2, colors='b')
    plt.hlines(eigenvalues[-(i+1)], 0, 2, colors='r', alpha=0.5)

plt.ylabel("Eigenvalues (Energy eV)")
plt.text(1.5,-4,"Red: Unoccupied Levels\nBlue: Occupied Fermi Levels")
plt.xticks([])
plt.title("Eigenvalue Spectrum")
plt.show()

# plot frequency of eigen values
bin = np.arange(eigenvalues.min(), eigenvalues.max() + dE, dE)
plt.hist(eigenvalues, bins = bin ,orientation='horizontal', color='blue', rwidth=0.85)
plt.title('Histogram of Eigen Values')
plt.ylabel('Eigen Value')
plt.xlabel('Frequency')
plt.show()

def plot_eigenstates(M):

    E,V=np.linalg.eigh(M)
    N=len(E)
    x=np.arange(1,N+1)
    idx={"Minimum Energy":0,"HOMO":N//2-1,"LUMO":N//2,"Maximum Energy":-1}

    plt.figure(figsize=(10,6))
    for k,i in idx.items():
        plt.plot(x,V[:,i],'o-',label=f"{k} (E={E[i]:.2f} eV)")

    plt.xlabel("Atomic site index")
    plt.ylabel("Eigenstate amplitude")
    plt.title(f"Eigenstates for 1D Chain (N={N})")
    plt.legend()
    plt.grid()
    plt.show()

plot_eigenstates(M1)

