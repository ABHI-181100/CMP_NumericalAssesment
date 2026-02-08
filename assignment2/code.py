import numpy as np
import matplotlib.pyplot as plt

#parameters
a = 1.0                                
m_vals = np.arange(-3, 4)                
k_vals = np.linspace(-6*np.pi/a, 6*np.pi/a, 500)

#Q1

E_a = np.array([
    (k_vals + 2*np.pi*m/a)**2
    for m in m_vals
])

plt.figure(figsize=(8, 6))
for i, m in enumerate(m_vals):
    plt.plot(k_vals, E_a[i], label=f"m = {m}")

plt.xlabel("k")
plt.ylabel("E(k)")
plt.title("Free-electron bands (periodicity a)")
plt.legend(ncol=2, fontsize=9)
plt.grid(True)
plt.tight_layout()
plt.show()


# Q2: Periodicity = 2a

E_2a = np.array([
    (k_vals + np.pi*m/a)**2
    for m in m_vals
])

plt.figure(figsize=(8, 6))
for i, m in enumerate(m_vals):
    plt.plot(k_vals, E_2a[i], label=f"m = {m}")

plt.xlabel("k")
plt.ylabel("E(k)")
plt.title("Free-electron bands (periodicity 2a)")
plt.legend(ncol=2, fontsize=9)
plt.grid(True)
plt.tight_layout()
plt.show()
