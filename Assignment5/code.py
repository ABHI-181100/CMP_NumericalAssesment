import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

%matplotlib inline

# ── Global plot aesthetics ──────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi"  : 110,
    "font.size"   : 11,
    "axes.grid"   : True,
    "grid.alpha"  : 0.25,
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
})

# ═══════════════════════════════════════════════════════════════════════════
#  QUESTION 1 — Two-leg ladder: band structure vs hopping ratio
# ═══════════════════════════════════════════════════════════════════════════

def compute_ladder_bands(q, t_leg, t_rung):
    """
    Dispersion of the two-leg ladder model.
    
    Hamiltonian in momentum space has two eigenvalues:
        E±(q) = 2·t_leg·cos(q)  ±  t_rung
    
    Parameters
    ----------
    q      : 1-D array of crystal momenta
    t_leg  : intra-leg (longitudinal) hopping
    t_rung : inter-leg (transverse) hopping
    
    Returns
    -------
    upper, lower bands as arrays
    """
    dispersion_base = 2.0 * t_leg * np.cos(q)
    band_upper = dispersion_base + t_rung
    band_lower = dispersion_base - t_rung
    return band_upper, band_lower


# Fixed rung hopping; scan t_leg / t_rung
t_rung_fixed = -1.0          # eV
ratio_scan   = [-0.1, -0.3, -0.5, -0.7, -1.0, -2.0]
q_vals       = np.linspace(-1.4 * np.pi, 1.4 * np.pi, 4000)

fig, panel = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
fig.suptitle(
    "Q1 — Two-Leg Ladder Dispersion\n"
    r"(fixed $t'$ = 1 eV, varying $t$)",
    fontsize=13, fontweight="bold"
)

for ax, r in zip(panel.flat, ratio_scan):
    t_leg_val = r * t_rung_fixed
    E_up, E_dn = compute_ladder_bands(q_vals, t_leg_val, t_rung_fixed)

    ax.plot(q_vals / np.pi, E_up, color="#3A86FF", lw=2.0, label=r"$E_+$")
    ax.plot(q_vals / np.pi, E_dn, color="#FF6361", lw=2.0, label=r"$E_-$")
    ax.axhline(0, color="gray", lw=0.9, ls="--", alpha=0.7, label="$E=0$")

    # Classify as metal or insulator based on band overlap
    spectral_overlap = E_dn.max() - E_up.min()
    tolerance        = 1e-3
    is_metal         = spectral_overlap > tolerance
    phase_label      = "Metal" if is_metal else "Insulator"

    if is_metal:
        annotation = f"overlap = {spectral_overlap:.2f} eV"
    else:
        annotation = f"gap = {-spectral_overlap:.2f} eV"

    ax.set_title(
        rf"$t/t'$ = {r:.1f}  →  **{phase_label}**" + f"\n{annotation}",
        fontsize=9
    )
    ax.set_xlabel(r"$q\,/\,\pi$")
    ax.set_ylabel("Energy (eV)")
    ax.legend(fontsize=8, loc="upper right")


# ═══════════════════════════════════════════════════════════════════════════
#  QUESTION 2 — Ladder with diagonal (next-nearest) hopping t''
# ═══════════════════════════════════════════════════════════════════════════

lattice_const = 1   # set a = 1

def bands_single_diagonal(q, t_leg, t_rung, t_diag):
    """
    Case (a): one diagonal bond per rung.
    The off-diagonal matrix element picks up a phase:
        |t' + t'' exp(iqa)|
    """
    base  = 2.0 * t_leg * np.cos(q * lattice_const)
    cross = np.abs(t_rung + t_diag * np.exp(1j * q * lattice_const))
    return base + cross, base - cross


def bands_double_diagonal(q, t_leg, t_rung, t_diag):
    """
    Case (b): both diagonals present (X-pattern per rung).
    The coupling becomes real:
        t' + 2·t''·cos(qa)
    """
    base  = 2.0 * t_leg * np.cos(q * lattice_const)
    cross = t_rung + 2.0 * t_diag * np.cos(q * lattice_const)
    return base + cross, base - cross


q_mesh   = np.linspace(-1.4 * np.pi / lattice_const,
                        1.4 * np.pi / lattice_const, 600)
t_nn     = -1.0    # eV  (nearest-neighbour leg hopping)
t_cross  = -1.5   # eV  (rung hopping)
t_nnn_range = [-0.4, 0.0, 0.4]   # diagonal hopping values to survey

fig2, panel2 = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
fig2.suptitle(
    r"Q2 — Ladder + Diagonal Hopping $t''$"
    "\n"
    r"($t = -1$ eV,  $t' = -1.5$ eV)",
    fontsize=12, fontweight="bold"
)

case_info = [
    ("(a) Single diagonal / rung", bands_single_diagonal),
    ("(b) X-pattern diagonals",    bands_double_diagonal),
]

for row_idx, (case_title, band_fn) in enumerate(case_info):
    for col_idx, t_nnn in enumerate(t_nnn_range):
        ax = panel2[row_idx, col_idx]
        E_p, E_m = band_fn(q_mesh, t_nn, t_cross, t_nnn)

        ax.plot(q_mesh / np.pi, E_p, "#3A86FF", lw=2, label=r"$E_+$")
        ax.plot(q_mesh / np.pi, E_m, "#FF6361", lw=2, label=r"$E_-$")
        ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.6)
        ax.fill_between(
            q_mesh / np.pi, E_m, E_p,
            alpha=0.07, color="slategray", label="gap region"
        )

        overlap_val  = E_m.max() - E_p.min()
        phase_str    = "Metal" if overlap_val > 0 else "Insulator"
        detail_str   = (f"overlap = {overlap_val:.2f} eV"
                        if overlap_val > 0
                        else f"gap = {-overlap_val:.2f} eV")

        ax.set_title(
            f"{case_title}\n"
            rf"$t''$ = {t_nnn:+.1f} eV  →  {phase_str}"
            f"\n({detail_str})",
            fontsize=8.5
        )
        ax.set_xlabel(r"$q\,/\,\pi$")
        ax.set_ylabel("Energy (eV)")
        ax.legend(fontsize=7.5)

plt.show()


# ═══════════════════════════════════════════════════════════════════════════
#  QUESTION 3(a) — Square lattice: nearest-neighbour only
# ═══════════════════════════════════════════════════════════════════════════

def sq_dispersion_nn(kx_grid, ky_grid, hop):
    """E(kx, ky) = 2t[cos(kx) + cos(ky)]  (nearest-neighbour square lattice)"""
    return 2.0 * hop * (np.cos(kx_grid) + np.cos(ky_grid))


def fermi_level_at_filling(energy_flat, fillings):
    """
    For each filling n (electrons per unit cell, 0 < n ≤ 2),
    locate the Fermi energy by sorting the DOS histogram.
    """
    sorted_E = np.sort(energy_flat)
    N_states = len(sorted_E)
    EF_vals  = []
    for n_fill in fillings:
        idx = min(int(np.ceil(n_fill / 2.0 * N_states)) - 1, N_states - 1)
        EF_vals.append(sorted_E[idx])
    return EF_vals


# Build k-space grid
Nk_pts    = 300
t_hop     = -2.0                                    # eV
k_axis    = np.linspace(-np.pi, np.pi, Nk_pts)
KX, KY    = np.meshgrid(k_axis, k_axis)
E_nn      = sq_dispersion_nn(KX, KY, t_hop)

# Fillings and corresponding Fermi energies
fill_vals = np.arange(0.5, 2.25, 0.25)
EF_nn     = fermi_level_at_filling(E_nn.ravel(), fill_vals)
palette   = cm.plasma(np.linspace(0.1, 0.9, len(fill_vals)))

fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
fig3.suptitle(
    r"Q3(a) — Square Lattice: NN only  ($t = -2$ eV)",
    fontsize=13, fontweight="bold"
)

# Left: full band contour map
ax = axes3[0]
cf = ax.contourf(KX / np.pi, KY / np.pi, E_nn, levels=60, cmap="RdYlBu_r")
plt.colorbar(cf, ax=ax, label="E (eV)")
ax.set_title(r"Band $E(k_x, k_y)$")
ax.set_xlabel(r"$k_x\,/\,\pi$")
ax.set_ylabel(r"$k_y\,/\,\pi$")
ax.set_aspect("equal")
for boundary in [-1, 1]:
    ax.axhline(boundary, color="k", lw=0.5, ls="--", alpha=0.4)
    ax.axvline(boundary, color="k", lw=0.5, ls="--", alpha=0.4)

# Right: Fermi surface evolution
ax = axes3[1]
ax.contourf(KX / np.pi, KY / np.pi, E_nn, levels=60, cmap="RdYlBu_r", alpha=0.2)
for n_e, EF_e, clr in zip(fill_vals, EF_nn, palette):
    ax.contour(KX / np.pi, KY / np.pi, E_nn,
               levels=[EF_e], colors=[clr], linewidths=2)
    ax.plot([], [], color=clr, lw=2,
            label=rf"$n={n_e:.2f}e$,  $E_F={EF_e:.2f}$ eV")

ax.set_title(r"Fermi Surfaces  ($n = 0.5e \to 2e$, step $0.25e$)")
ax.set_xlabel(r"$k_x\,/\,\pi$")
ax.set_ylabel(r"$k_y\,/\,\pi$")
ax.set_aspect("equal")
ax.legend(fontsize=7.5, loc="upper right", framealpha=0.85)
for boundary in [-1, 1]:
    ax.axhline(boundary, color="k", lw=0.5, ls="--", alpha=0.4)
    ax.axvline(boundary, color="k", lw=0.5, ls="--", alpha=0.4)

plt.show()

# Summary table for Q3(a)
header = f"{'n (e/u.c.)':>18}  {'E_F (eV)':>12}"
print(header)
print("-" * len(header))
for n_e, EF_e in zip(fill_vals, EF_nn):
    print(f"{n_e:>18.2f}  {EF_e:>12.4f}")


# ═══════════════════════════════════════════════════════════════════════════
#  QUESTION 3(b) — Square lattice: NN + NNN hopping
# ═══════════════════════════════════════════════════════════════════════════

def sq_dispersion_nnn(kx_grid, ky_grid, hop_nn, hop_nnn):
    """
    E(kx,ky) = 2·t·(cos kx + cos ky) + 4·t_nnn·cos kx·cos ky
    Second term arises from diagonal (next-nearest) bonds.
    """
    nn_term  = 2.0 * hop_nn  * (np.cos(kx_grid) + np.cos(ky_grid))
    nnn_term = 4.0 * hop_nnn * np.cos(kx_grid) * np.cos(ky_grid)
    return nn_term + nnn_term


hop_nnn_cases = [-1.0, +1.0]   # eV  (antiferromagnetic vs ferromagnetic NNN)

fig4, axes4 = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)
fig4.suptitle(
    r"Q3(b) — Square Lattice: NN + NNN Hopping  ($t = -2$ eV)",
    fontsize=13, fontweight="bold"
)

for row, t_nnn_val in enumerate(hop_nnn_cases):
    E_nnn   = sq_dispersion_nnn(KX, KY, t_hop, t_nnn_val)
    EF_nnn  = fermi_level_at_filling(E_nnn.ravel(), fill_vals)

    # Band structure
    ax_band = axes4[row, 0]
    cf2 = ax_band.contourf(KX / np.pi, KY / np.pi, E_nnn,
                            levels=60, cmap="RdYlBu_r")
    plt.colorbar(cf2, ax=ax_band, label="E (eV)")
    ax_band.set_title(rf"Band Structure  ($t''$ = {t_nnn_val:+.1f} eV)")
    ax_band.set_xlabel(r"$k_x\,/\,\pi$")
    ax_band.set_ylabel(r"$k_y\,/\,\pi$")
    ax_band.set_aspect("equal")

    # Fermi surfaces
    ax_fs = axes4[row, 1]
    ax_fs.contourf(KX / np.pi, KY / np.pi, E_nnn,
                   levels=60, cmap="RdYlBu_r", alpha=0.2)
    for n_e, EF_e, clr in zip(fill_vals, EF_nnn, palette):
        ax_fs.contour(KX / np.pi, KY / np.pi, E_nnn,
                      levels=[EF_e], colors=[clr], linewidths=2)
        ax_fs.plot([], [], color=clr, lw=2,
                   label=rf"$n={n_e:.2f}e$, $E_F={EF_e:.2f}$ eV")

    ax_fs.set_title(rf"Fermi Surfaces  ($t''$ = {t_nnn_val:+.1f} eV)")
    ax_fs.set_xlabel(r"$k_x\,/\,\pi$")
    ax_fs.set_ylabel(r"$k_y\,/\,\pi$")
    ax_fs.set_aspect("equal")
    ax_fs.legend(fontsize=7, loc="upper right", framealpha=0.85)
    ax_fs.set_xlim(-1, 1)
    ax_fs.set_ylim(-1, 1)

    for col in [0, 1]:
        for boundary in [-1, 1]:
            axes4[row, col].axhline(boundary, color="k", lw=0.5,
                                     ls="--", alpha=0.4)
            axes4[row, col].axvline(boundary, color="k", lw=0.5,
                                     ls="--", alpha=0.4)

plt.show()

# Summary table for Q3(b)
for t_nnn_val in hop_nnn_cases:
    E_nnn   = sq_dispersion_nnn(KX, KY, t_hop, t_nnn_val)
    EF_nnn  = fermi_level_at_filling(E_nnn.ravel(), fill_vals)
    print(f"\nt'' = {t_nnn_val:+.1f} eV  |  "
          f"Band min = {E_nnn.min():.3f} eV,  max = {E_nnn.max():.3f} eV")
    print(f"  {'n (e/u.c.)':>14}  {'E_F (eV)':>10}")
    for n_e, EF_e in zip(fill_vals, EF_nnn):
        print(f"  {n_e:>14.2f}  {EF_e:>10.4f}")
