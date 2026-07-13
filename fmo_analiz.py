"""
================================================================================
 FMO (FRONTIER MOLECULAR ORBITAL) THEORY — HUCKEL MOLECULAR ORBITAL MODULE
================================================================================
For conjugated pi-systems (Ethylene, Butadiene, Hexatriene, Benzene):

  1. Programmatic construction of the Huckel Hamiltonian matrix (H)
  2. Solving the secular equation (H - E S)C = 0, S = I, via scipy.linalg.eigh
  3. Electron placement under the Pauli Exclusion Principle, HOMO/LUMO
     determination, and the HOMO-LUMO gap (Delta E)
  4. Stark Effect: electric-field-dependent perturbation of the H diagonal
     and a full field-sweep simulation
  5. A 4-panel, publication-quality analysis figure built with
     matplotlib + seaborn
  6. Structuring and reporting of all computed raw data as pandas.DataFrame
     tables

Author: Computational Chemistry / Quantum Biology / Data Science module
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.linalg import eigh

# ==============================================================================
# 0. GLOBAL SETTINGS
# ==============================================================================
sns.set_theme(style="darkgrid", context="notebook", font_scale=0.95)
pd.set_option("display.float_format", lambda x: f"{x:8.4f}")
pd.set_option("display.width", 120)
pd.set_option("display.max_columns", None)

# Huckel parameters (used at full precision, with no simplification)
ALPHA_C = 0.0     # eV — Carbon 2p Coulomb integral (reference / zero-point energy)
BETA_CC = -2.71   # eV — resonance integral between neighboring p_z orbitals
CC_BOND_LENGTH_NM = 0.140   # nm — typical conjugated C-C bond length


# ==============================================================================
# 1. MOLECULAR TOPOLOGY AND GEOMETRY
# ==============================================================================
def _zigzag_positions(n):
    """2D zigzag backbone coordinates (nm) for an n-atom linear conjugated chain."""
    d = CC_BOND_LENGTH_NM
    pos = [np.array([0.0, 0.0])]
    for i in range(1, n):
        angle = np.radians(30 if i % 2 == 1 else -30)
        step = d * np.array([np.cos(angle), np.sin(angle)])
        pos.append(pos[-1] + step)
    pos = np.array(pos)
    pos -= pos.mean(axis=0)
    return pos


def _ring_positions(n):
    """Regular-polygon coordinates (nm) for an n-atom ring (benzene)."""
    R = CC_BOND_LENGTH_NM / (2 * np.sin(np.pi / n))
    angles = np.arange(n) * 2 * np.pi / n + np.pi / 2
    return np.column_stack([R * np.cos(angles), R * np.sin(angles)])


def build_molecule(name):
    """
    For the supported molecules, returns the atom count, bond (adjacency)
    list, number of pi-electrons, and 2D skeletal coordinates.
    """
    name = name.strip().lower()

    if name in ("ethylene", "ethene"):
        n, cyclic = 2, False
    elif name in ("butadiene", "1,3-butadiene"):
        n, cyclic = 4, False
    elif name in ("hexatriene", "1,3,5-hexatriene"):
        n, cyclic = 6, False
    elif name in ("benzene",):
        n, cyclic = 6, True
    else:
        raise ValueError(f"Unknown molecule: {name!r}. "
                          f"Options: Ethylene, Butadiene, Hexatriene, Benzene.")

    bonds = [(i, i + 1) for i in range(n - 1)]
    if cyclic:
        bonds.append((n - 1, 0))
        positions = _ring_positions(n)
    else:
        positions = _zigzag_positions(n)

    n_pi_electrons = n  # one pi-electron per carbon in a neutral alternant hydrocarbon

    return {
        "name": name.capitalize(),
        "n_atoms": n,
        "bonds": bonds,
        "positions": positions,       # (n, 2) in nm
        "n_pi_electrons": n_pi_electrons,
        "cyclic": cyclic,
    }


# ==============================================================================
# 2. HUCKEL HAMILTONIAN AND THE STARK EFFECT
# ==============================================================================
def build_hamiltonian(mol, field_V_per_nm=0.0, field_axis=0):
    """
    Constructs the Huckel Hamiltonian matrix (n x n).

        H_ii = alpha + q * F * x_i        (first-order Stark perturbation)
        H_ij = beta                        if i, j are bonded (sigma-linked) atoms
        H_ij = 0                           otherwise

    Parameters
    ----------
    mol : dict              output of build_molecule()
    field_V_per_nm : float  external electric field strength (V/nm)
    field_axis : int        axis along which the field is applied (0=x, 1=y)

    Note: F [V/nm] * x [nm] = energy [eV] (with unit charge q=+1e), so the
    unit analysis is direct and physically consistent.
    """
    n = mol["n_atoms"]
    x = mol["positions"][:, field_axis]

    H = np.zeros((n, n), dtype=float)
    np.fill_diagonal(H, ALPHA_C + field_V_per_nm * x)

    for i, j in mol["bonds"]:
        H[i, j] = BETA_CC
        H[j, i] = BETA_CC

    return H


def solve_huckel(H):
    """
    Solves the secular equation (H - E S) C = 0, under the S = I assumption.
    scipy.linalg.eigh is a numerically stable, exact (non-approximate)
    eigenvalue/eigenvector solver for symmetric/Hermitian matrices.

    Returns
    -------
    energies : (n,) eigenvalues in ascending order (exact energy levels, eV)
    coeffs   : (n,n) coefficient matrix C_ni whose columns are eigenvectors
               (C[:, k] = atomic coefficients of the k-th molecular orbital)
    """
    S = np.eye(H.shape[0])  # overlap matrix (Huckel approximation: S = I)
    energies, coeffs = eigh(H, S)   # generalized symmetric eigenproblem A x = lambda B x
    return energies, coeffs


def fill_electrons(energies, n_pi_electrons):
    """
    Places electrons in pairs into the lowest-energy orbitals first
    (Aufbau principle), respecting the Pauli Exclusion Principle.

    Returns
    -------
    occupations : (n,) electron count per orbital (0, 1, or 2)
    homo_idx, lumo_idx : int
    """
    n = len(energies)
    occupations = np.zeros(n, dtype=int)
    remaining = n_pi_electrons
    for i in range(n):
        if remaining <= 0:
            break
        put = min(2, remaining)
        occupations[i] = put
        remaining -= put

    occupied_idx = np.where(occupations > 0)[0]
    homo_idx = int(occupied_idx.max()) if len(occupied_idx) else -1
    lumo_idx = homo_idx + 1 if homo_idx + 1 < n else -1
    return occupations, homo_idx, lumo_idx


# ==============================================================================
# 3. SINGLE-CALL ANALYSIS FUNCTION FOR A FULL FMO WORKUP
# ==============================================================================
def analyze_molecule(name, field_V_per_nm=0.0):
    """
    Runs the complete Huckel FMO analysis for one molecule and returns all
    the raw data in a structured dictionary.
    """
    mol = build_molecule(name)
    H = build_hamiltonian(mol, field_V_per_nm=field_V_per_nm)
    energies, coeffs = solve_huckel(H)
    occupations, homo_idx, lumo_idx = fill_electrons(energies, mol["n_pi_electrons"])

    total_pi_energy = float(np.sum(energies * occupations))
    e_homo = float(energies[homo_idx]) if homo_idx >= 0 else np.nan
    e_lumo = float(energies[lumo_idx]) if lumo_idx >= 0 else np.nan
    gap = e_lumo - e_homo if (homo_idx >= 0 and lumo_idx >= 0) else np.nan
    hardness = gap / 2 if not np.isnan(gap) else np.nan
    softness = 1 / hardness if (not np.isnan(hardness) and hardness != 0) else np.nan
    electronegativity = -(e_homo + e_lumo) / 2 if not np.isnan(gap) else np.nan

    return {
        "mol": mol,
        "H": H,
        "energies": energies,
        "coeffs": coeffs,
        "occupations": occupations,
        "homo_idx": homo_idx,
        "lumo_idx": lumo_idx,
        "total_pi_energy": total_pi_energy,
        "e_homo": e_homo,
        "e_lumo": e_lumo,
        "gap": gap,
        "hardness": hardness,
        "softness": softness,
        "electronegativity": electronegativity,
        "field_V_per_nm": field_V_per_nm,
    }


# ==============================================================================
# 4. STARK EFFECT — ELECTRIC FIELD SWEEP
# ==============================================================================
def stark_field_sweep(name, field_max=5.0, n_points=26):
    """
    Ramps the electric field from 0 to field_max (V/nm) and computes the
    HOMO-LUMO gap at every step.

    Returns: pandas.DataFrame [Field_V_per_nm, E_HOMO, E_LUMO, Gap_eV, Hardness_eV]
    """
    field_values = np.linspace(0.0, field_max, n_points)
    rows = []
    for F in field_values:
        result = analyze_molecule(name, field_V_per_nm=F)
        rows.append({
            "Field_V_per_nm": F,
            "E_HOMO_eV": result["e_homo"],
            "E_LUMO_eV": result["e_lumo"],
            "Gap_eV": result["gap"],
            "Hardness_eV": result["hardness"],
        })
    return pd.DataFrame(rows)


# ==============================================================================
# 5. DATA FRAMES (pandas.DataFrame REPORTING)
# ==============================================================================
def build_energy_dataframe(result):
    n = len(result["energies"])
    homo_idx, lumo_idx = result["homo_idx"], result["lumo_idx"]
    rows = []
    for i in range(n):
        if i == homo_idx:
            role = "HOMO"
        elif i == lumo_idx:
            role = "LUMO"
        elif result["occupations"][i] > 0:
            role = "occupied"
        else:
            role = "virtual (empty)"
        rows.append({
            "MO": f"psi_{i+1}",
            "Energy_eV": result["energies"][i],
            "Electron_Count": result["occupations"][i],
            "Role": role,
        })
    df = pd.DataFrame(rows)
    return df.sort_values("Energy_eV").reset_index(drop=True)


def build_coefficient_dataframe(result):
    mol = result["mol"]
    n = mol["n_atoms"]
    cols = {f"psi_{k+1}": result["coeffs"][:, k] for k in range(n)}
    df = pd.DataFrame(cols, index=[f"C{i+1}" for i in range(n)])
    df.index.name = "Atom"
    return df


def build_summary_dataframe(result):
    return pd.DataFrame([{
        "Molecule": result["mol"]["name"],
        "Atom_Count": result["mol"]["n_atoms"],
        "Pi_Electrons": result["mol"]["n_pi_electrons"],
        "Field_V_per_nm": result["field_V_per_nm"],
        "Total_Pi_Energy_eV": result["total_pi_energy"],
        "E_HOMO_eV": result["e_homo"],
        "E_LUMO_eV": result["e_lumo"],
        "Gap_eV": result["gap"],
        "Hardness_eta_eV": result["hardness"],
        "Softness_S_eV-1": result["softness"],
        "Electronegativity_chi_eV": result["electronegativity"],
    }])


# ==============================================================================
# 6. VISUALIZATION — 4-PANEL ANALYSIS FIGURE
# ==============================================================================
PALETTE = {
    "homo": "#2E86AB",
    "lumo": "#D64545",
    "occupied": "#5C7A99",
    "virtual": "#9AA7B0",
    "spin_up": "#1B4965",
    "spin_down": "#B23A48",
}


def _plot_energy_levels(ax, result):
    energies = result["energies"]
    occ = result["occupations"]
    homo_idx, lumo_idx = result["homo_idx"], result["lumo_idx"]
    n = len(energies)

    # separate near-degenerate levels horizontally
    groups, used = [], [False] * n
    for i in range(n):
        if used[i]:
            continue
        grp = [i]
        used[i] = True
        for j in range(i + 1, n):
            if not used[j] and abs(energies[j] - energies[i]) < 1e-2:
                grp.append(j)
                used[j] = True
        groups.append(grp)

    for grp in groups:
        total_w = len(grp) * 0.7
        start_x = -total_w / 2
        for k, idx in enumerate(grp):
            xc = start_x + k * 0.7 + 0.35
            e = energies[idx]
            if idx == homo_idx:
                color, lw, ls = PALETTE["homo"], 3.0, "-"
            elif idx == lumo_idx:
                color, lw, ls = PALETTE["lumo"], 3.0, "-"
            elif occ[idx] > 0:
                color, lw, ls = PALETTE["occupied"], 2.0, "-"
            else:
                color, lw, ls = PALETTE["virtual"], 2.0, "-"

            ax.hlines(e, xc - 0.28, xc + 0.28, colors=color, linewidth=lw, linestyle=ls, zorder=3)
            ax.annotate(f"$\\psi_{{{idx+1}}}$", xy=(xc + 0.30, e), fontsize=8, va="center", color="0.3")

            # spin arrows (Pauli principle)
            if occ[idx] >= 1:
                ax.annotate("", xy=(xc - 0.10, e + 0.045 * (energies.max() - energies.min() + 1e-9)),
                            xytext=(xc - 0.10, e - 0.045 * (energies.max() - energies.min() + 1e-9)),
                            arrowprops=dict(arrowstyle="-|>", color=PALETTE["spin_up"], lw=1.6))
            if occ[idx] == 2:
                ax.annotate("", xy=(xc + 0.10, e - 0.045 * (energies.max() - energies.min() + 1e-9)),
                            xytext=(xc + 0.10, e + 0.045 * (energies.max() - energies.min() + 1e-9)),
                            arrowprops=dict(arrowstyle="-|>", color=PALETTE["spin_down"], lw=1.6))

    # HOMO / LUMO dashed reference lines + gap label
    if homo_idx >= 0:
        ax.axhline(energies[homo_idx], color=PALETTE["homo"], linestyle="--", linewidth=1.1, alpha=0.55, zorder=1)
    if lumo_idx >= 0:
        ax.axhline(energies[lumo_idx], color=PALETTE["lumo"], linestyle="--", linewidth=1.1, alpha=0.55, zorder=1)
        gap = energies[lumo_idx] - energies[homo_idx]
        ymid = (energies[homo_idx] + energies[lumo_idx]) / 2
        ax.annotate(
            "", xy=(1.55, energies[lumo_idx]), xytext=(1.55, energies[homo_idx]),
            arrowprops=dict(arrowstyle="<->", color="0.2", lw=1.3),
        )
        ax.text(1.65, ymid, f"$\\Delta E$ = {gap:.3f} eV", fontsize=8.5, color="0.15",
                va="center", rotation=90)

    ax.set_xlim(-1.6, 2.4)
    ax.set_xticks([])
    ax.set_ylabel("Energy (eV)")
    ax.set_title(f"{result['mol']['name']} — MO Energy Diagram (F = {result['field_V_per_nm']:.2f} V/nm)",
                 fontsize=10.5, fontweight="bold")
    pad = 0.12 * (energies.max() - energies.min() + 1e-9) + 0.4
    ax.set_ylim(energies.min() - pad, energies.max() + pad)


def _plot_hamiltonian_heatmap(ax, result):
    H = result["H"]
    n = H.shape[0]
    labels = [f"C{i+1}" for i in range(n)]
    sns.heatmap(
        H, ax=ax, cmap="RdBu_r", center=0, annot=True, fmt=".2f",
        xticklabels=labels, yticklabels=labels, cbar_kws={"label": "Energy (eV)"},
        linewidths=0.6, linecolor="white", square=True, annot_kws={"fontsize": 8},
    )
    ax.set_title(f"Huckel Hamiltonian $H$ — Heatmap\n"
                 f"($\\alpha$={ALPHA_C:.2f} eV, $\\beta$={BETA_CC:.2f} eV)",
                 fontsize=10.5, fontweight="bold")


def _plot_orbital_coefficients(ax, result):
    coeffs = result["coeffs"]
    mol = result["mol"]
    n = mol["n_atoms"]
    homo_idx, lumo_idx = result["homo_idx"], result["lumo_idx"]

    atom_labels = [f"C{i+1}" for i in range(n)]
    homo_vals = coeffs[:, homo_idx] if homo_idx >= 0 else np.zeros(n)
    lumo_vals = coeffs[:, lumo_idx] if lumo_idx >= 0 else np.zeros(n)

    df_bar = pd.DataFrame({
        "Atom": atom_labels * 2,
        "Coefficient": np.concatenate([homo_vals, lumo_vals]),
        "Orbital": [f"HOMO ($\\psi_{{{homo_idx+1}}}$)"] * n + [f"LUMO ($\\psi_{{{lumo_idx+1}}}$)"] * n,
    })

    sns.barplot(
        data=df_bar, x="Atom", y="Coefficient", hue="Orbital", ax=ax,
        palette=[PALETTE["homo"], PALETTE["lumo"]], edgecolor="0.2", linewidth=0.7,
    )
    ax.axhline(0, color="0.3", linewidth=1.0)
    ax.set_title("HOMO / LUMO Orbital Coefficient Distribution ($C_{ni}$)", fontsize=10.5, fontweight="bold")
    ax.set_ylabel("Coefficient Amplitude $C_{ni}$")
    ax.set_xlabel("Atom (pi-center)")
    ax.legend(fontsize=8, loc="best")


def _plot_field_sweep(ax, df_sweep, mol_name):
    ax.plot(df_sweep["Field_V_per_nm"], df_sweep["Gap_eV"], marker="o", markersize=4,
            color="#6A3D9A", linewidth=2.0, label="HOMO-LUMO Gap")
    ax.fill_between(df_sweep["Field_V_per_nm"], df_sweep["Gap_eV"], alpha=0.12, color="#6A3D9A")
    ax.set_xlabel("External Electric Field $F$ (V/nm)")
    ax.set_ylabel("$\\Delta E_{HOMO-LUMO}$ (eV)")
    ax.set_title(f"{mol_name} — Stark Effect: Field Sweep (0 -> {df_sweep['Field_V_per_nm'].max():.1f} V/nm)",
                 fontsize=10.5, fontweight="bold")
    ax.legend(fontsize=8.5)


def build_full_report_figure(result, df_sweep, save_path=None):
    """
    Builds the full 4-panel analysis figure in a 2x2 GridSpec layout:
      [0,0] Energy Level Diagram          [0,1] Hamiltonian Heatmap
      [1,0] Orbital Coefficient Bar Plot  [1,1] Stark Field Sweep
    """
    fig = plt.figure(figsize=(15, 11))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.28)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    _plot_energy_levels(ax1, result)
    _plot_hamiltonian_heatmap(ax2, result)
    _plot_orbital_coefficients(ax3, result)
    _plot_field_sweep(ax4, df_sweep, result["mol"]["name"])

    fig.suptitle(
        f"Frontier Molecular Orbital (FMO) Analysis — {result['mol']['name']}  |  Huckel Method (S = I)",
        fontsize=14, fontweight="bold", y=0.985,
    )

    if save_path:
        fig.savefig(save_path, dpi=160, bbox_inches="tight", facecolor="white")
    return fig


# ==============================================================================
# 7. REPORTING (terminal output)
# ==============================================================================
def print_report(result, df_sweep):
    mol = result["mol"]
    sep = "=" * 88

    print(sep)
    print(f" FMO ANALYSIS REPORT — {mol['name'].upper()}  (Huckel Molecular Orbital Method)")
    print(sep)
    print(f"  Atom count            : {mol['n_atoms']}")
    print(f"  Pi-electron count     : {mol['n_pi_electrons']}")
    print(f"  Coulomb integral a    : {ALPHA_C:.3f} eV")
    print(f"  Resonance integral b  : {BETA_CC:.3f} eV")
    print(f"  Applied field F       : {result['field_V_per_nm']:.3f} V/nm")
    print()

    print("-- Hamiltonian Matrix H (eV) --")
    df_H = pd.DataFrame(result["H"],
                         index=[f"C{i+1}" for i in range(mol['n_atoms'])],
                         columns=[f"C{i+1}" for i in range(mol['n_atoms'])])
    print(df_H.to_string())
    print()

    print("-- Molecular Orbital Energy Levels --")
    df_E = build_energy_dataframe(result)
    print(df_E.to_string(index=False))
    print()

    print("-- Orbital Coefficient Matrix C_ni (rows=atom, columns=MO) --")
    df_C = build_coefficient_dataframe(result)
    print(df_C.to_string())
    print()

    print("-- Summary of Quantum Chemical Descriptors --")
    df_S = build_summary_dataframe(result)
    print(df_S.to_string(index=False))
    print()

    print(f"-- Stark Effect Field Sweep (0 -> {df_sweep['Field_V_per_nm'].max():.1f} V/nm, "
          f"{len(df_sweep)} steps) --")
    print(df_sweep.iloc[[0, len(df_sweep)//4, len(df_sweep)//2, (3*len(df_sweep))//4, -1]]
          .to_string(index=False))
    print(sep)


def compare_all_molecules():
    """Summary table comparing the zero-field HOMO-LUMO gap of all four molecules."""
    names = ["Ethylene", "Butadiene", "Hexatriene", "Benzene"]
    rows = []
    for name in names:
        r = analyze_molecule(name, field_V_per_nm=0.0)
        rows.append(build_summary_dataframe(r).iloc[0])
    df = pd.DataFrame(rows).reset_index(drop=True)
    print("\n" + "=" * 88)
    print(" CROSS-MOLECULE COMPARISON — Zero Field (F = 0 V/nm) ")
    print("=" * 88)
    print(df.to_string(index=False))
    print("=" * 88)
    return df


# ==============================================================================
# 8. MAIN EXECUTION BLOCK
# ==============================================================================
if __name__ == "__main__":

    # --- Primary molecule for the full analysis ---
    TARGET_MOLECULE = "Benzene"

    result = analyze_molecule(TARGET_MOLECULE, field_V_per_nm=0.0)
    df_sweep = stark_field_sweep(TARGET_MOLECULE, field_max=5.0, n_points=26)

    print_report(result, df_sweep)
    compare_all_molecules()

    fig = build_full_report_figure(result, df_sweep, save_path="fmo_analysis_report_en.png")
    plt.show()