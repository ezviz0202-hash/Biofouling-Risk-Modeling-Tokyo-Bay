import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
from scipy.integrate import solve_ivp

os.makedirs("output_deb", exist_ok=True)

PARAMS = dict(
    T_ref   = 293.15,
    T_A     = 8_000.0,
    T_AL    = 45_000.0,
    T_AH    = 90_000.0,
    T_L     = 278.15,
    T_H     = 303.15,
    p_Am    = 22.5,
    v_dot   = 0.065,
    kappa   = 0.80,
    k_M     = 0.012,
    E_G     = 4000.0,
    K_food  = 2.5,
    C_D     = 1.2,
    rho_w   = 1025.0,
    tau_0   = 12.5,
    sigma_t = 4.0,
    L_ref   = 1.0,    
    alpha_L = 0.8,    
    L_inf   = 4.0,   
    L0_res  = 0.10,   
)

def arrhenius_correction(T_K, p):
    T   = T_K
    s1  = np.exp(p['T_A'] / p['T_ref'] - p['T_A'] / T)
    s_L = 1.0 + np.exp(p['T_AL'] / T - p['T_AL'] / p['T_L'])
    s_H = 1.0 + np.exp(p['T_AH'] / p['T_H'] - p['T_AH'] / T)
    denom_ref = (1.0 + np.exp(p['T_AL'] / p['T_ref'] - p['T_AL'] / p['T_L'])
                     + np.exp(p['T_AH'] / p['T_H'] - p['T_AH'] / p['T_ref']))
    T_corr = s1 / (s_L + s_H) * denom_ref
    return np.clip(T_corr, 0.0, 3.0)

def functional_response(X, U, p):
    U_crit = 0.20
    U_pen  = 0.35
    beta   = 0.8
    gamma  = 0.6

    enhancement = 1.0 + beta * np.minimum(U / U_crit, 1.0)
    penalty = np.exp(-gamma * np.maximum(U - U_pen, 0.0) / U_pen)

    X_eff = X * enhancement * penalty
    return X_eff / (p['K_food'] + X_eff)

def deb_ode(t, y, T_K, X, U, p):
    e, L = y
    L = max(L, 1e-6)

    T_corr = arrhenius_correction(T_K, p)
    f      = functional_response(X, U, p)

    v_T   = p['v_dot'] * T_corr
    kM_T  = p['k_M']   * T_corr
    pAm_T = p['p_Am']  * T_corr

    
    de_dt = (f - e) * v_T / (L + p['L0_res'])

    
    assim = p['kappa'] * e * v_T
    maint = kM_T * L
    size_factor = max(0.0, 1.0 - L / p['L_inf'])

    numerator = (assim - maint) * size_factor
    dL_dt = numerator / (3.0 * (1.0 + p['E_G'] / (pAm_T / v_T)))

    dL_dt = max(dL_dt, -0.05 * L)
    return [de_dt, dL_dt]

def detachment_probability(U_arr, L_cm, p):
    L_m = L_cm * 0.01

    A_proj = np.pi * (L_m / 2.0) ** 2
    A_adh  = L_m**2 + 1e-12

    tau = 0.5 * p['C_D'] * p['rho_w'] * U_arr**2 * A_proj / A_adh

  
    tau0_eff = p['tau_0'] * (L_cm / p['L_ref']) ** p['alpha_L']

    z = (tau - tau0_eff) / p['sigma_t']
    P = 1.0 - np.exp(-np.exp(z))
    return np.clip(P, 0.0, 1.0)

t_span = (0, 180)
t_eval = np.linspace(0, 180, 500)

baseline_cases = {
    "T=15°C, X=3.0, U=0.05": (288.15, 3.0, 0.05),
    "T=20°C, X=3.0, U=0.05": (293.15, 3.0, 0.05),
    "T=25°C, X=3.0, U=0.05": (298.15, 3.0, 0.05),
    "T=20°C, X=1.5, U=0.05": (293.15, 1.5, 0.05),
    "T=20°C, X=5.0, U=0.05": (293.15, 5.0, 0.05),
}

y0 = [0.5, 0.05]
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(baseline_cases)))

for (label, (T_K, X, U)), color in zip(baseline_cases.items(), colors):
    sol = solve_ivp(deb_ode, t_span, y0, t_eval=t_eval,
                    args=(T_K, X, U, PARAMS), method='RK45', rtol=1e-6)
    axes[0].plot(sol.t, sol.y[1], label=label, color=color, lw=2)
    axes[1].plot(sol.t, sol.y[0], color=color, lw=2, ls='--', label=label)

for ax, ylabel, title in zip(axes,
    ["Structural length L (cm)", "Scaled reserve density e"],
    ["DEB Growth Curves", "DEB Reserve Dynamics"]):
    ax.set_xlabel("Time (days)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight='bold')
    ax.legend(fontsize=7.5, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig("output_deb/growth_curves.png", dpi=150)
plt.close()

T_range = np.linspace(10, 28, 20)
X_range = np.linspace(0.5, 8.0, 20)
U_range = np.linspace(0.01, 0.5, 20)
t_target = 120
t_eval_sa = np.linspace(0, t_target, 300)

def final_length(T_C, X, U):
    T_K  = T_C + 273.15
    sol  = solve_ivp(deb_ode, (0, t_target), y0, t_eval=t_eval_sa,
                     args=(T_K, X, U, PARAMS), method='RK45', rtol=1e-6, max_step=1.0)
    return sol.y[1, -1]

grid_TX = np.zeros((len(T_range), len(X_range)))
for i, T in enumerate(T_range):
    for j, X in enumerate(X_range):
        grid_TX[i, j] = final_length(T, X, U=0.05)

grid_TU = np.zeros((len(T_range), len(U_range)))
for i, T in enumerate(T_range):
    for j, U in enumerate(U_range):
        grid_TU[i, j] = final_length(T, X=3.0, U=U)

for grid, x_arr, x_label, fname, title in [
    (grid_TX, X_range, "Food density X (µg chl-a/L)",
     "sensitivity_heatmap_T_X.png",
     "Sensitivity: L at Day 120 (T vs X)"),
    (grid_TU, U_range, "Flow velocity U (m/s)",
     "sensitivity_heatmap_T_U.png",
     "Sensitivity: L at Day 120 (T vs U)"),
]:
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.contourf(x_arr, T_range, grid, levels=20, cmap='YlOrRd')
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("Structural length L (cm)")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Water temperature T (°C)")
    ax.set_title(title, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"output_deb/{fname}", dpi=150)
    plt.close()

U_sweep = np.linspace(0, 3.0, 300)
sizes   = [0.5, 1.0, 2.0, 3.5, 5.0]
fig, ax = plt.subplots(figsize=(8, 5))
palette = plt.cm.plasma(np.linspace(0.1, 0.9, len(sizes)))
for L_cm, color in zip(sizes, palette):
    P = detachment_probability(U_sweep, L_cm, PARAMS)
    ax.plot(U_sweep, P, lw=2.2, color=color, label=f"L = {L_cm} cm")

ax.set_xlabel("Flow velocity U (m/s)")
ax.set_ylabel("Detachment probability P_detach")
ax.set_ylim(-0.02, 1.05)
ax.grid(True, alpha=0.3)
ax.spines[['top', 'right']].set_visible(False)
ax.legend()
plt.tight_layout()
plt.savefig("output_deb/detachment_probability.png", dpi=150)
plt.close()

L_base = final_length(20.0, 3.0, 0.05)
delta_T, delta_X, delta_U = 2.0, 0.5, 0.05
dL_dT = (final_length(22.0, 3.0, 0.05) - L_base) / delta_T
dL_dX = (final_length(20.0, 3.5, 0.05) - L_base) / delta_X
dL_dU = (final_length(20.0, 3.0, 0.10) - L_base) / delta_U

rows = [
    ("Temperature T", "°C", delta_T, f"{dL_dT:+.4f}"),
    ("Food density X", "µg/L", delta_X, f"{dL_dX:+.4f}"),
    ("Flow velocity U", "m/s", delta_U, f"{dL_dU:+.4f}"),
]
df = pd.DataFrame(rows, columns=["Parameter", "Unit", "Delta", "Sensitivity"])
df["Relative (%)"] = np.abs([dL_dT*delta_T, dL_dX*delta_X, dL_dU*delta_U]) / L_base * 100
df.to_csv("output_deb/sensitivity_summary.csv", index=False)