# Methodology Notes

This document provides the complete mathematical derivation,
parameter choices, and modeling assumptions underlying both
modules in this repository.

---

## Module 1 — DEB Model for Sessile Marine Organisms

### 1.1 Standard DEB Framework

The model follows the standard Dynamic Energy Budget (DEB) theory
(Kooijman, 2010) for sessile organisms, using two state variables:

| Symbol | Name | Unit |
|--------|------|------|
| `e` | Scaled reserve density | dimensionless (0–1) |
| `L` | Structural length | cm |

The volumetric energy reserve density `[E]` relates to `e` by:
`e = [E] / [Em]` where `[Em] = p_Am / v_dot` is the maximum
reserve density.

### 1.2 Reserve Dynamics

```
de/dt = (f(X, U) − e) · v̇(T) / L
```

The functional response `f` is Holling type-II, extended to include
a flow-velocity-dependent food encounter rate:

```
f(X, U) = X · h(U) / (K + X · h(U))

h(U) = 1 + 0.8 · min(U / U_crit, 1)    [U_crit = 0.25 m/s]
```

At `U = 0`, `h = 1` (no flow enhancement); for `U ≥ U_crit`,
`h = 1.8` (80% enhancement due to increased particle encounter).
This captures the documented positive effect of moderate water flow
on filter-feeder food acquisition (barnacles, mussels).

### 1.3 Growth Dynamics

```
dL/dt = [κ · ṁC − ṗM · L³] / (3 · EG · L²)
```

Simplified for implementation:

```
dL/dt = [κ_eff · p_Am_T · e · v_T / p_Am_T − k_M_T · L] 
        / (3 · (1 + EG / (p_Am_T / v_T)))
```

where all rates carry the Arrhenius temperature correction (§1.4).

Shrinkage is limited to −5% of L per day, consistent with observed
starvation responses in adult barnacles.

### 1.4 Extended Arrhenius Temperature Correction

The five-parameter correction accounts for upper and lower thermal
tolerance limits:

```
c_T = exp(T_A/T_ref − T_A/T) 
      · [1 + exp(T_AL/T_ref − T_AL/T_L) + exp(T_AH/T_H − T_AH/T_ref)]
      / [1 + exp(T_AL/T   − T_AL/T_L) + exp(T_AH/T_H  − T_AH/T)]
```

| Parameter | Value | Description |
|-----------|-------|-------------|
| `T_ref` | 293.15 K (20°C) | Reference temperature |
| `T_A` | 8,000 K | Arrhenius temperature |
| `T_AL` | 45,000 K | Lower boundary Arrhenius temp |
| `T_AH` | 90,000 K | Upper boundary Arrhenius temp |
| `T_L` | 278.15 K (5°C) | Lower tolerance boundary |
| `T_H` | 303.15 K (30°C) | Upper tolerance boundary |

### 1.5 Gumbel Detachment Model

Drag force on the organism:

```
F_drag = ½ · C_D · ρ_w · U² · A_proj
```

For a disc-shaped organism with diameter L: `A_proj = π(L/2)²`

Drag stress (force normalized by adhesion area, also ∝ L²):

```
τ(U, L) = F_drag / A_adh ∝ ½ · C_D · ρ_w · U²
```

Note: area scaling alone makes τ approximately independent of L for
geometrically similar organisms, so the size effect is introduced
through a size-dependent adhesion threshold:

```
τ₀,eff(L) = τ₀,ref · (L / L_ref)^α
```

with `L_ref = 1 cm` and `α = 0.8` in the prototype implementation.
This shifts the detachment curve to higher flow velocity for larger
organisms.

Gumbel CDF:

```
P_detach = 1 − exp(−exp((τ − τ₀) / σ))
```

| Parameter | Value | Unit | Source |
|-----------|-------|------|--------|
| `C_D` | 1.2 | — | Bluff body (barnacle shape) |
| `ρ_w` | 1025 | kg/m³ | Seawater |
| `τ₀,ref` | 12.5 | N/m² | Reference adhesion stress at `L_ref = 1 cm` |
| `L_ref` | 1.0 | cm | Reference organism size |
| `α` | 0.8 | — | Size-scaling exponent for adhesion threshold |
| `σ` | 4.0 | N/m² | Gumbel scale (to be calibrated) |

**Calibration plan**: water tank oscillating-flow experiments
with *Balanus amphitrite* settlers at U = 0.1–1.5 m/s, measuring
detachment onset by video analysis.

### 1.6 Sensitivity Analysis Method

First-order finite-difference approximation:

```
∂L/∂p ≈ [L(p + Δp) − L(p)] / Δp
```

evaluated at `t = 120` days (mid-growth-season reference point).
To avoid unrealistically fast reserve equilibration at very small size,
the reserve equation is regularized as `de/dt = (f - e)·v̇(T)/(L + L0)`
with `L0 = 0.10 cm`. Growth is also slowed near an asymptotic structural
length `L_inf = 4.0 cm` through a multiplicative factor
`max(0, 1 - L/L_inf)`. Perturbations: `ΔT = 2°C`, `ΔX = 0.5 µg/L`,
`ΔU = 0.05 m/s`.

---

## Module 2 — Tokyo Bay Site-Specific Risk Prediction

### 2.1 Environmental Variables

Six predictor variables per station-day:

| Variable | Symbol | Unit | Primary effect on biofouling |
|----------|--------|------|------------------------------|
| Water temperature | T | °C | Controls DEB growth rate via Arrhenius |
| Chlorophyll-a | X | µg/L | Proxy for food availability (phytoplankton) |
| Salinity | S | PSU | Osmotic stress; affects settlement |
| Turbidity | Turb | NTU | Light attenuation; inorganic particle interference |
| Flow velocity | U | m/s | Food encounter rate; detachment forcing |
| Dissolved oxygen | DO | mg/L | Metabolic constraint in hypoxic conditions |

Day-of-year (DOY) is included as an additional feature capturing
unmeasured seasonal forcing (photoperiod, planktonic larval supply).

### 2.2 Risk Score Construction

The continuous biofouling risk score `R ∈ [0, 100]` is constructed
from DEB-derived sub-scores:

```
R = T_score · f_score · U_adh · DO_score · 100

T_score  = exp(−0.5 · ((T − T_opt) / 6)²)        [T_opt = 22°C]
f_score  = X / (K + X)                             [K = 2.5 µg/L]
U_adh    = exp(−0.5 · ((U − U_opt) / 0.12)²)      [U_opt = 0.08 m/s]
DO_score = clip((DO − 2.0) / 8.0,  0, 1)
```

Binary label: `risk_label = 1` if `R > median(R)`.
The 50th percentile threshold is ecologically motivated —
it separates "actively growing season" (high risk) from
"growth-suppressed season" (low risk).

### 2.3 Random Forest Configuration

| Hyperparameter | Value | Rationale |
|----------------|-------|-----------|
| n_estimators | 300 | Sufficient for stable OOB error |
| max_depth | 10 | Prevents overfitting on 11K records |
| min_samples_leaf | 10 | Regularization for coastal data noise |
| class_weight | balanced | Equal training weight for risk classes |
| CV | 5-fold stratified | Preserves class balance in splits |

### 2.4 Spatial Interpolation

Station-level risk scores are interpolated to a continuous
`200 × 200` grid (139.60°E–140.02°E, 35.18°N–35.72°N) using
Radial Basis Function interpolation with a thin-plate-spline kernel
and smoothing factor 5.

```python
interp = RBFInterpolator(station_coords, risk_values,
                         kernel='thin_plate_spline', smoothing=5)
```

### 2.5 Data Sources (for real deployment)

Replace synthetic data generation with:

```python
# Example — MLIT Tokyo Bay data (requires registration)
# https://www.pa.ktr.mlit.go.jp/tokyowan/

# J-DOSS API
# https://jdoss1.jodc.go.jp/vpage/point_data_d.html
```

The synthetic dataset is generated to match published Tokyo Bay
seasonal statistics from Hayami & Fujiwara (2012), Fujii & Chai (2007),
and MLIT annual environmental monitoring reports.

---

## Planned: Coupled FEM–CFD–DEB Model

The full research plan extends this prototype with a three-way
coupled solver:

```
Wave forcing → FEM membrane deformation → CFD internal flow field
                                              ↓
                                       DEB biological model
                                       (spatially distributed)
                                              ↓
                                    Biofouling attachment map
```

**FEM component**: HDPE membrane modeled as nonlinear elastic shell;
OpenFOAM structural solver or Abaqus (to be determined based on
licensing at IIS).

**CFD component**: Volume-of-fluid (VOF) free surface within the cage;
k-ω SST turbulence model; one-way coupling via moving boundary
condition from FEM deformation field.

**DEB component**: Spatially distributed at cage surface nodes;
forcing updated from CFD velocity and passive scalar (nutrient) fields
at each time step.

---

## References

Kooijman, S.A.L.M. (2010). *Dynamic Energy Budget Theory for
Metabolic Organisation* (3rd ed.). Cambridge University Press.

Hayami, Y. & Fujiwara, T. (2012). Seasonal variation of dissolved
oxygen in Tokyo Bay. *Journal of Oceanography*, 68, 641–655.

Fujii, M. & Chai, F. (2007). Modeling carbon and silicon cycling in
the equatorial Pacific. *Deep-Sea Research II*, 54, 496–520.
