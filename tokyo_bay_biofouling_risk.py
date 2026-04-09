import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import RBFInterpolator

os.makedirs("output_tokyobay", exist_ok=True)
np.random.seed(42)

STATIONS = {
    "ST01_Odaiba": (139.775, 35.628, "inner"),
    "ST02_Keihin": (139.795, 35.555, "inner"),
    "ST03_Tokyo": (139.765, 35.598, "inner"),
    "ST04_Yokohama": (139.655, 35.442, "middle"),
    "ST05_Kawasaki": (139.820, 35.508, "middle"),
    "ST06_Kanazawa": (139.640, 35.345, "middle"),
    "ST07_Kisarazu": (139.910, 35.382, "outer"),
    "ST08_Futtsu": (139.848, 35.305, "outer"),
    "ST09_Uraga_N": (139.748, 35.253, "mouth"),
    "ST10_Uraga_S": (139.720, 35.216, "mouth"),
}

station_chars = {
    "inner":  dict(T_bias=1.5, chl_bias=3.5, turb_bias=5.0, U_mean=0.05),
    "middle": dict(T_bias=0.5, chl_bias=1.0, turb_bias=1.5, U_mean=0.12),
    "outer":  dict(T_bias=0.0, chl_bias=0.0, turb_bias=0.0, U_mean=0.20),
    "mouth":  dict(T_bias=-1.0, chl_bias=-1.5, turb_bias=-2.0, U_mean=0.38),
}

def seasonal(doy, amp, phase, base, noise=0.0):
    return base + amp * np.sin(2 * np.pi * (doy - phase) / 365) + np.random.normal(0, noise, len(doy))

def generate_station_data(name, lon, lat, zone):
    ch = station_chars[zone]
    doy = np.tile(np.arange(1, 366), 3)
    
    T = np.clip(seasonal(doy, 9.0, 200, 17.5 + ch['T_bias'], 0.8), 5.0, 31.0)
    chl = np.clip(1.2 + (6.0 + ch['chl_bias']) * np.exp(-0.5 * ((doy - 100) / 25)**2) + 
                  (4.5 + ch['chl_bias'] * 0.7) * np.exp(-0.5 * ((doy - 210) / 45)**2) + 
                  np.random.normal(0, 0.5, len(doy)), 0.2, 22.0)
    sal = np.clip(seasonal(doy, -1.2, 220, 31.5 - ch['T_bias'] * 0.3, 0.3), 26.0, 34.5)
    turb = np.clip(3.0 + ch['turb_bias'] + 2.0 * np.exp(-0.5 * ((doy - 100) / 40)**2) + 
                   np.random.exponential(0.8, len(doy)), 0.5, 30.0)
    U = np.clip(ch['U_mean'] + 0.02 * np.sin(2 * np.pi * np.arange(len(doy)) / 14.8) + 
                np.random.exponential(0.01, len(doy)), 0.005, 0.8)
    do = np.clip(9.0 - ch['T_bias'] * 0.2 - 3.5 * np.exp(-0.5 * ((doy - 215) / 35)**2) + 
                 np.random.normal(0, 0.4, len(doy)), 1.0, 12.5)

    risk = (np.exp(-0.5 * ((T - 22.0) / 6.0)**2) * (chl / (2.5 + chl)) * np.exp(-0.5 * ((U - 0.08) / 0.12)**2) * np.clip((do - 2.0) / 8.0, 0, 1) * 100.0)
    risk = np.clip(risk + np.random.normal(0, 3.0, len(risk)), 0, 100)
    
    return pd.DataFrame({"station": name, "zone": zone, "lon": lon, "lat": lat, "doy": doy,
                         "T_C": T, "chl_a": chl, "salinity": sal, "turbidity": turb, "flow_U": U, 
                         "do_mgl": do, "risk_score": risk, 
                         "risk_label": (risk > np.percentile(risk, 50)).astype(int)})

df_all = pd.concat([generate_station_data(n, lo, la, z) for n, (lo, la, z) in STATIONS.items()], ignore_index=True)
df_all.to_csv("output_tokyobay/tokyo_bay_risk_data.csv", index=False)

features = ["T_C", "chl_a", "salinity", "turbidity", "flow_U", "do_mgl", "doy"]
X_train, X_test, y_train, y_test = train_test_split(df_all[features].values, df_all["risk_label"].values, 
                                                    test_size=0.25, random_state=42)

scaler = StandardScaler()
X_train_s, X_test_s = scaler.fit_transform(X_train), scaler.transform(X_test)

rf = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train_s, y_train)

importance = pd.Series(rf.feature_importances_, index=features).sort_values()
fig, ax = plt.subplots(figsize=(8, 5))
importance.plot(kind='barh', ax=ax)
ax.set_title("Feature Importance", fontweight='bold')
plt.savefig("output_tokyobay/feature_importance.png", dpi=150)
plt.close()

y_prob = rf.predict_proba(X_test_s)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
fig, ax = plt.subplots()
ax.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.3f}")
ax.plot([0,1], [0,1], '--', color='grey')
ax.set_title("ROC Curve", fontweight='bold')
ax.legend()
plt.savefig("output_tokyobay/model_evaluation.png", dpi=150)
plt.close()

lon_grid, lat_grid = np.meshgrid(np.linspace(139.6, 140.0, 200), np.linspace(35.2, 35.7, 200))
cmap_risk = LinearSegmentedColormap.from_list("risk", ["#2ecc71", "#f9ca24", "#e74c3c"])

for label, doy_c in [("Summer", 210), ("Winter", 30)]:
    sub = df_all[(df_all.doy >= doy_c - 15) & (df_all.doy <= doy_c + 15)].groupby("station").agg(
        {"lon":"mean", "lat":"mean", "risk_score":"mean"}).reset_index()
    interp = RBFInterpolator(sub[["lon","lat"]].values, sub["risk_score"].values, kernel='thin_plate_spline', smoothing=5)
    grid_vals = np.clip(interp(np.column_stack([lon_grid.ravel(), lat_grid.ravel()])).reshape(lon_grid.shape), 0, 100)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    cf = ax.contourf(lon_grid, lat_grid, grid_vals, levels=20, cmap=cmap_risk)
    plt.colorbar(cf, ax=ax, label="Risk Score")
    ax.scatter(sub.lon, sub.lat, c=sub.risk_score, cmap=cmap_risk, edgecolors='black', s=100)
    ax.set_title(f"Tokyo Bay Risk Map - {label}", fontweight='bold')
    plt.savefig(f"output_tokyobay/risk_map_{label.lower()}.png", dpi=150)
    plt.close()