"""
Paracetamol PK Modeling Pipeline
=================================
Three models fitted to subject2 plasma concentration-time data:
  1. One-compartment oral model  (Bateman function, scipy curve_fit)
  2. Two-compartment oral model  (ODE system, scipy solve_ivp + curve_fit)
  3. ODE-centered Gaussian Process (GPyTorch, Bateman mean + RBF kernel)

All models are plotted together for comparison.
"""

import json
import warnings
import numpy as np
import pandas as pd
import torch
import gpytorch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit
from scipy.integrate import solve_ivp

warnings.filterwarnings("ignore")

# ============================================================
# 0)  Load & flatten JSON
# ============================================================
SUBJECT_JSON = "subject2.json"

with open(SUBJECT_JSON) as f:
    data = json.load(f)

records = []
for row in data["array"]:
    for entry in row:
        records.append({
            "time_hr":    float(entry["time"]),
            "conc_g_per_L": float(entry["value"]),
            "subject":    entry["individual"]["name"],
            "dose":       entry["interventions"][0]["name"],
            "tissue":     entry["tissue"]["name"],
        })

df = (pd.DataFrame(records)
        .sort_values("time_hr")
        .reset_index(drop=True))

df["conc_mg_per_L"] = df["conc_g_per_L"] * 1000.0   # g/L → mg/L

t_obs = df["time_hr"].values
y_obs = df["conc_mg_per_L"].values

# Dense evaluation grid
t_dense = np.linspace(0.0, t_obs.max() + 0.5, 300)


# ============================================================
# 1)  ONE-COMPARTMENT ORAL MODEL  (Bateman function)
# ============================================================
#
#   C(t) = A * [ka / (ka - ke)] * (exp(-ke*t) - exp(-ka*t))
#
#   A = F * Dose / V   (apparent amplitude, lumped)
#   ka = absorption rate constant  [hr⁻¹]
#   ke = elimination rate constant [hr⁻¹]

def bateman(t, A, ka, ke):
    """One-compartment oral Bateman function."""
    denom = ka - ke
    if abs(denom) < 1e-9:          # ka ≈ ke edge case
        return A * ka * t * np.exp(-ke * t)
    return A * (ka / denom) * (np.exp(-ke * t) - np.exp(-ka * t))

bateman_vec = np.vectorize(bateman)   # allows scalar ke etc.

# Initial guesses: A~peak conc, ka fast, ke slow
p0_1cpt  = [y_obs.max(), 1.5, 0.3]
bounds_1cpt = ([0, 1e-3, 1e-3], [1e4, 50, 50])

popt_1cpt, pcov_1cpt = curve_fit(
    bateman_vec, t_obs, y_obs,
    p0=p0_1cpt, bounds=bounds_1cpt, maxfev=10_000
)
A_1, ka_1, ke_1 = popt_1cpt
perr_1cpt = np.sqrt(np.diag(pcov_1cpt))

print("=" * 60)
print("ONE-COMPARTMENT MODEL")
print(f"  A  = {A_1:.4f} ± {perr_1cpt[0]:.4f} mg/L")
print(f"  ka = {ka_1:.4f} ± {perr_1cpt[1]:.4f} hr⁻¹")
print(f"  ke = {ke_1:.4f} ± {perr_1cpt[2]:.4f} hr⁻¹")
print(f"  t½ (elimination) = {np.log(2)/ke_1:.3f} hr")
print(f"  t_max = {np.log(ka_1/ke_1)/(ka_1-ke_1):.3f} hr")

y_1cpt = bateman_vec(t_dense, *popt_1cpt)

# Residuals & RMSE
y_1cpt_obs = bateman_vec(t_obs, *popt_1cpt)
rmse_1cpt  = np.sqrt(np.mean((y_obs - y_1cpt_obs) ** 2))
print(f"  RMSE = {rmse_1cpt:.4f} mg/L")


# ============================================================
# 2)  TWO-COMPARTMENT ORAL MODEL  (ODE system)
# ============================================================
#
#   GI compartment (depot):
#     dX_gut/dt = -ka * X_gut           X_gut(0) = Dose_abs (mg)
#
#   Central compartment (1):
#     dX_c/dt  =  ka * X_gut  - (k12 + ke) * X_c  +  k21 * X_p
#     C_c(t)   =  X_c / V1
#
#   Peripheral compartment (2):
#     dX_p/dt  =  k12 * X_c  - k21 * X_p
#
#   Parameters to fit:   ka, ke, k12, k21, V1
#   (Dose_abs is derived from first data point scale; treated as free param here)

def two_cpt_ode(t, y, ka, ke, k12, k21):
    """RHS of 2-compartment oral ODE.  State: [X_gut, X_c, X_p]."""
    X_gut, X_c, X_p = y
    dX_gut = -ka * X_gut
    dX_c   =  ka * X_gut - (k12 + ke) * X_c + k21 * X_p
    dX_p   =  k12 * X_c - k21 * X_p
    return [dX_gut, X_c, dX_p]   # NOTE: corrected below

def two_cpt_ode_fixed(t, y, ka, ke, k12, k21):
    X_gut, X_c, X_p = y
    dX_gut = -ka * X_gut
    dX_c   =  ka * X_gut - (k12 + ke) * X_c + k21 * X_p
    dX_p   =  k12 * X_c  - k21 * X_p
    return [dX_gut, dX_c, dX_p]

def two_cpt_curve(t_eval, log_Dose_V1, log_ka, log_ke, log_k12, log_k21):
    """
    Wrapper for curve_fit.  Works in log-space for positivity.
    Returns C_c(t) = X_c / V1  (but V1 is absorbed into Dose/V1).
    """
    Dose_V1 = np.exp(log_Dose_V1)   # effectively Dose/V1
    ka  = np.exp(log_ka)
    ke  = np.exp(log_ke)
    k12 = np.exp(log_k12)
    k21 = np.exp(log_k21)

    y0 = [Dose_V1, 0.0, 0.0]   # gut=Dose_V1, central=0, periph=0

    sol = solve_ivp(
        two_cpt_ode_fixed,
        [0.0, max(t_eval) + 1e-3],
        y0,
        args=(ka, ke, k12, k21),
        t_eval=t_eval,
        method="RK45",
        rtol=1e-8, atol=1e-10,
        dense_output=False,
    )
    if not sol.success:
        return np.full_like(t_eval, np.inf, dtype=float)

    X_c = sol.y[1]    # central compartment mass / V1 → concentration
    return X_c

# Initial guesses in log-space
p0_2cpt = [
    np.log(y_obs.max()),   # log(Dose/V1) – initial guess from peak
    np.log(1.5),           # log(ka)
    np.log(0.3),           # log(ke)
    np.log(0.5),           # log(k12)
    np.log(0.3),           # log(k21)
]

popt_2cpt, pcov_2cpt = curve_fit(
    two_cpt_curve, t_obs, y_obs,
    p0=p0_2cpt,
    maxfev=20_000,
    method="lm",
)
perr_2cpt = np.sqrt(np.diag(pcov_2cpt))

Dose_V1_2 = np.exp(popt_2cpt[0])
ka_2      = np.exp(popt_2cpt[1])
ke_2      = np.exp(popt_2cpt[2])
k12_2     = np.exp(popt_2cpt[3])
k21_2     = np.exp(popt_2cpt[4])

print("\n" + "=" * 60)
print("TWO-COMPARTMENT MODEL")
print(f"  Dose/V1 = {Dose_V1_2:.4f} mg/L")
print(f"  ka  = {ka_2:.4f}  hr⁻¹")
print(f"  ke  = {ke_2:.4f}  hr⁻¹")
print(f"  k12 = {k12_2:.4f} hr⁻¹  (central → peripheral)")
print(f"  k21 = {k21_2:.4f} hr⁻¹  (peripheral → central)")

y_2cpt = two_cpt_curve(t_dense, *popt_2cpt)

y_2cpt_obs = two_cpt_curve(t_obs, *popt_2cpt)
rmse_2cpt  = np.sqrt(np.mean((y_obs - y_2cpt_obs) ** 2))
print(f"  RMSE = {rmse_2cpt:.4f} mg/L")


# ============================================================
# 3)  ODE-CENTERED GAUSSIAN PROCESS
# ============================================================
# Mean function  =  Bateman (1-compartment, learned jointly)
# Covariance     =  ScaleKernel( RBFKernel )
# Likelihood     =  GaussianLikelihood (noise learned)

class OneCompOralMean(gpytorch.means.Mean):
    """Bateman function as a learnable GP mean."""
    def __init__(self, A0=10.0, ka0=1.0, ke0=0.2):
        super().__init__()
        self.log_A  = torch.nn.Parameter(torch.log(torch.tensor(float(A0))))
        self.log_ka = torch.nn.Parameter(torch.log(torch.tensor(float(ka0))))
        self.log_ke = torch.nn.Parameter(torch.log(torch.tensor(float(ke0))))

    def forward(self, x):
        t  = x.squeeze(-1)
        A  = torch.exp(self.log_A)
        ka = torch.exp(self.log_ka)
        ke = torch.exp(self.log_ke)
        denom = (ka - ke).clamp(min=1e-6)
        return A * (ka / denom) * (torch.exp(-ke * t) - torch.exp(-ka * t))


class BatemanGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, A0, ka0, ke0):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module  = OneCompOralMean(A0=A0, ka0=ka0, ke0=ke0)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(0.5, 3.0)
            )
        )

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x),
            self.covar_module(x),
        )


# ---- tensors ----
x_train = torch.tensor(t_obs, dtype=torch.float32).unsqueeze(-1)
y_train = torch.tensor(y_obs, dtype=torch.float32)

torch.manual_seed(42)
likelihood_gp = gpytorch.likelihoods.GaussianLikelihood()
likelihood_gp.noise = 0.5

model_gp = BatemanGP(
    x_train, y_train, likelihood_gp,
    A0  = float(y_obs.max()),
    ka0 = 1.5,
    ke0 = 0.3,
)
model_gp.covar_module.base_kernel.lengthscale = 1.5
model_gp.covar_module.outputscale = 1.0

# ---- train ----
model_gp.train()
likelihood_gp.train()

optimizer_gp = torch.optim.Adam(model_gp.parameters(), lr=0.05)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_gp, model_gp)

N_ITER = 300
print("\n" + "=" * 60)
print("ODE-CENTERED GP  (training)")
for i in range(N_ITER):
    optimizer_gp.zero_grad()
    output = model_gp(x_train)
    loss   = -mll(output, y_train)
    loss.backward()
    optimizer_gp.step()
    if (i + 1) % 75 == 0:
        ka_ = torch.exp(model_gp.mean_module.log_ka).item()
        ke_ = torch.exp(model_gp.mean_module.log_ke).item()
        A_  = torch.exp(model_gp.mean_module.log_A).item()
        ls_ = model_gp.covar_module.base_kernel.lengthscale.item()
        ns_ = likelihood_gp.noise.item()
        print(f"  Iter {i+1:3d} | loss {loss.item():8.3f} | "
              f"A={A_:.3f}  ka={ka_:.3f}  ke={ke_:.3f} | "
              f"ℓ={ls_:.3f}  σ²={ns_:.4f}")

# ---- predict ----
model_gp.eval()
likelihood_gp.eval()

x_test = torch.tensor(t_dense, dtype=torch.float32).unsqueeze(-1)
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    gp_pred = likelihood_gp(model_gp(x_test))

gp_mean  = gp_pred.mean.numpy()
gp_lower, gp_upper = gp_pred.confidence_region()
gp_lower, gp_upper = gp_lower.numpy(), gp_upper.numpy()

with torch.no_grad():
    bateman_mean_gp = model_gp.mean_module(x_test).numpy()

# GP RMSE
with torch.no_grad():
    gp_pred_obs = likelihood_gp(model_gp(x_train)).mean.numpy()
rmse_gp = np.sqrt(np.mean((y_obs - gp_pred_obs) ** 2))

ka_gp = torch.exp(model_gp.mean_module.log_ka).item()
ke_gp = torch.exp(model_gp.mean_module.log_ke).item()
A_gp  = torch.exp(model_gp.mean_module.log_A).item()

print("\nFinal GP parameters:")
print(f"  A  = {A_gp:.4f} mg/L")
print(f"  ka = {ka_gp:.4f} hr⁻¹")
print(f"  ke = {ke_gp:.4f} hr⁻¹")
print(f"  RMSE = {rmse_gp:.4f} mg/L")


# ============================================================
# 4)  Summary table
# ============================================================
print("\n" + "=" * 60)
print(f"{'Model':<25}  {'RMSE (mg/L)':>12}")
print("-" * 40)
print(f"{'1-Compartment':<25}  {rmse_1cpt:>12.4f}")
print(f"{'2-Compartment':<25}  {rmse_2cpt:>12.4f}")
print(f"{'ODE-Centered GP':<25}  {rmse_gp:>12.4f}")


# ============================================================
# 5)  Plot
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
fig.suptitle("Paracetamol PK Modeling — Subject 2", fontsize=14, fontweight="bold")

obs_kw   = dict(color="black", s=60, zorder=5, label="Observed")
dense_kw = dict(linewidth=2)

# --- Panel 1: 1-compartment ---
ax = axes[0]
ax.scatter(t_obs, y_obs, **obs_kw)
ax.plot(t_dense, y_1cpt, color="steelblue", label="1-cpt fit", **dense_kw)
ax.set_title("1-Compartment Oral\n(Bateman)", fontsize=12)
ax.set_xlabel("Time (hr)")
ax.set_ylabel("Paracetamol (mg/L)")
ax.legend(fontsize=9)
ax.text(0.97, 0.97,
        f"ka={ka_1:.3f} hr⁻¹\nke={ke_1:.3f} hr⁻¹\nRMSE={rmse_1cpt:.3f}",
        transform=ax.transAxes, va="top", ha="right", fontsize=8,
        bbox=dict(boxstyle="round", fc="white", alpha=0.7))

# --- Panel 2: 2-compartment ---
ax = axes[1]
ax.scatter(t_obs, y_obs, **obs_kw)
ax.plot(t_dense, y_2cpt, color="darkorange", label="2-cpt fit", **dense_kw)
ax.set_title("2-Compartment Oral\n(ODE)", fontsize=12)
ax.set_xlabel("Time (hr)")
ax.legend(fontsize=9)
ax.text(0.97, 0.97,
        f"ka={ka_2:.3f}\nke={ke_2:.3f}\nk12={k12_2:.3f}\nk21={k21_2:.3f}\nRMSE={rmse_2cpt:.3f}",
        transform=ax.transAxes, va="top", ha="right", fontsize=8,
        bbox=dict(boxstyle="round", fc="white", alpha=0.7))

# --- Panel 3: ODE-centered GP ---
ax = axes[2]
ax.scatter(t_obs, y_obs, **obs_kw)
ax.plot(t_dense, bateman_mean_gp, linestyle="--", color="grey",
        linewidth=1.5, label="Bateman mean (GP)")
ax.plot(t_dense, gp_mean, color="seagreen", label="GP posterior mean", **dense_kw)
ax.fill_between(t_dense, gp_lower, gp_upper,
                alpha=0.25, color="seagreen", label="95% CI")
ax.set_title("ODE-Centered GP\n(Bateman mean + RBF kernel)", fontsize=12)
ax.set_xlabel("Time (hr)")
ax.legend(fontsize=9)
ax.text(0.97, 0.97,
        f"ka={ka_gp:.3f}\nke={ke_gp:.3f}\nRMSE={rmse_gp:.3f}",
        transform=ax.transAxes, va="top", ha="right", fontsize=8,
        bbox=dict(boxstyle="round", fc="white", alpha=0.7))

for ax in axes:
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("pk_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nFigure saved to pk_comparison.png")