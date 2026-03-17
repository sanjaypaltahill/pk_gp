"""
Paracetamol PK Modeling Pipeline
=================================
Four models fitted to subject2 plasma concentration-time data:
  1. One-compartment oral model  (Bateman function, scipy curve_fit)
  2. Two-compartment oral model  (ODE system, scipy solve_ivp + curve_fit)
  3. ODE-centered GP #1  (GPyTorch, Bateman/1-cpt mean + RBF kernel)
  4. ODE-centered GP #2  (GPyTorch, 2-compartment analytical mean + RBF kernel)

All models are plotted together for comparison.
"""

import json
import warnings
import numpy as np
import pandas as pd
import torch
import gpytorch
import matplotlib.pyplot as plt
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
            "time_hr":      float(entry["time"]),
            "conc_g_per_L": float(entry["value"]),
            "subject":      entry["individual"]["name"],
            "dose":         entry["interventions"][0]["name"],
            "tissue":       entry["tissue"]["name"],
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
def bateman(t, A, ka, ke):
    """One-compartment oral Bateman function."""
    denom = ka - ke
    if abs(denom) < 1e-9:
        return A * ka * t * np.exp(-ke * t)
    return A * (ka / denom) * (np.exp(-ke * t) - np.exp(-ka * t))

bateman_vec = np.vectorize(bateman)

p0_1cpt     = [y_obs.max(), 1.5, 0.3]
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

y_1cpt     = bateman_vec(t_dense, *popt_1cpt)
y_1cpt_obs = bateman_vec(t_obs,   *popt_1cpt)
rmse_1cpt  = np.sqrt(np.mean((y_obs - y_1cpt_obs) ** 2))
print(f"  RMSE = {rmse_1cpt:.4f} mg/L")


# ============================================================
# 2)  TWO-COMPARTMENT ORAL MODEL  (ODE system)
# ============================================================
def two_cpt_ode_fixed(t, y, ka, ke, k12, k21):
    X_gut, X_c, X_p = y
    dX_gut = -ka * X_gut
    dX_c   =  ka * X_gut - (k12 + ke) * X_c + k21 * X_p
    dX_p   =  k12 * X_c  - k21 * X_p
    return [dX_gut, dX_c, dX_p]

def two_cpt_curve(t_eval, log_Dose_V1, log_ka, log_ke, log_k12, log_k21):
    Dose_V1 = np.exp(log_Dose_V1)
    ka  = np.exp(log_ka)
    ke  = np.exp(log_ke)
    k12 = np.exp(log_k12)
    k21 = np.exp(log_k21)
    y0  = [Dose_V1, 0.0, 0.0]
    sol = solve_ivp(
        two_cpt_ode_fixed,
        [0.0, max(t_eval) + 1e-3],
        y0, args=(ka, ke, k12, k21),
        t_eval=t_eval, method="RK45",
        rtol=1e-8, atol=1e-10,
    )
    if not sol.success:
        return np.full_like(t_eval, np.inf, dtype=float)
    return sol.y[1]

p0_2cpt = [
    np.log(y_obs.max()),
    np.log(1.5),
    np.log(0.3),
    np.log(0.5),
    np.log(0.3),
]

popt_2cpt, pcov_2cpt = curve_fit(
    two_cpt_curve, t_obs, y_obs,
    p0=p0_2cpt, maxfev=20_000, method="lm",
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

y_2cpt     = two_cpt_curve(t_dense, *popt_2cpt)
y_2cpt_obs = two_cpt_curve(t_obs,   *popt_2cpt)
rmse_2cpt  = np.sqrt(np.mean((y_obs - y_2cpt_obs) ** 2))
print(f"  RMSE = {rmse_2cpt:.4f} mg/L")


# ============================================================
# 3)  GP HELPERS
# ============================================================

# ---------- 1-compartment mean (Bateman) ----------
class OneCompOralMean(gpytorch.means.Mean):
    """Bateman function as a learnable GP mean."""
    def __init__(self, A0=10.0, ka0=1.0, ke0=0.2):
        super().__init__()
        self.log_A  = torch.nn.Parameter(torch.log(torch.tensor(float(A0))))
        self.log_ka = torch.nn.Parameter(torch.log(torch.tensor(float(ka0))))
        self.log_ke = torch.nn.Parameter(torch.log(torch.tensor(float(ke0))))

    def forward(self, x):
        t      = x.squeeze(-1)
        A      = torch.exp(self.log_A)
        ka     = torch.exp(self.log_ka)
        ke     = torch.exp(self.log_ke)
        denom  = (ka - ke).clamp(min=1e-6)
        return A * (ka / denom) * (torch.exp(-ke * t) - torch.exp(-ka * t))


# ---------- 2-compartment mean via differentiable fixed-step RK4 ----------
class TwoCompOralMean(gpytorch.means.Mean):
    """
    2-compartment oral model solved with a fixed-step RK4 in pure PyTorch.
    Fully differentiable — no eigenvalue algebra, no division-by-near-zero.

    State: s = [X_gut, X_c, X_p]
      dX_gut/dt = -ka * X_gut
      dX_c/dt   =  ka * X_gut  - (k12 + ke) * X_c  + k21 * X_p
      dX_p/dt   =  k12 * X_c  - k21 * X_p

    C_c(t) = X_c(t)   (D/V1 lumped into a single amplitude parameter)

    All rate constants stored in log-space for positivity.
    Parameter bounds enforced via clamping in forward().
    """
    # Physiologically reasonable bounds for paracetamol [hr⁻¹]
    KA_MIN,  KA_MAX  = 0.1,  10.0
    KE_MIN,  KE_MAX  = 0.05,  5.0
    K12_MIN, K12_MAX = 0.01,  5.0
    K21_MIN, K21_MAX = 0.01,  5.0

    def __init__(self, DV1_0, ka0, ke0, k12_0, k21_0):
        super().__init__()
        self.log_DV1 = torch.nn.Parameter(torch.log(torch.tensor(float(DV1_0))))
        self.log_ka  = torch.nn.Parameter(torch.log(torch.tensor(float(ka0))))
        self.log_ke  = torch.nn.Parameter(torch.log(torch.tensor(float(ke0))))
        self.log_k12 = torch.nn.Parameter(torch.log(torch.tensor(float(k12_0))))
        self.log_k21 = torch.nn.Parameter(torch.log(torch.tensor(float(k21_0))))

    def _ode_rhs(self, s, ka, ke, k12, k21):
        """s: (3,) tensor [X_gut, X_c, X_p]"""
        dX_gut = -ka * s[0]
        dX_c   =  ka * s[0] - (k12 + ke) * s[1] + k21 * s[2]
        dX_p   =  k12 * s[1] - k21 * s[2]
        return torch.stack([dX_gut, dX_c, dX_p])

    def forward(self, x):
        t   = x.squeeze(-1)                      # (N,)
        DV1 = torch.exp(self.log_DV1)

        # Clamp rates to physiological range (smooth — still differentiable)
        ka  = torch.exp(self.log_ka ).clamp(self.KA_MIN,  self.KA_MAX)
        ke  = torch.exp(self.log_ke ).clamp(self.KE_MIN,  self.KE_MAX)
        k12 = torch.exp(self.log_k12).clamp(self.K12_MIN, self.K12_MAX)
        k21 = torch.exp(self.log_k21).clamp(self.K21_MIN, self.K21_MAX)

        # Build a uniform time grid from 0 to t.max(), ~200 steps
        T_end  = t.max() + 1e-3
        N_step = 200
        dt     = T_end / N_step
        t_grid = torch.linspace(0.0, T_end.item(), N_step + 1,
                                dtype=t.dtype, device=t.device)

        # Fixed-step RK4 — integrate the ODE on t_grid
        s = torch.stack([DV1,
                         torch.zeros_like(DV1),
                         torch.zeros_like(DV1)])   # (3,)
        X_c_grid = [s[1]]

        for i in range(N_step):
            k1 = self._ode_rhs(s,              ka, ke, k12, k21)
            k2 = self._ode_rhs(s + 0.5*dt*k1, ka, ke, k12, k21)
            k3 = self._ode_rhs(s + 0.5*dt*k2, ka, ke, k12, k21)
            k4 = self._ode_rhs(s +     dt*k3, ka, ke, k12, k21)
            s  = s + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            X_c_grid.append(s[1])

        X_c_grid = torch.stack(X_c_grid)          # (N_step+1,)

        # Interpolate X_c at the requested time points
        idx_f  = (t / dt).clamp(0, N_step - 1)
        idx_lo = idx_f.long()
        idx_hi = (idx_lo + 1).clamp(max=N_step)
        frac   = idx_f - idx_lo.float()
        C_c    = X_c_grid[idx_lo] * (1 - frac) + X_c_grid[idx_hi] * frac

        return C_c.clamp(min=0.0)


# ---------- Generic GP model ----------
class OdeCenteredGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, mean_module):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module  = mean_module
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


def train_gp(model, likelihood, x_train, y_train, n_iter=300, lr=0.05, label="GP"):
    model.train(); likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    print(f"\n{'='*60}\n{label}  (training)")
    for i in range(n_iter):
        optimizer.zero_grad()
        loss = -mll(model(x_train), y_train)
        loss.backward()
        optimizer.step()
        if (i + 1) % 75 == 0:
            print(f"  Iter {i+1:3d} | loss {loss.item():8.3f}")
    model.eval(); likelihood.eval()


def predict_gp(model, likelihood, x_test):
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = likelihood(model(x_test))
    mean  = pred.mean.numpy()
    lo, hi = pred.confidence_region()
    return mean, lo.numpy(), hi.numpy()


def gp_mean_curve(model, x_test):
    with torch.no_grad():
        return model.mean_module(x_test).numpy()


def gp_rmse(model, likelihood, x_train, y_obs):
    with torch.no_grad():
        pred_obs = likelihood(model(x_train)).mean.numpy()
    return np.sqrt(np.mean((y_obs - pred_obs) ** 2))


# ============================================================
# 4)  GP #1 — 1-compartment (Bateman) mean
# ============================================================
x_train = torch.tensor(t_obs,   dtype=torch.float32).unsqueeze(-1)
y_train = torch.tensor(y_obs,   dtype=torch.float32)
x_test  = torch.tensor(t_dense, dtype=torch.float32).unsqueeze(-1)

torch.manual_seed(42)

lik_1cpt_gp = gpytorch.likelihoods.GaussianLikelihood()
lik_1cpt_gp.noise = 0.5
mean_1cpt = OneCompOralMean(A0=float(y_obs.max()), ka0=1.5, ke0=0.3)
model_1cpt_gp = OdeCenteredGP(x_train, y_train, lik_1cpt_gp, mean_1cpt)
model_1cpt_gp.covar_module.base_kernel.lengthscale = 1.5
model_1cpt_gp.covar_module.outputscale = 1.0

train_gp(model_1cpt_gp, lik_1cpt_gp, x_train, y_train,
         n_iter=300, label="ODE-CENTERED GP #1  (1-cpt Bateman mean)")

gp1_mean, gp1_lo, gp1_hi   = predict_gp(model_1cpt_gp, lik_1cpt_gp, x_test)
gp1_prior_mean              = gp_mean_curve(model_1cpt_gp, x_test)
rmse_gp1                    = gp_rmse(model_1cpt_gp, lik_1cpt_gp, x_train, y_obs)

ka_gp1 = torch.exp(model_1cpt_gp.mean_module.log_ka).item()
ke_gp1 = torch.exp(model_1cpt_gp.mean_module.log_ke).item()
A_gp1  = torch.exp(model_1cpt_gp.mean_module.log_A).item()
print(f"\nFinal GP#1 params:  A={A_gp1:.4f}  ka={ka_gp1:.4f}  ke={ke_gp1:.4f}  RMSE={rmse_gp1:.4f}")


# ============================================================
# 5)  GP #2 — 2-compartment analytical mean
# ============================================================
torch.manual_seed(42)

lik_2cpt_gp = gpytorch.likelihoods.GaussianLikelihood()
lik_2cpt_gp.noise = 0.5
mean_2cpt = TwoCompOralMean(
    DV1_0 = Dose_V1_2,
    ka0   = ka_2,
    ke0   = ke_2,
    k12_0 = k12_2,
    k21_0 = k21_2,
)
model_2cpt_gp = OdeCenteredGP(x_train, y_train, lik_2cpt_gp, mean_2cpt)
model_2cpt_gp.covar_module.base_kernel.lengthscale = 1.5
model_2cpt_gp.covar_module.outputscale = 1.0

train_gp(model_2cpt_gp, lik_2cpt_gp, x_train, y_train,
         n_iter=300, label="ODE-CENTERED GP #2  (2-cpt analytical mean)")

gp2_mean, gp2_lo, gp2_hi   = predict_gp(model_2cpt_gp, lik_2cpt_gp, x_test)
gp2_prior_mean              = gp_mean_curve(model_2cpt_gp, x_test)
rmse_gp2                    = gp_rmse(model_2cpt_gp, lik_2cpt_gp, x_train, y_obs)

ka_gp2  = torch.exp(model_2cpt_gp.mean_module.log_ka).item()
ke_gp2  = torch.exp(model_2cpt_gp.mean_module.log_ke).item()
k12_gp2 = torch.exp(model_2cpt_gp.mean_module.log_k12).item()
k21_gp2 = torch.exp(model_2cpt_gp.mean_module.log_k21).item()
DV1_gp2 = torch.exp(model_2cpt_gp.mean_module.log_DV1).item()
print(f"\nFinal GP#2 params:  D/V1={DV1_gp2:.4f}  ka={ka_gp2:.4f}  ke={ke_gp2:.4f}  "
      f"k12={k12_gp2:.4f}  k21={k21_gp2:.4f}  RMSE={rmse_gp2:.4f}")


# ============================================================
# 6)  Summary table
# ============================================================
print("\n" + "=" * 60)
print(f"{'Model':<30}  {'RMSE (mg/L)':>12}")
print("-" * 45)
print(f"{'1-Compartment (Bateman)':<30}  {rmse_1cpt:>12.4f}")
print(f"{'2-Compartment (ODE)':<30}  {rmse_2cpt:>12.4f}")
print(f"{'GP (1-cpt Bateman mean)':<30}  {rmse_gp1:>12.4f}")
print(f"{'GP (2-cpt analytical mean)':<30}  {rmse_gp2:>12.4f}")


# ============================================================
# 7)  Plot — 4 panels
# ============================================================
fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=False)
fig.suptitle("Paracetamol PK Modeling — Subject 2", fontsize=14, fontweight="bold")

obs_kw   = dict(color="black", s=55, zorder=5, label="Observed")
line_kw  = dict(linewidth=2)

# ---- Panel 1: 1-compartment ----
ax = axes[0]
ax.scatter(t_obs, y_obs, **obs_kw)
ax.plot(t_dense, y_1cpt, color="steelblue", label="1-cpt fit", **line_kw)
ax.set_title("1-Compartment Oral\n(Bateman)", fontsize=11)
ax.set_xlabel("Time (hr)")
ax.set_ylabel("Paracetamol (mg/L)")
ax.legend(fontsize=8)
ax.text(0.97, 0.97,
        f"ka={ka_1:.3f} hr⁻¹\nke={ke_1:.3f} hr⁻¹\nRMSE={rmse_1cpt:.3f}",
        transform=ax.transAxes, va="top", ha="right", fontsize=8,
        bbox=dict(boxstyle="round", fc="white", alpha=0.7))

# ---- Panel 2: 2-compartment ----
ax = axes[1]
ax.scatter(t_obs, y_obs, **obs_kw)
ax.plot(t_dense, y_2cpt, color="darkorange", label="2-cpt fit", **line_kw)
ax.set_title("2-Compartment Oral\n(ODE)", fontsize=11)
ax.set_xlabel("Time (hr)")
ax.set_ylabel("Paracetamol (mg/L)")
ax.legend(fontsize=8)
ax.text(0.97, 0.97,
        f"ka={ka_2:.3f}\nke={ke_2:.3f}\nk12={k12_2:.3f}\nk21={k21_2:.3f}\nRMSE={rmse_2cpt:.3f}",
        transform=ax.transAxes, va="top", ha="right", fontsize=8,
        bbox=dict(boxstyle="round", fc="white", alpha=0.7))

# ---- Panel 3: GP (1-cpt Bateman mean) ----
ax = axes[2]
ax.scatter(t_obs, y_obs, **obs_kw)
ax.plot(t_dense, gp1_prior_mean, linestyle="--", color="grey",
        linewidth=1.5, label="Bateman mean (prior)")
ax.plot(t_dense, gp1_mean, color="seagreen", label="GP posterior mean", **line_kw)
ax.fill_between(t_dense, gp1_lo, gp1_hi,
                alpha=0.25, color="seagreen", label="95% CI")
ax.set_title("ODE-Centered GP\n(1-cpt Bateman mean + RBF)", fontsize=11)
ax.set_xlabel("Time (hr)")
ax.set_ylabel("Paracetamol (mg/L)")
ax.legend(fontsize=8)
ax.text(0.97, 0.97,
        f"ka={ka_gp1:.3f}\nke={ke_gp1:.3f}\nRMSE={rmse_gp1:.3f}",
        transform=ax.transAxes, va="top", ha="right", fontsize=8,
        bbox=dict(boxstyle="round", fc="white", alpha=0.7))

# ---- Panel 4: GP (2-cpt analytical mean) ----
ax = axes[3]
ax.scatter(t_obs, y_obs, **obs_kw)
ax.plot(t_dense, gp2_prior_mean, linestyle="--", color="grey",
        linewidth=1.5, label="2-cpt mean (prior)")
ax.plot(t_dense, gp2_mean, color="mediumpurple", label="GP posterior mean", **line_kw)
ax.fill_between(t_dense, gp2_lo, gp2_hi,
                alpha=0.25, color="mediumpurple", label="95% CI")
ax.set_title("ODE-Centered GP\n(2-cpt analytical mean + RBF)", fontsize=11)
ax.set_xlabel("Time (hr)")
ax.set_ylabel("Paracetamol (mg/L)")
ax.legend(fontsize=8)
ax.text(0.97, 0.97,
        f"ka={ka_gp2:.3f}\nke={ke_gp2:.3f}\nk12={k12_gp2:.3f}\nk21={k21_gp2:.3f}\nRMSE={rmse_gp2:.3f}",
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