"""
Coverage Verification for ODE-Centered GP Models
==================================================
Verifies empirical 95% credible interval coverage for:
  1. 1-compartment (Bateman) ODE-centered GP
  2. 2-compartment (RK4) ODE-centered GP

For each model, we:
  - Simulate data from the true PK curve + Gaussian noise
  - Fit the GP
  - Measure fraction of the true curve that falls inside the 95% CI
  - Repeat over many random seeds
  - Report mean coverage (target ≈ 0.95)
"""

import warnings
import numpy as np
import torch
import gpytorch
from scipy.integrate import solve_ivp

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# True PK parameters & evaluation grid
# ─────────────────────────────────────────────
A_true,  ka_true,  ke_true              = 15.0, 1.2, 0.25
DV1_true, ka2_true, ke2_true            = 12.0, 1.2, 0.25
k12_true, k21_true                      = 0.4,  0.3

T_MAX   = 12.0
t_grid  = np.linspace(0, T_MAX, 200)


# ─────────────────────────────────────────────
# True-curve generators
# ─────────────────────────────────────────────
def bateman_np(t, A, ka, ke):
    denom = max(ka - ke, 1e-9)
    return A * (ka / denom) * (np.exp(-ke * t) - np.exp(-ka * t))

bateman_vec = np.vectorize(bateman_np)


def two_cpt_np(t_eval, DV1, ka, ke, k12, k21):
    def rhs(t, y):
        dX_gut = -ka * y[0]
        dX_c   =  ka * y[0] - (k12 + ke) * y[1] + k21 * y[2]
        dX_p   =  k12 * y[1] - k21 * y[2]
        return [dX_gut, dX_c, dX_p]
    sol = solve_ivp(rhs, [0, t_eval[-1] + 1e-3], [DV1, 0, 0],
                    t_eval=t_eval, method="RK45", rtol=1e-9, atol=1e-11)
    return sol.y[1]


true_1cpt = bateman_vec(t_grid, A_true, ka_true, ke_true)
true_2cpt = two_cpt_np(t_grid, DV1_true, ka2_true, ke2_true, k12_true, k21_true)


# ─────────────────────────────────────────────
# GP mean modules (same as main pipeline)
# ─────────────────────────────────────────────
class OneCompOralMean(gpytorch.means.Mean):
    def __init__(self, A0=10.0, ka0=1.0, ke0=0.2):
        super().__init__()
        self.log_A  = torch.nn.Parameter(torch.log(torch.tensor(float(A0))))
        self.log_ka = torch.nn.Parameter(torch.log(torch.tensor(float(ka0))))
        self.log_ke = torch.nn.Parameter(torch.log(torch.tensor(float(ke0))))

    def forward(self, x):
        t     = x.squeeze(-1)
        A     = torch.exp(self.log_A)
        ka    = torch.exp(self.log_ka)
        ke    = torch.exp(self.log_ke)
        denom = (ka - ke).clamp(min=1e-6)
        return A * (ka / denom) * (torch.exp(-ke * t) - torch.exp(-ka * t))


class TwoCompOralMean(gpytorch.means.Mean):
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

    def _rhs(self, s, ka, ke, k12, k21):
        dX_gut = -ka * s[0]
        dX_c   =  ka * s[0] - (k12 + ke) * s[1] + k21 * s[2]
        dX_p   =  k12 * s[1] - k21 * s[2]
        return torch.stack([dX_gut, dX_c, dX_p])

    def forward(self, x):
        t   = x.squeeze(-1)
        DV1 = torch.exp(self.log_DV1)
        ka  = torch.exp(self.log_ka ).clamp(self.KA_MIN,  self.KA_MAX)
        ke  = torch.exp(self.log_ke ).clamp(self.KE_MIN,  self.KE_MAX)
        k12 = torch.exp(self.log_k12).clamp(self.K12_MIN, self.K12_MAX)
        k21 = torch.exp(self.log_k21).clamp(self.K21_MIN, self.K21_MAX)

        T_end  = (t.max() + 1e-3).item()   # .item() detaches scalar — prevents unbounded graph growth
        N_step = 200
        dt     = T_end / N_step

        s = torch.stack([DV1,
                         torch.zeros_like(DV1),
                         torch.zeros_like(DV1)])
        X_c_grid = [s[1]]
        for _ in range(N_step):
            k1 = self._rhs(s,              ka, ke, k12, k21)
            k2 = self._rhs(s + 0.5*dt*k1, ka, ke, k12, k21)
            k3 = self._rhs(s + 0.5*dt*k2, ka, ke, k12, k21)
            k4 = self._rhs(s +     dt*k3, ka, ke, k12, k21)
            s  = s + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            X_c_grid.append(s[1])

        X_c_grid = torch.stack(X_c_grid)
        idx_f  = (t / dt).clamp(0, N_step - 1)
        idx_lo = idx_f.long()
        idx_hi = (idx_lo + 1).clamp(max=N_step)
        frac   = idx_f - idx_lo.float()
        C_c    = X_c_grid[idx_lo] * (1 - frac) + X_c_grid[idx_hi] * frac
        return C_c.clamp(min=0.0)


# ─────────────────────────────────────────────
# Generic GP container
# ─────────────────────────────────────────────
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


# ─────────────────────────────────────────────
# Single-trial helper
# ─────────────────────────────────────────────
def run_trial(seed, model_type="1cpt",
              n_obs=10, noise_std=1.0, n_iter=150, lr=0.05):
    """
    Returns the fraction of the true curve covered by the GP's 95% CI.

    model_type : "1cpt"  → 1-compartment Bateman mean
                 "2cpt"  → 2-compartment RK4 mean
    """
    rng = np.random.default_rng(seed)

    # ── Sample observation times & noisy concentrations ──────────────────
    t_candidates = np.linspace(0, T_MAX, 50)
    t_obs = np.sort(rng.choice(t_candidates, size=n_obs, replace=False))

    if model_type == "1cpt":
        true_curve_full = true_1cpt
        y_obs = bateman_vec(t_obs, A_true, ka_true, ke_true) + \
                rng.normal(0, noise_std, size=n_obs)
    else:
        true_curve_full = true_2cpt
        y_obs = two_cpt_np(t_obs, DV1_true, ka2_true, ke2_true,
                            k12_true, k21_true) + \
                rng.normal(0, noise_std, size=n_obs)

    x_train = torch.tensor(t_obs,  dtype=torch.float32).unsqueeze(-1)
    y_train = torch.tensor(y_obs,  dtype=torch.float32)
    x_test  = torch.tensor(t_grid, dtype=torch.float32).unsqueeze(-1)

    # ── Build model ───────────────────────────────────────────────────────
    torch.manual_seed(seed)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood.noise = 0.5

    if model_type == "1cpt":
        mean_mod = OneCompOralMean(
            A0  = max(float(y_train.max().item()), 1.0),
            ka0 = 1.2,
            ke0 = 0.25,
        )
    else:
        mean_mod = TwoCompOralMean(
            DV1_0 = max(float(y_train.max().item()), 1.0),
            ka0   = 1.2,
            ke0   = 0.25,
            k12_0 = 0.4,
            k21_0 = 0.3,
        )

    model = OdeCenteredGP(x_train, y_train, likelihood, mean_mod)
    model.covar_module.base_kernel.lengthscale = 1.5
    model.covar_module.outputscale = 1.0

    # ── Train ─────────────────────────────────────────────────────────────
    model.train(); likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for _ in range(n_iter):
        optimizer.zero_grad()
        loss = -mll(model(x_train), y_train)
        loss.backward()
        optimizer.step()

    # ── Predict ───────────────────────────────────────────────────────────
    model.eval(); likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = likelihood(model(x_test))
    lo, hi = pred.confidence_region()          # ≈ mean ± 2·std  (≈95%)
    lo, hi = lo.numpy(), hi.numpy()

    covered = np.mean((true_curve_full >= lo) & (true_curve_full <= hi))
    return covered


# ─────────────────────────────────────────────
# Main: run both models
# ─────────────────────────────────────────────
if __name__ == "__main__":
    N_TRIALS  = 200   # increase to 1 000 for a tighter estimate
    N_OBS     = 10
    NOISE_STD = 1.0
    N_ITER    = 150   # training iterations per trial

    print("=" * 60)
    print(f"Coverage verification  |  {N_TRIALS} trials  |  "
          f"n_obs={N_OBS}  noise_std={NOISE_STD}")
    print("=" * 60)

    for label, mtype in [("1-Compartment GP (Bateman mean)", "1cpt"),
                          ("2-Compartment GP (RK4 mean)",    "2cpt")]:
        coverages = []
        for i in range(N_TRIALS):
            cov = run_trial(seed=i, model_type=mtype,
                            n_obs=N_OBS, noise_std=NOISE_STD, n_iter=N_ITER)
            coverages.append(cov)
            if (i + 1) % 25 == 0:
                print(f"  [{label}]  trial {i+1:4d}/{N_TRIALS}  "
                      f"running mean = {np.mean(coverages):.3f}")

        mean_cov = np.mean(coverages)
        std_cov  = np.std(coverages)
        p5, p95  = np.percentile(coverages, [5, 95])
        print(f"\n  ── {label} ──")
        print(f"     Mean coverage  : {mean_cov:.3f}  (target ≈ 0.950)")
        print(f"     Std            : {std_cov:.3f}")
        print(f"     5th–95th pct   : [{p5:.3f}, {p95:.3f}]")
        print()