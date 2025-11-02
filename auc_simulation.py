import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import trange
from importlib import import_module
import gpytorch
from scipy.optimize import curve_fit

# =====================================================
# 1. True model: Bateman (one-compartment oral)
# =====================================================
def bateman(t, A, ka, ke):
    denom = (ka - ke)
    return A * (ka / denom) * (np.exp(-ke * t) - np.exp(-ka * t))

def true_auc(A, ka, ke):
    """Analytical AUC for Bateman model = A / ke"""
    return A / ke

# =====================================================
# 2. Simulation helper
# =====================================================
def simulate_data(A=10.0, ka=1.2, ke=0.25, noise_std=0.5, n_points=25):
    t = np.linspace(0, 12, n_points)
    y_true = bateman(t, A, ka, ke)
    y_noisy = y_true + np.random.normal(0, noise_std, size=y_true.shape)
    return t, y_noisy, y_true

# =====================================================
# 3. Utility to fit and predict from GP model file
# =====================================================
def fit_gp_model(model_module, t, y):
    """Train a GP model from a given module (baseline or ODE-centered)."""
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = model_module.ExactGPModel(
        torch.tensor(t, dtype=torch.float32).unsqueeze(-1),
        torch.tensor(y, dtype=torch.float32),
        likelihood
    )

    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(150):
        optimizer.zero_grad()
        output = model(torch.tensor(t, dtype=torch.float32).unsqueeze(-1))
        loss = -mll(output, torch.tensor(y, dtype=torch.float32))
        loss.backward()
        optimizer.step()

    # Predict
    model.eval()
    likelihood.eval()
    t_test = torch.linspace(0, 12, 200).unsqueeze(-1)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = likelihood(model(t_test))
    mean_pred = pred.mean.detach().numpy().squeeze()
    return t_test.squeeze().numpy(), mean_pred

# =====================================================
# 4. Classic ODE-only fit
# =====================================================
def fit_classic_ode(t, y):
    """Fit A, ka, ke by nonlinear least squares."""
    def safe_bateman(t, A, ka, ke):
        # guard for numerical stability
        denom = np.clip(ka - ke, 1e-6, None)
        return A * (ka / denom) * (np.exp(-ke * t) - np.exp(-ka * t))

    # initial guesses
    p0 = [max(y), 1.0, 0.2]
    bounds = ([1e-3, 1e-3, 1e-3], [100.0, 10.0, 5.0])
    try:
        popt, _ = curve_fit(safe_bateman, t, y, p0=p0, bounds=bounds, maxfev=20000)
        A_hat, ka_hat, ke_hat = popt
    except RuntimeError:
        # fallback
        A_hat, ka_hat, ke_hat = p0
    t_dense = np.linspace(0, 12, 200)
    y_pred = bateman(t_dense, A_hat, ka_hat, ke_hat)
    return t_dense, y_pred, A_hat, ka_hat, ke_hat

# =====================================================
# 5. Estimate AUC distribution for each model
# =====================================================
def estimate_auc_distribution(model_name, model_module=None, n_sims=300, noise_std=0.5):
    A_true, ka_true, ke_true = 10.0, 1.2, 0.25
    auc_true = true_auc(A_true, ka_true, ke_true)
    auc_estimates = []

    for _ in trange(n_sims, desc=model_name):
        t, y_noisy, _ = simulate_data(A_true, ka_true, ke_true, noise_std=noise_std)

        if model_name == "Classic ODE":
            t_pred, mean_pred, *_ = fit_classic_ode(t, y_noisy)
        else:
            t_pred, mean_pred = fit_gp_model(model_module, t, y_noisy)

        auc_est = np.trapz(mean_pred, t_pred)
        auc_estimates.append(auc_est)

    auc_estimates = np.array(auc_estimates)
    return auc_true, auc_estimates

# =====================================================
# 6. Run comparison across models
# =====================================================
def main():
    ode_model = import_module("ode_centered_model")
    base_model = import_module("baseline_model")

    auc_true, auc_ode = estimate_auc_distribution("ODE-centered", ode_model)
    _, auc_base = estimate_auc_distribution("Baseline", base_model)
    _, auc_classic = estimate_auc_distribution("Classic ODE")

    # Plot all 3
    plt.figure(figsize=(8,5))
    plt.hist(auc_base, bins=30, alpha=0.5, label=f"Baseline GP (mean={auc_base.mean():.2f})")
    plt.hist(auc_ode, bins=30, alpha=0.5, label=f"ODE-centered GP (mean={auc_ode.mean():.2f})")
    plt.hist(auc_classic, bins=30, alpha=0.5, label=f"Classic ODE (mean={auc_classic.mean():.2f})")
    plt.axvline(auc_true, color='k', linestyle='--', label=f"True AUC = {auc_true:.2f}")
    plt.xlabel("Estimated AUC")
    plt.ylabel("Frequency")
    plt.title("AUC Sampling Distributions Across Models")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("\nSummary:")
    print(f"True AUC: {auc_true:.3f}")
    print(f"Baseline GP: {auc_base.mean():.3f} ± {auc_base.std():.3f}")
    print(f"ODE-centered GP: {auc_ode.mean():.3f} ± {auc_ode.std():.3f}")
    print(f"Classic ODE: {auc_classic.mean():.3f} ± {auc_classic.std():.3f}")

if __name__ == "__main__":
    main()
