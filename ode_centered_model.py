import json
import torch
import gpytorch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# ---------------------------
# 1) Load & flatten JSON
# ---------------------------
with open("subject2.json") as f:
    data = json.load(f)

records = []
for row in data["array"]:
    for entry in row:
        records.append({
            "time_hr": float(entry["time"]),
            "conc_g_per_L": float(entry["value"]),
            "subject": entry["individual"]["name"],
            "dose": entry["interventions"][0]["name"],
            "tissue": entry["tissue"]["name"],
        })

df = pd.DataFrame(records).sort_values("time_hr").reset_index(drop=True)

# Rescale to mg/L (helps conditioning)
df["conc_mg_per_L"] = df["conc_g_per_L"] * 1000.0

# ---------------------------------
# 2) Prepare tensors for GPyTorch
# ---------------------------------
x_train = torch.tensor(df["time_hr"].values, dtype=torch.float32).unsqueeze(-1)
y_train = torch.tensor(df["conc_mg_per_L"].values, dtype=torch.float32)

# ---------------------------------
# 3) Define an ODE-centered mean and GP model
# ---------------------------------
class OneCompOralMean(gpytorch.means.Mean):
    """
    Bateman function mean:
      mu(t) = A * [ka/(ka - ke)] * (exp(-ke t) - exp(-ka t))
    A absorbs F * Dose / V. ka, ke > 0 are learned.
    """
    def __init__(self, initial_A=10.0, initial_ka=1.0, initial_ke=0.2):
        super().__init__()
        # Work in log-space to keep params positive
        self.log_A  = torch.nn.Parameter(torch.log(torch.tensor(initial_A)))
        self.log_ka = torch.nn.Parameter(torch.log(torch.tensor(initial_ka)))
        self.log_ke = torch.nn.Parameter(torch.log(torch.tensor(initial_ke)))

    def forward(self, x):
        t = x.squeeze(-1)  # [N]
        A  = torch.exp(self.log_A)
        ka = torch.exp(self.log_ka)
        ke = torch.exp(self.log_ke)
        # Avoid ka ~= ke numerical issue
        denom = (ka - ke).clamp(min=1e-6)
        bateman = (ka / denom) * (torch.exp(-ke * t) - torch.exp(-ka * t))
        return bateman * A

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        # ODE-centered mean
        self.mean_module = OneCompOralMean(
            initial_A=max(y_train.max().item(), 1.0),  # rough scale
            initial_ka=1.0,  # hr^-1 (fast-ish absorption guess)
            initial_ke=0.2,  # hr^-1 (elimination guess)
        )
        # Smooth residuals around ODE with RBF
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# ---------------------------------
# 4) Initialize likelihood & model
# ---------------------------------
torch.manual_seed(18)
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(x_train, y_train, likelihood)

# (Optional) put weak priors to keep params reasonable
model.covar_module.base_kernel.lengthscale = 1.0
model.covar_module.outputscale = 1.0
likelihood.noise = 0.5

# ---------------------------------
# 5) Train the GP
# ---------------------------------
model.train()
likelihood.train()

optimizer = torch.optim.Adam([
    {'params': model.parameters(), 'lr': 0.05},
], lr=0.05)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

for i in range(250):
    optimizer.zero_grad()
    output = model(x_train)
    loss = -mll(output, y_train)
    loss.backward()
    optimizer.step()
    if (i + 1) % 50 == 0:
        ka = torch.exp(model.mean_module.log_ka).item()
        ke = torch.exp(model.mean_module.log_ke).item()
        A  = torch.exp(model.mean_module.log_A).item()
        print(f"Iter {i+1}/250 - Loss: {loss.item():.3f} | "
              f"noise: {likelihood.noise.item():.4f} | "
              f"ℓ: {model.covar_module.base_kernel.lengthscale.item():.3f} | "
              f"A: {A:.3f}, ka: {ka:.3f}, ke: {ke:.3f}")

# ---------------------------------
# 6) Predict on a dense grid
# ---------------------------------
model.eval()
likelihood.eval()

x_min, x_max = float(df["time_hr"].min()), float(df["time_hr"].max())
pad = 0.5
x_test = torch.linspace(max(0.0, x_min - pad), x_max + pad, 200).unsqueeze(-1)

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    pred = likelihood(model(x_test))

mean = pred.mean
lower, upper = pred.confidence_region()

# Also plot the *parametric mean* alone (without kernel)
with torch.no_grad():
    ode_mean_only = model.mean_module(x_test).detach()

# ---------------------------------
# 7) Plot (in mg/L)
# ---------------------------------
plt.figure(figsize=(8,5))
plt.scatter(df["time_hr"], df["conc_mg_per_L"], label="Observed (mg/L)")
plt.plot(x_test.squeeze().numpy(), ode_mean_only.numpy(), linestyle="--", label="ODE mean (Bateman)")
plt.plot(x_test.squeeze().numpy(), mean.numpy(), label="GP posterior mean")
plt.fill_between(
    x_test.squeeze().numpy(),
    lower.numpy(),
    upper.numpy(),
    alpha=0.3,
    label="95% CI"
)
plt.xlabel("Time (hr)")
plt.ylabel("Paracetamol (mg/L)")
plt.title("GP centered on one-compartment oral model")
plt.legend()
plt.tight_layout()
plt.show()
