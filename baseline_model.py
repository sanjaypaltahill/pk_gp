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

# Optional: rescale to mg/L for better conditioning (values ~ 1–20 instead of ~0.001–0.01)
df["conc_mg_per_L"] = df["conc_g_per_L"] * 1000.0

# ---------------------------------
# 2) Prepare tensors for GPyTorch
# ---------------------------------
# Train inputs must be shape [N, 1]
x_train = torch.tensor(df["time_hr"].values, dtype=torch.float32).unsqueeze(-1)
y_train = torch.tensor(df["conc_mg_per_L"].values, dtype=torch.float32)

# ---------------------------------
# 3) Define an Exact GP model
# ---------------------------------
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        # RBF kernel with outputscale (amplitude) learned; add small jitter through default settings
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

# ---------------------------------
# 5) Train the GP
# ---------------------------------
model.train()
likelihood.train()

optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

for i in range(100):
    optimizer.zero_grad()
    output = model(x_train)
    loss = -mll(output, y_train)
    loss.backward()
    optimizer.step()
    if (i + 1) % 20 == 0:
        print(f"Iter {i+1}/100 - Loss: {loss.item():.3f} | "
              f"noise: {likelihood.noise.item():.4f} | "
              f"lengthscale: {model.covar_module.base_kernel.lengthscale.item():.3f}")

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

# ---------------------------------
# 7) Plot (in mg/L)
# ---------------------------------
plt.figure(figsize=(8,5))
plt.scatter(df["time_hr"], df["conc_mg_per_L"], label="Observed (mg/L)")
plt.plot(x_test.squeeze().numpy(), mean.numpy(), label="GP mean")
plt.fill_between(
    x_test.squeeze().numpy(),
    lower.numpy(),
    upper.numpy(),
    alpha=0.3,
    label="95% CI"
)
plt.xlabel("Time (hr)")
plt.ylabel("Paracetamol (mg/L)")
plt.title("Gaussian Process fit to saliva concentrations")
plt.legend()
plt.tight_layout()
plt.show()
