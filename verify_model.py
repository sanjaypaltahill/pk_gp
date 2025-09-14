import torch
import gpytorch
import numpy as np

# ---------------------------
# 1) True Bateman model
# ---------------------------
def bateman(t, A, ka, ke):
    denom = max(ka - ke, 1e-6)
    return A * (ka/denom) * (np.exp(-ke*t) - np.exp(-ka*t))

# True parameters
A_true, ka_true, ke_true = 15.0, 1.2, 0.25
t_grid = np.linspace(0, 12, 200)
true_curve = bateman(t_grid, A_true, ka_true, ke_true)

# ---------------------------
# 2) GP model classes
# ---------------------------
class OneCompOralMean(gpytorch.means.Mean):
    def __init__(self, initial_A=10.0, initial_ka=1.0, initial_ke=0.2):
        super().__init__()
        self.log_A  = torch.nn.Parameter(torch.log(torch.tensor(initial_A)))
        self.log_ka = torch.nn.Parameter(torch.log(torch.tensor(initial_ka)))
        self.log_ke = torch.nn.Parameter(torch.log(torch.tensor(initial_ke)))

    def forward(self, x):
        t = x.squeeze(-1)
        A  = torch.exp(self.log_A)
        ka = torch.exp(self.log_ka)
        ke = torch.exp(self.log_ke)
        denom = (ka - ke).clamp(min=1e-6)
        return A * (ka/denom) * (torch.exp(-ke * t) - torch.exp(-ka * t))

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = OneCompOralMean(
            initial_A=max(train_y.max().item(), 1.0),
            initial_ka=1.0,
            initial_ke=0.2,
        )
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# ---------------------------
# 3) Coverage experiment
# ---------------------------
def run_one(seed=0, n_obs=10, noise_std=1.0):
    rng = np.random.default_rng(seed)
    # sample random timepoints
    t_obs = np.sort(rng.choice(np.arange(0, 13), size=n_obs, replace=False))
    y_obs = bateman(t_obs, A_true, ka_true, ke_true) + rng.normal(0, noise_std, size=n_obs)

    # convert to tensors
    x_train = torch.tensor(t_obs, dtype=torch.float32).unsqueeze(-1)
    y_train = torch.tensor(y_obs, dtype=torch.float32)

    # GP setup
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(x_train, y_train, likelihood)

    model.train(); likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # short training loop
    for _ in range(50):
        optimizer.zero_grad()
        output = model(x_train)
        loss = -mll(output, y_train)
        loss.backward()
        optimizer.step()

    # prediction
    model.eval(); likelihood.eval()
    x_test = torch.tensor(t_grid, dtype=torch.float32).unsqueeze(-1)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = likelihood(model(x_test))
    lower, upper = pred.confidence_region()
    lower, upper = lower.numpy(), upper.numpy()

    # coverage proportion
    covered = np.mean((true_curve >= lower) & (true_curve <= upper))
    return covered

# ---------------------------
# 4) Repeat many times
# ---------------------------
if __name__ == "__main__":
    n_trials = 10000
    coverages = [run_one(seed=i) for i in range(n_trials)]
    overall = np.mean(coverages)
    print(f"Average coverage over {n_trials} runs: {overall:.3f}")
