import torch 
import gpytorch
import numpy
from matplotlib import pyplot as plt

# Set seed for reproducibility
torch.manual_seed(18) 

## STEP ONE: SIMULATE A DATASET ## 
X_START = 0 ## Lowest x value ##
X_END = 10 ## Highest x value ## 
NUM_POINTS = 500 ## No. of x values ## 
NOISE_VAR = 0 ## CHOOSE NOISE LEVEL ##

# Training inputs must be 2D: [N, 1]
x_vals = torch.linspace(X_START, X_END, NUM_POINTS).unsqueeze(-1) 

y_true = torch.sin(x_vals).squeeze() ## SPECIFY TRUE FUNCTION f ##
noise_var = torch.tensor(NOISE_VAR) # Convert noise var to a tensor
noise_std = torch.sqrt(noise_var)   # Calculate std of noise
error_vals = noise_std * torch.randn_like(y_true) 
y = y_true + error_vals # Add Gaussian noise

## STEP TWO: Define GP Model ##
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)    
        
## STEP THREE: Initialize likelihood and model ##
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(x_vals, y, likelihood)

## STEP FOUR: Train the GP ##
model.train()
likelihood.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

training_iterations = 100
for i in range(training_iterations):
    optimizer.zero_grad()
    output = model(x_vals)
    loss = -mll(output, y)
    loss.backward()
    if (i+1) % 20 == 0:
        print(f"Iter {i+1}/{training_iterations} - Loss: {loss.item():.3f}")
    optimizer.step()

## STEP FIVE: Make predictions ##
model.eval()
likelihood.eval()
x_test = torch.linspace(X_START * -1, X_END*2, 50).unsqueeze(-1)  # test inputs also [M,1]

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred = likelihood(model(x_test))

## STEP SIX: Extract mean and confidence intervals ##
mean = observed_pred.mean
lower, upper = observed_pred.confidence_region()

plt.figure(figsize=(8,5))
plt.plot(x_vals.squeeze().numpy(), y_true.numpy(), 'r', label='True function')
plt.scatter(x_vals.squeeze().numpy(), y.numpy(), label='Noisy observations')
plt.plot(x_test.squeeze().numpy(), mean.numpy(), 'b', label='GP mean')
plt.fill_between(x_test.squeeze().numpy(), lower.numpy(), upper.numpy(), color='blue', alpha=0.3, label='Confidence')
plt.legend()
plt.title("GP Regression Fit")
plt.show()
