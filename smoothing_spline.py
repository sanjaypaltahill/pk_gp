import json
import numpy as np
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt

# Load JSON file
with open("subject2.json") as f:
    data = json.load(f)

# Flatten the nested array
flat_data = [point[0] for point in data["array"]]
x = np.array([d["time"] for d in flat_data])  # hours
y = np.array([d["value"] * 1000.0 for d in flat_data])  # convert g/L → mg/L

# Fit cubic smoothing spline
spline = UnivariateSpline(x, y, k=3, s=None)  # GCV smoothing

# Evaluate spline on dense grid
xnew = np.linspace(x.min(), x.max(), 500)
ynew = spline(xnew)

# Plot
plt.figure(figsize=(8,5))
plt.plot(x, y, 'o', label='Observed (mg/L)')
plt.plot(xnew, ynew, '-', label='Cubic Smoothing Spline')
plt.xlabel("Time (hr)")
plt.ylabel("Paracetamol (mg/L)")
plt.title("Cubic Smoothing Spline Fit")
plt.legend()
plt.show()
