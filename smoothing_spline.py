import json
import numpy as np
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt

import json
import numpy as np

# Load JSON file
with open("subject2.json") as f:
    data = json.load(f)

# Flatten the nested array and extract time and value
flat_data = [point[0] for point in data["array"]]  # each point is wrapped in a list
x = np.array([d["time"] for d in flat_data])
y = np.array([d["value"] for d in flat_data])

print(x)
print(y)


# Fit cubic smoothing spline (k=3) with default smoothing
spline = UnivariateSpline(x, y, k=3, s=None)  # s=None uses GCV to choose smoothing

# Evaluate the spline
xnew = np.linspace(x.min(), x.max(), 500)
ynew = spline(xnew)

# Plot
plt.figure(figsize=(8,5))
plt.plot(x, y, 'o', label='Data')
plt.plot(xnew, ynew, '-', label='Cubic Smoothing Spline')
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Cubic Smoothing Spline Fit")
plt.show()
