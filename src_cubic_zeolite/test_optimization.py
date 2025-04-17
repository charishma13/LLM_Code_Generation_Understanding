import numpy as np
from scipy.optimize import minimize

# Define the function to minimize
def objective_function(x):
    #  simple paraboloid (a bowl shape)
    # The minimum of this function is at 𝑥 = [ 3 , − 1 ] where both squared terms become zero.
    return (x[0] - 3)**2 + (x[1] + 1)**2

# Initial guess - Start searching from the point (0, 0)
x0 = np.array([0.0, 0.0])
# x0 is your starting point for the search.

# Run optimization using BFGS
result = minimize(objective_function, x0, method='BFGS',options={'disp': True} )

# Print result
print("Optimal parameters:", result.x)
print("Function value at optimum:", result.fun)

import matplotlib.pyplot as plt

# Store path during optimization
path = []

def objective_with_tracking(x):
    path.append(np.copy(x))
    return (x[0] - 3)**2 + (x[1] + 1)**2

# Reset and optimize
path = []
result = minimize(
    objective_with_tracking,
    x0,
    method='BFGS',
    options={'disp': False}
)

# Convert path to array
path = np.array(path)

# Plot contours
x = np.linspace(-2, 6, 100)
y = np.linspace(-4, 2, 100)
X, Y = np.meshgrid(x, y)
Z = (X - 3)**2 + (Y + 1)**2

plt.contour(X, Y, Z, levels=50, cmap='viridis')
plt.plot(path[:, 0], path[:, 1], marker='o', color='red')
plt.title('BFGS Optimization Path')
plt.xlabel('x[0]')
plt.ylabel('x[1]')
plt.grid(True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Function to generate y (target) values with noise
def generate_data(X, true_w, true_b, noise_std=1.0):
    noise = np.random.normal(0, noise_std, size=X.shape)
    return true_w * X + true_b + noise

# Step 1: Generate sample data
X = np.linspace(0, 10, 50)
true_w, true_b = 2.5, -1.0
y = generate_data(X, true_w, true_b)  # Using our function

# Step 2: Define the loss (Mean Squared Error)
def mse_loss(params):
    w, b = params
    y_pred = w * X + b
    return np.mean((y - y_pred)**2)

# Step 3: Initial guess
initial_guess = [0.0, 0.0]

# Step 4: Minimize the loss
result = minimize(mse_loss, initial_guess, method='BFGS')

# Step 5: Extract optimal parameters
opt_w, opt_b = result.x
print("Optimized w and b:", opt_w, opt_b)

# Plot results
plt.scatter(X, y, label='Data')
plt.plot(X, opt_w * X + opt_b, color='red', label='Fitted Line')
plt.legend()
plt.title("Linear Fit using BFGS Optimization")
plt.show()
