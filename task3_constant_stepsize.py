import numpy as np
import matplotlib.pyplot as plt

# Problem parameters
N = 4
r = np.array([100, 50, 10, 1])
p = np.array([1/3, 1/12, 1/4, 1/3])
lambda_rate = 1e5
c = 0.5

# Simulation parameters
T = 0.1
num_episodes = 1000
theta_init = 50

# Constant step size
epsilon = 5  # Try different values: 1, 5, 10, 20

# For tracking results
theta_history = []
theta = theta_init

# Main simulation loop
for n in range(1, num_episodes + 1):
    
    num_packets = int(lambda_rate * T)
    rewards = np.random.choice(r, size=num_packets, p=p)
    admitted = np.sum(rewards >= theta)
    admission_rate = admitted / num_packets
    
    # Update theta (NO n^gamma - just constant epsilon!)
    theta = theta + epsilon * (admission_rate - c)
    
    theta_history.append(theta)
    
    if n % 100 == 0:
        print(f"Episode {n}: theta = {theta:.2f}, admission_rate = {admission_rate:.2f}")

# Plot
plt.figure(figsize=(10, 6))
plt.plot(theta_history, label=f'θ (ε = {epsilon})')
plt.axhline(y=10, color='r', linestyle='--', label='θ* = 10 (optimal)')
plt.xlabel('Episode')
plt.ylabel('Threshold θ')
plt.title(f'Robbins-Monro with Constant Step Size (ε = {epsilon})')
plt.legend()
plt.grid(True)
plt.show()