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

# Robbins-Monro parameters (decreasing step size)
epsilon_0 = 100
gamma = 0.7

# For tracking results
theta_history = []
polyak_history = []  # NEW: track the running average

theta = theta_init
theta_sum = 0  # NEW: sum of all theta values

# Main simulation loop
for n in range(1, num_episodes + 1):
    
    num_packets = int(lambda_rate * T)
    rewards = np.random.choice(r, size=num_packets, p=p)
    admitted = np.sum(rewards >= theta)
    admission_rate = admitted / num_packets
    
    # Update theta (same as Task 2)
    epsilon = epsilon_0 / (n ** gamma)
    theta = theta + epsilon * (admission_rate - c)
    
    # NEW: Calculate Polyak average
    theta_sum = theta_sum + theta
    polyak_avg = theta_sum / n
    
    theta_history.append(theta)
    polyak_history.append(polyak_avg)
    
    if n % 100 == 0:
        print(f"Episode {n}: theta = {theta:.2f}, polyak_avg = {polyak_avg:.2f}")

# Plot both
plt.figure(figsize=(10, 6))
plt.plot(theta_history, label='θ (raw)', alpha=0.7)
plt.plot(polyak_history, label='Polyak average', linewidth=2)
plt.axhline(y=10, color='r', linestyle='--', label='θ* = 10 (optimal)')
plt.xlabel('Episode')
plt.ylabel('Threshold θ')
plt.title('Robbins-Monro with Polyak Averaging')
plt.legend()
plt.grid(True)
plt.show()