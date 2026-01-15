import numpy as np
import matplotlib.pyplot as plt

# Parameeters
N=4                              # Number of job classes
r=np.array([100, 50, 10, 1])     # Rewards (already sorted highest to lowest)
p=np.array([1/3, 1/12, 1/4, 1/3])  # Probability of each class
lambda_rate=1e5                  # Arrival rate (packets/second)
c=0.5                            # Capacity fraction

# Simulation parameters
T=0.1                          # Total time (seconds) per episode
num_episodes=1000              # How many episodes to run
theta_init=50                # Initial threshold

# Robbins-Monro parameters
epsilon_0=100                 # Initial step size
gamma=0.7                    # Decay rate (between 0.5 and 1)

# For tracking results
theta_history = []

# Initialize theta
theta = theta_init

# Main simulation loop
for n in range(1, num_episodes + 1):
    
    # Step 1: Calculate number of packets in this episode
    num_packets = int(lambda_rate * T)
    
    # Step 2: Generate random rewards for each packet
    rewards = np.random.choice(r, size=num_packets, p=p)
    
    # Step 3: Apply threshold policy - count admitted packets
    admitted = np.sum(rewards >= theta)
    
    # Step 4: Calculate admission rate
    admission_rate = admitted / num_packets
    
    # Step 5: Calculate step size (decreasing)
    epsilon = epsilon_0 / (n ** gamma)
    
    # Step 6: Update theta using Robbins-Monro rule
    theta = theta + epsilon * (admission_rate - c)
    
    # Save theta for plotting
    theta_history.append(theta)
    
    # Print progress every 100 episodes
    if n % 100 == 0:
        print(f"Episode {n}: theta = {theta:.2f}, admission_rate = {admission_rate:.2f}")

# Plot theta over episodes
plt.figure(figsize=(10, 6))
plt.plot(theta_history, label='θ (learned threshold)')
plt.axhline(y=10, color='r', linestyle='--', label='θ* = 10 (optimal)')
plt.xlabel('Episode')
plt.ylabel('Threshold θ')
plt.title('Robbins-Monro with Decreasing Step Size')
plt.legend()
plt.grid(True)
plt.show()