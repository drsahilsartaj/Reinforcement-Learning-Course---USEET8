import numpy as np

# Problem parameters from Table 1
N = 4                              # Number of job classes
r = np.array([100, 50, 10, 1])     # Rewards (already sorted highest to lowest)
p = np.array([1/3, 1/12, 1/4, 1/3])  # Probability of each class
lambda_rate = 1e5                  # Arrival rate (packets/second)
c = 0.5                            # Capacity fraction

# Derived values
capacity = c * lambda_rate         # Actual capacity: 50,000 packets/sec
arrival_rates = lambda_rate * p    # Arrivals per second for each class

print("\nArrival rates per class:", np.round(arrival_rates, 3))
print("\nTotal capacity:", capacity)

# Fractional Knapsack: Find optimal threshold
cumulative_admitted = 0
optimal_threshold = 0

print()
for i in range(N):
    if cumulative_admitted + arrival_rates[i] <= capacity:
        # Can admit entire class
        cumulative_admitted += arrival_rates[i]
        print(f"Class {i+1} (reward={r[i]}): Admit ALL {arrival_rates[i]:.0f} -> Total: {cumulative_admitted:.0f}")
    else:
        # Can only admit part of this class (we hit capacity here)
        remaining_capacity = capacity - cumulative_admitted
        print(f"Class {i+1} (reward={r[i]}): Admit PARTIAL {remaining_capacity:.0f} -> Total: {capacity:.0f}")
        optimal_threshold = r[i]  # This is our threshold!
        break

print(f"\n*** Optimal threshold θ* = {optimal_threshold} ***")