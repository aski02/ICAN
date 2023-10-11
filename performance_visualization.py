# -------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------

import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# Visualizations
# -------------------------------------------------------------------------

# Load runtimes from files
def load_runtimes(filename):
    with open(filename + ".txt", "r") as file:
        lines = file.readlines()[1:]
        return [float(line.strip()) for line in lines]

# Data points
data_points = [20, 30, 40, 50, 60, 70, 80, 90, 100]

# Load runtimes
runtimes_0_gpr = load_runtimes("runtimes_0_gpr")
runtimes_1_gpr = load_runtimes("runtimes_1_gpr")
runtimes_2_gpr = load_runtimes("runtimes_2_gpr")
runtimes_3_gpr = load_runtimes("runtimes_3_gpr")

runtimes_0_xgb = load_runtimes("runtimes_0_xgb")
runtimes_1_xgb = load_runtimes("runtimes_1_xgb")
runtimes_2_xgb = load_runtimes("runtimes_2_xgb")
runtimes_3_xgb = load_runtimes("runtimes_3_xgb")


# Plot for GPR runtimes
plt.figure(figsize=(10, 6))
plt.plot(data_points, runtimes_0_gpr, marker='o', label='Dataset 0 (Air-pressure)')
plt.plot(data_points, runtimes_1_gpr, marker='o', label='Dataset 1 (Stocks)')
plt.plot(data_points, runtimes_2_gpr, marker='o', label='Dataset 2 (Simulated)')
plt.plot(data_points, runtimes_3_gpr, marker='o', label='Dataset 3 (Simulated)')

plt.xlabel('Number of datapoints')
plt.ylabel('Runtime (seconds)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("gpr_runtimes.png")

# Plot for XGBoost runtimes
plt.figure(figsize=(10, 6))
plt.plot(data_points, runtimes_0_xgb, marker='o', label='Dataset 0 (Air-pressure)')
plt.plot(data_points, runtimes_1_xgb, marker='o', label='Dataset 1 (Stocks)')
plt.plot(data_points, runtimes_2_xgb, marker='o', label='Dataset 2 (Simulated)')
plt.plot(data_points, runtimes_3_xgb, marker='o', label='Dataset 3 (Simulated))')

plt.xlabel('Number of datapoints')
plt.ylabel('Runtime (seconds)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("xgb_runtimes.png")

