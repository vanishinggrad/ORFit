
import main
import numpy as np
np.set_printoptions(precision=4)

num_runs = 10
try:
    data = np.load(f'logs_{num_runs}_runs.npz')
    logs = data["logs"]
except:
    logs = np.array([main.ORfit() for i in range(num_runs)])
    np.savez(f'logs_{num_runs}_runs', logs = logs)

import matplotlib.pyplot as plt
plt.figure(dpi=100, figsize=(6, 4))
plt.plot(logs[:, :, 0].mean(axis=0), label = "train error")
plt.fill_between(range(logs.shape[1]), y1=logs[:, :, 0].mean(axis=0) + logs[:, :, 0].std(axis=0),
                 y2 = logs[:, :, 0].mean(axis=0) - logs[:, :, 0].std(axis=0), alpha=0.2)
plt.plot(logs[:, :, 1].mean(axis=0), label = "test error", ls='--')
plt.fill_between(range(logs.shape[1]), y1=logs[:, :, 1].mean(axis=0) + logs[:, :, 1].std(axis=0),
                 y2 = logs[:, :, 1].mean(axis=0) - logs[:, :, 1].std(axis=0), alpha=0.2)
plt.xlabel("Training steps (samples)")
plt.xlabel("RMSE (deg)")
plt.legend(loc='best')
plt.grid(visible=True)
plt.show()