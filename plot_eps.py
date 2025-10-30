import numpy as np, matplotlib.pyplot as plt
from dqn_common import epsilon_by_frame
cfgs = [("decay30k", dict(epsilon_start=1.0, epsilon_final=0.001, epsilon_decay=30000)),
        ("decay50k", dict(epsilon_start=1.0, epsilon_final=0.001, epsilon_decay=50000)),
        ("decay80k", dict(epsilon_start=1.0, epsilon_final=0.001, epsilon_decay=80000))]
F = 200_000
for name,p in cfgs:
    xs = np.arange(1, F+1); ys = [epsilon_by_frame(i, p) for i in xs]
    plt.plot(xs, ys, label=name)
plt.xlabel("Frame"); plt.ylabel("epsilon"); plt.legend()
plt.savefig("saved_models/figs/Q6_epsilon_curves.pdf", bbox_inches="tight")