import os, numpy as np, matplotlib.pyplot as plt
figs = "saved_models/figs"
v0 = np.load(os.path.join(figs, "CartPole-v0_single-hidden_r100.npy"))
v1 = np.load(os.path.join(figs, "CartPole-v1_single-hidden_r100.npy"))
m = min(len(v0), len(v1)); x = np.arange(1, m+1)
plt.plot(x, v0[:m], label="CartPole-v0 (R100)")
plt.plot(x, v1[:m], label="CartPole-v1 (R100)")
plt.xlabel("Episode"); plt.ylabel("R100"); plt.legend(); plt.title("Q2: v0 vs v1 (R100)")
plt.savefig(os.path.join(figs, "Q2_v0_vs_v1_R100.pdf"), bbox_inches="tight")
plt.savefig(os.path.join(figs, "Q2_v0_vs_v1_R100.png"), bbox_inches="tight")