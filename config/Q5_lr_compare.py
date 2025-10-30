import os, numpy as np, matplotlib.pyplot as plt
figs = ["saved_models/lr1e4/figs/CartPole-v1_single-hidden_r100.npy",
        "saved_models/lr5e4/figs/CartPole-v1_single-hidden_r100.npy",
        "saved_models/lr1e3/figs/CartPole-v1_single-hidden_r100.npy"]
labels = ["lr=1e-4","lr=5e-4","lr=1e-3"]
r = [np.load(p) for p in figs]; m = min(map(len,r)); x = range(1,m+1)
for y,l in zip(r,labels): plt.plot(x, y[:m], label=l)
plt.xlabel("Episode"); plt.ylabel("R100"); plt.legend(); plt.title("Q5: LR compare")
plt.savefig("saved_models/figs/Q5_lr_compare.pdf", bbox_inches="tight")