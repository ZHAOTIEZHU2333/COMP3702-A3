import os, numpy as np, matplotlib.pyplot as plt

soft = "saved_models/q4_soft/figs/CartPole-v1_single-hidden_r100.npy"
hard = "saved_models/q4_hard/figs/CartPole-v1_single-hidden_r100.npy"

r_soft = np.load(soft); r_hard = np.load(hard)
m = min(len(r_soft), len(r_hard)); x = range(1, m+1)

plt.figure()
plt.plot(x, r_soft[:m], label="Soft update (tau=0.005)")
plt.plot(x, r_hard[:m], label="Hard sync (N=1000)")
plt.xlabel("Episode"); plt.ylabel("R100"); plt.title("Q4(d): Target Network – Soft vs Hard"); plt.legend()
os.makedirs("saved_models/figs", exist_ok=True)
plt.savefig("saved_models/figs/Q4_soft_vs_hard_R100.pdf", bbox_inches="tight")
plt.savefig("saved_models/figs/Q4_soft_vs_hard_R100.png", bbox_inches="tight")

import numpy as np

def first_hit(r100, thr=475):
    idx = np.where(r100 >= thr)[0]
    return int(idx[0]+1) if idx.size>0 else None

def tail_stats(r100, start=800):
    t = r100[start:] if len(r100)>start else r100
    return float(np.mean(t)), float(np.std(t))

print("[Soft] first hit 475:", first_hit(r_soft), "  mean/std after 800:", tail_stats(r_soft))
print("[Hard] first hit 475:", first_hit(r_hard), "  mean/std after 800:", tail_stats(r_hard))