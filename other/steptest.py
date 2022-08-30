import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import auxiliary as aux

res_01_equal = np.load("results/s-0.1-equal/states.npz")
res_01_less = np.load("results/s-0.1-less/states.npz")
res_01_greater = np.load("results/s-0.1-greater/states.npz")

res_03_equal = np.load("results/s-0.3-equal/states.npz")
res_03_less = np.load("results/s-0.3-less/states.npz")
res_03_greater = np.load("results/s-0.3-greater/states.npz")

res_05_equal = np.load("results/s-0.5-equal/states.npz")
res_05_less = np.load("results/s-0.5-less/states.npz")
res_05_greater = np.load("results/s-0.5-greater/states.npz")

res_07_equal = np.load("results/s-0.7-equal/states.npz")
res_07_less = np.load("results/s-0.7-less/states.npz")
res_07_greater = np.load("results/s-0.7-greater/states.npz")

res_10_equal = np.load("results/s-1.0-equal/states.npz")
res_10_less = np.load("results/s-1.0-less/states.npz")
res_10_greater = np.load("results/s-1.0-greater/states.npz")

t01 = res_01_equal["t"] * 10**9
t03 = res_03_equal["t"] * 10**9
t05 = res_05_equal["t"] * 10**9
t07 = res_07_equal["t"] * 10**9
t10 = res_10_equal["t"] * 10**9

x, y = res_01_equal["x"], res_01_equal["y"]
r01 = (np.sqrt(x**2 + y**2) - aux.r_magic) * 1000
x, y = res_03_equal["x"], res_03_equal["y"]
r03 = (np.sqrt(x**2 + y**2) - aux.r_magic) * 1000
x, y = res_05_equal["x"], res_05_equal["y"]
r05 = (np.sqrt(x**2 + y**2) - aux.r_magic) * 1000
x, y = res_07_equal["x"], res_07_equal["y"]
r07 = (np.sqrt(x**2 + y**2) - aux.r_magic) * 1000
x, y = res_10_equal["x"], res_10_equal["y"]
r10 = (np.sqrt(x**2 + y**2) - aux.r_magic) * 1000

vx, vy = res_01_less["vx"], res_01_less["vy"]
less01 = aux.v_to_p(np.sqrt(vx**2 + vy**2))
vx, vy = res_03_less["vx"], res_03_less["vy"]
less03 = aux.v_to_p(np.sqrt(vx**2 + vy**2))
vx, vy = res_05_less["vx"], res_05_less["vy"]
less05 = aux.v_to_p(np.sqrt(vx**2 + vy**2))
vx, vy = res_07_less["vx"], res_07_less["vy"]
less07 = aux.v_to_p(np.sqrt(vx**2 + vy**2))
vx, vy = res_10_less["vx"], res_10_less["vy"]
less10 = aux.v_to_p(np.sqrt(vx**2 + vy**2))

vx, vy = res_01_equal["vx"], res_01_equal["vy"]
equal01 = aux.v_to_p(np.sqrt(vx**2 + vy**2))
vx, vy = res_03_equal["vx"], res_03_equal["vy"]
equal03 = aux.v_to_p(np.sqrt(vx**2 + vy**2))
vx, vy = res_05_equal["vx"], res_05_equal["vy"]
equal05 = aux.v_to_p(np.sqrt(vx**2 + vy**2))
vx, vy = res_07_equal["vx"], res_07_equal["vy"]
equal07 = aux.v_to_p(np.sqrt(vx**2 + vy**2))
vx, vy = res_10_equal["vx"], res_10_equal["vy"]
equal10 = aux.v_to_p(np.sqrt(vx**2 + vy**2))

vx, vy = res_01_greater["vx"], res_01_greater["vy"]
greater01 = aux.v_to_p(np.sqrt(vx**2 + vy**2))
vx, vy = res_03_greater["vx"], res_03_greater["vy"]
greater03 = aux.v_to_p(np.sqrt(vx**2 + vy**2))
vx, vy = res_05_greater["vx"], res_05_greater["vy"]
greater05 = aux.v_to_p(np.sqrt(vx**2 + vy**2))
vx, vy = res_07_greater["vx"], res_07_greater["vy"]
greater07 = aux.v_to_p(np.sqrt(vx**2 + vy**2))
vx, vy = res_10_greater["vx"], res_10_greater["vy"]
greater10 = aux.v_to_p(np.sqrt(vx**2 + vy**2))

plt.rcParams["figure.figsize"] = (12,6)
plt.rcParams["lines.linewidth"] = 0.8

fig1, ax1 = plt.subplots()
fig1.suptitle("Momentum change over propagation: $p - p_{magic} = -0.01$")
ax1.set_xlabel("ns")
ax1.set_ylabel("GeV")
ax1.plot(t01, less01 - aux.p_magic + 0.01, label = "0.1")
ax1.plot(t03, less03 - aux.p_magic + 0.01, label = "0.3")
ax1.plot(t05, less05 - aux.p_magic + 0.01, label = "0.5")
ax1.plot(t07, less07 - aux.p_magic + 0.01, label = "0.7")
ax1.plot(t10, less10 - aux.p_magic + 0.01, label = "1.0")
ax1.plot([0, 1000], [0, 0], label = "truth", linestyle = "dashed")
ax1.ticklabel_format(style = 'sci', useMathText = True)
ax1.legend()

fig2, ax2 = plt.subplots()
fig2.suptitle("Momentum chang over propagation: $p - p_{magic} = +0.01$")
ax2.set_xlabel("ns")
ax2.set_ylabel("GeV")
ax2.plot(t01, greater01 - aux.p_magic - 0.01, label = "0.1")
ax2.plot(t03, greater03 - aux.p_magic - 0.01, label = "0.3")
ax2.plot(t05, greater05 - aux.p_magic - 0.01, label = "0.5")
ax2.plot(t07, greater07 - aux.p_magic - 0.01, label = "0.7")
ax2.plot(t10, greater10 - aux.p_magic - 0.01, label = "1.0")
ax2.plot([0, 1000], [0, 0], label = "truth", linestyle = "dashed")
ax2.ticklabel_format(style = 'sci', useMathText = True)
ax2.legend()

fig3, ax3 = plt.subplots()
fig3.suptitle("Momentum change over propagation: $p - p_{magic} = 0$")
ax3.set_xlabel("ns")
ax3.set_ylabel("GeV")
ax3.plot(t01, equal01 - aux.p_magic, label = "0.1")
ax3.plot(t03, equal03 - aux.p_magic, label = "0.3")
ax3.plot(t05, equal05 - aux.p_magic, label = "0.5")
ax3.plot(t07, equal07 - aux.p_magic, label = "0.7")
ax3.plot(t10, equal10 - aux.p_magic, label = "1.0")
ax3.plot([0, 1000], [0, 0], label = "truth", linestyle = "dashed")
ax3.ticklabel_format(style = 'sci', useMathText = True)
ax3.legend()

fig4, ax4 = plt.subplots()
fig4.suptitle("radial position, $p - p_{magic} = 0$")
ax4.set_xlabel("ns")
ax4.set_ylabel("mm")
ax4.plot(t01, r01, label = "0.1")
ax4.plot(t03, r03, label = "0.3")
ax4.plot(t05, r05, label = "0.5")
ax4.plot(t07, r07, label = "0.7")
ax4.plot(t10, r10, label = "1.0")
ax4.plot([0, 1000], [0, 0], label = "truth", linestyle = "dashed")
ax4.ticklabel_format(style = 'sci', useMathText = True)
ax4.legend()

comparison = PdfPages('results/t_step test.pdf')
comparison.savefig(fig1)
comparison.savefig(fig2)
comparison.savefig(fig3)
comparison.savefig(fig4)
comparison.close()
plt.show()