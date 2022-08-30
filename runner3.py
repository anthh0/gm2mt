import numpy as np
import numpy.linalg as la
import gm2fr.constants as const
import matplotlib.pyplot as plt

width = 150
df = 2

hist = np.load('results/s2-alg_source_5e7_100k/f_dist.npz')
edges = np.arange(const.info["f"].magic - width / 2, const.info["f"].magic + width / 2 + df, df)
centers = (edges[:-1] + edges[1:]) / 2
mask = const.physical(centers)
heights = hist['heights']
cov = hist['cov'][mask][:, mask]

print(cov)

errors = np.zeros(shape = 43)
for i in range(len(errors)):
    errors[i] = cov[i,i]
print(errors)

# inv_cov = la.inv(cov)
# inverr = np.zeros(shape = 75)
# for i in range(len(inverr)):
#     inverr[i] = inv_cov[i,i]
# print(inverr)

fig, ax = plt.subplots()
ax.errorbar(centers, heights, yerr = errors)

fig.savefig('f.png')