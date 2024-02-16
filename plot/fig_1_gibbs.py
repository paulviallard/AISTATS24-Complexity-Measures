import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
from scipy.interpolate import interp1d

###############################################################################

path = os.path.dirname(__file__)
path = os.path.abspath(path)+os.sep
if(os.name == 'nt'):
    path = path.replace("\\", "/")

f = open(os.path.join(os.path.dirname(__file__), "header_standalone.tex"), "r")
preamble = f.read()
preamble = preamble.replace("\\input{", "\\input{"+path)


plt.rcParams.update({
    "font.size": 20,
    "font.family": "serif",
    "text.usetex": True,
    "pgf.rcfonts": False,
    "text.latex.preamble": preamble,
    "pgf.preamble": preamble,

})

###############################################################################

WHITE = "#FFFFFF"
BLACK = "#000000"
BLUE = "#0077BB"
CYAN = "#009988"
GREEN = "#009988"
ORANGE = "#EE7733"
RED = "#CC3311"
MAGENTA = "#EE3377"
GREY = "#BBBBBB"

###############################################################################

seed = 42
rng = default_rng(seed=seed)

x = np.arange(0, 10)
y = rng.normal(0.0, 1.0, size=(10,))
f = interp1d(x, y, kind="quadratic")
y = f(np.arange(0, 9, 0.01))
comp = (y-np.min(y))/(np.max(y)-np.min(y))

# --------------------------------------------------------------------------- #

distrib = np.exp(-comp)
distrib = distrib/np.sum(distrib)
distrib = (distrib-np.min(distrib))/(np.max(distrib)-np.min(distrib))

fig, ax = plt.subplots(1, 1, figsize=((0.49*8.27*2.5, 0.04*11.69*2.5)))

l_comp = ax.plot(
    np.arange(0, 9, 0.01), comp,
    label=r"$\comp(h, \Scal)$", color=BLUE)
ax.plot(np.arange(0, 9, 0.01), distrib, zorder=-1,
        label=r"$\AQ(h) \propto \exp[-\!\comp(h, \Scal)]$", c=ORANGE)
ax.fill_between(
    np.arange(0, 9, 0.01), distrib, zorder=-1, fc=ORANGE, alpha=0.5,
)
ax.set_xlim(0, 9.01)
ax.set_ylim(0, 1.1)

ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.62),
          frameon=False, ncol=3)

ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.tick_params(left=False, labelleft=False,
               labelbottom=False, bottom=False)

ax.set_xlabel(r"Hypothesis set $\Hcal$")
os.makedirs("figures/", exist_ok=True)
fig.savefig("figures/fig_1.pdf", bbox_inches="tight", transparent=True)
