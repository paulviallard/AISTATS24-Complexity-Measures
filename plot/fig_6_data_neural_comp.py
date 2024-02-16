import os
import numpy as np
import matplotlib.pyplot as plt
from h5py import File

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
    "text.usetex": True,
    "font.family": "serif",
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

data_list = ["mnist", "fashion"]
data_name = {"mnist": "MNIST", "fashion": "FashionMNIST"}

fig, ax_list = plt.subplots(2, 1, figsize=((1.0*8.27*2.0, 0.5*11.69*2.0)))

for i, data_ in enumerate(data_list):
    data = File("data/dataset_neural_comp_{}.h5".format(data_), "r")

    # ----------------------------------------------------------------------- #

    ax_list[i].hist(
        np.abs(np.array(data["risk_val"])-np.array(data["risk_train"])),
        bins=50)
    ax_list[i].set_title("{}".format(data_name[data_]), fontsize=16)

    # ----------------------------------------------------------------------- #

os.makedirs("figures/", exist_ok=True)
fig.savefig("figures/fig_6.pdf".format(data_),
            bbox_inches="tight", transparent=True)
