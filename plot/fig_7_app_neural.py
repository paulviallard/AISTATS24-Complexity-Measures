import sys
import os
import matplotlib.pyplot as plt
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from core.nd_data import NDData

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

seed_size = 5
data_text_dict = {
    "mnist": "MNIST",
    "fashion": "FashionMNIST",
}

for data in ["mnist", "fashion"]:

    d = NDData(
        os.path.join(
            os.path.join(os.path.dirname(__file__), ".."),
            "result_fig_3.csv"))

    for comp in ["neural", "dist_neural"]:

        # We initialize the plot
        fig, ax = plt.subplots(1, 1, figsize=((1.0*8.27*2.0, 1.0*11.69*2.0)))
        fig.subplots_adjust(wspace=0.2, hspace=0.28)

        X = d.get("risk_post_test", "seeger", "neural_path",
                  data=data, prior="0.0", comp_post=comp, alpha_post="60000.0")
        X = X.sort_values(by=["neural_path"])
        X.to_numpy()

        X = np.reshape(X, (-1, seed_size, 3))
        X_str = X[:, 0, 2]

        for i in range(len(X_str)):
            tab_ = X_str[i].replace(".h5", "").split("_")
            X_str[i] = tab_[-3]+" / 0."+tab_[-2]+" / 0."+tab_[-1]

        X_mean = np.mean(X[:, :, :2], axis=1)
        X_std = np.std(X[:, :, :2].astype(float), axis=1)
        X_pos = np.arange(len(X_str))

        ax.invert_yaxis()

        ax.barh(X_pos, X_mean[:, 1], xerr=X_std[:, 1],
                alpha=1.0, hatch="/", color=WHITE, edgecolor=BLACK,
                error_kw=dict(lw=3, ecolor=BLACK, capsize=10, capthick=1))
        ax.barh(X_pos, X_mean[:, 0], xerr=X_std[:, 0],
                alpha=0.6, color=ORANGE,
                error_kw=dict(lw=3, ecolor=BLACK, capsize=10, capthick=1))

        ax.set_xlim(0.0)
        ax.set_ylim(-0.4, X_pos[-1]+0.4)

        if(comp == "neural"):
            ax.set_title(r"{} ($\neural$)".format(
                data_text_dict[data]), fontsize=25)
        elif(comp == "dist_neural"):
            ax.set_title(r"{} ($\distneural$)".format(
                data_text_dict[data]), fontsize=25)

        ax.set_yticks(X_pos, labels=X_str)

        os.makedirs("figures/", exist_ok=True)
        if(comp == "neural"):
            fig.savefig(
                "figures/fig_7_{}.pdf".format(data),
                bbox_inches="tight")
        elif(comp == "dist_neural"):
            fig.savefig(
                "figures/fig_7_dist_{}.pdf".format(data),
                bbox_inches="tight")
        plt.close(fig)
