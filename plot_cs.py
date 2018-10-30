

import pandas as pd
import numpy as np
import plots
from matplotlib import pyplot as plt

data = pd.read_csv('4-1.csv', sep="|", header=None)
iter = data.iloc[:, 0].values
tr_losses = data.iloc[:, 1].values
te_losses = data.iloc[:, 2].values

# iter  = np.loadtxt( '2.csv' , delimiter='|', usecols=(0) )
title="Exp4:- Batch size=" + str(64) + " Learning Rate=" + str(0.005)
plots.linear_plot( iter , tr_losses, te_losses, "Iterations", "Losses",
                  title, 1 + 1)
plt.savefig( title +".png")
plt.show();


