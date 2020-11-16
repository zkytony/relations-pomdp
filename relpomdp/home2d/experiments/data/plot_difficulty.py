# Difficulty score plot

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

filename = "difficulty-train-envs-6x6-random-20201113134115484.csv"
df = pd.read_csv(filename)
df = df.sort_values(['difficulty']).reset_index(drop=True)
sns.barplot(x="class", y="difficulty",
            data=df, palette="rocket")

xvals = df["class"]
plt.xticks(np.arange(len(xvals)), xvals, rotation='vertical')
plt.title("Difficulty Score (Random Explorer, 6x6)")
plt.tight_layout()
plt.savefig("Plot_Difficulty_%s.png" % filename)
