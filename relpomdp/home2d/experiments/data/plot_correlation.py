# Difficulty score plot

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

filename = "correlation_train-envs-6x6_20201113135539885.csv"
df = pd.read_csv(filename)
df2 = df[(df["corr_score"] == 0) & (df["corr_score"] < 20)][5:6]
df3 = df[(df["corr_score"] > 20) & (df["corr_score"] < 50)][1:2]
df = df[df["corr_score"] > 50][1:2].append(df2).append(df3)
df = df.sort_values(['corr_score']).reset_index(drop=True)
df["combo"] = df[["class1", "class2"]].agg("-".join, axis=1)
sns.barplot(x="combo", y="corr_score",
            data=df, palette="rocket")

xvals = df["combo"]
plt.xticks(np.arange(len(xvals)), xvals, rotation=45)
plt.title("Correlation Score (6x6)")
plt.tight_layout()
plt.savefig("Plot_Correlatoin_%s.png" % filename)
