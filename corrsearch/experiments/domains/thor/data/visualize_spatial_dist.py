import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def main():
    df = pd.read_csv("./robothor_object_stats.csv")

    scenes = df["scene"].unique()

    fig, ax = plt.subplots(figsize=(20,15))
    os.makedirs("plots", exist_ok=True)

    for scene in scenes:
        print(scene)
        subdf = df.loc[df["scene"] == scene]
        x = []
        z = []
        t = []
        for index, row in subdf.iterrows():
            x.append(row["pos_x"])
            z.append(row["pos_z"])
            t.append(row["type"])
        ax.scatter(x, z)
        for i, label in enumerate(t):
            ax.annotate(label, (x[i], z[i]))
        ax.set_title(scene)
        plt.savefig(os.path.join("plots", "{}-objects.png".format(scene)))
        ax.clear()

if __name__ == "__main__":
    main()
