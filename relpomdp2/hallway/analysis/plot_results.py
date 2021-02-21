import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
timestamp = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")

def readlist(filenames):
    df = None
    for filename in filenames:
        if df is None:
            df = pd.read_pickle(filename)
        else:
            dftmp = pd.read_pickle(filename)
            df = pd.concat([dftmp, df])
    return df

files_perfect_sensor_increasing_size = [
    # TODO
    # "./EXP_perfect-sensor-increasing-size_01-22-2021-11-25-05.pkl"
    # "./EXP_perfect-sensor-increasing-size_01-22-2021-14-15-54.pkl",
    "./EXP_perfect-sensor-increasing-size_01-22-2021-15-07-25.pkl",
    "./EXP_perfect-sensor-increasing-size_02-09-2021-07-06-35.pkl"
]

files_same_size_increasing_noise = [
    # TODO
    # "./EXP_same-size-increasing-noise_01-22-2021-11-42-40.pkl"
    # "./EXP_same-size-increasing-noise_01-22-2021-15-00-20.pkl",
    "./EXP_same-size-increasing-noise_01-22-2021-15-11-55.pkl",
    "./EXP_same-size-increasing-noise_02-09-2021-07-13-44.pkl"
]

detectable_to_label = {
    "A": "A (target)",
    "AB": "A, B",
    "AB-manual": "A, B (must lookB first)"
}

df_hwl = readlist(files_perfect_sensor_increasing_size)
df_hwl = df_hwl.loc[df_hwl["Detectable"].isin(["A", "AB"])]
df_hwl["Detectable"] = df_hwl["Detectable"].replace(detectable_to_label)

df_noise = readlist(files_same_size_increasing_noise)
df_noise = df_noise.loc[df_noise["Detectable"].isin(["A", "AB"])]
df_noise["Detectable"] = df_noise["Detectable"].replace(detectable_to_label)

# Plot return vs. hallway size and value vs. hallway size
fig, axes = plt.subplots(1, 2, sharey=True, figsize=(12, 8))
sns.pointplot(x="hallway_length", y="return",
              hue="Detectable", data=df_hwl, ax=axes[0])
sns.pointplot(x="hallway_length", y="Vb0",
              hue="Detectable",
              data=df_hwl.loc[df_hwl["Detectable"] != detectable_to_label["AB-manual"]],
              ax=axes[1])
axes[0].set_title("Actual return vs. hallway length\n"
                  "(Perfect sensor for A and B. B's sensor is bigger )")
axes[1].set_title("Expected return (i.e. V(b0)) vs. hallway length\n"
                  "(Perfect sensor for A and B. B's sensor is bigger)")
plt.savefig("perfect_sensor_increasing_size_%s.png" % timestamp)
plt.clf()

# Plot return vs. noise and value vs. noise
fig, axes = plt.subplots(1, 2, sharey=True, figsize=(12, 8))
sns.pointplot(x="NL", y="return",
              hue="Detectable", data=df_noise, ax=axes[0])
sns.pointplot(x="NL", y="Vb0",
              hue="Detectable",
              data=df_noise.loc[df_noise["Detectable"] != detectable_to_label["AB-manual"]],
              ax=axes[1])
axes[0].set_title("Actual return vs. Noise level (NL) for A's sensor\n"
                  "(Same hallway length. B's sensor is perfect and bigger)")
axes[1].set_title("Expected return (i.e. V(b0)) vs. Noise level (NL) for A's sensor\n"
                  "(Same hallway length. B's sensor is perfect and bigger)")
plt.savefig("same_size_increasing_noise_%s.png" % timestamp)
