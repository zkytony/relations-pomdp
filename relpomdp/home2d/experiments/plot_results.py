import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt

output_dir = "results"
timestr = "20201104034301218"

df_rewards = pd.read_csv(os.path.join(output_dir, "results_%s_disc-rewards.csv" % (timestr)),
                         index_col=0)
df_success = pd.read_csv(os.path.join(output_dir, "results_%s_success.csv" % (timestr)),
                         index_col=0)

# Transform the data for easier plotting
# headers = ["index", "method", "discounted_reward"]
# rows = []
# for index, row in df_rewards.iterrows():
#     for column in df_rewards:
#         print(column)
#     # rows.append([index, "POMDP", ])

sns.barplot(data=df_rewards)
plt.show()

sns.barplot(data=df_success)
plt.show()
