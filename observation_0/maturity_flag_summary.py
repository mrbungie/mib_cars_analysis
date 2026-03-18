import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

full_path = "observation_0/observation0_full_with_maturity.csv"
df = pd.read_csv(full_path)

plt.figure(figsize=(8, 4))
sns.countplot(data=df, x="age_bucket", hue="is_mature", palette="viridis")
plt.title("Maturity flag distribution across age buckets")
plt.xlabel("Age bucket (days)")
plt.ylabel("Count")
plt.tight_layout()
output_path = "observation_0/maturity_flag_counts.png"
plt.savefig(output_path, dpi=150)
print("Saved", output_path)
