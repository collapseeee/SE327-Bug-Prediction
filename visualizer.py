import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr

# ---------------------------------------------------------
# Step 1: Load and Inspect log4j-1.0.csv
# ---------------------------------------------------------
df_v10 = pd.read_csv("log4j-1.0.csv")

# Identify numerical metric columns (excluding non-numeric metadata and target)
ignore_cols = {"name", "version", "bug"}
metric_cols = [
    c for c in df_v10.select_dtypes(include=[np.number]).columns 
    if c not in ignore_cols
]

# 1.1 Distribution plots for all metrics (log4j-1.0 baseline)
n_cols = len(metric_cols)
n_rows = int(np.ceil(n_cols / 5)) if n_cols > 0 else 1

fig, axes = plt.subplots(n_rows, 5, figsize=(16, 3 * n_rows))
axes = np.array(axes).flatten()

for i, col in enumerate(metric_cols):
    df_v10[col].hist(ax=axes[i], bins=20, color="#0b5ed7")
    axes[i].set_title(col, fontsize=10)

# Hide any unused subplots
for j in range(len(metric_cols), len(axes)):
    fig.delaxes(axes[j])

plt.suptitle("Metric Distributions - Log4j 1.0", fontsize=16)
plt.tight_layout()
plt.savefig("metric_distributions_log4j10.png", dpi=300)
plt.close()

# 1.2 Calculate Spearman and Pearson correlations with 'bug'
corr_results = []
for col in metric_cols:
    if df_v10[col].nunique() <= 1 or df_v10["bug"].nunique() <= 1:
        s_corr, s_p, p_corr, p_p = np.nan, np.nan, np.nan, np.nan
    else:
        s_corr, s_p = spearmanr(df_v10[col], df_v10["bug"])
        p_corr, p_p = pearsonr(df_v10[col], df_v10["bug"])

    corr_results.append({
        "metric": col,
        "spearman_rho": s_corr,
        "spearman_p": s_p,
        "pearson_r": p_corr,
        "pearson_p": p_p
    })

df_corr = pd.DataFrame(corr_results).dropna(subset=["spearman_rho"]).sort_values(by="spearman_rho", ascending=False)
print("Correlation Analysis with Bug Count (log4j-1.0):")
print(df_corr.to_string(index=False))

# 1.3 Identify Top 5 Positive & Negative Metrics and Plot
top_pos = df_corr.sort_values(by="spearman_rho", ascending=False).head(5)
top_neg = df_corr.sort_values(by="spearman_rho", ascending=True).head(5)

# A. Top 5 Positive Correlations Scatter Plot
fig, axes = plt.subplots(1, 5, figsize=(22.5, 4))
for ax, (_, row) in zip(axes, top_pos.iterrows()):
    metric = row["metric"]
    rho = row["spearman_rho"]
    p_val = row["spearman_p"]
    sns.regplot(data=df_v10, x=metric, y="bug", ax=ax, scatter_kws={'alpha': 0.6}, color="#1f77b4")
    ax.set_title(f"{metric} vs Bug\n(rho={rho:.2f}, p={p_val:.1e})", fontsize=11)
    ax.set_xlabel(metric)
    ax.set_ylabel("bug")
plt.tight_layout()
plt.savefig("top_positive_scatter_log4j10.png", dpi=300)
plt.close()

# B. Top 5 Negative Correlations Scatter Plot
fig, axes = plt.subplots(1, 5, figsize=(22.5, 4))
for ax, (_, row) in zip(axes, top_neg.iterrows()):
    metric = row["metric"]
    rho = row["spearman_rho"]
    p_val = row["spearman_p"]
    sns.regplot(data=df_v10, x=metric, y="bug", ax=ax, scatter_kws={'alpha': 0.6}, color="#d62728")
    ax.set_title(f"{metric} vs Bug\n(rho={rho:.2f}, p={p_val:.2f})", fontsize=11)
    ax.set_xlabel(metric)
    ax.set_ylabel("bug")
plt.tight_layout()
plt.savefig("top_negative_scatter_log4j10.png", dpi=300)
plt.close()

print("Generated top_positive_scatter_log4j10.png and top_negative_scatter_log4j10.png")

# ---------------------------------------------------------
# Step 2: Version Generalization (1.0 vs 1.1 vs 1.2)
# ---------------------------------------------------------
def get_correlations(df, name):
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ignore_cols]
    res = {}
    for c in numeric_cols:
        if df[c].nunique() > 1 and df["bug"].nunique() > 1:
            rho, _ = spearmanr(df[c], df["bug"])
        else:
            rho = np.nan
        res[c] = rho
    return pd.Series(res, name=name)

try:
    df_v11 = pd.read_csv("log4j-1.1.csv")
    df_v12 = pd.read_csv("log4j-1.2.csv")

    s_10 = get_correlations(df_v10, "log4j-1.0")
    s_11 = get_correlations(df_v11, "log4j-1.1")
    s_12 = get_correlations(df_v12, "log4j-1.2")

    version_comparison = pd.concat([s_10, s_11, s_12], axis=1).sort_values(by="log4j-1.0", ascending=False)
    print("\nCorrelation Consistency Across Log4j Versions (Spearman rho):")
    print(version_comparison)
    print("log4j-1.1 and 1.2 analyzed")

    # Figure 1: Clean summary bar chart for all metrics across releases
    plt.figure(figsize=(12, 5))
    version_comparison.plot(kind="bar", figsize=(12, 5), colormap="Blues_r", edgecolor="black")
    plt.title("Spearman Correlation with Bug Counts Across Log4j Releases", fontsize=12)
    plt.xlabel("Object-Oriented Metric")
    plt.ylabel("Spearman Rho")
    plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig("version_generalization_bar.png", dpi=300)
    plt.close()

    # Figure 2: Focused 3x3 Grid comparing core structural metrics (CBO, NPM, CAM)
    dfs = {"Log4j 1.0": df_v10, "Log4j 1.1": df_v11, "Log4j 1.2": df_v12}
    key_metrics = ["cbo", "npm", "cam"]
    fig, axes = plt.subplots(3, 3, figsize=(13, 10))

    for r_idx, metric in enumerate(key_metrics):
        for c_idx, (v_name, df_ver) in enumerate(dfs.items()):
            ax = axes[r_idx, c_idx]
            color = "#1f77b4" if metric != "cam" else "#d62728"
            sns.regplot(data=df_ver, x=metric, y="bug", ax=ax, scatter_kws={'alpha': 0.5, 's': 25}, color=color)
            rho, _ = spearmanr(df_ver[metric], df_ver["bug"])
            ax.set_title(f"{v_name}: {metric.upper()} vs Bug (rho={rho:.2f})", fontsize=11)
            ax.set_xlabel(metric)
            ax.set_ylabel("bug" if c_idx == 0 else "")

    plt.suptitle("Cross-Version Comparison of Core Metrics (CBO, NPM, CAM)", fontsize=13, y=0.99)
    plt.tight_layout()
    plt.savefig("core_metrics_cross_version.png", dpi=300)
    plt.close()

    print("Generated version_generalization_bar.png and core_metrics_cross_version.png")
except FileNotFoundError as e:
    print(f"\nMissing version file: {e}")
    version_comparison = pd.DataFrame(get_correlations(df_v10, "log4j-1.0"))

# ---------------------------------------------------------
# Step 3: Project Generalization (Tomcat)
# ---------------------------------------------------------
try:
    df_tomcat = pd.read_csv("tomcat.csv")
    s_tomcat = get_correlations(df_tomcat, "Tomcat")
    
    project_comparison = pd.concat([version_comparison, s_tomcat], axis=1)
    print("\nCross-Project Comparison with Tomcat (Spearman rho):")
    print(project_comparison.round(3))
    
    # Generate Cross-Project Comparison Bar Chart
    plt.figure(figsize=(14, 5.5))
    ax = project_comparison.plot(
        kind='bar', 
        figsize=(14, 5.5), 
        colormap='tab10', 
        width=0.8, 
        edgecolor='black', 
        linewidth=0.5
    )
    plt.title("Cross-Project Metric Correlation Comparison: Log4j vs. Tomcat", fontsize=13)
    plt.xlabel("Object-Oriented Metric", fontsize=11)
    plt.ylabel("Spearman Correlation (rho)", fontsize=11)
    plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.legend(title="Dataset")
    plt.tight_layout()
    plt.savefig("cross_project_generalization_bar.png", dpi=300)
    plt.close()
    
    print("Generated cross_project_generalization_bar.png successfully.")
except FileNotFoundError:
    print("\nNote: 'tomcat.csv' not found in working directory.")