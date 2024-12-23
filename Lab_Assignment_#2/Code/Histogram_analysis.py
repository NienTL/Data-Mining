import pandas as pd
import matplotlib.pyplot as plt
import os

data = pd.read_csv('./train_X.csv')
# Select numerical columns
numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns

# Plot histograms for each numerical column
cnt = 0
for col in numeric_cols:
    cnt += 1
    plt.figure(figsize=(8, 6))
    # Drop NA values for plotting
    plt.hist(data[col].dropna(), bins=30, edgecolor='k', alpha=0.7, color="#004777")  
    plt.title(f'Histogram of {col}', fontsize=16)
    plt.xlabel(col, fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    if not os.path.exists("./visualization_analysis"):
        os.makedirs("./visualization_analysis")
    plt.savefig(f'./visualization_analysis/histogram_{cnt}.png')
    plt.show()

# Select categorical columns
categorical_cols = data.select_dtypes(include=['object']).columns

# Generate histograms for each categorical column
for col in categorical_cols:
    cnt += 1
    plt.figure(figsize=(8, 6))
    data[col].value_counts().plot(kind='bar', alpha=0.7, edgecolor='k', color="#EFD28D")
    plt.title(f'Histogram of {col}', fontsize=16)
    plt.xlabel(col, fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    if not os.path.exists("./visualization_analysis"):
        os.makedirs("./visualization_analysis")
    plt.savefig(f'./visualization_analysis/histogram_{cnt}.png')
    plt.show()