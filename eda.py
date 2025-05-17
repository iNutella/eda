import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import os

# 1. Load and clean dataset
# Read CSV, use first column as index, then standardize column names
df = pd.read_csv('Top-50-musicality-global.csv', index_col=0)
# Strip whitespace, lowercase, replace spaces with underscores
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
# Rename columns to match analysis conventions
df.rename(columns={
    'tsignature': 'time_signature',
    'positiveness': 'valence'
}, inplace=True)

# 2. Inspect data
print("Data Info:")
df.info()
print("\nColumns:\", df.columns.tolist())
print("\nHead of Data:")
print(df.head())

# 3. Prepare output directory
os.makedirs('plots', exist_ok=True)

# 4. Category distributions
plt.figure(figsize=(6,4))
sns.countplot(x='key', data=df)
plt.title('Distribution of Key')
plt.savefig('plots/key_countplot.png')
plt.clf()

plt.figure(figsize=(6,4))
sns.countplot(x='time_signature', data=df)
plt.title('Distribution of Time Signature')
plt.savefig('plots/time_signature_countplot.png')
plt.clf()

# 5. Continuous variable histograms
numeric_cols = [
    'danceability','energy','loudness','speechiness',
    'acousticness','instrumentalness','liveness','valence','tempo'
]
df[numeric_cols].hist(bins=15, figsize=(12,10))
plt.tight_layout()
plt.savefig('plots/numeric_histograms.png')
plt.clf()

# 6. Correlation matrix against Popularity
corr = df[numeric_cols + ['popularity']].corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix (including Popularity)')
plt.savefig('plots/correlation_heatmap.png')
plt.clf()

# 7. Scatter plots vs. popularity
for col in ['energy', 'danceability', 'valence', 'loudness']:
    plt.figure(figsize=(6,4))
    sns.scatterplot(x=col, y='popularity', data=df)
    plt.title(f'{col.capitalize()} vs. Popularity')
    plt.savefig(f'plots/{col}_vs_popularity.png')
    plt.clf()

# 8. Feature importances with Random Forest
# Prepare data
X = df[numeric_cols]
y = df['popularity']
# Handle any missing values
X = X.apply(pd.to_numeric, errors='coerce').fillna(X.mean())
# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Extract and plot importances
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(8,4))
importances.plot(kind='bar')
plt.title('Feature Importances for Popularity Prediction')
plt.tight_layout()
plt.savefig('plots/feature_importances.png')
plt.clf()

print("EDA complete. Plots saved in the 'plots' directory.")
