"""Simplified Data Preprocessing - For Intermediate Students"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
import math

def load_data(file_path):
    """Load CSV data and assign feature names"""
    print(f"Loading data: {file_path}")

    # [修改] header=None: 原始文件没有表头，必须指定，否则第一行数据会被当成列名
    data = pd.read_csv(file_path, header=None)

    # [新增] 根据作业文档(Task 1)手动添加 18 个特征名称
    column_names = [
        "Temperature (Min)", "Temperature (Max)", "Temperature (Mean)",
        "Relative Humidity (Min)", "Relative Humidity (Max)", "Relative Humidity (Mean)",
        "Sea Level Pressure (Min)", "Sea Level Pressure (Max)", "Sea Level Pressure (Mean)",
        "Precipitation Total", "Snowfall Amount", "Sunshine Duration",
        "Wind Gust (Min)", "Wind Gust (Max)", "Wind Gust (Mean)",
        "Wind Speed (Min)", "Wind Speed (Max)", "Wind Speed (Mean)"
    ]

    # 检查列数是否匹配
    if len(data.columns) == len(column_names):
        data.columns = column_names
        print("  ✓ Feature names assigned successfully.")
    else:
        print(f"  ⚠ Warning: Column count mismatch. Expected {len(column_names)}, got {len(data.columns)}")

    print(f"Data shape: {data.shape[0]} rows x {data.shape[1]} columns")
    return data


def handle_missing_values(data):
    """Handle missing values - Fill with median and plot heatmap"""
    print("\nChecking missing values...")

    # Count missing values
    missing_count = data.isnull().sum()
    print(f"Total missing values: {missing_count.sum()}")

    # Visualize missing data
    plt.figure(figsize=(10, 6))
    sns.heatmap(data.isnull(), cbar=False, yticklabels=False, cmap='viridis')
    plt.title('Missing Data Heatmap')
    plt.xlabel('Features')
    plt.tight_layout()
    plt.savefig('output/01_missing_data_heatmap.png')
    print("  Visualization saved: output/01_missing_data_heatmap.png")
    plt.close()

    # Fill missing values with median for each column
    for col in data.columns:
        if missing_count[col] > 0:
            median_value = data[col].median()
            data[col].fillna(median_value, inplace=True)
            print(f"  Column '{col}': filled {missing_count[col]} missing values")

    return data


def remove_outliers(data):
    """Remove outliers - Using IQR method and plot boxplots"""
    print("\nDetecting and removing outliers...")

    # Visualize outliers before removal
    n_cols = len(data.columns)
    n_rows = math.ceil(n_cols / 3)

    plt.figure(figsize=(15, 5 * n_rows))
    for idx, col in enumerate(data.columns):
        plt.subplot(n_rows, 3, idx + 1)
        sns.boxplot(y=data[col])
        plt.title(col)
        plt.grid(True, alpha=0.3)

    plt.suptitle('Outlier Detection (Boxplots)', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('output/02_outliers_boxplot.png')
    print("  Visualization saved: output/02_outliers_boxplot.png")
    plt.close()

    original_rows = len(data)
    cleaned_data = data.copy()

    # Apply IQR method for each column
    for col in data.columns:
        # Calculate quartiles
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1

        # Define bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Keep data within bounds
        cleaned_data = cleaned_data[(cleaned_data[col] >= lower_bound) &
                                   (cleaned_data[col] <= upper_bound)]

    removed_rows = original_rows - len(cleaned_data)
    print(f"Removed {removed_rows} rows of outliers")
    print(f"Remaining {len(cleaned_data)} rows of data")

    return cleaned_data


def normalize_data(data):
    """Data normalization - Using Z-score normalization"""
    print("\nNormalizing data...")

    normalized_data = data.copy()

    for col in data.columns:
        mean = data[col].mean()
        std = data[col].std()

        # Check if standard deviation is zero to avoid division by zero
        if std == 0:
            print(f"  Warning: Column '{col}' has zero standard deviation, skipping normalization")
            normalized_data[col] = 0  # Set to 0 for constant columns
        else:
            normalized_data[col] = (data[col] - mean) / std

    print("Normalization completed")
    return normalized_data


def show_correlation(data, save_path=None):
    """Show correlation matrix"""
    print("\nCalculating feature correlations...")

    # Calculate correlation coefficients
    corr_matrix = data.corr()

    # Plot heatmap
    plt.figure(figsize=(14, 12)) # [修改] 稍微调大画布，以容纳长标签
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True,
                xticklabels=True, yticklabels=True) # [修改] 确保显示标签
    plt.title('Feature Correlation Matrix')
    plt.xticks(rotation=45, ha='right') # [修改] 旋转X轴标签防止重叠
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Correlation plot saved: {save_path}")
    else:
        plt.show()

    plt.close()

    # Find highly correlated feature pairs
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.8:
                high_corr_pairs.append((corr_matrix.columns[i],
                                       corr_matrix.columns[j],
                                       corr_matrix.iloc[i, j]))

    if high_corr_pairs:
        print(f"\nFound {len(high_corr_pairs)} highly correlated feature pairs (|r| > 0.8):")
        for feat1, feat2, corr_val in high_corr_pairs:
            print(f"  {feat1} <-> {feat2}: {corr_val:.3f}")

    return corr_matrix


def remove_correlated_features(data, corr_matrix, threshold=0.9):
    """Remove highly correlated features"""
    print(f"\nRemoving features with correlation > {threshold}...")

    # Find columns to remove
    to_remove = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                col_name = corr_matrix.columns[j]
                to_remove.add(col_name)
                print(f"  Removing: {col_name} (correlation with {corr_matrix.columns[i]} is {corr_matrix.iloc[i,j]:.3f})")

    # Remove columns
    cleaned_data = data.drop(columns=list(to_remove))
    print(f"Retained {len(cleaned_data.columns)} features")

    return cleaned_data


def perform_pca(data, save_path=None):
    """Perform PCA and plot explained variance"""
    print("\nPerforming PCA analysis...")

    # Initialize PCA
    pca = PCA()
    pca.fit(data)

    # Get explained variance ratio
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    # Plotting
    plt.figure(figsize=(10, 6))

    # Bar plot: Individual explained variance
    plt.bar(range(1, len(explained_variance) + 1), explained_variance,
            alpha=0.5, align='center', label='Individual explained variance')

    # Line plot: Cumulative explained variance
    plt.step(range(1, len(cumulative_variance) + 1), cumulative_variance,
             where='mid', label='Cumulative explained variance', color='red')

    # Mark 95% threshold
    plt.axhline(y=0.95, color='green', linestyle='--', label='95% Threshold')

    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.title('PCA Variance Explained (Elbow Plot)')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path)
        print(f"PCA plot saved: {save_path}")
    else:
        plt.show()

    plt.close()

    # Print statistics
    n_95 = np.argmax(cumulative_variance >= 0.95) + 1
    print(f"  Number of components for 95% variance: {n_95}")
    print(f"  Variance explained by first 2 components: {cumulative_variance[1]:.2%}")

    return data


def preprocess_pipeline(file_path, output_path=None):
    """Complete preprocessing pipeline"""
    print("=" * 50)
    print("Starting Data Preprocessing Pipeline")
    print("=" * 50)

    # 1. Load data
    data = load_data(file_path)

    # 2. Handle missing values
    data = handle_missing_values(data)

    # 3. Remove outliers
    data = remove_outliers(data)

    # 4. Analyze correlations
    corr_matrix = show_correlation(data, 'output/correlation_matrix.png')

    # 5. Remove highly correlated features
    data = remove_correlated_features(data, corr_matrix, threshold=0.9)

    # 6. Normalize
    data = normalize_data(data)

    # 7. Perform PCA Analysis
    perform_pca(data, 'output/pca_variance.png')

    # Save preprocessed data
    if output_path:
        # [修改] 保存时包含表头，以便后续聚类分析能读取到列名
        data.to_csv(output_path, index=False)
        print(f"\nPreprocessed data saved: {output_path}")

    print("\n" + "=" * 50)
    print("Preprocessing Complete!")
    print("=" * 50)

    return data


# If running this file directly
if __name__ == "__main__":
    # Example usage
    input_file = "ClimateDataBasel.csv"  # Replace with actual file path
    output_file = "output/preprocessed_data.csv"

    processed_data = preprocess_pipeline(input_file, output_file)
    print(f"\nFinal data shape: {processed_data.shape}")