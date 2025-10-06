from sklearn.datasets import load_diabetes
from scipy.stats import boxcox, skew
import pandas as pd
import matplotlib.pyplot as plt


def find_most_skewed_positive_feature(df):
    """
    Find the most skewed positive feature in a DataFrame.

    Args:
        df (pandas.DataFrame): DataFrame with numeric features.

    Returns:
        str: Name of the most skewed positive feature.

    Raises:
        ValueError: If no positive features are available.
    """
    positive_features = [col for col in df.columns if (df[col] > 0).all()]
    if not positive_features:
        raise ValueError("No positive features available for Box-Cox transformation.")

    skew_values = df[positive_features].apply(lambda x: skew(x, nan_policy="omit"))
    most_skewed = skew_values.abs().idxmax()
    return most_skewed


def shift_to_positive(df):
    """
    Shift all columns to make values strictly positive.

    Args:
        df (pandas.DataFrame): Original DataFrame.

    Returns:
        pandas.DataFrame: Shifted DataFrame with positive values.
    """
    return df - df.min() + 1e-6


def apply_boxcox(df, feature):
    """
    Apply Box-Cox transformation to a feature.

    Args:
        df (pandas.DataFrame): DataFrame containing the feature.
        feature (str): Name of the feature.

    Returns:
        tuple: (Transformed values as numpy.ndarray, optimal lambda as float)
    """
    transformed_values, lambda_opt = boxcox(df[feature])
    return transformed_values, lambda_opt


def plot_before_after(original, transformed, feature):
    """
    Plot histograms before and after Box-Cox transformation.

    Args:
        original (array-like): Original feature values.
        transformed (array-like): Box-Cox transformed values.
        feature (str): Feature name.

    Returns:
        None
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(original, bins=50, color="skyblue", edgecolor="black")
    axes[0].set_title(f"Before Box-Cox ({feature})")

    axes[1].hist(transformed, bins=50, color="lightgreen", edgecolor="black")
    axes[1].set_title(f"After Box-Cox ({feature})")

    fig.suptitle("Effect of Box-Cox Transformation", fontsize=14)
    plt.tight_layout()
    plt.show()


def main():
    """
    Perform Box-Cox transformation on the most skewed positive feature.

    Returns:
        None
    """
    data = load_diabetes()
    df = pd.DataFrame(data.data, columns=data.feature_names)

    df_shifted = shift_to_positive(df)
    feature = find_most_skewed_positive_feature(df_shifted)
    print(f"Most skewed positive feature: {feature}")

    transformed_values, lambda_opt = apply_boxcox(df_shifted, feature)
    print(f"Optimal lambda for Box-Cox: {lambda_opt:.4f}")

    original_skew = skew(df_shifted[feature])
    transformed_skew = skew(transformed_values)
    print(f"Skewness before Box-Cox: {original_skew:.4f}")
    print(f"Skewness after Box-Cox: {transformed_skew:.4f}")

    plot_before_after(df_shifted[feature], transformed_values, feature)
    # print(plt.__file__)


if __name__ == "__main__":
    main()
