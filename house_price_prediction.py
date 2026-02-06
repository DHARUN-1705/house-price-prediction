# =========================================
# House Price Prediction using Machine Learning
# =========================================

def main():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score

    # -----------------------------------------
    # Load Dataset
    # -----------------------------------------
    housing = fetch_california_housing()
    X = housing.data
    y = housing.target

    df = pd.DataFrame(X, columns=housing.feature_names)
    df["Price"] = y

    print("Dataset Preview:")
    print(df.head())

    print("\nDataset Statistics:")
    print(df.describe())

    # -----------------------------------------
    # Data Visualization
    # -----------------------------------------
    sns.histplot(df["Price"], kde=True)
    plt.title("House Price Distribution")
    plt.xlabel("Price")
    plt.ylabel("Count")
    plt.show()

    # -----------------------------------------
    # Check Missing Values
    # -----------------------------------------
    print("\nMissing Values:")
    print(df.isnull().sum())

    # -----------------------------------------
    # Split Features and Target
    # -----------------------------------------
    X = df.drop("Price", axis=1)
    y = df["Price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # -----------------------------------------
    # Feature Scaling
    # -----------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # -----------------------------------------
    # Train Linear Regression Model
    # -----------------------------------------
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    # -----------------------------------------
    # Model Evaluation
    # -----------------------------------------
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\nModel Performance:")
    print("Mean Squared Error:", mse)
    print("R2 Score:", r2)

    # -----------------------------------------
    # Feature Importance (Coefficients)
    # -----------------------------------------
    coeff_df = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": model.coef_
    }).sort_values(by="Coefficient", ascending=False)

    print("\nFeature Importance:")
    print(coeff_df)

    # -----------------------------------------
    # Residual Plot
    # -----------------------------------------
    residuals = y_test - y_pred

    plt.scatter(y_pred, residuals)
    plt.axhline(y=0, color="red")
    plt.xlabel("Predicted Price")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.show()


if __name__ == "__main__":
    main()
