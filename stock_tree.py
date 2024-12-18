import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def train_and_evaluate(features, max_depth=None, min_samples_leaf=1, cv=5):
    """Trains and evaluates a decision tree model with specified features and parameters using cross-validation."""
    # 1. Define the stock we're interested in
    ticker_symbol = "AAPL"

    # 2. Create a Ticker object
    ticker = yf.Ticker(ticker_symbol)

    # 3. Fetch historical data (let's get 5 years worth of data)
    historical_data = ticker.history(period="5y")

    # 4. Select relevant columns and rename
    clean_data = historical_data[['Close']].copy()
    clean_data.rename(columns={'Close': 'Price'}, inplace=True)

    # 5. Feature Engineering
    clean_data['MA_7'] = clean_data['Price'].rolling(window=7).mean()
    clean_data['MA_21'] = clean_data['Price'].rolling(window=21).mean()
    clean_data['Price_Diff'] = clean_data['Price'].diff()
    clean_data['Volatility'] = clean_data['Price'].rolling(window=21).std()

    # 6. Drop NaN values
    clean_data.dropna(inplace=True)

    # 7. Create Target variable
    clean_data['Price_Change'] = clean_data['Price'].diff().shift(-1)  # Price difference with next day
    clean_data['Target'] = (clean_data['Price_Change'] > 0).astype(int) # 1 = up, 0 = down or same
    clean_data.dropna(inplace=True)  # Drop last row that has NaNs as well

    # 8. Define Target
    target = 'Target'

    # 9. Split data into features and target
    X = clean_data[features]
    y = clean_data[target]

    # 10. Create Model
    model = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=42)

    # 11. Perform cross-validation
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    print(f"\nFeatures: {features}, max_depth: {max_depth}, min_samples_leaf: {min_samples_leaf}")
    print(f"Cross-Validation Accuracy Scores: {cv_scores}")
    print(f"Mean Cross-Validation Accuracy: {np.mean(cv_scores)}")

    # 12. Split data into training and testing for evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 13. Train Model
    model.fit(X_train, y_train)

    # 14. Make Predictions
    y_pred = model.predict(X_test)

    # 15. Evaluate Model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # # 16. Confusion Matrix
    # cm = confusion_matrix(y_test, y_pred)
    # sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    # plt.title('Confusion Matrix')
    # plt.xlabel('Predicted Label')
    # plt.ylabel('True Label')
    # plt.show()


if __name__ == "__main__":
    # Define feature combinations
    feature_options = [
        ['MA_7', 'MA_21'],
        ['MA_7', 'MA_21', 'Price_Diff'],
        ['MA_7', 'MA_21', 'Volatility'],
        ['MA_7', 'MA_21', 'Price_Diff', 'Volatility']
    ]

    # Define hyperparameter options
    max_depth_options = [None, 5, 10]
    min_samples_leaf_options = [1, 5, 10]

    # Iterate over all feature combinations and parameters
    for features in feature_options:
      for max_depth in max_depth_options:
        for min_samples_leaf in min_samples_leaf_options:
            train_and_evaluate(features, max_depth, min_samples_leaf)
