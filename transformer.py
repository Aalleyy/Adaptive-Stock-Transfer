import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from stock_transformer import StockTransformer


class StockDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len - 1

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + self.seq_len, 0]  # Target is the next day price
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def train_and_evaluate(features, seq_len=20, n_epochs=50, batch_size=32, learning_rate=0.001, hidden_dim=64, num_heads=4):
    """Trains and evaluates a Transformer model for stock prediction."""
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

    # 8. Define Features and Target
    target = 'Target'
    all_features = ['Price', 'MA_7', 'MA_21', 'Price_Diff', 'Volatility']
    X = clean_data[all_features].values
    y = clean_data[target].values

    # 9. Data Normalization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 10. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

    # 11. Prepare Dataset and Dataloader
    train_dataset = StockDataset(X_train, seq_len)
    test_dataset = StockDataset(X_test, seq_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 12. Define the Model
    model = StockTransformer(input_size=len(all_features), hidden_dim=hidden_dim, num_heads=num_heads)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # 13. Train the model
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(x_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch: {epoch+1}, Loss: {total_loss/len(train_loader)}")

    # 14. Evaluate the model
    model.eval()
    test_loss = 0
    y_true_list = []
    y_pred_list = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            outputs = model(x_batch).squeeze()
            loss = criterion(outputs, y_batch)
            test_loss += loss.item()
            y_true_list.extend(y_batch.tolist())
            y_pred_list.extend(outputs.tolist())

    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)
    y_pred_binary = (y_pred > 0.5).astype(int)
    accuracy = accuracy_score(y_true, y_pred_binary)
    precision = precision_score(y_true, y_pred_binary)
    recall = recall_score(y_true, y_pred_binary)
    f1 = f1_score(y_true, y_pred_binary)
    print(f"\nTest Loss: {test_loss/len(test_loader)}")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("\nClassification Report:\n", classification_report(y_true, y_pred_binary))

    cm = confusion_matrix(y_true, y_pred_binary)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()



if __name__ == "__main__":
    # Define hyperparameter options
    seq_len_options = [10, 20, 30]
    n_epochs_options = [20, 50, 100]
    batch_size_options = [16, 32, 64]
    learning_rate_options = [0.001, 0.0001]
    hidden_dim_options = [32, 64, 128]
    num_heads_options = [2, 4, 8]

    # Iterate over all parameter combinations
    for seq_len in seq_len_options:
      for n_epochs in n_epochs_options:
        for batch_size in batch_size_options:
          for learning_rate in learning_rate_options:
            for hidden_dim in hidden_dim_options:
              for num_heads in num_heads_options:
                train_and_evaluate(
                    features = [], #No features passed here, will take them all.
                    seq_len=seq_len,
                    n_epochs=n_epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    hidden_dim=hidden_dim,
                    num_heads=num_heads
                )
