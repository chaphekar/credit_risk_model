# model_transition_regression.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

"""
DATA PREPARATION
----------------
Assume you have:
X: shape (N_years, 12)              # macroeconomic features
Y: shape (N_years, 7, 8)            # transition matrices for each year (probabilities)

You MUST provide these when calling train_model().
"""


# ------- MODEL DEFINITION -------- #

class TransitionMatrixRegressor(nn.Module):
    """
    Linear regression:
        logits = X @ W + b
    Then reshape logits → (7 rows, 8 columns)
    Then apply softmax row-wise.
    """
    def __init__(self, in_features=12, out_rows=7, out_cols=8):
        super().__init__()
        self.out_rows = out_rows
        self.out_cols = out_cols
        self.linear = nn.Linear(in_features, out_rows * out_cols, bias=True)

    def forward(self, x):
        """
        x: (batch_size, 12)
        returns: (batch_size, 7, 8) logits (NOT softmaxed)
        """
        logits = self.linear(x)  # (batch_size, 56)
        logits = logits.view(-1, self.out_rows, self.out_cols)
        return logits


# ------- TRAINING FUNCTION -------- #

def train_model(X, Y, num_epochs=2000, lr=0.01):
    """
    X: numpy array (N, 12)
    Y: numpy array (N, 7, 8)
    """

    # Convert to tensor
    X_t = torch.tensor(X, dtype=torch.float32)
    Y_t = torch.tensor(Y, dtype=torch.float32)

    model = TransitionMatrixRegressor()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Use distributional loss: KL divergence (expects log-probs as input and
    # target probabilities). Monitor Brier (MSE) and top-1 accuracy too.
    loss_fn = nn.KLDivLoss(reduction="batchmean")
    mse_fn = nn.MSELoss(reduction="mean")

    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()

        preds = model(X_t)                       # logits shape: (N,7,8)
        log_probs = torch.log_softmax(preds, dim=2)

        loss = loss_fn(log_probs, Y_t)           # KL divergence
        loss.backward()
        optimizer.step()

        # Monitoring metrics (vectorized)
        if epoch % 10 == 0 or epoch == 1 or epoch == num_epochs:
            model.eval()
            with torch.no_grad():
                probs = torch.softmax(preds, dim=2)
                brier = mse_fn(probs, Y_t)          # mean squared error across elements
                pred_classes = probs.argmax(dim=2)
                true_classes = Y_t.argmax(dim=2)
                acc = (pred_classes == true_classes).float().mean()

            print(f"Epoch {epoch:4d} | KL loss: {loss.item():.6f} | Brier(MSE): {brier.item():.6f} | top1 acc: {acc.item():.4f}")

    return model


# ------- PREDICTION WRAPPER -------- #

def predict_transition_matrix(model, x_single):
    """
    x_single: shape (12,) numpy array
    returns: (7, 8) probability matrix
    """
    x_t = torch.tensor(x_single, dtype=torch.float32).unsqueeze(0)
    logits = model(x_t).detach()
    probs = torch.softmax(logits, dim=2).numpy()[0]
    return probs

if __name__ == "__main__":
    # Example usage
    feature_map = {
        "Unemp.Rt": ["lag0_Unemp.Rt"],
        "BBB.Rt": ["lag0_BBB.Rt"],
        "Mort.Rt": ["lag0_Mort.Rt"],
        "Prime.Rt": ["lag0_Prime.Rt"],
        "DJIA": ["lag0_DJIA"],
        "VIX": ["lag0_VIX"],
        "RGDP": ["lag0_RGDP.Ygr"],
        "NGDP": ["lag0_NGDP.Ygr"],
        "NDI": ["lag0_NDI.Ygr"],
        "RDI": ["lag0_RDI.Ygr"],
        "CPI": ["lag0_CPI.Ygr"],
        "BBB.Spd": ["lag0_BBB.Spd"]
    }
    
    # read the data from macro_data.csv
    df = pd.read_csv("macro_data.csv")
    selected_features = []
    for key, options in feature_map.items():
        for option in options:
            if option in df.columns:
                selected_features.append(option)
                break

    X = df[selected_features].values

    # read the Y data from ../csv/transition.csv
    df_Y = pd.read_csv("../csv/transition.csv")
    print(df_Y.columns)

    # df_Y has columns ['Year', 'Rating', 'Cnt', 'AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'Default'
    # we need to make a matrix of shape (N_years, 7, 8) from this
    # each year has 7 rows (AAA to CCC) and 8 columns (AAA to Default)

    years = df_Y['Year'].unique()
    N_years = len(years)
    Y = np.zeros((N_years, 7, 8))
    rating_to_row = {'AAA': 0, 'AA': 1, 'A': 2, 'BBB': 3, 'BB': 4, 'B': 5, 'CCC': 6}
    for i, year in enumerate(years):
        df_year = df_Y[df_Y['Year'] == year]
        for _, row in df_year.iterrows():
            rating = row['Rating']
            if rating in rating_to_row:
                r = rating_to_row[rating]
                Y[i, r, 0] = row['AAA']
                Y[i, r, 1] = row['AA']
                Y[i, r, 2] = row['A']
                Y[i, r, 3] = row['BBB']
                Y[i, r, 4] = row['BB']
                Y[i, r, 5] = row['B']
                Y[i, r, 6] = row['CCC']
                Y[i, r, 7] = row['Default']

    print("X shape:", X.shape)
    print("Y shape:", Y.shape)

    print(Y)

    model = train_model(X, Y, lr=0.001)
    print("Training complete.")
    
    # Example prediction
    # read the 12 features from macro_pred.csv for years 2016 to 2019
    df_pred = pd.read_csv("macro_pred.csv")
    X_pred = df_pred[selected_features].values
    for i in range(X_pred.shape[0]):
        x_single = X_pred[i]
        pred_matrix = predict_transition_matrix(model, x_single)
        print(f"Predicted transition matrix for sample {i}:")
        print(pred_matrix)
        print()