"""
macro_regression_forecast.py
-----------------------------
Computes normalized Z-scores and Default Frequency (DF) forecasts for 2016–2019
using pre-estimated beta coefficients and macroeconomic data.

Steps:
1. Loads beta coefficients CSV (Variable,Beta)
2. Loads macro data CSV (Year + macro variable columns)
3. Computes raw Z = const + sum(beta_i * X_i)
4. Normalizes Z-scores (mean 0, std 1)
5. Maps normalized Z → Default Frequency (DF) using a logistic mapping
   (higher Z → lower DF)
6. Plots Z and DF by year (2016–2019)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# ------------------ Mapping Z → DF ------------------ #
def compute_df_from_z(z_values, a=-1.0, b=1.5):
    """
    Map Z-scores to default frequency using a logistic function:
    Higher Z → lower default frequency.
    DF = 1 / (1 + exp(-(a - b*z)))
    """
    df = 1 / (1 + np.exp(-(a - b * z_values)))
    return df


# ------------------ Plotting ------------------ #
def plot_z_df_forecasts(df):
    """
    Plot normalized Z-forecast and DF-forecast for each year.
    """
    plt.figure(figsize=(10, 6))

    # Plot normalized Z-forecast
    plt.subplot(2, 1, 1)
    plt.plot(df["Year"], df["Z_forecast_norm"], marker="o", color="blue")
    plt.title("Forecasted Normalized Z-Score (2016–2019)")
    plt.ylabel("Normalized Z")
    plt.grid(True)

    # Plot DF-forecast
    plt.subplot(2, 1, 2)
    plt.plot(df["Year"], 100 * df["DF_forecast"], marker="o", color="red")
    plt.title("Forecasted Default Frequency (2016–2019)")
    plt.ylabel("DF (%)")
    plt.xlabel("Year")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# ------------------ Main computation ------------------ #
def run_macro_forecast():
    """
    Loads betas and macro data, computes Z and DF forecasts for 2016–2019.
    """
    beta_path = "outputs/macro_betas.csv"
    macro_path = "../csv/macro_a.csv"

    # Load files
    beta_df = pd.read_csv(beta_path)
    macro_df = pd.read_csv(macro_path)

    # Filter only 2016–2019
    macro_df = macro_df[(macro_df["Year"] >= 2016) & (macro_df["Year"] <= 2019)].copy()

    # Extract intercept
    intercept_row = beta_df[beta_df["Variable"].str.lower().isin(["const", "intercept"])]
    intercept = float(intercept_row["Beta"].values[0]) if not intercept_row.empty else 0.0

    # Extract variable betas
    betas = (
        beta_df[~beta_df["Variable"].str.lower().isin(["const", "intercept"])]
        .set_index("Variable")["Beta"]
        .to_dict()
    )

    print("\n✅ Loaded coefficients:")
    print(f"Intercept (α): {intercept:.6f}")
    for var, coef in betas.items():
        print(f"{var}: {coef:.6f}")

    # Compute raw Z for each year
    z_forecasts = []
    for _, row in macro_df.iterrows():
        z_val = intercept
        for var, coef in betas.items():
            if var in row:
                z_val += coef * row[var]
        z_forecasts.append(z_val)

    macro_df["Z_forecast_raw"] = z_forecasts

    # Normalize Z-scores (mean 0, std 1)
    scaler = StandardScaler()
    macro_df["Z_forecast_norm"] = scaler.fit_transform(macro_df[["Z_forecast_raw"]])

    # Compute DF from normalized Z (higher Z → lower DF)
    macro_df["DF_forecast"] = compute_df_from_z(macro_df["Z_forecast_norm"], a=-1.0, b=1.5)

    print("\n📊 Forecast results (2016–2019):")
    print(macro_df[["Year", "Z_forecast_raw", "Z_forecast_norm", "DF_forecast"]].round(4))

    plot_z_df_forecasts(macro_df)


# ------------------ Entry point ------------------ #
if __name__ == "__main__":
    print("===== Macro Z/DF Forecast (2016–2019) =====")
    run_macro_forecast()
