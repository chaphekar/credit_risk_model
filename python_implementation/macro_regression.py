"""
macro_regression.py
-------------------
Performs regression of Z-scores on macroeconomic factors.

Steps:
1. Loads macroeconomic data (with lag0/lag1 columns)
2. Selects best representation for each of 11 macro factors
3. Chooses top statistically significant predictors
4. Fits OLS regression model
5. Saves beta coefficients for future scenario analysis
"""

import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import numpy as np
import os


def choose_best_features(df):
    """
    Chooses the best column for each macro factor among its lag0/lag1 and Ydf/Ygr variants.
    """
    feature_map = {
        "Unemp.Rt": ["lag0_Unemp.Rt", "lag0_Unemp.Rt_Ydf", "lag1_Unemp.Rt", "lag1_Unemp.Rt_Ydf"],
        "BBB.Rt": ["lag0_BBB.Rt", "lag0_BBB.Rt_Ydf", "lag1_BBB.Rt", "lag1_BBB.Rt_Ydf"],
        "Mort.Rt": ["lag0_Mort.Rt", "lag0_Mort.Rt_Ydf", "lag1_Mort.Rt", "lag1_Mort.Rt_Ydf"],
        "Prime.Rt": ["lag0_Prime.Rt", "lag1_Prime.Rt", "lag0_Prime.Rt_Ydf", "lag1_Prime.Rt_Ydf"],
        "DJIA": ["lag0_DJIA", "lag0_DJIA_Ygr", "lag1_DJIA", "lag1_DJIA_Ygr"],
        "VIX": ["lag0_VIX", "lag0_VIX_Ygr", "lag1_VIX", "lag1_VIX_Ygr"],
        "RGDP": ["lag0_RGDP.Ygr", "lag1_RGDP.Ygr"],
        "NGDP": ["lag0_NGDP.Ygr", "lag1_NGDP.Ygr"],
        "NDI": ["lag0_NDI.Ygr", "lag1_NDI.Ygr"],
        "RDI": ["lag0_RDI.Ygr", "lag1_RDI.Ygr"],
        "CPI": ["lag0_CPI.Ygr", "lag1_CPI.Ygr"],
        "BBB.Spd": ["lag0_BBB.Spd", "lag0_BBB.Spd_Ydf", "lag1_BBB.Spd", "lag1_BBB_Spd_Ydf"]
    }

    selected = {}
    for key, options in feature_map.items():
        available = [c for c in options if c in df.columns]
        if len(available) == 0:
            continue
        # pick lag1_Ydf if available, else lag0_Ydf, else lag1, else lag0
        priority = [c for c in available if "lag1" in c and ("Ydf" in c or "Ygr" in c)]
        if not priority:
            priority = [c for c in available if "lag0" in c and ("Ydf" in c or "Ygr" in c)]
        if not priority:
            priority = [c for c in available if "lag1" in c]
        if not priority:
            priority = [c for c in available if "lag0" in c]
        selected[key] = priority[0]

    return list(selected.values())


def run_macro_regression(csv_path, dep_var="Z", top_k=10, output_dir="outputs"):
    """
    Runs regression of Z-factor on selected macroeconomic variables.
    Saves and returns the beta coefficients.
    """

    # Load data
    df = pd.read_csv(csv_path)
    if dep_var not in df.columns:
        raise ValueError(f"Dependent variable '{dep_var}' not found in CSV!")

    # Step 1: Choose the best columns for each macro factor
    selected_features = choose_best_features(df)
    print("✅ Selected macro features:\n", selected_features)

    # Step 2: Filter for available columns
    X = df[selected_features].copy()
    y = df[dep_var]

    # Drop rows with NaNs
    data = pd.concat([X, y], axis=1).dropna()
    X, y = data[selected_features], data[dep_var]

    # Step 3: Standardize predictors
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=selected_features,
        index=X.index
    )
    X_scaled = sm.add_constant(X_scaled)

    model = sm.OLS(y, X_scaled).fit()
    
    # Step 5: Get coefficients (including intercept)
    betas = model.params.rename_axis("Variable").reset_index(name="Beta")

    # Step 6: Rank by t-stats (absolute value)
    tstats = model.tvalues.drop("const", errors="ignore")  # skip intercept for ranking
    top_vars = tstats.abs().sort_values(ascending=False).index[:top_k].tolist()

    # Always include the intercept in output
    keep_vars = ["const"] + top_vars
    betas = betas[betas["Variable"].isin(keep_vars)]

    # Step 7: Save results
    os.makedirs(output_dir, exist_ok=True)
    beta_path = os.path.join(output_dir, "macro_betas.csv")
    betas.to_csv(beta_path, index=False)

    print("\n✅ Regression completed successfully.")
    print(f"📁 Betas saved to: {beta_path}")
    print("\nTop Variables and Coefficients:\n", betas)
    print("\nModel Summary:\n", model.summary())

    print(f"📁 Betas saved to: {beta_path}")
    print("\nTop Variables and Coefficients:\n", betas)
    print("\nModel Summary:\n", model.summary())

    return betas, model, scaler


if __name__ == "__main__":
    # === Example Usage ===
    # Provide your CSV file path and run this script.
    # Ensure your data includes a dependent variable column "Z".

    # --- Data Loading and Merging ---
    z_score = pd.read_csv('../csv/z_score.csv')
    macro_history = pd.read_csv('../csv/macro_history.csv')

    z_score.rename(columns={'Period': 'Year'}, inplace=True)

    merged_df = pd.merge(
        left=macro_history,
        right=z_score[['Year', 'Zscore']],
        on='Year',
        how='left'
    )

    output_file_path = 'macro_data.csv'
    merged_df.to_csv(output_file_path, index=False)
    
    # --- Regression Execution ---
    csv_path = 'macro_data.csv'

    betas, model, scaler = run_macro_regression(
        csv_path=csv_path,
        dep_var="Zscore",     # The dependent variable column is 'Zscore'
        top_k=8,              # number of top macro factors to keep
        output_dir="outputs"
    )