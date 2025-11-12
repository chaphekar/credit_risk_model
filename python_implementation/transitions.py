import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from portfolio_simulator import PortfolioSimulator

# =========================================
# ---- Utility / Core Functions ------------
# =========================================

def load_data(file_path: str) -> pd.DataFrame:
    """Load the credit rating transition dataset."""
    return pd.read_csv(file_path)


def get_states(df: pd.DataFrame) -> list:
    """Extract list of rating states."""
    cols = list(df.columns)
    start_idx = cols.index("Cnt") + 1
    stop_idx = [i for i, c in enumerate(cols) if c.endswith("_cnt")]
    stop_idx = stop_idx[0] if stop_idx else len(cols)
    return cols[start_idx:stop_idx]


def build_transition_matrices(df: pd.DataFrame, states: list, default_state: str = None) -> dict:
    """Build transition matrices by year."""
    matrices = {}
    count_cols = [f"{s}_cnt" for s in states]

    for year, group in df.groupby("Year"):
        prob_matrix = pd.DataFrame(group[states].values, index=group["Rating"], columns=states)
        if default_state and default_state in prob_matrix.index:
            prob_matrix.loc[default_state] = 0
            prob_matrix.loc[default_state, default_state] = 1.0

        count_matrix = (
            pd.DataFrame(group[count_cols].values, index=group["Rating"], columns=states)
            if all(col in group.columns for col in count_cols)
            else None
        )

        matrices[year] = {"prob": prob_matrix, "count": count_matrix}

    return matrices


def build_weighted_average_matrix(df: pd.DataFrame, states: list, default_state: str = None) -> pd.DataFrame:
    """Compute weighted average transition matrix."""
    P = np.zeros((len(states), len(states)))
    weight_sum = np.zeros(len(states))

    for _, row in df.iterrows():
        i = states.index(row["Rating"])
        P[i] += np.array([row[s] * row["Cnt"] for s in states])
        weight_sum[i] += row["Cnt"]

    for i in range(len(states)):
        if weight_sum[i] > 0:
            P[i] /= weight_sum[i]

    row_sums = P.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    P /= row_sums

    df_P = pd.DataFrame(P, index=states, columns=states)

    if default_state and default_state in df_P.index:
        df_P.loc[default_state] = 0
        df_P.loc[default_state, default_state] = 1.0

    return df_P


def compute_n_step_transition(P: pd.DataFrame, n: int) -> pd.DataFrame:
    """Compute n-step transition matrix."""
    Pn = np.linalg.matrix_power(P.values, n)
    return pd.DataFrame(Pn, index=P.index, columns=P.columns)


def compute_probability_of_default(P: pd.DataFrame, n: int = 1, default_state: str = None) -> pd.Series:
    """Compute n-year PD."""
    if default_state is None:
        default_state = P.columns[-1]
    states = list(P.index)
    non_default_states = [s for s in states if s != default_state]
    Q = P.loc[non_default_states, non_default_states].values
    R = P.loc[non_default_states, [default_state]].values
    I = np.eye(Q.shape[0])
    N = np.linalg.inv(I - Q)
    B = N @ R
    B_df = pd.DataFrame(B, index=non_default_states, columns=["Prob(default)"])
    print("\n--- Limiting Distribution Matrix ---")
    print(B_df.round(6))
    Pn = compute_n_step_transition(P, n)
    pd_series = Pn[default_state]
    pd_series.name = f"{n}-year PD"
    return pd_series


def compute_expected_time_to_default(P: pd.DataFrame, default_state: str = None) -> pd.Series:
    """Expected time to default."""
    if default_state is None:
        default_state = P.columns[-1]
    states = list(P.index)
    non_default_states = [s for s in states if s != default_state]
    Q = P.loc[non_default_states, non_default_states].values
    I = np.eye(Q.shape[0])
    N = np.linalg.inv(I - Q)
    t = N.sum(axis=1)
    print("\n--- Fundamental Matrix ---")
    print(pd.DataFrame(N, index=non_default_states, columns=non_default_states).round(6))
    return pd.Series(t, index=non_default_states, name="Expected Time to Default")


# =========================================
# ---- NEW: Macro + Z/DF Modeling ----------
# =========================================

def fit_z_model(macro_df: pd.DataFrame, z_col: str = "Z") -> LinearRegression:
    """Fit Z = α + β1*GDP + β2*Unemp + ..."""
    X = macro_df.drop(columns=[z_col])
    y = macro_df[z_col]
    model = LinearRegression().fit(X, y)
    print("\n--- Z-Model Coefficients ---")
    for var, coef in zip(X.columns, model.coef_):
        print(f"{var}: {coef:.4f}")
    print(f"Intercept: {model.intercept_:.4f}")
    return model


def forecast_z(model: LinearRegression, macro_scenarios: dict) -> pd.DataFrame:
    """Forecast Z under Base/Adverse/Severe macro scenarios."""
    results = []
    for scenario, df in macro_scenarios.items():
        Z_hat = model.predict(df.drop(columns=["Year"]))
        tmp = df.copy()
        tmp["Z_forecast"] = Z_hat
        tmp["Scenario"] = scenario
        results.append(tmp)
    return pd.concat(results, ignore_index=True)


def compute_df_from_z(Z: np.ndarray, a: float = 0.5, b: float = 2.5) -> np.ndarray:
    """Map Z to Default Frequency (DF) using inverse relationship (lower Z → higher DF)."""
    return 1 / (1 + np.exp(-(a - b * Z)))


def apply_df_forecast(z_forecast_df: pd.DataFrame, a: float = -1.0, b: float = 1.5) -> pd.DataFrame:
    """Compute DF for each scenario/year from forecasted Z."""
    z_forecast_df["DF_forecast"] = compute_df_from_z(z_forecast_df["Z_forecast"].values, a, b)
    return z_forecast_df


def plot_z_df_forecasts(df: pd.DataFrame, year_col="Year"):
    """Plot Z and DF forecasts in separate subplots for clarity."""
    scenarios = df["Scenario"].unique()

    plt.figure(figsize=(8, 6))

    # --- Plot 1: Z forecasts ---
    plt.subplot(2, 1, 1)
    for sc in scenarios:
        sub = df[df["Scenario"] == sc]
        plt.plot(sub[year_col], sub["Z_forecast"], label=f"{sc} (Z)")
    plt.title("Z Forecasts under Macro Scenarios")
    plt.xlabel("Year")
    plt.ylabel("Z-score")
    plt.legend()

    # --- Plot 2: DF forecasts ---
    plt.subplot(2, 1, 2)
    for sc in scenarios:
        sub = df[df["Scenario"] == sc]
        plt.plot(sub[year_col], sub["DF_forecast"] * 100, label=f"{sc} (DF%)")
    plt.title("Default Frequency Forecasts under Macro Scenarios")
    plt.xlabel("Year")
    plt.ylabel("Default Frequency (%)")
    plt.legend()

    plt.tight_layout()
    plt.show()



def run_macro_scenario_demo():
    """Example of macro scenario Z/DF forecasting pipeline."""
    hist = pd.DataFrame({
        "Year": [2015, 2016, 2017, 2018],
        "GDP_Growth": [3.0, 2.5, 2.8, 3.2],
        "Unemp_Rate": [5.0, 5.2, 5.1, 4.9],
        "Z": [0.85, 0.82, 0.83, 0.86]
    })

    model = fit_z_model(hist.drop(columns=["Year"]))  # ✅ FIX: now this has GDP_Growth, Unemp_Rate, Z

    base = pd.DataFrame({
        "Year": [2019, 2020, 2021],
        "GDP_Growth": [3.0, 3.1, 3.2],
        "Unemp_Rate": [5.0, 5.1, 5.0],
    })
    adverse = pd.DataFrame({
        "Year": [2019, 2020, 2021],
        "GDP_Growth": [2.0, 1.0, 0.5],
        "Unemp_Rate": [6.0, 6.5, 7.0],
    })
    severe = pd.DataFrame({
        "Year": [2019, 2020, 2021],
        "GDP_Growth": [0.0, -1.5, -2.0],
        "Unemp_Rate": [8.0, 9.0, 10.0],
    })

    scenarios = {"Base": base, "Adverse": adverse, "Severe": severe}

    z_forecast_df = forecast_z(model, scenarios)
    z_forecast_df = apply_df_forecast(z_forecast_df)
    plot_z_df_forecasts(z_forecast_df)
    print("\nForecasted Z and DF values:")
    print(z_forecast_df[["Year", "Scenario", "Z_forecast", "DF_forecast"]].round(4))


# =========================================
# ---- CLI Menu ---------------------------
# =========================================

def print_menu():
    print("\n===== Credit Rating Markov + Macro Model =====")
    print("1. View transition PROBABILITY matrix for a given year")
    print("2. View transition COUNT matrix for a given year")
    print("3. View weighted average transition matrix")
    print("4. Compute n-year transition matrix")
    print("5. Compute probability of default (PD)")
    print("6. Compute expected time to default (ETTD)")
    print("7. Run macro scenario Z/DF demo")
    print("8. Simulate portfolio credit rating transitions")
    print("9. Exit")


def main():
    print("\n=== Enhanced Credit Risk Model ===")
    # file_path = input("Enter transition CSV path: ").strip()
    file_path = "../csv/loan__loan.csv"
    df = load_data(file_path)
    states = get_states(df)
    # print("Available rating states:", states)
    # default_state = input("Enter default state name: ").strip()
    default_state = "Default"

    matrices = build_transition_matrices(df, states, default_state)
    avg_matrix = build_weighted_average_matrix(df, states, default_state)

    while True:
        print_menu()
        choice = input("Choice: ").strip()

        if choice == "1":
            year = int(input("Enter year: "))
            if year in matrices:
                print(f"\nTransition PROBABILITY matrix for {year}:")
                print(matrices[year]["prob"].round(4))
            else:
                print("Year not found.")
        elif choice == "2":
            year = int(input("Enter year: "))
            if year in matrices and matrices[year]["count"] is not None:
                print(f"\nTransition COUNT matrix for {year}:")
                print(matrices[year]["count"].round(2))
            else:
                print("Year not found or count data missing.")
        elif choice == "3":
            print("\nWeighted Average Transition Matrix:")
            print(avg_matrix.round(4))
        elif choice == "4":
            n = int(input("Enter n (years): "))
            print(compute_n_step_transition(avg_matrix, n).round(4))
        elif choice == "5":
            n = int(input("Enter number of years for PD: "))
            print(compute_probability_of_default(avg_matrix, n, default_state))
        elif choice == "6":
            print(compute_expected_time_to_default(avg_matrix, default_state))
        elif choice == "7":
            run_macro_scenario_demo()
        elif choice == "8":
            n = int(input("Enter the number of firms: "))
            start_year = int(input("Enter the start year: "))
            duration = int(input("Enter the duration: "))
            portfolio = []
            for i in range(n):
                rating = input(f"Enter initial rating for firm {i+1}: ").strip()
                portfolio.append(rating)
            simulator = PortfolioSimulator({year: matrices[year]["prob"].values for year in matrices}, states)
            results = simulator.simulate_portfolio(start_year, duration, portfolio)
            print("Simulation results:")
            print(results)
            # print("Trajectories:", results["trajectories"])
            # print("Defaults:", results["defaults"])
            # print("Yearly default counts:", results["yearly_default_counts"])
            # print("Yearly rating counts:", results["yearly_rating_counts"])
        elif choice == "9":
            break
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()
