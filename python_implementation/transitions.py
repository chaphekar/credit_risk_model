import pandas as pd
import numpy as np


# ================================
# ---- Utility / Core Functions ---
# ================================

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the credit rating transition dataset.
    """
    return pd.read_csv(file_path)


def get_states(df: pd.DataFrame) -> list:
    """
    Extract the list of states (rating classes) from the dataframe columns.
    """
    cols = list(df.columns)
    start_idx = cols.index("Cnt") + 1
    stop_idx = [i for i, c in enumerate(cols) if c.endswith("_cnt")]
    stop_idx = stop_idx[0] if stop_idx else len(cols)
    return cols[start_idx:stop_idx]


def build_transition_matrices(df: pd.DataFrame, states: list, default_state: str = None) -> dict:
    """
    Build transition probability and count matrices for each year.
    Makes default an absorbing state (P[default, default] = 1).
    """
    matrices = {}
    count_cols = [f"{s}_cnt" for s in states]

    for year, group in df.groupby("Year"):
        prob_matrix = pd.DataFrame(group[states].values, index=group["Rating"], columns=states)

        # Make default absorbing
        if default_state and default_state in prob_matrix.index:
            prob_matrix.loc[default_state] = 0
            prob_matrix.loc[default_state, default_state] = 1.0

        if all(col in group.columns for col in count_cols):
            count_matrix = pd.DataFrame(group[count_cols].values, index=group["Rating"], columns=states)
        else:
            count_matrix = None

        matrices[year] = {"prob": prob_matrix, "count": count_matrix}

    return matrices


def build_weighted_average_matrix(df: pd.DataFrame, states: list, default_state: str = None) -> pd.DataFrame:
    """
    Compute weighted average transition matrix across all years.
    Ensures default is absorbing.
    """
    P = np.zeros((len(states), len(states)))
    weight_sum = np.zeros(len(states))

    for _, row in df.iterrows():
        i = states.index(row["Rating"])
        P[i] += np.array([row[s] * row["Cnt"] for s in states])
        weight_sum[i] += row["Cnt"]

    # Normalize rows and handle zero-weights
    for i in range(len(states)):
        if weight_sum[i] > 0:
            P[i] /= weight_sum[i]
        else:
            P[i] = np.zeros(len(states))

    # Normalize rows again
    row_sums = P.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    P /= row_sums

    df_P = pd.DataFrame(P, index=states, columns=states)

    # Make default absorbing
    if default_state and default_state in df_P.index:
        df_P.loc[default_state] = 0
        df_P.loc[default_state, default_state] = 1.0

    return df_P


def compute_n_step_transition(P: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Compute n-step transition matrix as P^n.
    """
    Pn = np.linalg.matrix_power(P.values, n)
    return pd.DataFrame(Pn, index=P.index, columns=P.columns)


def compute_probability_of_default(P: pd.DataFrame, n: int = 1, default_state: str = None) -> pd.Series:
    """
    Compute n-year probability of default (PD) and display limiting distribution matrix [(I - Q)^-1 * R].
    """
    if default_state is None:
        default_state = P.columns[-1]

    if default_state not in P.columns:
        raise ValueError(f"Default state '{default_state}' not found in transition matrix.")

    # Separate Q and R
    states = list(P.index)
    non_default_states = [s for s in states if s != default_state]

    Q = P.loc[non_default_states, non_default_states].values
    R = P.loc[non_default_states, [default_state]].values

    # Fundamental matrix
    I = np.eye(Q.shape[0])
    N = np.linalg.inv(I - Q)

    # Limiting distribution: B = N * R
    B = N @ R
    B_df = pd.DataFrame(B, index=non_default_states, columns=[f"Prob(default | start={s})" for s in [default_state]])

    print("\n--- Limiting Distribution Matrix [(I - Q)^-1 * R] ---")
    print(B_df.round(6))

    # Compute n-step PD
    Pn = compute_n_step_transition(P, n)
    pd_series = Pn[default_state]
    pd_series.name = f"{n}-year PD"
    return pd_series


def compute_expected_time_to_default(P: pd.DataFrame, default_state: str = None) -> pd.Series:
    """
    Compute expected number of years before default using the fundamental matrix.
    ETTD = sum of each row of (I - Q)^-1
    """
    if default_state is None:
        default_state = P.columns[-1]

    states = list(P.index)
    if default_state not in states:
        raise ValueError(f"Default state '{default_state}' not found in matrix.")

    non_default_states = [s for s in states if s != default_state]
    Q = P.loc[non_default_states, non_default_states].values

    I = np.eye(Q.shape[0])
    N = np.linalg.inv(I - Q)  # Fundamental matrix

    # Expected number of years before default = sum of each row of N
    t = N.sum(axis=1)

    print("\n--- Fundamental Matrix (I - Q)^-1 ---")
    print(pd.DataFrame(N, index=non_default_states, columns=non_default_states).round(6))

    return pd.Series(t, index=non_default_states, name="Expected Time to Default")


# ================================
# ---- Display / CLI Section -----
# ================================

def print_menu():
    print("\n===== Credit Rating Markov Model =====")
    print("1. View transition PROBABILITY matrix for a given year")
    print("2. View transition COUNT matrix for a given year")
    print("3. View weighted average transition matrix")
    print("4. Compute n-year transition matrix")
    print("5. Compute probability of default (PD)")
    print("6. Compute expected time to default (ETTD)")
    print("7. Exit")
    print("=======================================")


def main():
    file_path = input("Enter the CSV file path: ").strip()
    df = load_data(file_path)
    states = get_states(df)
    default_state = input("Enter the default state name (e.g., 'D' or 'Default'): ").strip()

    matrices = build_transition_matrices(df, states, default_state)
    avg_matrix = build_weighted_average_matrix(df, states, default_state)

    while True:
        print_menu()
        choice = input("Choose an option: ").strip()

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
            n = int(input("Enter number of years (n): "))
            Pn = compute_n_step_transition(avg_matrix, n)
            print(f"\n{n}-Year Transition Matrix:")
            print(Pn.round(4))

        elif choice == "5":
            n = int(input("Enter number of years for PD (default=1): ") or "1")
            pd_series = compute_probability_of_default(avg_matrix, n, default_state)
            print(f"\n{n}-Year Probability of Default (PD):")
            print(pd_series.round(6))

        elif choice == "6":
            ettd = compute_expected_time_to_default(avg_matrix, default_state)
            print("\nExpected Time to Default (years):")
            print(ettd.round(3))

        elif choice == "7":
            print("Exiting... Goodbye!")
            break

        else:
            print("Invalid choice. Try again.")


if __name__ == "__main__":
    main()
