import numpy as np
import random
from collections import defaultdict

NUM_SIMULATIONS = 10000

def cumulative_ranges(a):
    """
    Convert an array `a` of length 8 into array `b` of length 8 where
    b[0] = a[0], and b[i] - b[i-1] = a[i] for i=1..7 (i.e. b is the cumulative sum).

    Parameters
    ----------
    a : array-like of length 8
        Input values (not required to sum to 1).

    Returns
    -------
    numpy.ndarray
        Cumulative array b of length 8.
    """
    import numpy as _np

    a = _np.asarray(a, dtype=float)
    if a.size != 8:
        raise ValueError("Input array 'a' must have length 8")
    return _np.cumsum(a)

class PortfolioSimulator:
    """
    Simulates multi-firm credit migration over time using yearly transition matrices.
    Each company transitions independently using the corresponding year's transition matrix.
    """

    def __init__(self, transition_matrices: dict[int, np.ndarray], rating_labels: list[str]):
        """
        Parameters
        ----------
        transition_matrices : dict[int, np.ndarray]
            A dictionary mapping years to their transition matrices.
        rating_labels : list[str]
            The list of rating categories (e.g., ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "D"]).
            The last one is usually "D" for default.
        """
        # print(transition_matrices)
        self.transition_matrices = transition_matrices
        #for all transition matrices, append a new row with probabilities of staying in default (1.0)
        for year in self.transition_matrices:
            default_row = np.zeros((1, len(rating_labels)))
            default_row[0, -1] = 1.0  # 100% chance of staying in default
            self.transition_matrices[year] = np.vstack([self.transition_matrices[year], default_row])

        self.rating_labels = rating_labels
        self.rating_to_index = {r: i for i, r in enumerate(rating_labels)}

    def simulate_portfolio(self, start_year: int, duration: int, portfolio: list[str], seed: int | None = None):
        """
        Simulates rating evolution for a portfolio of firms over multiple years.

        Parameters
        ----------
        start_year : int
            The starting year (must be present in transition_matrices).
        duration : int
            Number of years to simulate.
        portfolio : list[str]
            List of initial credit ratings for each firm.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        results : dict
            {
                "trajectories": dict[int, list[str]],  # firm index -> list of yearly ratings
                "defaults": list[int],                 # firm indices that defaulted
                "yearly_default_counts": dict[int, int],
                "yearly_rating_counts": dict[int, dict[str, int]]
            }
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        firms = len(portfolio)

        # in the resulting dictionary, the count of res[firm_idx][final label] should be incremented by 1 if the final rating of firm_idx is final label
        
        # results should be a list of firms size
        results = [{} for _ in range(firms)]

        for i in range(NUM_SIMULATIONS):
            current_ratings = portfolio.copy()
            for year_offset in range(duration):
                current_year = start_year + year_offset
                for firm_idx in range(firms):
                    curr_rating = current_ratings[firm_idx]
                    if curr_rating == "Default":
                        continue  # Already defaulted

                    #generate a random number between 0 and 1
                    rand_val = random.random()
                    cum_range = cumulative_ranges(self.transition_matrices[current_year][self.rating_to_index[curr_rating]])
                    if rand_val <= cum_range[0]:
                        new_rating = self.rating_labels[0]
                    elif rand_val <= cum_range[1]:
                        new_rating = self.rating_labels[1]
                    elif rand_val <= cum_range[2]:
                        new_rating = self.rating_labels[2]
                    elif rand_val <= cum_range[3]:  
                        new_rating = self.rating_labels[3]
                    elif rand_val <= cum_range[4]:
                        new_rating = self.rating_labels[4]
                    elif rand_val <= cum_range[5]:
                        new_rating = self.rating_labels[5]
                    elif rand_val <= cum_range[6]:
                        new_rating = self.rating_labels[6]
                    else:
                        new_rating = "Default"

                    current_ratings[firm_idx] = new_rating
            
            # print(f"Final ratings after {duration} years: {current_ratings}")
            
            # in the resulting dictionary, the count of res[firm_idx][final label] should be incremented by 1 if the final rating of firm_idx is final label
            for firm_idx in range(firms):
                final_rating = current_ratings[firm_idx]
                if final_rating in results[firm_idx]:
                    results[firm_idx][final_rating] += 1
                else:
                    results[firm_idx][final_rating] = 1

        # need to print a theoretical result matrix as well which would be the expected distribution after duration years
        theoretical_results = {}
        for firm_idx in range(firms):
            initial_rating = portfolio[firm_idx]
            initial_vector = np.zeros(len(self.rating_labels))
            initial_vector[self.rating_to_index[initial_rating]] = 1.0

            # Multiply the initial vector by the transition matrices for each year
            result_vector = initial_vector
            for year_offset in range(duration):
                current_year = start_year + year_offset
                result_vector = np.dot(result_vector, self.transition_matrices[current_year])

            theoretical_results[firm_idx] = {self.rating_labels[i]: result_vector[i] for i in range(len(self.rating_labels))}

        print("Theoretical Results (Expected Distribution after simulations):")
        for firm_idx in range(firms):
            print(f"Firm {firm_idx} starting with rating {portfolio[firm_idx]}:")
            for rating, prob in theoretical_results[firm_idx].items():
                print(f"  {rating}: {prob:.4f}")
            print()
        return results
            


            

                    
        