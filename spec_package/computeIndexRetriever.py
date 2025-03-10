import numpy as np
from collections import defaultdict, deque

class AdaptiveStabilityRetriever:
    def __init__(self, window_size=3, penalty_factor=0.1, decay_rate=0.9, stability_threshold=0.9):
        """
        Parameters:
            window_size (int): Number of steps to consider before retrieval.
            penalty_factor (float): Base penalty applied when chunks drop out of common indexes.
            decay_rate (float): Exponential decay factor for ranking importance.
            stability_threshold (float): Minimum proportion of common indexes that must be stable to trigger retrieval.
        """
        self.window_size = window_size
        self.penalty_factor = penalty_factor
        self.decay_rate = decay_rate
        self.stability_threshold = stability_threshold  # Stability requirement

        self.chunk_ranking = defaultdict(float)  # Rank scores
        self.chunk_penalty = defaultdict(int)  # Penalty count
        self.retrieval_points = defaultdict(list)  # Store retrieval points per query
        self.common_index_history = defaultdict(lambda: deque(maxlen=window_size))  # Store recent common indexes per query

    def update_rankings(self, common_indexes):
        """Update rankings using exponential decay for frequently appearing indexes."""
        for idx in common_indexes:
            self.chunk_ranking[idx] = self.chunk_ranking[idx] * self.decay_rate + 1  # Reward frequent appearances
        for idx in list(self.chunk_ranking.keys()):
            if idx not in common_indexes:
                self.chunk_penalty[idx] += 1  # Increase penalty for missing
                self.chunk_ranking[idx] -= self.penalty_factor * np.log(1 + self.chunk_penalty[idx])  # Apply penalty

    def should_retrieve(self, prefix_index_map):
        """
        Determines when to retrieve based on **stable** common indexes and returns those indexes.

        Parameters:
            prefix_index_map (dict): Dictionary where keys are query IDs, and values contain:
                - "prefixes": OrderedDict with prefix vectors as keys and corresponding top indexes as values.

        Returns:
            tuple: (Dict of stable indexes per query, Dictionary of chunk rankings)
        """
        self.retrieval_points.clear()  # Reset retrieval points per query
        stable_index_map = defaultdict(list)  # Store stable indexes instead of retrieval points

        for query_id, prefix_data in prefix_index_map.items():
            prev_indices = None
            prev_stable_common_indexes = set()  # Track last stable common indexes

            # ✅ Reset ranking for each new query
            self.chunk_ranking.clear()  

            for idx, (_, indices) in enumerate(reversed(prefix_data["prefixes"].items())):
                current_indices = np.array(indices).flatten().tolist()  # Flatten into list

                if prev_indices is not None:
                    # Compute common indexes between previous and current prefix
                    common_indexes = set(current_indices[:5]) & set(prev_indices[:5])
                    self.common_index_history[query_id].append(common_indexes)

                    # ✅ Update ranking scores per query (reset for each new query)
                    self.update_rankings(common_indexes)

                    # Check if common indexes remained stable over the window
                    if len(self.common_index_history[query_id]) == self.window_size:
                        stable_common_indexes = set.intersection(*self.common_index_history[query_id])

                        # **Store Stable Indexes Instead of Retrieval Points**
                        if stable_common_indexes and stable_common_indexes != prev_stable_common_indexes:
                            stability_ratio = len(stable_common_indexes) / len(current_indices[:5])  # Stability score

                            if stability_ratio >= self.stability_threshold:  # Only store if stable enough
                                stable_index_map[query_id].append(list(stable_common_indexes))  # Store indexes
                                prev_stable_common_indexes = stable_common_indexes  # Track for next iteration

                prev_indices = current_indices
                last_prefix_index = current_indices[:5]

            stable_index_map[query_id].append(last_prefix_index)

            # ✅ Reset chunk penalty for each new query
            self.chunk_penalty = defaultdict(int)

        return stable_index_map, dict(self.chunk_ranking)  # ✅ Ranking is per query now

