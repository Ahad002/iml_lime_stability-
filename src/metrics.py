"""
Stability Metrics
=================
Calculate stability metrics for LIME explanations.
"""

import numpy as np
from scipy.stats import spearmanr


class StabilityMetrics:
    """Calculate stability metrics for LIME explanations"""

    @staticmethod
    def top_k_agreement(explanations, k=3):
        """What % of top-k words overlap across runs?"""
        top_k_lists = []
        for exp in explanations:
            sorted_words = sorted(exp.items(),
                                 key=lambda x: abs(x[1]),
                                 reverse=True)
            top_k_lists.append(set([w for w, _ in sorted_words[:k]]))

        # Pairwise overlap
        agreements = []
        n = len(top_k_lists)
        for i in range(n):
            for j in range(i+1, n):
                overlap = len(top_k_lists[i] & top_k_lists[j]) / k
                agreements.append(overlap)

        return np.mean(agreements) if agreements else 0.0

    @staticmethod
    def rank_correlation(explanations):
        """Average Spearman correlation of word rankings"""
        # Get all unique words
        all_words = set()
        for exp in explanations:
            all_words.update(exp.keys())
        all_words = sorted(list(all_words))

        # Create rank vectors
        rank_vectors = []
        for exp in explanations:
            ranks = []
            sorted_words = sorted(exp.items(),
                                 key=lambda x: abs(x[1]),
                                 reverse=True)
            word_to_rank = {w: i for i, (w, _) in enumerate(sorted_words)}

            for word in all_words:
                ranks.append(word_to_rank.get(word, len(sorted_words)))
            rank_vectors.append(ranks)

        # Pairwise correlations
        correlations = []
        n = len(rank_vectors)
        for i in range(n):
            for j in range(i+1, n):
                corr, _ = spearmanr(rank_vectors[i], rank_vectors[j])
                if not np.isnan(corr):
                    correlations.append(corr)

        return np.mean(correlations) if correlations else 0.0

    @staticmethod
    def coefficient_of_variation(explanations):
        """How much do importance scores vary?"""
        all_words = set()
        for exp in explanations:
            all_words.update(exp.keys())

        cvs = []
        for word in all_words:
            scores = [exp.get(word, 0) for exp in explanations]
            mean_abs = np.mean(np.abs(scores))
            std = np.std(scores)

            if mean_abs > 0:
                cvs.append(std / mean_abs)

        return np.mean(cvs) if cvs else 0.0
