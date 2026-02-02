"""
LIME Explainer Wrapper
======================
Provides stability analysis functionality for LIME explanations.
"""

from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt


class LIMEStabilityAnalyzer:
    """Analyze LIME explanation stability"""

    def __init__(self, model):
        self.model = model
        self.explainer = LimeTextExplainer(class_names=['negative', 'positive'])

    def explain_once(self, text, num_samples=1000):
        """Get single LIME explanation"""
        exp = self.explainer.explain_instance(
            text,
            self.model.predict_proba,
            num_features=10,
            num_samples=num_samples
        )
        return dict(exp.as_list())

    def explain_multiple(self, text, num_samples=1000, num_runs=30):
        """Run LIME multiple times"""
        explanations = []
        for _ in range(num_runs):
            exp = self.explain_once(text, num_samples)
            explanations.append(exp)
        return explanations

    def get_top_k(self, explanation, k=3):
        """Get top-k important words"""
        sorted_words = sorted(explanation.items(),
                             key=lambda x: abs(x[1]),
                             reverse=True)
        return [word for word, _ in sorted_words[:k]]

    def visualize_explanation(self, explanation, title="LIME Explanation"):
        """Plot word importances"""
        sorted_items = sorted(explanation.items(),
                             key=lambda x: abs(x[1]),
                             reverse=True)[:10]
        words, scores = zip(*sorted_items)

        colors = ['red' if s < 0 else 'green' for s in scores]

        plt.figure(figsize=(10, 6))
        plt.barh(words, scores, color=colors, alpha=0.7)
        plt.xlabel('Importance Score')
        plt.title(title)
        plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
        plt.tight_layout()
        plt.show()
