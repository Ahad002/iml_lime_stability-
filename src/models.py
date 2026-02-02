"""
Model Wrappers
==============
Provides unified interface for Logistic Regression and DistilBERT models.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class SimpleLogisticModel:
    """Logistic Regression with TF-IDF for sentiment analysis"""

    def __init__(self, seed=42):
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.model = LogisticRegression(random_state=seed, max_iter=1000)
        self.is_trained = False

    def fit(self, texts, labels):
        print("🔄 Vectorizing text...")
        X = self.vectorizer.fit_transform(texts)
        print(f"📊 Feature matrix: {X.shape}")

        print("🔄 Training model...")
        self.model.fit(X, labels)
        self.is_trained = True

        train_acc = accuracy_score(labels, self.model.predict(X))
        print(f"✅ Training accuracy: {train_acc:.3f}")

    def predict_proba(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        X = self.vectorizer.transform(texts)
        return self.model.predict_proba(X)


class DistilBERTModel:
    """DistilBERT model for sentiment classification"""

    def __init__(self, model_name="distilbert/distilbert-base-uncased-finetuned-sst-2-english"):
        """
        Initialize pre-trained DistilBERT
        
        Parameters:
        -----------
        model_name : str
            Hugging Face model identifier
        """
        print(f"📥 Loading {model_name}...")
        # Load tokenizer (converts text to numbers)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Load pre-trained model
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        # Set to evaluation mode (no training)
        self.model.eval()
        # Check if GPU available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"✅ Model loaded on {self.device}")
        self.is_trained = True  # Already pre-trained

    def predict_proba(self, texts):
        """
        Predict probabilities for text(s)
        
        Parameters:
        -----------
        texts : str or list
            Single text or list of texts
            
        Returns:
        --------
        numpy array : Probabilities [P(negative), P(positive)] for each text
        """
        # Handle single string input
        if isinstance(texts, str):
            texts = [texts]

        # Tokenize texts
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",      # Return PyTorch tensors
            padding=True,              # Pad to same length
            truncation=True,           # Truncate if too long
            max_length=512             # Max sequence length
        )

        # Move to GPU if available
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get predictions (no gradient calculation)
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Convert logits to probabilities
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # Convert to numpy and return
        return probs.cpu().numpy()
