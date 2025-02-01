import json
import logging
import numpy as np
from typing import List, Dict, Union
from dataclasses import dataclass
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from scipy.spatial import distance
import torch
import nltk


def ensure_nltk_data():
    """Ensure NLTK data is available"""
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        logging.warning("VADER lexicon not found, downloading...")
        nltk.download('vader_lexicon', quiet=True)


# Check NLTK data availability at import time
ensure_nltk_data()


@dataclass
class RecommendationMetrics:
    top_1_hit_rate: float
    top_3_hit_rate: float
    top_5_hit_rate: float
    average_hit_rate: float
    total_scenarios: int
    top_1_hits: int
    top_3_hits: int
    top_5_hits: int


@dataclass
class SimulationMetrics:
    preference_estimation: float
    review_generation: float
    overall_quality: float


class BaseEvaluator:
    """Base class for evaluation tools"""

    def __init__(self):
        self.metrics_history: List[Union[RecommendationMetrics, SimulationMetrics]] = []

    def save_metrics(self, metrics: Union[RecommendationMetrics, SimulationMetrics]):
        """Save metrics to history"""
        self.metrics_history.append(metrics)

    def get_metrics_history(self):
        """Get all historical metrics"""
        return self.metrics_history


class RecommendationEvaluator(BaseEvaluator):
    """Evaluator for recommendation tasks"""

    def __init__(self):
        super().__init__()
        self.n_values = [1, 3, 5]  # 预定义的n值数组

    def calculate_hr_at_n(
            self,
            ground_truth: List[str],
            predictions: List[List[str]]
    ) -> RecommendationMetrics:
        """Calculate Hit Rate at different N values"""
        total = len(ground_truth)
        hits = {n: 0 for n in self.n_values}

        for gt, pred in zip(ground_truth, predictions):
            for n in self.n_values:
                if gt in pred[:n]:
                    hits[n] += 1

        top_1_hit_rate = hits[1] / total if total > 0 else 0
        top_3_hit_rate = hits[3] / total if total > 0 else 0
        top_5_hit_rate = hits[5] / total if total > 0 else 0
        average_hit_rate = (top_1_hit_rate + top_3_hit_rate + top_5_hit_rate) / 3
        metrics = RecommendationMetrics(
            top_1_hit_rate=top_1_hit_rate,
            top_3_hit_rate=top_3_hit_rate,
            top_5_hit_rate=top_5_hit_rate,
            average_hit_rate=average_hit_rate,
            total_scenarios=total,
            top_1_hits=hits[1],
            top_3_hits=hits[3],
            top_5_hits=hits[5]
        )

        self.save_metrics(metrics)
        return metrics


class SimulationEvaluator(BaseEvaluator):
    """Evaluator for simulation tasks"""

    def __init__(self, device: str = "auto"):
        super().__init__()
        self.device = self._get_device(device)

        pipeline_device = self.device
        st_device = "cuda" if self.device == 0 else "cpu"

        self.sia = SentimentIntensityAnalyzer()
        self.emotion_classifier = pipeline(
            "text-classification",
            model="cardiffnlp/twitter-roberta-base-emotion",
            top_k=5,
            device=pipeline_device
        )
        self.topic_model = SentenceTransformer(
            'paraphrase-MiniLM-L6-v2',
            device=st_device
        )

    def _get_device(self, device: str) -> int:
        """Parse device from string"""
        if device == "gpu":
            if torch.cuda.is_available():
                return 0  # GPU
            else:
                logging.warning("GPU is not available, falling back to CPU")
                return -1  # CPU
        elif device == "cpu":
            return -1  # CPU
        elif device == "auto":
            return 0 if torch.cuda.is_available() else -1
        else:
            raise ValueError("Device type must be 'cpu', 'gpu' or 'auto'")

    def calculate_metrics(
            self,
            simulated_data: List[Dict],
            real_data: List[Dict]
    ) -> SimulationMetrics:
        """Calculate all simulation metrics"""
        # Calculate star error
        simulated_stars = [item['stars'] for item in simulated_data]
        real_stars = [item['stars'] for item in real_data]
        star_error = 0
        for sim_star, real_star in zip(simulated_stars, real_stars):
            if sim_star > 5:
                sim_star = 5
            elif sim_star < 0:
                sim_star = 0
            star_error += abs(sim_star - real_star) / 5
        star_error = star_error / len(real_stars)
        preference_estimation = 1 - star_error

        # Calculate review metrics
        simulated_reviews = [item['review'] for item in simulated_data]
        real_reviews = [item['review'] for item in real_data]
        review_details = self._calculate_review_metrics(
            simulated_reviews,
            real_reviews
        )

        sentiment_error = review_details['sentiment_error']
        emotion_error = review_details['emotion_error']
        topic_error = review_details['topic_error']
        review_generation = 1 - (sentiment_error * 0.25 + emotion_error * 0.25 + topic_error * 0.5)
        overall_quality = (preference_estimation + review_generation) / 2

        metrics = SimulationMetrics(
            preference_estimation=preference_estimation,
            review_generation=review_generation,
            overall_quality=overall_quality
        )

        self.save_metrics(metrics)
        return metrics

    def _calculate_review_metrics(
            self,
            simulated_reviews: List[str],
            real_reviews: List[str]
    ) -> Dict[str, float]:
        """Calculate detailed review metrics between two texts"""
        # sentiment analysis
        sentiment_error = []
        emotion_error = []
        topic_error = []
        for simulated_review, real_review in zip(simulated_reviews, real_reviews):
            # sentiment analysis
            sentiment1 = self.sia.polarity_scores(simulated_review)['compound']
            sentiment2 = self.sia.polarity_scores(real_review)['compound']
            sentiment_error_single = abs(sentiment1 - sentiment2) / 2
            sentiment_error.append(sentiment_error_single)

            # Topic analysis
            embeddings = self.topic_model.encode([simulated_review, real_review])
            topic_error_single = distance.cosine(embeddings[0], embeddings[1]) / 2
            topic_error.append(topic_error_single)

        # Emotion analysis
        for i in range(len(simulated_reviews)):
            if len(simulated_reviews[i]) > 300:
                simulated_reviews[i] = simulated_reviews[i][:300]
            if len(real_reviews[i]) > 300:
                real_reviews[i] = real_reviews[i][:300]
        simulated_emotions = self.emotion_classifier(simulated_reviews)
        real_emotions = self.emotion_classifier(real_reviews)
        for sim_emotion, real_emotion in zip(simulated_emotions, real_emotions):
            emotion_error_single = self._calculate_emotion_error(sim_emotion, real_emotion)
            emotion_error.append(emotion_error_single)

        sentiment_error = np.mean(sentiment_error)
        emotion_error = np.mean(emotion_error)
        topic_error = np.mean(topic_error)
        return {
            'sentiment_error': sentiment_error,
            'emotion_error': emotion_error,
            'topic_error': topic_error,
        }

    def _calculate_emotion_error(
            self,
            emotions1: List[Dict],
            emotions2: List[Dict]
    ) -> float:
        """Calculate similarity between two emotion distributions"""
        # Convert emotions to vectors
        emotion_dict1 = {e['label']: e['score'] for e in emotions1}
        emotion_dict2 = {e['label']: e['score'] for e in emotions2}

        # Get all unique emotions
        all_emotions = set(emotion_dict1.keys()) | set(emotion_dict2.keys())

        # Create vectors
        vec1 = np.array([emotion_dict1.get(e, 0) for e in all_emotions])
        vec2 = np.array([emotion_dict2.get(e, 0) for e in all_emotions])

        # Calculate emotion error
        return float(np.mean(np.abs(vec1 - vec2)))