import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional, Tuple
import logging
from models.cricket_data_collector import CricketDataCollector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PlayerStats:
    def __init__(self):
        self.data_collector = CricketDataCollector()
        self.player_cache = {}
        self.scaler = StandardScaler()

        # Initialize ML models with optimized parameters
        self.batting_model = GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            random_state=42
        )

        self.bowling_model = GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            random_state=42
        )

        # Train models with available data
        self._train_models()

    def _train_models(self):
        """Train ML models with available data"""
        try:
            logger.info("Training ML models...")

            # Generate training data
            batting_X, batting_y = self._prepare_batting_training_data()
            bowling_X, bowling_y = self._prepare_bowling_training_data()

            # Train batting model
            if batting_X is not None and batting_y is not None:
                self.batting_model.fit(batting_X, batting_y)
                logger.info("Batting model trained successfully")

            # Train bowling model
            if bowling_X is not None and bowling_y is not None:
                self.bowling_model.fit(bowling_X, bowling_y)
                logger.info("Bowling model trained successfully")

        except Exception as e:
            logger.error(f"Error training models: {str(e)}")
            self._train_with_sample_data()

    def _prepare_batting_training_data(self):
        """Prepare training data using real player stats"""
        try:
            # Collect real player data for training
            sample_players = [
                "Virat Kohli", "Babar Azam", "Kane Williamson",
                "Jos Buttler", "David Warner", "Rohit Sharma"
            ]

            X_data = []
            y_data = []

            for player in sample_players:
                stats = self.data_collector.get_player_stats(player)
                if stats:
                    # Extract features
                    features = [
                        stats['batting_stats']['average'],
                        stats['batting_stats']['strike_rate'],
                        stats['matches'],
                        stats['batting_stats'].get('impact_score', 0),
                        stats['batting_stats'].get('consistency_rating', 0)
                    ]
                    X_data.append(features)

                    # Use actual performance as target
                    recent_performances = self.data_collector.get_recent_performances(player)
                    if recent_performances:
                        avg_runs = sum(p.get('runs', 0) for p in recent_performances) / len(recent_performances)
                        y_data.append(avg_runs)
                    else:
                        y_data.append(stats['batting_stats']['average'])

            if X_data and y_data:
                return np.array(X_data), np.array(y_data)

            # Fallback to sample data if no real data available
            return self._generate_sample_training_data()

        except Exception as e:
            logger.error(f"Error preparing batting training data: {str(e)}")
            return self._generate_sample_training_data()

    def _prepare_bowling_training_data(self):
        """Prepare bowling training data using real player stats"""
        try:
            # Collect real player data for training
            sample_bowlers = [
                "Jasprit Bumrah", "Rashid Khan", "Mitchell Starc",
                "Trent Boult", "Shaheen Afridi", "Adam Zampa"
            ]

            X_data = []
            y_data = []

            for player in sample_bowlers:
                stats = self.data_collector.get_player_stats(player)
                if stats:
                    # Extract features
                    features = [
                        stats['bowling_stats']['average'],
                        stats['bowling_stats']['economy'],
                        stats['matches'],
                        stats['bowling_stats'].get('effectiveness_score', 0),
                        stats['bowling_stats'].get('pressure_rating', 0)
                    ]
                    X_data.append(features)

                    # Use actual performance as target
                    recent_performances = self.data_collector.get_recent_performances(player)
                    if recent_performances:
                        avg_wickets = sum(p.get('wickets', 0) for p in recent_performances) / len(recent_performances)
                        y_data.append(avg_wickets)
                    else:
                        y_data.append(stats['bowling_stats'].get('wickets', 0) / stats['matches'])

            if X_data and y_data:
                return np.array(X_data), np.array(y_data)

            # Fallback to sample data if no real data available
            return self._generate_sample_training_data()

        except Exception as e:
            logger.error(f"Error preparing bowling training data: {str(e)}")
            return self._generate_sample_training_data()

    def _generate_sample_training_data(self):
        """Generate sample training data as fallback"""
        n_samples = 1000
        X = np.random.rand(n_samples, 5)  # 5 features
        y = 30 + 40 * X[:, 0] + 10 * X[:, 1] + np.random.normal(0, 5, n_samples)
        return X, y

    def _train_with_sample_data(self):
        """Train models with basic sample data as fallback"""
        logger.info("Training models with sample data...")
        n_samples = 100
        X = np.random.rand(n_samples, 10)
        y_batting = np.random.randint(0, 100, n_samples)
        y_bowling = np.random.randint(0, 5, n_samples)

        self.batting_model.fit(X, y_batting)
        self.bowling_model.fit(X, y_bowling)

    def get_player_stats(self, player_name: str) -> Optional[Dict]:
        """Get comprehensive statistics for a specific player with enhanced metrics"""
        try:
            # Check cache first
            if player_name in self.player_cache:
                return self.player_cache[player_name]

            # Fetch fresh data
            stats = self.data_collector.get_player_stats(player_name)
            if stats:
                # Add advanced metrics
                stats = self._enhance_player_stats(stats)
                # Cache the results
                self.player_cache[player_name] = stats
                return stats

            logger.error(f"No stats found for player {player_name}")
            return None

        except Exception as e:
            logger.error(f"Error getting stats for player {player_name}: {str(e)}")
            return None

    def _enhance_player_stats(self, stats: Dict) -> Dict:
        """Add advanced metrics to player statistics"""
        try:
            # Enhanced batting metrics
            if 'batting_stats' in stats:
                batting = stats['batting_stats']
                batting['impact_score'] = self._calculate_impact_score(
                    batting['average'],
                    batting['strike_rate']
                )
                batting['consistency_rating'] = self._calculate_consistency(
                    batting['average'],
                    batting['fifties'],
                    batting['hundreds']
                )

            # Enhanced bowling metrics
            if 'bowling_stats' in stats:
                bowling = stats['bowling_stats']
                bowling['effectiveness_score'] = self._calculate_bowling_effectiveness(
                    bowling['average'],
                    bowling['economy'],
                    bowling['wickets']
                )
                bowling['pressure_rating'] = self._calculate_pressure_rating(
                    bowling['economy'],
                    bowling['wickets'],
                    stats['matches']
                )

            return stats
        except Exception as e:
            logger.error(f"Error enhancing player stats: {str(e)}")
            return stats

    def predict_performance(self, player_name: str, opposition: str, venue: str, match_type: str = 'T20') -> Dict:
        """Enhanced performance prediction with multiple factors"""
        player_stats = self.get_player_stats(player_name)
        if not player_stats:
            return {}

        try:
            # Get recent form and historical data
            recent_performances = self.data_collector.get_recent_performances(player_name)

            # Prepare advanced feature set
            features = self._prepare_prediction_features(
                player_stats,
                recent_performances,
                opposition,
                venue,
                match_type
            )

            # Scale features
            scaled_features = self.scaler.fit_transform(features)

            # Generate predictions with confidence intervals
            batting_pred, batting_ci = self._predict_with_confidence(
                self.batting_model,
                scaled_features,
                'batting'
            )

            bowling_pred, bowling_ci = (0, (0, 0))
            if player_stats.get('role') in ['Bowler', 'All-rounder']:
                bowling_pred, bowling_ci = self._predict_with_confidence(
                    self.bowling_model,
                    scaled_features,
                    'bowling'
                )

            # Calculate match influence probability
            influence_prob = self._calculate_match_influence(
                player_stats,
                recent_performances,
                opposition
            )

            return {
                'batting_prediction': {
                    'runs': max(0, round(batting_pred)),
                    'confidence_interval': (max(0, round(batting_ci[0])), round(batting_ci[1])),
                    'strike_rate_prediction': self._predict_strike_rate(player_stats, opposition)
                },
                'bowling_prediction': {
                    'wickets': max(0, round(bowling_pred)),
                    'confidence_interval': (max(0, round(bowling_ci[0])), round(bowling_ci[1])),
                    'economy_prediction': self._predict_economy(player_stats, opposition)
                },
                'match_influence_probability': influence_prob,
                'form_factor': self._calculate_form_factor(recent_performances),
                'opposition_specific_rating': self._calculate_opposition_rating(player_stats, opposition),
                'venue_performance_index': self._calculate_venue_index(player_stats, venue),
                'confidence_score': self._calculate_enhanced_confidence(
                    recent_performances,
                    player_stats,
                    opposition,
                    venue
                )
            }

        except Exception as e:
            logger.error(f"Error predicting performance for {player_name}: {str(e)}")
            return {}

    def _prepare_prediction_features(self, stats: Dict, recent_performances: List, opposition: str, venue: str, match_type: str) -> np.ndarray:
        """Prepare enhanced feature set for prediction"""
        features = [
            stats['batting_stats']['average'],
            stats['batting_stats']['strike_rate'],
            stats['matches'],
            len(recent_performances),
            stats['batting_stats'].get('impact_score', 0),
            stats['batting_stats'].get('consistency_rating', 0),
            self._calculate_form_factor(recent_performances),
            self._calculate_opposition_rating(stats, opposition),
            self._calculate_venue_index(stats, venue),
            1 if match_type == 'T20' else 0  # Match type encoding
        ]
        return np.array(features).reshape(1, -1)

    def _predict_with_confidence(self, model, features: np.ndarray, prediction_type: str) -> Tuple[float, Tuple[float, float]]:
        """Generate predictions with confidence intervals"""
        # For simplicity, using a basic confidence interval calculation
        pred = model.predict(features)[0]
        std_dev = np.std([tree.predict(features)[0] for tree in model.estimators_])
        confidence_interval = (pred - 1.96 * std_dev, pred + 1.96 * std_dev)
        return pred, confidence_interval

    def _calculate_impact_score(self, average: float, strike_rate: float) -> float:
        """Calculate player's impact score"""
        return (average * strike_rate) / 100

    def _calculate_consistency(self, average: float, fifties: int, hundreds: int) -> float:
        """Calculate player's consistency rating"""
        milestone_ratio = (fifties + 2 * hundreds) / max(1, average)
        return min(10, milestone_ratio * 5)

    def _calculate_bowling_effectiveness(self, average: float, economy: float, wickets: int) -> float:
        """Calculate bowler's effectiveness score"""
        return (wickets * 10) / (average * economy)

    def _calculate_pressure_rating(self, economy: float, wickets: int, matches: int) -> float:
        """Calculate bowler's pressure rating"""
        wickets_per_match = wickets / max(1, matches)
        return (wickets_per_match * 10) / economy

    def _calculate_match_influence(self, stats: Dict, recent_performances: List, opposition: str) -> float:
        """Calculate probability of player influencing the match outcome"""
        recent_form = self._calculate_form_factor(recent_performances)
        opposition_factor = self._calculate_opposition_rating(stats, opposition)

        return (recent_form + opposition_factor) / 2

    def _predict_strike_rate(self, stats: Dict, opposition: str) -> float:
        """Predict expected strike rate against specific opposition"""
        base_sr = stats['batting_stats']['strike_rate']
        opposition_factor = self._calculate_opposition_rating(stats, opposition)

        return base_sr * (0.8 + 0.4 * opposition_factor)

    def _predict_economy(self, stats: Dict, opposition: str) -> float:
        """Predict expected economy rate against specific opposition"""
        if 'bowling_stats' not in stats:
            return 0.0

        base_economy = stats['bowling_stats']['economy']
        opposition_factor = self._calculate_opposition_rating(stats, opposition)

        return base_economy * (1.1 - 0.2 * opposition_factor)

    def _calculate_enhanced_confidence(self, recent_performances: List, stats: Dict, opposition: str, venue: str) -> float:
        """Calculate enhanced confidence score considering multiple factors"""
        if not recent_performances:
            return 0.5

        # Weight factors
        recency_weight = 0.4
        consistency_weight = 0.2
        opposition_weight = 0.2
        venue_weight = 0.2

        # Calculate individual factors
        recency_factor = min(len(recent_performances) / 10, 1.0)
        consistency = stats['batting_stats'].get('consistency_rating', 5) / 10
        opposition_factor = self._calculate_opposition_rating(stats, opposition)
        venue_factor = self._calculate_venue_index(stats, venue)

        # Weighted sum
        confidence = (
            recency_weight * recency_factor +
            consistency_weight * consistency +
            opposition_weight * opposition_factor +
            venue_weight * venue_factor
        )

        return min(max(confidence, 0), 1)

    def _calculate_opposition_rating(self, stats: Dict, opposition: str) -> float:
        """Calculate performance rating against specific opposition"""
        # Simplified implementation - could be enhanced with actual head-to-head data
        return 0.7  # Default rating

    def _calculate_venue_index(self, stats: Dict, venue: str) -> float:
        """Calculate performance index for specific venue"""
        # Simplified implementation - could be enhanced with actual venue statistics
        return 0.6  # Default index

    def _calculate_form_factor(self, recent_performances: List) -> float:
        """Calculate current form factor based on recent performances"""
        if not recent_performances:
            return 0.5

        recent_scores = [match.get('runs', 0) for match in recent_performances]
        if not recent_scores:
            return 0.5

        # Calculate weighted average (more recent matches have higher weight)
        weights = np.exp(-np.arange(len(recent_scores)) / 2)
        weighted_avg = np.average(recent_scores, weights=weights)

        # Normalize to [0, 1]
        return min(weighted_avg / 100, 1.0)

    def get_form_analysis(self, player_name: str, last_n_matches: int = 5) -> Dict:
        """Analyze player's recent form"""
        try:
            recent_matches = self.data_collector.get_recent_performances(player_name, last_n_matches)
            if not recent_matches:
                return {}

            # Calculate form metrics
            recent_runs = [match.get('runs', 0) for match in recent_matches]
            recent_wickets = [match.get('wickets', 0) for match in recent_matches]

            form_analysis = {
                'recent_average': np.mean(recent_runs) if recent_runs else 0,
                'recent_wickets_per_match': np.mean(recent_wickets) if recent_wickets else 0,
                'form_rating': self._calculate_form_rating(recent_matches),
                'trend': self._calculate_trend(recent_matches)
            }

            return form_analysis

        except Exception as e:
            logger.error(f"Error analyzing form for {player_name}: {str(e)}")
            return {}

    def _calculate_confidence(self, recent_performances: List[Dict]) -> float:
        """Calculate confidence score based on recent performances"""
        if not recent_performances:
            return 0.5

        # More recent matches = higher confidence
        recency_factor = min(len(recent_performances) / 10, 1.0)

        # Consistency factor
        scores = [match.get('runs', 0) for match in recent_performances]
        consistency = 1 - (np.std(scores) / (np.mean(scores) + 1e-6)) if scores else 0

        return min(max((recency_factor + consistency) / 2, 0), 1)

    def _calculate_form_rating(self, recent_matches: List[Dict]) -> str:
        """Calculate form rating based on recent performances"""
        if not recent_matches:
            return "Unknown"

        recent_scores = [match.get('runs', 0) for match in recent_matches]
        avg_score = np.mean(recent_scores) if recent_scores else 0

        if avg_score >= 40:
            return "Excellent"
        elif avg_score >= 30:
            return "Good"
        elif avg_score >= 20:
            return "Average"
        else:
            return "Poor"

    def _calculate_trend(self, recent_matches: List[Dict]) -> str:
        """Calculate performance trend"""
        if len(recent_matches) < 2:
            return "Stable"

        recent_scores = [match.get('runs', 0) for match in recent_matches]
        if len(recent_scores) >= 2:
            slope = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
            if slope > 5:
                return "Upward"
            elif slope < -5:
                return "Downward"

        return "Stable"