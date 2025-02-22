import time
import logging
from datetime import datetime
from typing import Dict, Optional
import pandas as pd
from models.data_processor import DataProcessor
from models.ml_models import MLModels

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LiveMatchProcessor:
    def __init__(self, data_processor: DataProcessor, ml_models: MLModels):
        self.dp = data_processor
        self.ml_models = ml_models
        self.current_match: Optional[Dict] = None
        self.prediction_history = []

    def start_match(self, team1: str, team2: str) -> bool:
        """Initialize a new live match"""
        try:
            team1_stats = self.dp.get_team_stats(team1)
            team2_stats = self.dp.get_team_stats(team2)

            if not team1_stats or not team2_stats:
                logger.error(f"Could not get stats for {team1} or {team2}")
                return False

            self.current_match = {
                'team1': team1,
                'team2': team2,
                'start_time': datetime.now(),
                'team1_stats': team1_stats,
                'team2_stats': team2_stats,
                'current_score': {'team1': 0, 'team2': 0},
                'current_overs': {'team1': 0, 'team2': 0},
                'wickets': {'team1': 0, 'team2': 0},
                'batting_first': team1,
                'status': 'in_progress'
            }

            # Initial prediction
            self._update_prediction()
            return True
        except Exception as e:
            logger.error(f"Error starting match: {e}")
            return False

    def update_match_status(self, 
                          team1_score: int, 
                          team1_overs: float,
                          team1_wickets: int,
                          team2_score: int = 0,
                          team2_overs: float = 0,
                          team2_wickets: int = 0) -> Dict:
        """Update current match status and get new prediction"""
        if not self.current_match:
            logger.error("No active match to update")
            return {}

        try:
            self.current_match['current_score']['team1'] = team1_score
            self.current_match['current_score']['team2'] = team2_score
            self.current_match['current_overs']['team1'] = team1_overs
            self.current_match['current_overs']['team2'] = team2_overs
            self.current_match['wickets']['team1'] = team1_wickets
            self.current_match['wickets']['team2'] = team2_wickets

            return self._update_prediction()
        except Exception as e:
            logger.error(f"Error updating match status: {e}")
            return {}

    def _update_prediction(self) -> Dict:
        """Calculate updated prediction based on current match state"""
        try:
            if not self.current_match:
                return {}

            # Check if match data is available
            if not hasattr(self.dp, 'match_data') or self.dp.match_data is None:
                logger.error("Match data not available")
                return {}

            if self.dp.match_data.empty:
                logger.error("Match data is empty")
                return {}

            # Get the first ground name as default if available
            default_ground = self.dp.match_data['Ground'].iloc[0] if not self.dp.match_data['Ground'].empty else 'Unknown'

            # Prepare feature data for prediction with safer access
            match_data = pd.DataFrame([{
                'Team': self.current_match['team1'],
                'Opposition': self.current_match['team2'],
                'winner': self.current_match['team1'],  # placeholder
                'Ground': default_ground,
                'Toss': 'won',
                'Bat': '1st' if self.current_match['batting_first'] == self.current_match['team1'] else '2nd',
                'apperance_team': self.current_match['team1_stats'].get('apperance', 0),
                'Title_team': self.current_match['team1_stats'].get('Title', 0),
                'Finals_team': self.current_match['team1_stats'].get('Finals', 0),
                'Semi finals_team': self.current_match['team1_stats'].get('Semi finals', 0),
                'Current ranking_team': self.current_match['team1_stats'].get('Current ranking', 0),
                'apperance_opposition': self.current_match['team2_stats'].get('apperance', 0),
                'Title_opposition': self.current_match['team2_stats'].get('Title', 0),
                'Finals_opposition': self.current_match['team2_stats'].get('Finals', 0),
                'Semi finals_opposition': self.current_match['team2_stats'].get('Semi finals', 0),
                'Current ranking_opposition': self.current_match['team2_stats'].get('Current ranking', 0)
            }])

            # Transform categorical variables with error handling
            for col, le in self.dp.label_encoders.items():
                if col in match_data.columns:
                    try:
                        match_data[col] = le.transform(match_data[col].astype(str))
                    except Exception as e:
                        logger.error(f"Error transforming column {col}: {e}")
                        return {}

            predictions = self.ml_models.predict_match(match_data)

            # Add current match state to prediction history
            prediction_entry = {
                'timestamp': datetime.now(),
                'predictions': predictions,
                'match_state': {
                    'scores': self.current_match['current_score'].copy(),
                    'overs': self.current_match['current_overs'].copy(),
                    'wickets': self.current_match['wickets'].copy()
                }
            }
            self.prediction_history.append(prediction_entry)

            return prediction_entry
        except Exception as e:
            logger.error(f"Error updating prediction: {e}")
            return {}

    def get_prediction_history(self):
        """Get the history of predictions for the current match"""
        return self.prediction_history