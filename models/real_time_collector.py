import os
import requests
import logging
from datetime import datetime
from typing import Dict, Optional, List
from utils.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealTimeDataCollector:
    def __init__(self):
        """Initialize the collector with API key and settings"""
        self.base_url = "https://api.cricapi.com/v1"

        # Get API key from environment
        self.api_key = config.cricinfo_api_key
        if not self.api_key:
            logger.error("CRICINFO_API_KEY not found in environment variables. Please set it in your .env file.")
            logger.info("To get an API key, visit https://cricapi.com/ and create an account.")

        # Get settings from config
        settings = config.get_live_match_settings()
        self.refresh_interval = settings['refresh_interval']
        self.cache_duration = settings['cache_duration']

        # Initialize session for better performance
        self.session = requests.Session()
        self.last_request_time = datetime.now()
        self._cache = {}
        self._cache_timestamps = {}

    def _check_api_key(self) -> bool:
        """Verify if API key is available"""
        if not self.api_key:
            logger.error("CRICINFO_API_KEY is not set. Please add it to your .env file.")
            return False
        return True

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self._cache_timestamps:
            return False
        age = (datetime.now() - self._cache_timestamps[cache_key]).total_seconds()
        return age < self.cache_duration

    def _get_from_cache(self, cache_key: str) -> Optional[Dict]:
        """Get data from cache if valid"""
        if self._is_cache_valid(cache_key):
            return self._cache.get(cache_key)
        return None

    def _store_in_cache(self, cache_key: str, data: Dict):
        """Store data in cache with timestamp"""
        self._cache[cache_key] = data
        self._cache_timestamps[cache_key] = datetime.now()

    def _respect_rate_limit(self) -> bool:
        """Implement smarter rate limiting with exponential backoff"""
        time_since_last = (datetime.now() - self.last_request_time).total_seconds()
        if time_since_last < self.refresh_interval:
            logger.debug(f"Rate limit: Waiting {self.refresh_interval - time_since_last} seconds")
            return False
        self.last_request_time = datetime.now()
        return True

    def get_live_matches(self) -> Optional[List[Dict]]:
        """Fetch current live T20 matches with caching"""
        if not self._check_api_key():
            return None

        cache_key = 'live_matches'
        cached_data = self._get_from_cache(cache_key)
        if cached_data:
            return cached_data

        try:
            if not self._respect_rate_limit():
                return None

            endpoint = f"{self.base_url}/currentMatches"
            params = {
                "apikey": self.api_key,
                "offset": 0,
                "format": "t20"
            }

            response = self.session.get(endpoint, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "success":
                    matches = data.get("data", [])
                    live_matches = [
                        match for match in matches 
                        if match.get('matchStarted') and not match.get('matchEnded')
                    ]
                    self._store_in_cache(cache_key, live_matches)
                    return live_matches

            logger.error(f"Error fetching live matches: {response.status_code}")
            return None

        except Exception as e:
            logger.error(f"Error in get_live_matches: {str(e)}")
            return None

    def get_match_details(self, match_id: str) -> Optional[Dict]:
        """Get detailed information for a specific match with caching"""
        cache_key = f'match_details_{match_id}'
        cached_data = self._get_from_cache(cache_key)
        if cached_data:
            return cached_data

        try:
            if not self._check_api_key() or not self._respect_rate_limit():
                return None

            endpoint = f"{self.base_url}/match_info"
            params = {
                "apikey": self.api_key,
                "id": match_id
            }

            response = self.session.get(endpoint, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "success":
                    processed_data = self._process_match_data(data.get("data", {}))
                    self._store_in_cache(cache_key, processed_data)
                    return processed_data

            return None

        except Exception as e:
            logger.error(f"Error getting match details: {str(e)}")
            return None

    def _process_match_data(self, match_data: Dict) -> Dict:
        """Process raw match data into a structured format"""
        try:
            return {
                'match_id': match_data.get('id'),
                'status': match_data.get('status'),
                'team1': {
                    'name': match_data.get('team1'),
                    'score': match_data.get('score', {}).get('team1', '0/0'),
                    'overs': match_data.get('overs', {}).get('team1', 0),
                    'run_rate': match_data.get('runRate', {}).get('team1', 0),
                },
                'team2': {
                    'name': match_data.get('team2'),
                    'score': match_data.get('score', {}).get('team2', '0/0'),
                    'overs': match_data.get('overs', {}).get('team2', 0),
                    'run_rate': match_data.get('runRate', {}).get('team2', 0),
                },
                'venue': match_data.get('venue'),
                'match_type': match_data.get('match_type'),
                'toss': match_data.get('toss'),
                'current_innings': match_data.get('currentInnings'),
                'live_score': {
                    'recent_balls': match_data.get('recentBalls', []),
                    'current_partnership': match_data.get('currentPartnership'),
                    'required_rate': match_data.get('requiredRunRate'),
                },
                'last_updated': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error processing match data: {str(e)}")
            return {}

    def get_live_score(self, match_id: str) -> Optional[Dict]:
        """Get live score updates for a match"""
        try:
            if not self._check_api_key() or not self._respect_rate_limit():
                return None

            endpoint = f"{self.base_url}/live-score"
            params = {
                "apikey": self.api_key,
                "id": match_id
            }

            response = self.session.get(endpoint, params=params)
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "success":
                    return self._process_live_score(data.get("data", {}))

            return None

        except Exception as e:
            logger.error(f"Error getting live score: {str(e)}")
            return None

    def _process_live_score(self, score_data: Dict) -> Dict:
        """Process live score data"""
        try:
            return {
                'match_id': score_data.get('id'),
                'current_innings': score_data.get('current_innings'),
                'current_score': score_data.get('score'),
                'current_overs': score_data.get('overs'),
                'run_rate': score_data.get('run_rate'),
                'required_rate': score_data.get('required_rate'),
                'batting_team': {
                    'name': score_data.get('batting_team'),
                    'current_batsmen': score_data.get('current_batsmen', []),
                    'recent_overs': score_data.get('recent_overs', [])
                },
                'bowling_team': {
                    'name': score_data.get('bowling_team'),
                    'current_bowler': score_data.get('current_bowler', {}),
                    'recent_wickets': score_data.get('recent_wickets', [])
                },
                'last_updated': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error processing live score: {str(e)}")
            return {}

    def get_ball_by_ball(self, match_id: str, innings: int = 1) -> Optional[List[Dict]]:
        """Get ball by ball commentary"""
        try:
            if not self._check_api_key() or not self._respect_rate_limit():
                return None

            endpoint = f"{self.base_url}/ballByBall"
            params = {
                "apikey": self.api_key,
                "id": match_id,
                "innings": innings
            }

            response = self.session.get(endpoint, params=params)
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "success":
                    return data.get("data", [])

            return None

        except Exception as e:
            logger.error(f"Error getting ball by ball data: {str(e)}")
            return None