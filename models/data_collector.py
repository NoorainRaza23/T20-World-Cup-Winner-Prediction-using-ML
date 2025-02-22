import requests
import pandas as pd
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class CricketDataCollector:
    def __init__(self):
        self.base_url = "https://hs-consumer-api.espncricinfo.com/v1/pages"
        self.stats_url = "https://stats.espncricinfo.com/ci/engine/stats"
        self.rankings_url = "https://hs-consumer-api.espncricinfo.com/v1/pages/rankings/teams"

    def get_team_rankings(self, format_type='t20i'):
        """Fetch current team rankings"""
        try:
            url = f"{self.rankings_url}/{format_type}"
            response = requests.get(url)
            if response.status_code == 200:
                rankings = response.json().get('rankings', [])
                return pd.DataFrame([{
                    'Team': r.get('team', {}).get('name'),
                    'Ranking': r.get('rank'),
                    'Points': r.get('points'),
                    'Rating': r.get('rating')
                } for r in rankings])
            return None
        except Exception as e:
            logger.error(f"Error fetching rankings: {str(e)}")
            return None

    def get_player_stats(self, player_name: str) -> Optional[Dict]:
        """Fetch detailed player statistics for a player"""
        try:
            url = f"{self.base_url}/player/{player_name}/stats"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                return {
                    'batting_stats': self._process_batting_stats(data.get('batting', [])),
                    'bowling_stats': self._process_bowling_stats(data.get('bowling', [])),
                    'recent_form': self._process_recent_form(data.get('recent_matches', [])),
                    'player_rankings': self._get_player_rankings(player_name)
                }
            return None
        except Exception as e:
            logger.error(f"Error fetching player stats: {str(e)}")
            return None

    def _process_batting_stats(self, data: List[Dict]) -> pd.DataFrame:
        """Process batting statistics"""
        return pd.DataFrame([{
            'player_name': item.get('player', {}).get('name'),
            'matches': item.get('matches', 0),
            'runs': item.get('runs', 0),
            'average': item.get('average', 0),
            'strike_rate': item.get('strike_rate', 0),
            'hundreds': item.get('hundreds', 0),
            'fifties': item.get('fifties', 0)
        } for item in data])

    def _process_bowling_stats(self, data: List[Dict]) -> pd.DataFrame:
        """Process bowling statistics"""
        return pd.DataFrame([{
            'player_name': item.get('player', {}).get('name'),
            'matches': item.get('matches', 0),
            'wickets': item.get('wickets', 0),
            'average': item.get('average', 0),
            'economy': item.get('economy', 0),
            'strike_rate': item.get('strike_rate', 0)
        } for item in data])

    def _process_recent_form(self, matches: List[Dict]) -> pd.DataFrame:
        """Process recent match data"""
        return pd.DataFrame([{
            'date': match.get('date'),
            'opposition': match.get('opposition', {}).get('name'),
            'runs': match.get('batting', {}).get('runs', 0),
            'wickets': match.get('bowling', {}).get('wickets', 0),
            'catches': match.get('fielding', {}).get('catches', 0),
            'match_result': match.get('result')
        } for match in matches])

    def _get_player_rankings(self, player_name: str) -> Optional[pd.DataFrame]:
        """Get ICC rankings for a player"""
        try:
            url = f"{self.base_url}/player/{player_name}/rankings"
            response = requests.get(url)
            if response.status_code == 200:
                rankings = response.json().get('rankings', [])
                return pd.DataFrame([{
                    'player_name': r.get('player', {}).get('name'),
                    'batting_rank': r.get('batting_rank'),
                    'bowling_rank': r.get('bowling_rank'),
                    'all_rounder_rank': r.get('all_rounder_rank')
                } for r in rankings])
            return None
        except Exception as e:
            logger.error(f"Error fetching player rankings: {str(e)}")
            return None

    def get_recent_matches(self, format_type="t20") -> Optional[pd.DataFrame]:
        """Fetch recent T20 match data"""
        try:
            url = f"{self.base_url}/matches/current?format={format_type}"
            response = requests.get(url)
            if response.status_code == 200:
                matches = response.json().get('matches', [])
                return self._process_match_data(matches)
            return None
        except Exception as e:
            logger.error(f"Error fetching match data: {str(e)}")
            return None

    def _process_match_data(self, matches: List[Dict]) -> pd.DataFrame:
        """Process raw match data into DataFrame with enhanced features"""
        processed_matches = []
        for match in matches:
            if match.get('format') == 'T20':
                teams = match.get('teams', [{}, {}])
                team1 = teams[0].get('team', {})
                team2 = teams[1].get('team', {})

                processed_matches.append({
                    'Team': team1.get('name'),
                    'Opposition': team2.get('name'),
                    'Ground': match.get('venue', {}).get('name'),
                    'winner': match.get('winner', {}).get('name'),
                    'Date': match.get('startDate'),
                    'Team_score': teams[0].get('score'),
                    'Opposition_score': teams[1].get('score'),
                    'Toss_winner': match.get('toss', {}).get('winner', {}).get('name'),
                    'Toss_decision': match.get('toss', {}).get('decision'),
                    'match_type': match.get('stage_name', 'regular')
                })
        return pd.DataFrame(processed_matches)

    def get_venue_stats(self, venue: str) -> Optional[Dict]:
        """Fetch venue statistics"""
        try:
            url = f"{self.base_url}/venue/{venue}/stats"
            response = requests.get(url)
            if response.status_code == 200:
                stats = response.json()
                return {
                    'matches_played': stats.get('matches_played', 0),
                    'average_first_innings': stats.get('first_innings_average', 0),
                    'average_second_innings': stats.get('second_innings_average', 0),
                    'toss_win_bat_first_ratio': stats.get('toss_bat_first_ratio', 0.5)
                }
            return None
        except Exception as e:
            logger.error(f"Error fetching venue stats: {str(e)}")
            return None