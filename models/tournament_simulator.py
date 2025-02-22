
import numpy as np
from typing import List, Dict, Tuple
import pandas as pd
from models.data_processor import DataProcessor
from models.ml_models import MLModels

class TournamentSimulator:
    def __init__(self, data_processor: DataProcessor, ml_models: MLModels):
        self.dp = data_processor
        self.ml_models = ml_models
        
    def simulate_match(self, team1: str, team2: str, knockout: bool = False) -> Tuple[str, float]:
        """Simulate a single match between two teams"""
        team1_stats = self.dp.get_team_stats(team1)
        team2_stats = self.dp.get_team_stats(team2)
        
        match_data = {
            'Team': team1,
            'Opposition': team2,
            'winner': team1,
            'Toss': 'won',
            'Bat': '1st',
            'Ground': self.dp.match_data['Ground'].iloc[0],
            'apperance_team': team1_stats['apperance'],
            'Title_team': team1_stats['Title'],
            'Finals_team': team1_stats['Finals'],
            'Semi finals_team': team1_stats['Semi finals'],
            'Current ranking_team': team1_stats['Current ranking'],
            'apperance_opposition': team2_stats['apperance'],
            'Title_opposition': team2_stats['Title'],
            'Finals_opposition': team2_stats['Finals'],
            'Semi finals_opposition': team2_stats['Semi finals'],
            'Current ranking_opposition': team2_stats['Current ranking']
        }
        
        X_match = pd.DataFrame([match_data])
        predictions = self.ml_models.predict_match(X_match)
        
        # Average predictions from all models
        win_prob = np.mean([pred['win_probability'] for pred in predictions.values()])
        
        # In knockout matches, we need a definitive winner
        if knockout:
            winner = team1 if win_prob > 0.5 else team2
            return winner, max(win_prob, 1 - win_prob)
        
        # For group matches, return winner based on probability
        random_outcome = np.random.random()
        winner = team1 if random_outcome < win_prob else team2
        return winner, win_prob
        
    def simulate_group_stage(self, groups: Dict[str, List[str]]) -> Dict[str, List[Tuple[str, int]]]:
        """Simulate entire group stage and return standings"""
        standings = {}
        
        for group_name, teams in groups.items():
            group_standings = {team: 0 for team in teams}
            
            # Each team plays against every other team in the group
            for i, team1 in enumerate(teams):
                for team2 in teams[i+1:]:
                    winner, _ = self.simulate_match(team1, team2)
                    group_standings[winner] += 2
                    
            # Sort teams by points
            sorted_standings = sorted(
                group_standings.items(),
                key=lambda x: (x[1], -self.dp.get_team_stats(x[0])['Current ranking']),
                reverse=True
            )
            standings[group_name] = sorted_standings
            
        return standings
        
    def simulate_knockout_stage(self, qualified_teams: List[str]) -> Dict[str, str]:
        """Simulate knockout stage matches"""
        results = {
            'semi_finals': [],
            'final': None,
            'winner': None
        }
        
        # Simulate semi-finals
        semi_final1 = self.simulate_match(qualified_teams[0], qualified_teams[3], knockout=True)
        semi_final2 = self.simulate_match(qualified_teams[1], qualified_teams[2], knockout=True)
        
        results['semi_finals'] = [
            (qualified_teams[0], qualified_teams[3], semi_final1[0]),
            (qualified_teams[1], qualified_teams[2], semi_final2[0])
        ]
        
        # Simulate final
        final_winner, win_prob = self.simulate_match(semi_final1[0], semi_final2[0], knockout=True)
        results['final'] = (semi_final1[0], semi_final2[0], final_winner)
        results['winner'] = final_winner
        
        return results
        
    def simulate_tournament(self, groups: Dict[str, List[str]], teams_per_group: int = 2) -> Dict:
        """Simulate entire tournament"""
        # Simulate group stage
        group_standings = self.simulate_group_stage(groups)
        
        # Get qualified teams
        qualified_teams = []
        for group, standings in group_standings.items():
            qualified_teams.extend([team for team, _ in standings[:teams_per_group]])
            
        # Simulate knockout stage
        knockout_results = self.simulate_knockout_stage(qualified_teams)
        
        return {
            'group_standings': group_standings,
            'knockout_results': knockout_results
        }
