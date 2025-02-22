"""
Data Processor Module for T20 Cricket World Cup Predictor

This module handles all data processing operations including:
- Loading and cleaning match data
- Database operations
- Feature engineering for ML models
- Caching mechanisms
"""

try:
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import LabelEncoder
except ImportError as e:
    raise ImportError(f"Required package not found. Please install missing packages: {str(e)}")

import psycopg2
from psycopg2 import sql
import os
import logging
from pathlib import Path
from functools import lru_cache
from typing import Optional, Tuple, Dict, List, Any
import requests
from threading import Lock

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Singleton class for processing cricket match data and managing database operations.

    This class handles:
    - Data loading and cleaning
    - Database connections and queries
    - Feature engineering for ML models
    - Caching of processed data
    """
    _instance: Optional['DataProcessor'] = None
    _initialized: bool = False
    _lock: Lock = Lock()
    _data_cache: Dict[str, Any] = {}

    def __new__(cls) -> 'DataProcessor':
        if cls._instance is None:
            cls._instance = super(DataProcessor, cls).__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if not DataProcessor._initialized:
            with DataProcessor._lock:
                if not DataProcessor._initialized:
                    self.label_encoders: Dict[str, LabelEncoder] = {}
                    self.worldcup_data: Optional[pd.DataFrame] = None
                    self.match_data: Optional[pd.DataFrame] = None
                    self.bilateral_data: Optional[pd.DataFrame] = None
                    self.venue_data: Optional[pd.DataFrame] = None
                    self.conn: Optional[psycopg2.extensions.connection] = None
                    self._connect_to_db()
                    DataProcessor._initialized = True

    @staticmethod
    def _read_excel_cached(file_path: str) -> Optional[pd.DataFrame]:
        """Read Excel file with caching"""
        if file_path in DataProcessor._data_cache:
            return DataProcessor._data_cache[file_path].copy()

        try:
            df = pd.read_excel(file_path, engine='openpyxl')
            if df is not None and not df.empty:
                DataProcessor._data_cache[file_path] = df.copy()
                return df
        except Exception as e:
            logger.error(f"Error reading {file_path}: {str(e)}")
        return None

    def load_data(self) -> bool:
        """Load data with improved caching and parallel processing"""
        try:
            data_path = Path("attached_assets")
            if not data_path.exists():
                logger.error("attached_assets directory not found")
                return False

            success = True

            # Load World Cup data
            worldcup_file = data_path / "T20 worldcup overall.xlsx"
            if not worldcup_file.exists():
                logger.error(f"World Cup data file not found: {worldcup_file}")
                success = False
            else:
                self.worldcup_data = self._read_excel_cached(str(worldcup_file))
                if self.worldcup_data is not None:
                    self._clean_worldcup_data()
                    logger.info("Successfully loaded World Cup data")
                else:
                    success = False
                    logger.error("Failed to load World Cup data")

            # Load match data with caching
            match_file = data_path / "matchresultupdate2.xlsx"
            if not match_file.exists():
                logger.error(f"Match data file not found: {match_file}")
                success = False
            else:
                self.match_data = self._read_excel_cached(str(match_file))
                if self.match_data is not None:
                    self._clean_match_data()
                    logger.info("Successfully loaded match data")
                else:
                    success = False
                    logger.error("Failed to load match data")

            # Load bilateral data if available
            bilateral_file = data_path / "t20i_bilateral.xlsx"
            if bilateral_file.exists():
                self.bilateral_data = self._read_excel_cached(str(bilateral_file))
                if self.bilateral_data is not None and not self.bilateral_data.empty:
                    if self.match_data is not None:
                        self.match_data = pd.concat([self.match_data, self.bilateral_data], ignore_index=True)
                        self._clean_match_data()
                        logger.info("Successfully loaded bilateral data")

            return success and self.worldcup_data is not None and self.match_data is not None

        except Exception as e:
            logger.error(f"Error in load_data: {str(e)}")
            return False

    def _clean_worldcup_data(self) -> None:
        """Optimized World Cup data cleaning"""
        if self.worldcup_data is None:
            return

        try:
            # Clean Team column efficiently
            if 'Team' in self.worldcup_data.columns:
                mask = self.worldcup_data['Team'].notna()
                self.worldcup_data.loc[mask, 'Team'] = (
                    self.worldcup_data.loc[mask, 'Team']
                    .astype(str)
                    .str.strip()
                )
                self.worldcup_data = self.worldcup_data.dropna(subset=['Team'])

            # Efficient numeric conversion
            numeric_cols = ['apperance', 'Title', 'Finals', 'Semi finals', 'Current ranking']
            self.worldcup_data[numeric_cols] = (
                self.worldcup_data[numeric_cols]
                .apply(pd.to_numeric, errors='coerce')
                .fillna(0)
                .astype(int)
            )

            # Fill remaining NaN values
            self.worldcup_data['group'] = self.worldcup_data['group'].fillna('Unknown')
            logger.info("Successfully cleaned World Cup data")

        except Exception as e:
            logger.error(f"Error cleaning World Cup data: {str(e)}")
            self.worldcup_data = None

    def _clean_match_data(self) -> None:
        """Optimized match data cleaning with type safety"""
        if self.match_data is None:
            return

        try:
            # Clean team names efficiently
            for col in ['Team', 'Opposition', 'winner']:
                if col in self.match_data.columns:
                    mask = self.match_data[col].notna()
                    self.match_data.loc[mask, col] = (
                        self.match_data.loc[mask, col]
                        .astype(str)
                        .str.strip()
                    )
            logger.info("Successfully cleaned match data")

        except Exception as e:
            logger.error(f"Error cleaning match data: {str(e)}")
            self.match_data = None

    @lru_cache(maxsize=32)
    def get_team_stats(self, team: str) -> Dict:
        """Get team statistics with caching and type safety"""
        try:
            if team in self._data_cache:
                return self._data_cache[team].copy()

            if self.worldcup_data is None:
                if not self.load_data():
                    return self._get_default_stats()

            team = str(team).strip()
            if self.worldcup_data is not None:
                team_data = self.worldcup_data[self.worldcup_data['Team'].str.strip() == team]

                if team_data.empty:
                    stats = self._get_default_stats()
                else:
                    try:
                        stats = {
                            'Current ranking': int(team_data['Current ranking'].iloc[0]),
                            'apperance': int(team_data['apperance'].iloc[0]),
                            'Title': int(team_data['Title'].iloc[0]),
                            'Finals': int(team_data['Finals'].iloc[0]),
                            'Semi finals': int(team_data['Semi finals'].iloc[0]),
                            'group': str(team_data['group'].iloc[0])
                        }
                    except (IndexError, KeyError):
                        stats = self._get_default_stats()

                self._data_cache[team] = stats.copy()
                return stats
            return self._get_default_stats()

        except Exception as e:
            logger.error(f"Error getting team stats for {team}: {str(e)}")
            return self._get_default_stats()

    def _get_default_stats(self) -> Dict:
        """Get default statistics with type hints"""
        return {
            'Current ranking': 0,
            'apperance': 0,
            'Title': 0,
            'Finals': 0,
            'Semi finals': 0,
            'group': 'Unknown'
        }

    def get_all_teams(self) -> List[str]:
        """Get list of all teams with caching and type safety"""
        try:
            if 'all_teams' in self._data_cache:
                return self._data_cache['all_teams'].copy()

            if self.worldcup_data is None:
                if not self.load_data():
                    return []

            if self.worldcup_data is not None and 'Team' in self.worldcup_data.columns:
                teams = sorted(list(
                    self.worldcup_data['Team']
                    .dropna()
                    .astype(str)
                    .str.strip()
                    .unique()
                ))

                self._data_cache['all_teams'] = teams.copy()
                return teams

            return []

        except Exception as e:
            logger.error(f"Error getting team list: {str(e)}")
            return []

    def _connect_to_db(self):
        """Connect to PostgreSQL database with proper error handling"""
        try:
            if not os.environ.get('DATABASE_URL'):
                logger.warning("DATABASE_URL not found, proceeding with file-based data only")
                return
            self.conn = psycopg2.connect(
                os.environ.get('DATABASE_URL'),
                connect_timeout=5
            )
            self.conn.autocommit = True
            logger.info("Successfully connected to database")
        except Exception as e:
            logger.error(f"Database connection error: {str(e)}")
            self.conn = None

    def _ensure_db_connection(self):
        """Ensure database connection is active"""
        if self.conn is None:
            self._connect_to_db()
        try:
            if self.conn:
                with self.conn.cursor() as cur:
                    cur.execute("SELECT 1")
        except Exception:
            self._connect_to_db()

    def create_tables(self):
        """Create necessary database tables if they don't exist"""
        self._ensure_db_connection()
        if not self.conn:
            logger.warning("No database connection available, skipping table creation")
            return False

        try:
            with self.conn.cursor() as cur:
                # Create teams table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS teams (
                        id SERIAL PRIMARY KEY,
                        name VARCHAR(100) UNIQUE NOT NULL,
                        group_name VARCHAR(10),
                        appearances INTEGER DEFAULT 0,
                        titles INTEGER DEFAULT 0,
                        finals INTEGER DEFAULT 0,
                        semi_finals INTEGER DEFAULT 0,
                        current_ranking INTEGER DEFAULT 0
                    )
                """)

                cur.execute("""
                    CREATE TABLE IF NOT EXISTS matches (
                        id SERIAL PRIMARY KEY,
                        team1_id INTEGER REFERENCES teams(id),
                        team2_id INTEGER REFERENCES teams(id),
                        winner_id INTEGER REFERENCES teams(id),
                        toss_winner_id INTEGER REFERENCES teams(id),
                        bat_first BOOLEAN,
                        ground VARCHAR(100),
                        match_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                self.conn.commit()
                logger.info("Database tables created successfully")
                return True
        except Exception as e:
            logger.error(f"Error creating tables: {str(e)}")
            if self.conn:
                self.conn.rollback()
            return False

    def _populate_database(self):
        """Internal method to populate the database with loaded data"""
        self._ensure_db_connection()
        if not self.conn:
            logger.warning("No database connection available, skipping database population")
            return False
        try:
            with self.conn.cursor() as cur:
                # Clear existing data
                cur.execute("TRUNCATE TABLE matches CASCADE")
                cur.execute("TRUNCATE TABLE teams CASCADE")

                # Populate teams table
                if self.worldcup_data is not None:
                    for _, row in self.worldcup_data.iterrows():
                        try:
                            if pd.isna(row['Team']) or pd.isna(row['group']):
                                continue

                            # Insert team with proper error handling
                            cur.execute("""
                                INSERT INTO teams (name, group_name, appearances, titles, 
                                                finals, semi_finals, current_ranking)
                                VALUES (%s, %s, %s, %s, %s, %s, %s)
                                RETURNING id
                            """, (
                                str(row['Team']).strip(),
                                str(row['group']).strip(),
                                int(row['apperance']),
                                int(row['Title']),
                                int(row['Finals']),
                                int(row['Semi finals']),
                                int(row['Current ranking'])
                            ))
                            result = cur.fetchone()
                            if result is None:
                                logger.error(f"Failed to insert team {row['Team']}")
                                continue
                            team_id = result[0]

                            # Populate matches table with proper error handling
                            if self.match_data is not None:
                                for _, match in self.match_data.iterrows():
                                    team1 = str(match['Team']).strip()
                                    team2 = str(match['Opposition']).strip()
                                    winner = str(match['winner']).strip()

                                    # Get team IDs with error handling
                                    cur.execute("SELECT id FROM teams WHERE name = %s", (team1,))
                                    result = cur.fetchone()
                                    if result is None:
                                        logger.error(f"Team not found: {team1}")
                                        continue
                                    team1_id = result[0]

                                    cur.execute("SELECT id FROM teams WHERE name = %s", (team2,))
                                    result = cur.fetchone()
                                    if result is None:
                                        logger.error(f"Team not found: {team2}")
                                        continue
                                    team2_id = result[0]

                                    cur.execute("SELECT id FROM teams WHERE name = %s", (winner,))
                                    result = cur.fetchone()
                                    if result is None:
                                        logger.error(f"Winner team not found: {winner}")
                                        continue
                                    winner_id = result[0]

                                    cur.execute("""
                                        INSERT INTO matches (team1_id, team2_id, winner_id, toss_winner_id, bat_first, ground)
                                        VALUES (%s, %s, %s, %s, %s, %s)
                                    """,(team1_id, team2_id, winner_id, team1_id, True, str(match.get('Ground', 'Unknown'))))

                        except Exception as e:
                            logger.error(f"Error inserting team {row.get('Team', 'Unknown')}: {e}")
                            continue

                self.conn.commit()
                logger.info("Database populated successfully")
                return True
        except Exception as e:
            logger.error(f"Error populating database: {str(e)}")
            if self.conn:
                self.conn.rollback()
            return False

    def preprocess_data(self) -> Tuple[Optional[pd.DataFrame], Optional[np.ndarray]]:
        """Preprocess data for model training with optimized performance"""
        try:
            if self.worldcup_data is None or self.match_data is None or self.worldcup_data.empty or self.match_data.empty:
                logger.error("Required data files are not loaded")
                return None, None

            features = []
            targets = []

            # Process each match with optimized data access
            if self.match_data is not None:
                for _, match in self.match_data.iterrows():
                    try:
                        team1 = str(match['Team']).strip()
                        team2 = str(match['Opposition']).strip()
                        winner = str(match['winner']).strip()

                        # Get team statistics with caching
                        team1_stats = self.get_team_stats(team1)
                        team2_stats = self.get_team_stats(team2)

                        if team1_stats and team2_stats:
                            feature_dict = {
                                'Team': team1,
                                'Opposition': team2,
                                'Ground': str(match.get('Ground', 'Unknown')),
                                'Toss': str(match.get('Toss', 'won')),
                                'Bat': str(match.get('Bat', '1st')),
                                'winner': winner,
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
                            features.append(feature_dict)
                            targets.append(1 if winner == team1 else 0)

                    except Exception as e:
                        logger.warning(f"Skipping match due to error: {str(e)}")
                        continue

            if not features:
                logger.error("No valid matches found for training")
                return None, None

            # Convert to DataFrame
            X = pd.DataFrame(features)

            # Encode categorical variables
            categorical_cols = ['Team', 'Opposition', 'Ground', 'Toss', 'Bat', 'winner']
            for col in categorical_cols:
                if col in X.columns:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                    self.label_encoders[col] = le

            y = np.array(targets)

            logger.info(f"Successfully preprocessed {len(X)} matches for training")
            return X, y

        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            return None, None

    def process_worldcup_data(self, df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        """Process World Cup data with type conversion"""
        try:
            if df is None:
                return None

            # Ensure required columns exist
            required_cols = ['Team', 'group', 'apperance', 'Title', 'Finals', 'Semi finals', 'Current ranking']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}")
                return None

            # Clean Team column
            df['Team'] = df['Team'].fillna('Unknown').astype(str)
            df['Team'] = df['Team'].apply(lambda x: x.strip() if isinstance(x, str) else x)

            # Convert numeric columns with error handling
            numeric_cols = ['apperance', 'Title', 'Finals', 'Semi finals', 'Current ranking']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

            return df
        except Exception as e:
            logger.error(f"Error processing World Cup data: {str(e)}")
            return None

    def get_recent_matches(self, team: str, n: int = 5) -> pd.DataFrame:
        """Get recent matches for a team with proper DataFrame operations"""
        try:
            if self.match_data is None or self.match_data.empty:
                return pd.DataFrame()

            # Create a proper copy of the filtered data
            team_matches = self.match_data[
                (self.match_data['Team'] == team) |
                (self.match_data['Opposition'] == team)
            ].copy()  # Create explicit copy

            # Handle date column with proper type conversion
            if 'Start Date' in team_matches.columns:
                team_matches['Start Date'] = pd.to_datetime(
                    team_matches['Start Date'],
                    errors='coerce'
                )
                team_matches = team_matches.sort_values('Start Date', ascending=False)

            return team_matches.head(n)

        except Exception as e:
            logger.error(f"Error getting recent matches for {team}: {str(e)}")
            return pd.DataFrame()

    def _calculate_head_to_head(self, team1: str, team2: str) -> float:
        """Calculate head-to-head win ratio"""
        if self.match_data is None:
            return 0.5
        matches = self.match_data[
            ((self.match_data['Team'] == team1) & (self.match_data['Opposition'] == team2)) |
            ((self.match_data['Team'] == team2) & (self.match_data['Opposition'] == team1))
        ]
        if len(matches) == 0:
            return 0.5
        team1_wins = len(matches[matches['winner'] == team1])
        return team1_wins / len(matches)

    def _calculate_recent_form(self, team: str, n: int = 5) -> float:
        """Calculate recent form based on last n matches"""
        recent_matches = self.get_recent_matches(team, n)
        if len(recent_matches) == 0:
            return 0.5
        wins = len(recent_matches[recent_matches['winner'] == team])
        return wins / len(recent_matches)

    def _get_venue_stats(self, team: str, venue: str) -> float:
        """Get comprehensive venue statistics including weather and pitch conditions"""
        if self.venue_data is None or self.venue_data.empty:
            return 0.5

        venue_matches = self.venue_data[
            (self.venue_data['Team'] == team) &
            (self.venue_data['Ground'] == venue)
        ]

        if len(venue_matches) == 0:
            return 0.5

        # Get weather data
        weather_data = self._get_weather_data(venue)

        # Get pitch conditions
        pitch_data = self._get_pitch_conditions(venue)

        # Calculate adjusted win rate based on conditions
        base_win_rate = venue_matches['win_rate'].iloc[0] if not venue_matches.empty and 'win_rate' in venue_matches.columns else 0.5
        weather_factor = self._calculate_weather_impact(weather_data)
        pitch_factor = self._calculate_pitch_impact(pitch_data, team)

        return base_win_rate * weather_factor * pitch_factor

    def _get_weather_data(self, venue: str) -> Optional[Dict]:
        """Get weather data for venue"""
        try:
            weather_api_url = f"https://api.weatherapi.com/v1/current.json?key={os.environ.get('WEATHER_API_KEY')}&q={venue}"
            response = requests.get(weather_api_url)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            logger.error(f"Error fetching weather data: {str(e)}")
            return None

    def _get_pitch_conditions(self, venue: str) -> Optional[Dict]:
        """Get pitch conditions for venue"""
        try:
            if self.venue_data is not None and not self.venue_data.empty:
                venue_info = self.venue_data[self.venue_data['Ground'] == venue]
                if not venue_info.empty:
                    return {
                        'pitch_type': venue_info['pitch_type'].iloc[0],
                        'bounce': venue_info['bounce_rating'].iloc[0],
                        'turn': venue_info['turn_rating'].iloc[0],
                        'pace': venue_info['pace_rating'].iloc[0]
                    }
            return None
        except Exception as e:
            logger.error(f"Error getting pitch conditions: {str(e)}")
            return None

    def get_model_data(self) -> Tuple[Optional[pd.DataFrame], Optional[np.ndarray]]:
        """Get data for model training"""
        self._ensure_db_connection()
        if not self.conn:
            logger.warning("No database connection available, skipping model data retrieval")
            return None, None
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT t1.name as team, t2.name as opposition,
                           w.name as winner,
                           CASE WHEN m.toss_winner_id = t1.id THEN 'won' ELSE 'lost' END as toss,
                           CASE WHEN m.bat_first THEN '1st' ELSE '2nd' END as bat,
                           m.ground,
                           t1.appearances as team_appearances,
                           t1.titles as team_titles,
                           t1.finals as team_finals,
                           t1.semi_finals as team_semi_finals,
                           t1.current_ranking as team_ranking,
                           t2.appearances as opposition_appearances,
                           t2.titles as opposition_titles,
                           t2.finals as opposition_finals,
                           t2.semi_finals as opposition_semi_finals,
                           t2.current_ranking as opposition_ranking
                    FROM matches m
                    JOIN teams t1 ON m.team1_id = t1.id
                    JOIN teams t2 ON m.team2_id = t2.id
                    JOIN teams w ON m.winner_id = w.id
                """)

                columns = ['Team', 'Opposition', 'winner', 'Toss', 'Bat', 'Ground',
                           'apperance_team', 'Title_team', 'Finals_team', 'Semi finals_team',
                           'Current ranking_team', 'apperance_opposition', 'Title_opposition',
                           'Finals_opposition', 'Semi finals_opposition', 'Current ranking_opposition']

                df = pd.DataFrame(cur.fetchall(), columns=columns)

                # Encode categorical variables
                categorical_columns = ['Team', 'Opposition', 'winner', 'Ground', 'Toss', 'Bat']
                for column in categorical_columns:
                    self.label_encoders[column] = LabelEncoder()
                    df[column] = self.label_encoders[column].fit_transform(df[column].astype(str))

                # Create target variable
                y = (df['Team'] == df['winner']).astype(int)

                return df, y
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            if self.conn:
                self._connect_to_db()
            return None, None

    def _calculate_weather_impact(self, weather_data: Optional[Dict]) -> float:
        # Placeholder for weather impact calculation
        return 1.0

    def _calculate_pitch_impact(self, pitch_data: Optional[Dict], team: str) -> float:
        # Placeholder for pitch impact calculation
        return 1.0