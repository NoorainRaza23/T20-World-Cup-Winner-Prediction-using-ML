import os
import psycopg2
from psycopg2.extras import RealDictCursor
import logging

logger = logging.getLogger(__name__)

def get_db_connection():
    """Create a database connection"""
    try:
        conn = psycopg2.connect(os.environ['DATABASE_URL'])
        return conn
    except Exception as e:
        logger.error(f"Error connecting to database: {e}")
        raise

def execute_query(query, params=None):
    """Execute a query and return results"""
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, params)
            if query.strip().upper().startswith('SELECT'):
                return cur.fetchall()
            conn.commit()
    except Exception as e:
        logger.error(f"Error executing query: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()

def get_all_teams():
    """Get all teams from the database"""
    query = "SELECT * FROM teams ORDER BY ranking"
    return execute_query(query)

def get_team_stats(team_id):
    """Get detailed stats for a specific team"""
    query = """
    SELECT t.*, 
           COUNT(DISTINCT p.id) as player_count,
           COUNT(DISTINCT m.id) as total_matches
    FROM teams t
    LEFT JOIN players p ON p.team_id = t.id
    LEFT JOIN matches m ON m.team1_id = t.id OR m.team2_id = t.id
    WHERE t.id = %s
    GROUP BY t.id
    """
    results = execute_query(query, (team_id,))
    return results[0] if results else None

def add_prediction(match_id, predicted_winner_id, confidence_score):
    """Add a new prediction"""
    query = """
    INSERT INTO predictions (match_id, predicted_winner_id, confidence_score)
    VALUES (%s, %s, %s)
    RETURNING id
    """
    result = execute_query(query, (match_id, predicted_winner_id, confidence_score))
    return result[0]['id'] if result else None
