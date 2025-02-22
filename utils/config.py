import os
from typing import Optional
import logging
from dotenv import load_dotenv
import sys

logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv(override=True)

class Config:
    """Configuration management for API keys and secrets"""

    def __init__(self):
        # API Keys
        self.cricinfo_api_key: Optional[str] = os.environ.get('CRICINFO_API_KEY')
        if not self.cricinfo_api_key:
            logger.warning("CRICINFO_API_KEY not found in .env file")

        self.weather_api_key: Optional[str] = os.environ.get('WEATHER_API_KEY')


        # Database Configuration
        self.database_url: Optional[str] = os.environ.get('DATABASE_URL')

        # Live Match Settings
        self.live_match_refresh_interval: int = int(os.environ.get('LIVE_MATCH_REFRESH_INTERVAL', 30))
        self.cache_duration: int = int(os.environ.get('CACHE_DURATION', 3600))

        # Application Settings
        self.debug_mode: bool = os.environ.get('DEBUG', 'False').lower() == 'true'
        self.environment: str = os.environ.get('ENVIRONMENT', 'development')

    def validate_required_keys(self) -> bool:
        """Validate that all required API keys are present"""
        missing_keys = []

        # Check required keys
        if not self.cricinfo_api_key:
            missing_keys.append('CRICINFO_API_KEY')

        if missing_keys:
            logger.warning(f"Missing required API keys in .env file: {', '.join(missing_keys)}")
            return False
        return True

    def get_database_url(self) -> Optional[str]:
        """Get database URL with fallback to SQLite for local development"""
        if self.database_url:
            return self.database_url

        logger.info("No DATABASE_URL found in .env file, using file-based data")
        return None

    def get_live_match_settings(self) -> dict:
        """Get live match configuration settings from .env"""
        return {
            'refresh_interval': self.live_match_refresh_interval,
            'cache_duration': self.cache_duration
        }

    @staticmethod
    def setup_local_env():
        """Setup instructions for local environment"""
        print("""
        To set up your local environment:

        1. Copy .env.template to .env:
           cp .env.template .env

        2. Edit .env and add your API keys:
           CRICINFO_API_KEY=your_api_key_here

        3. Optional: Adjust other settings in .env:
           LIVE_MATCH_REFRESH_INTERVAL=30
           CACHE_DURATION=3600
           DEBUG=True
           ENVIRONMENT=development

        Note: Never commit your .env file to version control
        """)

# Create a global config instance
config = Config()

# Validate configuration on import
if not config.validate_required_keys():
    logger.warning("Please check your .env file configuration")