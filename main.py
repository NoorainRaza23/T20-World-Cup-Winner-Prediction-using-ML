# Load environment variables first
from dotenv import load_dotenv
import os
import streamlit as st
from utils.config import config
from functools import lru_cache
from models.data_processor import DataProcessor

# Load environment variables before any other imports
load_dotenv(override=True)

# Function definitions moved to top
@lru_cache(maxsize=None)
def get_page_module(page_name: str):
    """Cached import of page modules"""
    try:
        if page_name == "Team Analysis":
            from pages.team_analysis import render_team_analysis
            return render_team_analysis
        elif page_name == "Team Comparison":
            from pages.team_comparison import render_team_comparison
            return render_team_comparison
        elif page_name == "Live Match":
            from pages.live_match import render_live_match
            return render_live_match
        elif page_name == "Tournament Simulation":
            from pages.tournament_simulation import render_tournament_simulation
            return render_tournament_simulation
        elif page_name == "Player Performance":
            from pages.player_performance import render_player_performance
            return render_player_performance
        else:
            from pages.predictions import render_predictions
            return render_predictions
    except Exception as e:
        st.error(f"Error loading page module: {str(e)}")
        return None

# Configure page settings
st.set_page_config(
    page_title="T20 World Cup Predictor | AI-Powered Cricket Analytics",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize DataProcessor with better error handling
@st.cache_resource
def get_data_processor():
    try:
        with st.spinner('Initializing data processor...'):
            dp = DataProcessor()
            return dp
    except Exception as e:
        st.error(f"Error initializing data processor: {str(e)}")
        return None

# Load data with improved error handling
def load_data():
    dp = get_data_processor()
    if dp is None:
        return False

    if not hasattr(st.session_state, 'data_loaded'):
        try:
            with st.spinner('Loading cricket data...'):
                success = dp.load_data()
                if success:
                    st.session_state.data_loaded = True
                    return True
                else:
                    st.error("Error loading data. Please check your internet connection and try again.")
                    return False
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return False
    return st.session_state.data_loaded

# Sidebar content
with st.sidebar:
    # Logo
    st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <span style="font-size: 4rem;">🏏</span>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("### Navigation")

    # Navigation with icons
    page = st.radio(
        "Select Page",
        ["📊 Team Analysis", "🔄 Team Comparison", "🎯 Predictions", 
         "🏆 Tournament Simulation", "⚡ Live Match", "👤 Player Performance"],
        format_func=lambda x: x.split(" ", 1)[1]
    )

    st.markdown("---")
    st.markdown("""
        <div style='margin: 2rem 0; text-align: center;'>
            <div style='margin-bottom: 1rem;'>Made with ❤️ using:</div>
            <div style='font-size: 0.9em; opacity: 0.8; margin-bottom: 0.5rem;'>
                • Streamlit • Scikit-learn<br>
                • Pandas • Plotly<br>
                • XGBoost • PostgreSQL
            </div>
            <div style='margin-top: 2rem; font-size: 0.8em;'>
                © 2024 T20 World Cup Predictor<br>
                <span style='font-size: 0.9em;'>All rights reserved</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

# Main content
st.title("🏏 T20 World Cup Predictor")
st.markdown("#### AI-Powered Cricket Analytics & Predictions")

# Load and render page content with error handling
try:
    # Load data first
    if load_data():
        # Get page name without emoji
        page_name = " ".join(page.split(" ")[1:])

        # Get the page module
        render_page = get_page_module(page_name)

        if render_page:
            # Render selected page
            with st.spinner(f'Loading {page_name}...'):
                render_page()
        else:
            st.warning("This page is under development. Please try another page.")

except Exception as e:
    st.error("An error occurred. Please try refreshing the page.")
    st.button("Refresh", on_click=lambda: st.rerun())

st.markdown("""
    <div style='padding: 2rem 0; margin-top: 4rem; text-align: center; border-radius: 10px;'>
        <p style='font-size: 0.9em; margin-bottom: 0.5rem;'>
            Data updated daily • AI-powered predictions based on historical match data and team statistics
        </p>
        <p style='font-size: 0.8em; margin: 0.5rem 0;'>
            Disclaimer: For educational purposes only • Not affiliated with ICC or any cricket board
        </p>
        <div style='display: flex; justify-content: center; gap: 1rem; margin-top: 1rem;'>
            <a href="#" style='text-decoration: none; font-size: 0.9em;'>Terms of Use</a>
            <a href="#" style='text-decoration: none; font-size: 0.9em;'>Privacy Policy</a>
            <a href="#" style='text-decoration: none; font-size: 0.9em;'>Contact Us</a>
        </div>
    </div>
""", unsafe_allow_html=True)

# Verify required environment variables
required_env_vars = ['CRICINFO_API_KEY', 'DATABASE_URL']
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    st.error(f"Missing required API keys in .env file: {', '.join(missing_vars)}")
    st.stop() # Stop execution if missing env vars