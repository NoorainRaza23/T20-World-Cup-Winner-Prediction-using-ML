import streamlit as st
import time
import plotly.graph_objects as go
from datetime import datetime
from models.data_processor import DataProcessor
from models.ml_models import MLModels
from models.live_match_processor import LiveMatchProcessor
from models.real_time_collector import RealTimeDataCollector
from utils.config import config

def plot_prediction_history(prediction_history):
    """Plot the prediction history over time with enhanced cricket metrics"""
    if not prediction_history:
        return None

    timestamps = [entry['timestamp'] for entry in prediction_history]
    team1_probs = [entry['predictions']['ensemble']['win_probability'] for entry in prediction_history]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=team1_probs,
        mode='lines+markers',
        name='Win Probability',
        line=dict(color='#6b46c1', width=2),
        marker=dict(size=8)
    ))

    fig.update_layout(
        title='Win Probability Tracker',
        xaxis_title='Match Progression',
        yaxis_title='Win Probability',
        yaxis=dict(
            range=[0, 1],
            tickformat='.0%',
            gridcolor='#f0f0f0'
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig

@st.cache_data(ttl=30)
def fetch_live_matches():
    """Cached wrapper for fetching live matches"""
    try:
        collector = st.session_state.real_time_collector
        if not collector:
            return None
        return collector.get_live_matches()
    except Exception as e:
        st.error(f"Error fetching live matches: {str(e)}")
        return None

@st.cache_data(ttl=60)
def fetch_match_details(match_id: str):
    """Cached wrapper for fetching match details"""
    try:
        collector = st.session_state.real_time_collector
        if not collector:
            return None
        return collector.get_match_details(match_id)
    except Exception as e:
        st.error(f"Error fetching match details: {str(e)}")
        return None

def render_live_match():
    """Main function to render the live match interface"""
    st.title("⚡ Live Match Predictions")

    try:
        # Initialize session state
        if 'live_processor' not in st.session_state:
            with st.spinner("Initializing match predictor..."):
                dp = DataProcessor()
                if not dp.load_data():
                    st.error("❌ Error loading match data files")
                    return

                # Add loading indicator for data preprocessing
                with st.spinner("Preprocessing training data..."):
                    X, y = dp.preprocess_data()
                    if X is None or y is None:
                        st.error("❌ Error preprocessing training data")
                        return

                # Add loading indicator for model training
                with st.spinner("Training prediction models..."):
                    ml_models = MLModels()
                    ml_models.train_models(X, y)

                    st.session_state.live_processor = LiveMatchProcessor(dp, ml_models)
                    st.session_state.real_time_collector = RealTimeDataCollector()
                    st.session_state.match_started = False

    except Exception as e:
        st.error(f"❌ Error initializing models: {str(e)}")
        st.exception(e)
        return

    # Custom CSS for better contrast
    st.markdown("""
        <style>
        .team-header {
            color: #ffffff;
            background-color: #1e293b;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
            text-align: center;
        }
        .score-display {
            font-size: 2.5rem;
            font-weight: bold;
            color: #f8fafc;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
        }
        .metric-card {
            background-color: #334155;
            padding: 1rem;
            border-radius: 8px;
            margin: 0.5rem 0;
        }
        .metric-label {
            color: #94a3b8;
            font-size: 0.9rem;
            margin-bottom: 0.25rem;
        }
        .metric-value {
            color: #f1f5f9;
            font-size: 1.5rem;
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)

    # Create tabs with enhanced styling
    tab1, tab2 = st.tabs(["📊 Manual Score Entry", "🔴 Live Match Data"])

    with tab1:
        render_manual_match()

    with tab2:
        render_live_data_match()

def render_live_data_match():
    """Enhanced live match data interface with better loading states"""
    st.subheader("🔴 Live Matches")

    if not hasattr(st.session_state, 'real_time_collector') or not st.session_state.real_time_collector:
        st.error("❌ Real-time collector not initialized")
        return

    if not config.cricinfo_api_key:
        st.error("❌ CricAPI key not set. Live match data is unavailable.")
        st.info("""
            To enable live match data:
            1. Sign up for a CricAPI account
            2. Get your API key
            3. Add it to your .env file as CRICINFO_API_KEY
        """)
        return

    # Fetch live matches with loading state
    with st.spinner("Fetching live matches..."):
        live_matches = fetch_live_matches()
        if live_matches is None:
            st.error("❌ Error fetching live matches")
            if st.button("🔄 Retry", key='retry_fetch'):
                st.experimental_rerun()
            return

    if not live_matches:
        st.info("ℹ️ No live matches available at the moment.")
        if st.button("🔄 Refresh", key='refresh_matches'):
            st.experimental_rerun()
        return

    # Display available live matches with enhanced UI
    match_options = {f"{m['team1']} vs {m['team2']}": m['id'] for m in live_matches}
    selected_match = st.selectbox(
        "Select Live Match",
        list(match_options.keys()),
        format_func=lambda x: "🏏 " + x
    )

    if selected_match:
        match_id = match_options[selected_match]

        # Add loading state for match details
        with st.spinner("Loading match details..."):
            match_details = fetch_match_details(match_id)
            if match_details:
                display_match_details(match_details)
            else:
                st.warning("⚠️ Unable to fetch match details. Please try again.")
                if st.button("🔄 Retry", key='retry_details'):
                    st.experimental_rerun()

def display_match_details(match_details):
    # Match header with enhanced contrast
    st.markdown(f"""
        <div class="team-header">
            <h2 style="color: #f8fafc;">{match_details['team1']['name']} vs {match_details['team2']['name']}</h2>
        </div>
    """, unsafe_allow_html=True)

    # Score display with better contrast
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Score</div>
                <div class="score-display">{match_details['team1']['score']}</div>
                <div class="metric-label">Run Rate: {match_details['team1'].get('run_rate', 0):.2f}</div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Score</div>
                <div class="score-display">{match_details['team2']['score']}</div>
                <div class="metric-label">Run Rate: {match_details['team2'].get('run_rate', 0):.2f}</div>
            </div>
        """, unsafe_allow_html=True)

    # Match status with improved visibility
    st.markdown(f"""
        <div class="metric-card" style="text-align: center;">
            <div class="metric-label">Match Status</div>
            <div class="metric-value">{match_details['status']}</div>
        </div>
    """, unsafe_allow_html=True)

    # Get live score updates
    live_score = st.session_state.real_time_collector.get_live_score(match_details['id'])
    if live_score:
        display_live_score(live_score)


def render_manual_match():
    """Enhanced manual score entry interface with cricket-specific metrics"""
    if not st.session_state.match_started:
        st.subheader("Start New Match")

        # Team selection with enhanced UI
        available_teams = sorted(list(set(st.session_state.live_processor.dp.match_data['Team'].unique())))
        col1, col2 = st.columns(2)

        with col1:
            team1 = st.selectbox(
                "🏏 Select Team 1",
                available_teams,
                key='manual_team1'
            )

        with col2:
            team2 = st.selectbox(
                "🏏 Select Team 2", 
                [t for t in available_teams if t != team1],
                key='manual_team2'
            )

        if st.button("🎯 Start Match", key='manual_start', use_container_width=True):
            if st.session_state.live_processor.start_match(team1, team2):
                st.session_state.match_started = True
                st.experimental_rerun()

    else:
        current_match = st.session_state.live_processor.current_match
        if not current_match:
            st.error("❌ No active match")
            return

        # Match header with team logos
        st.markdown(f"""
            <div style='text-align: center; padding: 1rem; background: white; border-radius: 10px; margin-bottom: 2rem;'>
                <h2>{current_match['team1']} 🏏 vs 🏏 {current_match['team2']}</h2>
            </div>
        """, unsafe_allow_html=True)

        # Match status input with enhanced layout
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"### {current_match['team1']} Status")
            team1_score = st.number_input("Runs", key='team1_score', min_value=0, help="Total runs scored")
            team1_overs = st.number_input("Overs", key='team1_overs', min_value=0.0, max_value=20.0, step=0.1, help="Overs completed (max 20)")
            team1_wickets = st.number_input("Wickets", key='team1_wickets', min_value=0, max_value=10, help="Wickets fallen")

            # Additional cricket metrics
            if team1_overs > 0:
                run_rate = team1_score / team1_overs
                st.metric("Current Run Rate", f"{run_rate:.2f}")

        with col2:
            st.markdown(f"### {current_match['team2']} Status")
            team2_score = st.number_input("Runs", key='team2_score', min_value=0, help="Total runs scored")
            team2_overs = st.number_input("Overs", key='team2_overs', min_value=0.0, max_value=20.0, step=0.1, help="Overs completed (max 20)")
            team2_wickets = st.number_input("Wickets", key='team2_wickets', min_value=0, max_value=10, help="Wickets fallen")

            # Additional cricket metrics
            if team2_overs > 0:
                run_rate = team2_score / team2_overs
                st.metric("Current Run Rate", f"{run_rate:.2f}")

        # Update button with loading state
        if st.button("🔄 Update Match Status", key='manual_update', use_container_width=True):
            with st.spinner("Updating predictions..."):
                try:
                    prediction = st.session_state.live_processor.update_match_status(
                        team1_score, team1_overs, team1_wickets,
                        team2_score, team2_overs, team2_wickets
                    )

                    if prediction:
                        st.success("✅ Match status updated successfully")
                        display_predictions(prediction, current_match)
                    else:
                        st.warning("⚠️ No prediction generated")
                except Exception as e:
                    st.error(f"❌ Error updating match status: {str(e)}")

        if st.button("🏁 End Match", key='manual_end', use_container_width=True):
            st.session_state.match_started = False
            st.experimental_rerun()

def display_predictions(prediction, current_match):
    """Enhanced prediction display with detailed analytics"""
    st.markdown("""
        <div style='margin: 2rem 0;'>
            <h3 style='color: #2d3748;'>Match Predictions</h3>
        </div>
    """, unsafe_allow_html=True)

    # Create metrics grid
    cols = st.columns(len(prediction['predictions']))
    for col, (model_name, pred) in zip(cols, prediction['predictions'].items()):
        with col:
            st.markdown(f"""
                <div class="metric-container" style="text-align: center;">
                    <p style="color: #718096; font-size: 0.9rem;">{model_name.title()} Model</p>
                    <p style="font-size: 1.8rem; font-weight: bold; color: #2d3748;">
                        {pred['win_probability']:.1%}
                    </p>
                    <p style="color: #718096;">Win probability for {current_match['team1']}</p>
                </div>
            """, unsafe_allow_html=True)

    # Plot prediction history
    st.markdown("""
        <div style='margin: 2rem 0;'>
            <h3 style='color: #2d3748;'>Prediction Trend</h3>
        </div>
    """, unsafe_allow_html=True)

    history = st.session_state.live_processor.get_prediction_history()
    if history:
        fig = plot_prediction_history(history)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("ℹ️ Not enough data points for prediction trend")

def display_live_score(live_score):
    """Enhanced live score display with better contrast"""
    st.markdown("""
        <div class="team-header">
            <h3 style="color: #f8fafc;">Live Score Details</h3>
        </div>
    """, unsafe_allow_html=True)

    # Main metrics with improved visibility
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Current Score</div>
                <div class="metric-value">{live_score['current_score']}</div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Overs</div>
                <div class="metric-value">{live_score['current_overs']}</div>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Run Rate</div>
                <div class="metric-value">{live_score['run_rate']:.2f}</div>
            </div>
        """, unsafe_allow_html=True)

    # Required rate if available
    if live_score.get('required_rate'):
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Required Run Rate</div>
                <div class="metric-value">{live_score['required_rate']:.2f}</div>
            </div>
        """, unsafe_allow_html=True)