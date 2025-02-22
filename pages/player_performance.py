import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from models.data_processor import DataProcessor
from models.player_stats import PlayerStats
import pandas as pd
import numpy as np

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_player_history(stats: dict, recent_form: dict):
    """Plot player history with caching"""
    fig = plot_player_history(stats, recent_form)
    return fig

def plot_player_history(stats: dict, recent_form: dict):
    """Enhanced plot with advanced metrics"""
    fig = go.Figure()

    # Career stats
    fig.add_trace(go.Bar(
        name='Career Average',
        x=['Batting Average'],
        y=[stats['batting_stats']['average']],
        marker_color='#805ad5'
    ))

    # Add impact score
    fig.add_trace(go.Bar(
        name='Impact Score',
        x=['Impact Score'],
        y=[stats['batting_stats'].get('impact_score', 0)],
        marker_color='#6b46c1'
    ))

    # Add consistency rating
    fig.add_trace(go.Bar(
        name='Consistency Rating',
        x=['Consistency'],
        y=[stats['batting_stats'].get('consistency_rating', 0)],
        marker_color='#9f7aea'
    ))

    if stats.get('role') in ['Bowler', 'All-rounder']:
        fig.add_trace(go.Bar(
            name='Bowling Effectiveness',
            x=['Effectiveness'],
            y=[stats['bowling_stats'].get('effectiveness_score', 0)],
            marker_color='#b794f4'
        ))

        fig.add_trace(go.Bar(
            name='Pressure Rating',
            x=['Pressure'],
            y=[stats['bowling_stats'].get('pressure_rating', 0)],
            marker_color='#e9d8fd'
        ))

    fig.update_layout(
        title='Advanced Performance Metrics',
        barmode='group',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=400,
        showlegend=True
    )

    return fig

@st.cache_data(ttl=300)  # Cache for 5 minutes
def plot_prediction_confidence(prediction: dict):
    """Plot prediction confidence intervals with caching"""
    fig = go.Figure()

    # Batting prediction
    batting_pred = prediction['batting_prediction']
    fig.add_trace(go.Scatter(
        name='Predicted Runs',
        x=['Runs'],
        y=[batting_pred['runs']],
        error_y=dict(
            type='data',
            symmetric=False,
            array=[batting_pred['confidence_interval'][1] - batting_pred['runs']],
            arrayminus=[batting_pred['runs'] - batting_pred['confidence_interval'][0]]
        ),
        marker_color='#805ad5'
    ))

    # Add bowling prediction if available
    if prediction['bowling_prediction']['wickets'] > 0:
        bowling_pred = prediction['bowling_prediction']
        fig.add_trace(go.Scatter(
            name='Predicted Wickets',
            x=['Wickets'],
            y=[bowling_pred['wickets']],
            error_y=dict(
                type='data',
                symmetric=False,
                array=[bowling_pred['confidence_interval'][1] - bowling_pred['wickets']],
                arrayminus=[bowling_pred['wickets'] - bowling_pred['confidence_interval'][0]]
            ),
            marker_color='#6b46c1'
        ))

    fig.update_layout(
        title='Performance Predictions with Confidence Intervals',
        showlegend=True,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=400
    )

    return fig

def render_player_performance():
    st.title("👤 Advanced Player Performance Analytics")

    st.markdown("""
        <div class="stats-card">
            <p>Get detailed player insights powered by advanced machine learning models. 
            Our system analyzes historical performance, current form, opposition strength, 
            and venue conditions to provide comprehensive predictions and performance metrics.</p>
        </div>
    """, unsafe_allow_html=True)

    # Initialize data processor and player stats with error handling
    try:
        dp = DataProcessor()
        if not dp.load_data():
            st.error("Error loading required data. Please check if the data files are present and try again.")
            return
    except Exception as e:
        st.error(f"Error initializing data processor: {str(e)}")
        return

    # Initialize player stats with enhanced prediction capabilities
    try:
        player_stats = PlayerStats()
    except Exception as e:
        st.error(f"Error initializing player statistics: {str(e)}")
        return

    # Get list of teams with proper error handling
    try:
        available_teams = dp.get_all_teams()
        if not available_teams:
            st.error("No teams available for selection. Please check the data files.")
            return
    except Exception as e:
        st.error(f"Error loading team data: {str(e)}")
        return

    st.markdown("""
        <div style='margin-top: 2rem;'>
            <h2 style='color: #2d3748;'>Player Analysis</h2>
        </div>
    """, unsafe_allow_html=True)

    # Team and player selection with error handling
    col1, col2 = st.columns([2, 3])

    with col1:
        try:
            selected_team = st.selectbox(
                "Select Team",
                available_teams,
                key="team_selector"
            )
        except Exception as e:
            st.error("Error loading team selector. Please refresh the page.")
            return

    with col2:
        player_name = st.text_input(
            "Enter Player Name",
            placeholder="Type player name here...",
            help="Enter the full name of the player you want to analyze"
        )

    if player_name:
        try:
            with st.spinner('Analyzing player data...'):
                stats = player_stats.get_player_stats(player_name)

                if stats:
                    # Player Overview section with enhanced metrics
                    st.markdown("""
                        <div class="stats-card">
                            <h3 style='color: #6b46c1; margin-bottom: 1.5rem;'>Advanced Player Metrics</h3>
                    """, unsafe_allow_html=True)

                    col3, col4, col5 = st.columns(3)

                    with col3:
                        st.markdown(f"""
                            <div class="metric-container">
                                <h4 style='color: #4a5568; margin-bottom: 1rem;'>Profile</h4>
                                <p><strong>Role:</strong> {stats.get('role', 'Unknown')}</p>
                                <p><strong>Matches:</strong> {stats.get('matches', 0)}</p>
                                <p><strong>Impact Score:</strong> {stats['batting_stats'].get('impact_score', 0):.2f}</p>
                            </div>
                        """, unsafe_allow_html=True)

                    with col4:
                        st.markdown(f"""
                            <div class="metric-container">
                                <h4 style='color: #4a5568; margin-bottom: 1rem;'>Batting Metrics</h4>
                                <p><strong>Average:</strong> {stats['batting_stats']['average']:.2f}</p>
                                <p><strong>Strike Rate:</strong> {stats['batting_stats']['strike_rate']:.2f}</p>
                                <p><strong>Consistency:</strong> {stats['batting_stats'].get('consistency_rating', 0):.2f}</p>
                            </div>
                        """, unsafe_allow_html=True)

                    with col5:
                        if stats.get('role') in ['Bowler', 'All-rounder']:
                            st.markdown(f"""
                                <div class="metric-container">
                                    <h4 style='color: #4a5568; margin-bottom: 1rem;'>Bowling Metrics</h4>
                                    <p><strong>Average:</strong> {stats['bowling_stats']['average']:.2f}</p>
                                    <p><strong>Economy:</strong> {stats['bowling_stats']['economy']:.2f}</p>
                                    <p><strong>Effectiveness:</strong> {stats['bowling_stats'].get('effectiveness_score', 0):.2f}</p>
                                </div>
                            """, unsafe_allow_html=True)

                    st.markdown("</div>", unsafe_allow_html=True)

                    # Advanced Performance Visualization
                    st.markdown("""
                        <div style='margin-top: 2rem;'>
                            <h2 style='color: #2d3748;'>Performance Analytics</h2>
                        </div>
                    """, unsafe_allow_html=True)

                    st.plotly_chart(get_player_history(stats, {}), use_container_width=True)

                    # Enhanced Prediction Section
                    st.markdown("""
                        <div style='margin-top: 2rem;'>
                            <h2 style='color: #2d3748;'>Advanced Performance Prediction</h2>
                        </div>
                    """, unsafe_allow_html=True)

                    # Additional prediction parameters
                    col6, col7, col8 = st.columns(3)

                    with col6:
                        opposition = st.selectbox(
                            "Select Opposition",
                            [team for team in available_teams if team != selected_team]
                        )

                    with col7:
                        venue = st.selectbox(
                            "Select Venue",
                            ["Home", "Away", "Neutral"]
                        )

                    with col8:
                        match_type = st.selectbox(
                            "Match Type",
                            ["T20", "ODI"]
                        )

                    if st.button("🎯 Generate Advanced Prediction", type="primary", use_container_width=True):
                        try:
                            with st.spinner('Analyzing match conditions and generating predictions...'):
                                prediction = player_stats.predict_performance(
                                    player_name,
                                    opposition,
                                    venue,
                                    match_type
                                )

                                if prediction:
                                    # Display prediction confidence intervals
                                    st.plotly_chart(plot_prediction_confidence(prediction), use_container_width=True)

                                    # Detailed predictions with error handling
                                    col9, col10, col11 = st.columns(3)

                                    with col9:
                                        st.markdown(f"""
                                            <div class="metric-container">
                                                <h4 style='color: #4a5568; margin-bottom: 1rem;'>Batting Prediction</h4>
                                                <p><strong>Predicted Runs:</strong> {prediction['batting_prediction']['runs']} ({prediction['batting_prediction']['confidence_interval'][0]}-{prediction['batting_prediction']['confidence_interval'][1]})</p>
                                                <p><strong>Strike Rate:</strong> {prediction['batting_prediction']['strike_rate_prediction']:.2f}</p>
                                            </div>
                                        """, unsafe_allow_html=True)

                                    with col10:
                                        if stats.get('role') in ['Bowler', 'All-rounder']:
                                            st.markdown(f"""
                                                <div class="metric-container">
                                                    <h4 style='color: #4a5568; margin-bottom: 1rem;'>Bowling Prediction</h4>
                                                    <p><strong>Predicted Wickets:</strong> {prediction['bowling_prediction']['wickets']} ({prediction['bowling_prediction']['confidence_interval'][0]}-{prediction['bowling_prediction']['confidence_interval'][1]})</p>
                                                    <p><strong>Economy Rate:</strong> {prediction['bowling_prediction']['economy_prediction']:.2f}</p>
                                                </div>
                                            """, unsafe_allow_html=True)

                                    with col11:
                                        st.markdown(f"""
                                            <div class="metric-container">
                                                <h4 style='color: #4a5568; margin-bottom: 1rem;'>Match Impact</h4>
                                                <p><strong>Influence Probability:</strong> {prediction['match_influence_probability']:.1%}</p>
                                                <p><strong>Form Factor:</strong> {prediction['form_factor']:.2f}</p>
                                                <p><strong>Confidence:</strong> {prediction['confidence_score']:.1%}</p>
                                            </div>
                                        """, unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"Error generating prediction: {str(e)}")

                else:
                    st.warning("Could not fetch player statistics. Please check the player name and try again.")
        except Exception as e:
            st.error(f"Error analyzing player data: {str(e)}")