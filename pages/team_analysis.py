import streamlit as st
import logging
from models.data_processor import DataProcessor
from utils.visualization import plot_team_performance, plot_head_to_head
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def render_team_analysis():
    try:
        # Container for better mobile layout
        with st.container():
            st.title("🏏 Team Analysis")

            # Description card with improved mobile layout
            st.markdown("""
                <div class="stats-card">
                    <p style="margin: 0;">Explore comprehensive team statistics, historical performance data, 
                    and head-to-head analysis to gain deeper insights into each team's strengths 
                    and potential tournament outcomes.</p>
                </div>
            """, unsafe_allow_html=True)

            dp = DataProcessor()
            logger.info("DataProcessor instance created")

            if not dp.load_data():
                st.error("Error loading data files. Please check if the data files are present in the attached_assets folder.")
                logger.error("Failed to load data files")
                return

            logger.info("Data loaded successfully")

            # Team selector with enhanced mobile UI
            st.markdown("""
                <div style='margin: 2rem 0 1rem;'>
                    <h2 style='color: #2d3748; font-size: calc(1.2rem + 0.5vw);'>Team Selection</h2>
                </div>
            """, unsafe_allow_html=True)

            teams = sorted(dp.get_all_teams())
            logger.info(f"Retrieved {len(teams)} teams")

            if not teams:
                st.error("No valid teams found in the data.")
                return

            team = st.selectbox(
                "Choose a team to analyze",
                options=teams,
                key="team_selector"
            )

            if team:
                logger.info(f"Selected team: {team}")
                stats = dp.get_team_stats(team)
                if stats:
                    st.markdown("""
                        <div class="stats-card">
                            <h3 style='color: #6b46c1; margin-bottom: 1.5rem; 
                                    font-size: calc(1.1rem + 0.3vw);'>Team Statistics</h3>
                        </div>
                    """, unsafe_allow_html=True)

                    # Responsive metrics grid
                    for i in range(0, 4, 2):  # Create 2 rows for mobile
                        col1, col2 = st.columns(2)

                        with col1:
                            if i == 0:
                                st.metric("Current Ranking", f"#{stats['Current ranking']}")
                                st.metric("World Cup Appearances", stats['apperance'])
                            else:
                                st.metric("Semi-Finals", stats['Semi finals'])
                                st.metric("Group", stats['group'])

                        with col2:
                            if i == 0:
                                st.metric("Titles Won", stats['Title'])
                                st.metric("Finals Appearances", stats['Finals'])
                            else:
                                recent_matches = dp.get_recent_matches(team, n=100)
                                if not recent_matches.empty and 'winner' in recent_matches.columns:
                                    wins = len(recent_matches[recent_matches['winner'] == team])
                                    total = len(recent_matches)
                                    if total > 0:
                                        win_percentage = (wins / total) * 100
                                        st.metric("Recent Win Rate", f"{win_percentage:.1f}%")

                    # Recent matches section with responsive table
                    st.markdown("""
                        <div style='margin: 2rem 0 1rem;'>
                            <h2 style='color: #2d3748; font-size: calc(1.2rem + 0.5vw);'>Recent Matches</h2>
                        </div>
                    """, unsafe_allow_html=True)

                    recent_matches = dp.get_recent_matches(team)
                    if not recent_matches.empty:
                        with st.container():
                            display_columns = ['Team', 'Opposition', 'Ground', 'Start Date']
                            if 'winner' in recent_matches.columns:
                                display_columns.insert(2, 'winner')

                            display_df = recent_matches[display_columns].copy()
                            st.dataframe(
                                display_df,
                                use_container_width=True,
                                height=min(len(display_df) * 35 + 38, 400)  # Responsive height
                            )

                    # Head-to-head analysis with responsive layout
                    st.markdown("""
                        <div style='margin: 2rem 0 1rem;'>
                            <h2 style='color: #2d3748; font-size: calc(1.2rem + 0.5vw);'>Head-to-Head Analysis</h2>
                        </div>
                    """, unsafe_allow_html=True)

                    opponents = [t for t in teams if t != team]
                    if opponents:
                        opponent = st.selectbox(
                            "Select an opponent",
                            options=opponents,
                            key="opponent_selector"
                        )

                        if opponent:
                            matches = dp.get_recent_matches(team, n=100)
                            if not matches.empty and 'winner' in matches.columns:
                                try:
                                    with st.container():
                                        fig = plot_head_to_head(matches, team, opponent)
                                        st.plotly_chart(fig, use_container_width=True)
                                except Exception as e:
                                    logger.error(f"Error plotting head-to-head: {str(e)}")
                                    st.error("Error generating head-to-head analysis")

                else:
                    st.error(f"Could not retrieve statistics for {team}")
                    logger.error(f"Failed to get stats for team: {team}")

    except Exception as e:
        logger.error(f"Error in render_team_analysis: {str(e)}")
        st.error("An error occurred while loading the team analysis. Please try again.")