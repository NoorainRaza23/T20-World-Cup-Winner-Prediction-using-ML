import streamlit as st
from models.data_processor import DataProcessor
from utils.visualization import plot_head_to_head, plot_team_form
import pandas as pd

def render_team_comparison():
    st.title("🔄 Team Comparison")

    # Add descriptive text with styled container
    st.markdown("""
        <div class="stats-card">
            <p>Compare any two teams head-to-head using comprehensive statistics, 
            historical performance data, and advanced analytics to understand their relative strengths.</p>
        </div>
    """, unsafe_allow_html=True)

    dp = DataProcessor()
    if not dp.load_data():
        st.error("Error loading data files. Please check if the Excel files are present.")
        return

    # Team selectors with enhanced UI
    col1, col2 = st.columns(2)

    # Get teams list with proper error handling
    teams = []
    if dp.worldcup_data is not None and not dp.worldcup_data.empty:
        teams = sorted(dp.worldcup_data['Team'].dropna().unique().tolist())

    if not teams:
        st.error("No team data available. Please check the data files.")
        return

    with col1:
        st.markdown("""
            <div class="metric-container">
                <h3 style='color: #6b46c1; margin-bottom: 1rem;'>Team 1</h3>
            </div>
        """, unsafe_allow_html=True)
        team1 = st.selectbox(
            "",
            options=teams,
            key='team1'
        )

        # Team 1 stats with enhanced styling
        stats1 = dp.get_team_stats(team1)
        if stats1:
            st.markdown("""
                <div class="metric-container">
                    <div style='display: grid; gap: 1rem;'>
                        <div>
                            <p style='color: #718096; font-size: 0.9rem;'>Current Ranking</p>
                            <p style='font-size: 1.5rem; font-weight: bold; color: #2d3748;'>#{rank}</p>
                        </div>
                        <div>
                            <p style='color: #718096; font-size: 0.9rem;'>World Cup Titles</p>
                            <p style='font-size: 1.5rem; font-weight: bold; color: #2d3748;'>{titles}</p>
                        </div>
                        <div>
                            <p style='color: #718096; font-size: 0.9rem;'>Win Rate</p>
                            <p style='font-size: 1.5rem; font-weight: bold; color: #2d3748;'>{rate:.1%}</p>
                        </div>
                    </div>
                </div>
            """.format(
                rank=stats1.get('Current ranking', 'N/A'),
                titles=stats1.get('Title', 0),
                rate=dp._calculate_recent_form(team1) or 0
            ), unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div class="metric-container">
                <h3 style='color: #6b46c1; margin-bottom: 1rem;'>Team 2</h3>
            </div>
        """, unsafe_allow_html=True)

        # Filter out team1 from options for team2
        team2_options = [t for t in teams if t != team1]
        team2 = st.selectbox(
            "",
            options=team2_options,
            key='team2'
        )

        # Team 2 stats with enhanced styling
        stats2 = dp.get_team_stats(team2)
        if stats2:
            st.markdown("""
                <div class="metric-container">
                    <div style='display: grid; gap: 1rem;'>
                        <div>
                            <p style='color: #718096; font-size: 0.9rem;'>Current Ranking</p>
                            <p style='font-size: 1.5rem; font-weight: bold; color: #2d3748;'>#{rank}</p>
                        </div>
                        <div>
                            <p style='color: #718096; font-size: 0.9rem;'>World Cup Titles</p>
                            <p style='font-size: 1.5rem; font-weight: bold; color: #2d3748;'>{titles}</p>
                        </div>
                        <div>
                            <p style='color: #718096; font-size: 0.9rem;'>Win Rate</p>
                            <p style='font-size: 1.5rem; font-weight: bold; color: #2d3748;'>{rate:.1%}</p>
                        </div>
                    </div>
                </div>
            """.format(
                rank=stats2.get('Current ranking', 'N/A'),
                titles=stats2.get('Title', 0),
                rate=dp._calculate_recent_form(team2) or 0
            ), unsafe_allow_html=True)

    # Head-to-head analysis with enhanced styling
    if team1 and team2 and dp.match_data is not None and not dp.match_data.empty:
        st.markdown("""
            <div style='margin-top: 2rem;'>
                <h2 style='color: #2d3748;'>Head-to-Head Analysis</h2>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='stats-card'>", unsafe_allow_html=True)
        st.plotly_chart(plot_head_to_head(dp.match_data, team1, team2), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Recent form comparison with enhanced styling
        st.markdown("""
            <div style='margin-top: 2rem;'>
                <h2 style='color: #2d3748;'>Recent Form Comparison</h2>
            </div>
        """, unsafe_allow_html=True)

        col3, col4 = st.columns(2)
        with col3:
            st.markdown("""
                <div class="metric-container">
                    <h3 style='color: #6b46c1; margin-bottom: 1rem;'>{team} Recent Form</h3>
                </div>
            """.format(team=team1), unsafe_allow_html=True)
            st.plotly_chart(plot_team_form(dp.match_data, team1), use_container_width=True)

        with col4:
            st.markdown("""
                <div class="metric-container">
                    <h3 style='color: #6b46c1; margin-bottom: 1rem;'>{team} Recent Form</h3>
                </div>
            """.format(team=team2), unsafe_allow_html=True)
            st.plotly_chart(plot_team_form(dp.match_data, team2), use_container_width=True)

        # Common opponents analysis with enhanced styling
        if dp.match_data is not None:
            st.markdown("""
                <div style='margin-top: 2rem;'>
                    <h2 style='color: #2d3748;'>Performance Against Common Opponents</h2>
                </div>
            """, unsafe_allow_html=True)

            team1_opponents = set(dp.match_data[dp.match_data['Team'] == team1]['Opposition'].dropna())
            team2_opponents = set(dp.match_data[dp.match_data['Team'] == team2]['Opposition'].dropna())
            common_opponents = team1_opponents & team2_opponents

            if common_opponents:
                st.markdown("<div class='stats-card'>", unsafe_allow_html=True)
                comparison_data = []
                for opponent in common_opponents:
                    team1_winrate = dp._calculate_head_to_head(team1, opponent)
                    team2_winrate = dp._calculate_head_to_head(team2, opponent)
                    comparison_data.append({
                        'Opponent': opponent,
                        f'{team1} Win Rate': f"{team1_winrate:.1%}",
                        f'{team2} Win Rate': f"{team2_winrate:.1%}"
                    })

                if comparison_data:
                    st.dataframe(
                        pd.DataFrame(comparison_data),
                        use_container_width=True
                    )
                st.markdown("</div>", unsafe_allow_html=True)