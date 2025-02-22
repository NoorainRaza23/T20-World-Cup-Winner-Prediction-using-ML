import streamlit as st
from models.data_processor import DataProcessor
from models.ml_models import MLModels
from models.tournament_simulator import TournamentSimulator
import plotly.graph_objects as go
import pandas as pd

def render_tournament_simulation():
    st.title("🏆 Tournament Simulation")

    # Description card
    st.markdown("""
        <div class="stats-card">
            <p>Simulate entire T20 World Cup tournaments with our advanced AI engine. 
            Select teams for each group and watch as our system predicts the outcomes 
            of every match through to the final.</p>
        </div>
    """, unsafe_allow_html=True)

    dp = DataProcessor()
    if not dp.load_data():
        st.error("Error loading data files. Please check if the data files are present in the attached_assets folder.")
        return

    # Train models with loading indicator
    with st.spinner('Preparing simulation engine...'):
        try:
            X, y = dp.preprocess_data()
            if X is None or y is None:
                st.error("Error preparing data for simulation. Please check the data files.")
                return

            ml_models = MLModels()
            ml_models.train_models(X, y)

            # Initialize simulator
            simulator = TournamentSimulator(dp, ml_models)
        except Exception as e:
            st.error(f"Error initializing simulation engine: {str(e)}")
            return

    # Group setup with enhanced UI
    st.markdown("""
        <div style='margin-top: 2rem;'>
            <h2 style='color: #2d3748;'>Group Setup</h2>
            <p style='color: #718096;'>Select at least 4 teams for each group to begin simulation</p>
        </div>
    """, unsafe_allow_html=True)

    # Get available teams
    available_teams = dp.get_all_teams()
    if not available_teams:
        st.error("No teams available for selection. Please check the data files.")
        return

    groups = {'A': [], 'B': []}
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
            <div class="metric-container">
                <h3 style='color: #6b46c1; margin-bottom: 1rem;'>Group A</h3>
            </div>
        """, unsafe_allow_html=True)
        groups['A'] = st.multiselect(
            "Select Teams for Group A",
            options=available_teams,
            key='groupA'
        )

    with col2:
        st.markdown("""
            <div class="metric-container">
                <h3 style='color: #6b46c1; margin-bottom: 1rem;'>Group B</h3>
            </div>
        """, unsafe_allow_html=True)
        remaining_teams = [t for t in available_teams if t not in groups['A']]
        groups['B'] = st.multiselect(
            "Select Teams for Group B",
            options=remaining_teams,
            key='groupB'
        )

    # Simulate button with enhanced styling
    if st.button("🎮 Simulate Tournament", type="primary", use_container_width=True):
        if all(len(teams) >= 4 for teams in groups.values()):
            with st.spinner('Simulating tournament matches...'):
                results = simulator.simulate_tournament(groups)

                # Display group standings in a grid
                st.markdown("""
                    <div style='margin-top: 2rem;'>
                        <h2 style='color: #2d3748;'>Group Stage Results</h2>
                    </div>
                """, unsafe_allow_html=True)

                col3, col4 = st.columns(2)

                with col3:
                    st.markdown("""
                        <div class="metric-container">
                            <h3 style='color: #6b46c1;'>Group A Standings</h3>
                        </div>
                    """, unsafe_allow_html=True)
                    standings_data = results['group_standings']['A']
                    fig1 = go.Figure(data=[
                        go.Bar(
                            x=[team for team, _ in standings_data],
                            y=[points for _, points in standings_data],
                            marker_color='#805ad5'
                        )
                    ])
                    fig1.update_layout(
                        title="Group A Points",
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        height=400
                    )
                    st.plotly_chart(fig1, use_container_width=True)

                with col4:
                    st.markdown("""
                        <div class="metric-container">
                            <h3 style='color: #6b46c1;'>Group B Standings</h3>
                        </div>
                    """, unsafe_allow_html=True)
                    standings_data = results['group_standings']['B']
                    fig2 = go.Figure(data=[
                        go.Bar(
                            x=[team for team, _ in standings_data],
                            y=[points for _, points in standings_data],
                            marker_color='#805ad5'
                        )
                    ])
                    fig2.update_layout(
                        title="Group B Points",
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        height=400
                    )
                    st.plotly_chart(fig2, use_container_width=True)

                # Display knockout results in styled containers
                st.markdown("""
                    <div style='margin-top: 2rem;'>
                        <h2 style='color: #2d3748;'>Knockout Stage Results</h2>
                    </div>
                """, unsafe_allow_html=True)

                # Semi-finals
                st.markdown("""
                    <div class="metric-container">
                        <h3 style='color: #6b46c1; margin-bottom: 1rem;'>Semi-finals</h3>
                    </div>
                """, unsafe_allow_html=True)

                for sf in results['knockout_results']['semi_finals']:
                    st.markdown(f"""
                        <div style='background: white; padding: 1rem; border-radius: 8px; margin: 0.5rem 0; 
                                    box-shadow: 0 2px 8px rgba(107, 70, 193, 0.05);'>
                            <p style='color: #4a5568; font-size: 1.1em;'>
                                {sf[0]} vs {sf[1]} - <span style='color: #6b46c1; font-weight: 600;'>
                                Winner: {sf[2]}</span>
                            </p>
                        </div>
                    """, unsafe_allow_html=True)

                # Final
                final = results['knockout_results']['final']
                st.markdown("""
                    <div class="metric-container">
                        <h3 style='color: #6b46c1; margin-bottom: 1rem;'>Final</h3>
                    </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                    <div style='background: white; padding: 1.5rem; border-radius: 8px; margin: 0.5rem 0;
                                box-shadow: 0 2px 8px rgba(107, 70, 193, 0.05);'>
                        <p style='color: #4a5568; font-size: 1.2em; text-align: center;'>
                            {final[0]} vs {final[1]}
                        </p>
                        <p style='color: #6b46c1; font-weight: 600; font-size: 1.4em; text-align: center;'>
                            Winner: {final[2]}
                        </p>
                    </div>
                """, unsafe_allow_html=True)

                # Tournament Winner announcement
                st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #6b46c1, #805ad5);
                                padding: 2rem; border-radius: 12px; margin: 2rem 0; text-align: center;
                                box-shadow: 0 4px 20px rgba(107, 70, 193, 0.2);'>
                        <h2 style='color: white; margin: 0;'>🏆 Tournament Winner</h2>
                        <p style='color: white; font-size: 2em; font-weight: 700; margin: 1rem 0;'>
                            {results['knockout_results']['winner']}
                        </p>
                    </div>
                """, unsafe_allow_html=True)

        else:
            st.error("Please select at least 4 teams for each group")