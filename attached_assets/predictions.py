import streamlit as st
import pandas as pd
from models.data_processor import DataProcessor
from models.ml_models import MLModels
from utils.visualization import plot_win_probability, plot_feature_importance

def render_predictions():
    st.title("T20 World Cup 2026 Predictions")

    # Add descriptive text with styled container
    st.markdown("""
        <div class="stats-card">
            <p>Our AI-powered prediction system analyzes historical match data, 
            team statistics, and player performance to predict match outcomes with high accuracy.</p>
        </div>
    """, unsafe_allow_html=True)

    dp = DataProcessor()
    if not dp.load_data():
        st.error("Error loading data files. Please check if the Excel files are present.")
        return

    # Preprocess data and train models
    X, y = dp.preprocess_data()
    if X is None or y is None:
        st.error("Error preprocessing data. Please check the data files and database connection.")
        return

    ml_models = MLModels()
    try:
        models = ml_models.train_models(X, y)
    except Exception as e:
        st.error(f"Error training models: {str(e)}")
        return

    # Display model accuracies in a grid
    st.subheader("🎯 Model Performance")

    # Create a grid container for model metrics
    cols = st.columns(len(models))
    for idx, (name, model_info) in enumerate(models.items()):
        with cols[idx]:
            st.markdown(f"""
                <div class="metric-container" style="text-align: center;">
                    <h3 style="color: #6b46c1; margin-bottom: 0.5rem;">{name}</h3>
                    <p style="font-size: 1.5rem; font-weight: bold; color: #2d3748;">
                        {model_info['accuracy']:.1%}
                    </p>
                    <p style="color: #718096; font-size: 0.9rem;">Accuracy Score</p>
                </div>
            """, unsafe_allow_html=True)

    # Match prediction interface with enhanced styling
    st.markdown("""
        <div style='margin-top: 2rem;'>
            <h2 style='color: #2d3748;'>🏏 Predict Match Winner</h2>
        </div>
    """, unsafe_allow_html=True)

    if dp.match_data is not None:
        available_teams = sorted(list(set(dp.match_data['Team'].unique()) | set(dp.match_data['Opposition'].unique())))

        # Team selection with improved layout
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
                <div style='padding: 1rem 0;'>
                    <p style='color: #4a5568; font-weight: 600;'>Select Team 1</p>
                </div>
            """, unsafe_allow_html=True)
            team1 = st.selectbox("", available_teams, key="team1_select", label_visibility="collapsed")

        with col2:
            st.markdown("""
                <div style='padding: 1rem 0;'>
                    <p style='color: #4a5568; font-weight: 600;'>Select Team 2</p>
                </div>
            """, unsafe_allow_html=True)
            team2 = st.selectbox("", [t for t in available_teams if t != team1], key="team2_select", label_visibility="collapsed")

        # Predict button with enhanced styling
        if st.button("🎯 Predict Winner", type="primary", use_container_width=True):
            if team1 != team2:
                # Prepare match data with loading spinner
                with st.spinner('Analyzing match data...'):
                    team1_stats = dp.get_team_stats(team1)
                    team2_stats = dp.get_team_stats(team2)

                    if team1_stats is None or team2_stats is None:
                        st.error("Error getting team statistics")
                        return

                    try:
                        match_data = pd.DataFrame([{
                            'Team': team1,
                            'Opposition': team2,
                            'winner': team1,  # placeholder
                            'Ground': dp.match_data['Ground'].iloc[0],
                            'Toss': 'won',
                            'Bat': '1st',
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
                        }])

                        # Ensure columns match training data
                        match_data = match_data[X.columns]

                        # Transform categorical variables
                        for col, le in dp.label_encoders.items():
                            if col in match_data.columns:
                                match_data[col] = le.transform(match_data[col].astype(str))

                        predictions = ml_models.predict_match(match_data)

                        # Display predictions with enhanced visuals
                        st.markdown("<div class='stats-card'>", unsafe_allow_html=True)
                        st.plotly_chart(plot_win_probability(predictions, team1, team2), use_container_width=True)
                        st.markdown("</div>", unsafe_allow_html=True)

                        # Feature importance
                        st.subheader("📊 Feature Importance Analysis")
                        importance = ml_models.get_feature_importance()
                        if importance is not None:
                            st.plotly_chart(plot_feature_importance(X.columns, importance), use_container_width=True)

                    except Exception as e:
                        st.error(f"Error making prediction: {str(e)}")
            else:
                st.error("Please select different teams")
    else:
        st.error("No match data available")