import streamlit as st
import pandas as pd
import logging
from models.data_processor import DataProcessor
from models.ml_models import MLModels
from utils.visualization import plot_win_probability, plot_feature_importance

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def render_predictions():
    st.title("T20 World Cup 2026 Predictions")

    # Add descriptive text with high-contrast styling
    st.markdown("""
        <div style='background: #1e293b; padding: 1.5rem; border-radius: 0.75rem; border: 1px solid #475569; margin-bottom: 2rem; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);'>
            <p style='color: #ffffff; margin: 0; font-size: 1.1rem; line-height: 1.6;'>
                Our AI-powered prediction system analyzes historical match data, 
                team statistics, and player performance to predict match outcomes with high accuracy.
            </p>
        </div>
    """, unsafe_allow_html=True)

    dp = DataProcessor()
    if not dp.load_data():
        logger.error("Failed to load data files")
        st.error("Error loading data files. Please check if the Excel files are present.")
        return

    # Preprocess data and train models with error handling
    logger.info("Preprocessing data...")
    X, y = dp.preprocess_data()
    if X is None or y is None:
        logger.error("Failed to preprocess data")
        st.error("Error preprocessing data. Please check the data files and database connection.")
        return

    logger.info(f"Data preprocessed successfully. X shape: {X.shape}, y shape: {y.shape}")

    ml_models = MLModels()
    try:
        logger.info("Training models...")
        models = ml_models.train_models(X, y)
        logger.info(f"Models trained successfully. Number of models: {len(models)}")
    except Exception as e:
        logger.error(f"Error training models: {str(e)}")
        st.error(f"Error training models: {str(e)}")
        return

    # Display model accuracies with enhanced contrast
    st.subheader("🎯 Model Performance")

    cols = st.columns(len(models))
    for idx, (name, model_info) in enumerate(models.items()):
        with cols[idx]:
            st.markdown(f"""
                <div style='background: #1e293b; padding: 1.5rem; border-radius: 0.75rem; border: 1px solid #475569; text-align: center;'>
                    <h3 style='color: #93c5fd; margin-bottom: 0.75rem; font-size: 1.25rem;'>{name}</h3>
                    <p style='font-size: 2rem; font-weight: bold; color: #ffffff; margin: 0.5rem 0;'>
                        {model_info['accuracy']:.1%}
                    </p>
                    <p style='color: #94a3b8; font-size: 0.9rem; margin: 0;'>Accuracy Score</p>
                </div>
            """, unsafe_allow_html=True)

    # Match prediction interface with enhanced contrast
    st.markdown("""
        <div style='margin-top: 3rem;'>
            <h2 style='color: #ffffff; font-size: 1.75rem; margin-bottom: 1.5rem;'>
                🏏 Predict Match Winner
            </h2>
        </div>
    """, unsafe_allow_html=True)

    if dp.match_data is not None:
        available_teams = sorted(list(set(dp.match_data['Team'].unique()) | set(dp.match_data['Opposition'].unique())))
        logger.info(f"Available teams: {len(available_teams)}")

        # Team selection with improved contrast
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
                <div style='padding: 1rem 0;'>
                    <p style='color: #e2e8f0; font-weight: 600; margin-bottom: 0.5rem;'>Select Team 1</p>
                </div>
            """, unsafe_allow_html=True)
            team1 = st.selectbox("", available_teams, key="team1_select", label_visibility="collapsed")

        with col2:
            st.markdown("""
                <div style='padding: 1rem 0;'>
                    <p style='color: #e2e8f0; font-weight: 600; margin-bottom: 0.5rem;'>Select Team 2</p>
                </div>
            """, unsafe_allow_html=True)
            team2 = st.selectbox("", [t for t in available_teams if t != team1], key="team2_select", label_visibility="collapsed")

        # Predict button with enhanced contrast
        if st.button("🎯 Predict Winner", type="primary", use_container_width=True):
            if team1 != team2:
                with st.spinner('Analyzing match data...'):
                    logger.info(f"Getting stats for teams: {team1} vs {team2}")
                    team1_stats = dp.get_team_stats(team1)
                    team2_stats = dp.get_team_stats(team2)

                    if team1_stats is None or team2_stats is None:
                        logger.error(f"Failed to get team stats for {team1} or {team2}")
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

                        logger.info("Match data created successfully")
                        match_data = match_data[X.columns]

                        # Transform categorical variables
                        for col, le in dp.label_encoders.items():
                            if col in match_data.columns:
                                match_data[col] = le.transform(match_data[col].astype(str))

                        logger.info("Making predictions...")
                        predictions = ml_models.predict_match(match_data)
                        logger.info(f"Predictions generated: {predictions}")

                        # Display predictions with enhanced contrast
                        st.markdown("""
                            <div style='background: #1e293b; padding: 1.5rem; border-radius: 0.75rem; border: 1px solid #475569; margin: 2rem 0; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);'>
                        """, unsafe_allow_html=True)
                        fig = plot_win_probability(predictions, team1, team2)
                        if fig is not None:
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            logger.error("Failed to create win probability plot")
                            st.error("Error creating visualization")
                        st.markdown("</div>", unsafe_allow_html=True)

                        # Feature importance with enhanced contrast
                        st.subheader("📊 Feature Importance Analysis")
                        importance = ml_models.get_feature_importance()
                        if importance is not None:
                            st.plotly_chart(plot_feature_importance(X.columns, importance), use_container_width=True)
                        else:
                            logger.error("Failed to get feature importance")

                    except Exception as e:
                        logger.error(f"Error making prediction: {str(e)}")
                        st.error(f"Error making prediction: {str(e)}")
            else:
                st.error("Please select different teams")
    else:
        logger.error("No match data available")
        st.error("No match data available")