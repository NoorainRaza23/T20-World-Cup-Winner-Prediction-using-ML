import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

def plot_win_probability(predictions, team1, team2):
    """Create an interactive bar chart showing win probabilities"""
    try:
        fig = go.Figure()
        colors = {
            'Logistic Regression': ['#6b46c1', '#f59e0b'],  # Purple & Orange
            'Random Forest': ['#10b981', '#3b82f6'],        # Green & Blue
            'Gradient Boosting': ['#ef4444', '#8b5cf6'],    # Red & Purple
            'XGBoost': ['#fb923c', '#22c55e'],             # Light Orange & Green
            'Ensemble': ['#4f46e5', '#ec4899']             # Indigo & Pink
        }

        # Calculate the optimal figure height based on number of models
        num_models = len(predictions)
        base_height = 300  # minimum height
        height_per_model = 50  # additional height per model
        figure_height = max(base_height, base_height + (num_models * height_per_model))

        for model_name, pred in predictions.items():
            prob1 = pred['win_probability']
            prob2 = 1 - prob1
            color_pair = colors.get(model_name, ['#6b46c1', '#f59e0b'])

            fig.add_trace(go.Bar(
                name=model_name,
                x=[team1, team2],
                y=[prob1, prob2],
                text=[f"{prob1:.1%}", f"{prob2:.1%}"],
                textposition='auto',
                marker_color=color_pair
            ))

        fig.update_layout(
            title={
                'text': 'Win Probability Prediction',
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=16)
            },
            barmode='group',
            template='plotly_white',
            height=figure_height,
            margin=dict(t=80, b=50, l=50, r=50),  # Increased top margin for legend
            yaxis=dict(
                title='Probability',
                tickformat=',.0%',
                range=[0, 1],
                automargin=True
            ),
            xaxis=dict(
                automargin=True,
                tickangle=-45  # Angle the team names for better fit
            ),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=10)
            ),
            autosize=True
        )

        # Make the layout more compact for mobile
        fig.update_layout(
            bargap=0.15,
            bargroupgap=0.1
        )

        return fig
    except Exception as e:
        print(f"Error in plot_win_probability: {str(e)}")
        return None

def plot_feature_importance(feature_names, importance):
    """Create a horizontal bar chart showing feature importance with high contrast"""
    df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    })
    df = df.sort_values('Importance', ascending=True)

    fig = px.bar(
        df,
        x='Importance',
        y='Feature',
        orientation='h',
        title='Feature Importance in Prediction',
        height=500,
        color='Importance',
        color_continuous_scale=['#4ade80', '#22c55e', '#16a34a']  # Green scale for better contrast
    )

    fig.update_layout(
        paper_bgcolor='#1e293b',
        plot_bgcolor='#1e293b',
        title={
            'font': dict(size=20, color='#ffffff'),
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis={
            'title': {
                'text': 'Relative Importance',
                'font': {'color': '#ffffff', 'size': 14}
            },
            'gridcolor': '#475569',
            'tickfont': {'color': '#ffffff'}
        },
        yaxis={
            'title': {
                'text': 'Feature',
                'font': {'color': '#ffffff', 'size': 14}
            },
            'gridcolor': '#475569',
            'tickfont': {'color': '#ffffff'}
        },
        font=dict(color='#ffffff'),
        showlegend=False,
        coloraxis_showscale=False
    )
    return fig

def plot_team_performance(data, team):
    """Plot team's historical performance metrics"""
    try:
        stats = data[data['Team'] == team].iloc[0]
        metrics = ['apperance', 'Title', 'Finals', 'Semi finals']
        values = [stats[metric] for metric in metrics]

        # Distinct colors for each metric
        colors = ['#6b46c1', '#f59e0b', '#10b981', '#3b82f6']

        fig = go.Figure(data=[
            go.Bar(
                name='Performance Metrics',
                x=metrics,
                y=values,
                marker_color=colors,
                text=values,
                textposition='auto'
            )
        ])

        fig.update_layout(
            title={
                'text': f'{team} Historical Performance',
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            template='plotly_white',
            height=400,
            margin=dict(t=50, b=50, l=50, r=50),
            yaxis_title='Count',
            showlegend=False,
            xaxis=dict(
                tickangle=-45,
                title=None,
                tickfont=dict(size=12)
            ),
            yaxis=dict(
                tickfont=dict(size=12)
            )
        )
        return fig
    except Exception as e:
        print(f"Error in plot_team_performance: {str(e)}")
        return None

def plot_head_to_head(matches, team1, team2):
    """Plot head-to-head statistics between two teams"""
    try:
        h2h_matches = matches[
            ((matches['Team'] == team1) & (matches['Opposition'] == team2)) |
            ((matches['Team'] == team2) & (matches['Opposition'] == team1))
        ]

        team1_wins = len(h2h_matches[h2h_matches['winner'] == team1])
        team2_wins = len(h2h_matches[h2h_matches['winner'] == team2])

        labels = [f"{team1}<br>{team1_wins} wins", f"{team2}<br>{team2_wins} wins"]
        values = [team1_wins, team2_wins]
        colors = ['#6b46c1', '#f59e0b']  # Purple and Orange for contrast

        fig = go.Figure(data=[
            go.Pie(
                labels=labels,
                values=values,
                marker=dict(colors=colors),
                textinfo='label+percent',
                hole=0.3,
                rotation=90,
                textfont=dict(size=14)
            )
        ])

        fig.update_layout(
            title={
                'text': f'Head-to-Head: {team1} vs {team2}',
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            template='plotly_white',
            height=400,
            margin=dict(t=50, b=50, l=50, r=50),
            showlegend=False,
            annotations=[
                dict(
                    text=f'Total Matches: {len(h2h_matches)}',
                    x=0.5,
                    y=0.5,
                    font=dict(size=12),
                    showarrow=False
                )
            ]
        )
        return fig
    except Exception as e:
        print(f"Error in plot_head_to_head: {str(e)}")
        return None

def plot_venue_analysis(venue_data, venue):
    """Plot venue statistics"""
    try:
        stats = venue_data[venue_data['Ground'] == venue].iloc[0]

        # Distinct colors for better visibility
        colors = ['#6b46c1', '#f59e0b', '#10b981']

        fig = go.Figure(data=[
            go.Bar(
                x=['First Innings Avg', 'Second Innings Avg', 'Toss Win Impact'],
                y=[stats['first_innings_avg'], stats['second_innings_avg'], stats['toss_win_rate']],
                marker_color=colors,
                text=[f"{val:.1f}" for val in [stats['first_innings_avg'], 
                                             stats['second_innings_avg'], 
                                             stats['toss_win_rate']]],
                textposition='auto'
            )
        ])

        fig.update_layout(
            title={
                'text': f'Venue Analysis: {venue}',
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            template='plotly_white',
            height=400,
            margin=dict(t=50, b=50, l=50, r=50),
            showlegend=False,
            yaxis_title='Value',
            xaxis_title=None,
            xaxis=dict(tickangle=-30)
        )
        return fig
    except Exception as e:
        print(f"Error in plot_venue_analysis: {str(e)}")
        return None

def plot_team_form(matches, team, n=5):
    """Plot team's recent form"""
    recent = matches[
        (matches['Team'] == team) | (matches['Opposition'] == team)
    ].tail(n)

    results = []
    for _, match in recent.iterrows():
        if match['winner'] == team:
            results.append(1)
        else:
            results.append(0)

    fig = go.Figure(data=[
        go.Scatter(
            y=results,
            mode='lines+markers',
            line=dict(color='#6b46c1', width=2),
            marker=dict(
                size=10,
                color='#9f7aea',
                line=dict(color='#6b46c1', width=2)
            )
        )
    ])

    fig.update_layout(
        title={
            'text': f'{team} Recent Form',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        template='plotly_white',
        height=300,
        margin=dict(t=50, b=50, l=50, r=50),
        yaxis=dict(
            ticktext=['Loss', 'Win'],
            tickvals=[0, 1],
            tickfont=dict(size=12)
        ),
        showlegend=False
    )
    return fig