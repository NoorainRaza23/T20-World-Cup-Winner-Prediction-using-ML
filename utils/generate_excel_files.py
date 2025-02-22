import pandas as pd

# Create bilateral series data
bilateral_data = {
    'Team': ['India', 'Afghanistan', 'Australia', 'West Indies', 'England', 'West Indies', 'Pakistan', 'New Zealand', 'South Africa', 'India'],
    'Opposition': ['Afghanistan', 'India', 'West Indies', 'Australia', 'West Indies', 'England', 'New Zealand', 'Pakistan', 'India', 'South Africa'],
    'Ground': ['Bengaluru', 'Indore', 'Adelaide', 'Hobart', 'Trinidad', 'Grenada', 'Rawalpindi', 'Hamilton', 'Durban', 'Gqeberha'],
    'Toss': ['won', 'lost', 'won', 'lost', 'won', 'won', 'won', 'lost', 'won', 'lost'],
    'Bat': ['1st', '2nd', '1st', '2nd', '1st', '1st', '1st', '2nd', '1st', '2nd'],
    'winner': ['India', 'India', 'Australia', 'Australia', 'West Indies', 'West Indies', 'Pakistan', 'New Zealand', 'South Africa', 'India'],
    'Start Date': ['2024-01-17', '2024-01-14', '2024-02-09', '2024-02-11', '2023-12-19', '2023-12-21', '2024-01-12', '2024-01-14', '2023-12-10', '2023-12-12']
}

bilateral_df = pd.DataFrame(bilateral_data)
bilateral_df.to_excel('attached_assets/t20i_bilateral.xlsx', index=False)

# Create venue statistics data
venue_data = {
    'Ground': ['Mumbai', 'Bengaluru', 'Barbados', 'Lahore', 'Adelaide', 'Hobart', 'Trinidad', 'Grenada', 'Rawalpindi', 'Hamilton', 'Durban', 'Gqeberha', 'Indore'],
    'Matches_Played': [25, 30, 20, 28, 22, 15, 18, 12, 24, 20, 26, 15, 22],
    'Avg_First_Innings': [165, 175, 145, 160, 170, 155, 150, 140, 165, 160, 155, 145, 170],
    'Avg_Second_Innings': [155, 165, 140, 150, 160, 145, 145, 135, 155, 150, 145, 135, 160],
    'Toss_Win_Percentage': [55, 52, 48, 51, 54, 49, 50, 51, 53, 52, 49, 48, 53],
    'Chasing_Win_Percentage': [48, 55, 45, 52, 51, 47, 52, 49, 50, 48, 51, 46, 54]
}

venue_df = pd.DataFrame(venue_data)
venue_df.to_excel('attached_assets/venue_stats.xlsx', index=False)
