import os
import shutil
from pathlib import Path

def setup_project():
    """Set up the T20 World Cup Predictor project structure"""
    # Create necessary directories
    directories = [
        'assets',
        'attached_assets',
        'models',
        'pages',
        '.streamlit'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

    # Copy .env.template to .env if it doesn't exist
    if not os.path.exists('.env') and os.path.exists('.env.template'):
        shutil.copy('.env.template', '.env')
        print("Created .env file from template")

    print("\nProject structure set up successfully!")
    print("\nNext steps:")
    print("1. Add your Excel files to the attached_assets/ directory:")
    print("   - T20 worldcup overall.xlsx")
    print("   - matchresultupdate2.xlsx")
    print("   - t20i_bilateral.xlsx")
    print("   - venue_stats.xlsx")
    print("2. Edit .env file with your API keys")
    print("3. Install required packages using:")
    print("   pip install streamlit pandas numpy scikit-learn plotly xgboost psycopg2-binary requests openpyxl")
    print("4. Run the application:")
    print("   streamlit run main.py")

if __name__ == "__main__":
    setup_project()
