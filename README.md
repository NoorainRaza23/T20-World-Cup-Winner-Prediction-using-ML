git clone <repository-url>
   cd t20-predictor
   ```

2. **Run Setup Script**
   ```bash
   python setup.py
   ```
   This will:
   - Create all necessary directories
   - Set up your .env file from template
   - Guide you through next steps

3. **Add Required Data Files**
   Place these Excel files in the `attached_assets` directory:
   - T20 worldcup overall.xlsx
   - matchresultupdate2.xlsx
   - t20i_bilateral.xlsx
   - venue_stats.xlsx

4. **Configure Environment**
   Edit the `.env` file with your API keys:
   ```
   CRICINFO_API_KEY=your_cricinfo_api_key_here
   WEATHER_API_KEY=your_weather_api_key_here
   ```

5. **Install Dependencies**
   ```bash
   pip install streamlit pandas numpy scikit-learn plotly xgboost psycopg2-binary requests openpyxl
   ```

6. **Run the Application**
   ```bash
   streamlit run main.py
   ```

### Manual Setup (Alternative)

If you prefer to set up manually, follow these steps:

1. Create the following directory structure:
   ```
   t20-predictor/
   ├── assets/
   ├── attached_assets/
   ├── models/
   ├── pages/
   └── .streamlit/
   ```

2. Copy `.env.template` to `.env` and configure your API keys

3. Follow steps 3-6 from the Quick Setup section above

## Features

- 📊 Team Analysis & Comparison
- 🏆 Tournament Simulation
- ⚡ Live Match Predictions
- 👤 Player Performance Analytics
- 🎯 Advanced ML-based Predictions

## Optional Database Setup

To use PostgreSQL database:
1. Install PostgreSQL on your system
2. Create a new database
3. Add database URL to your `.env` file:
   ```
   DATABASE_URL=postgresql://username:password@localhost:5432/database_name