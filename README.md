# T20 World Cup Winner Prediction

## 📌 Project Overview
This project leverages machine learning to predict the winner of the T20 World Cup based on team analysis, match predictions, tournament simulations, and live analytics. The system uses historical performance data, player statistics, and AI-driven models to forecast match outcomes and tournament progression probabilities.

## 🚀 Features
### 1️⃣ Team Analysis
- Historical performance metrics
- Head-to-head statistics
- Performance trends and patterns

### 2️⃣ Match Predictions
- AI-powered match outcome predictions
- Real-time odds updates
- Performance factor analysis

### 3️⃣ Tournament Simulation
- Complete tournament pathway simulation
- Team progression probabilities
- Multiple scenario analysis

### 4️⃣ Live Match Analytics
- Real-time match statistics
- Key performance indicators
- Live win probability updates

### 5️⃣ Player Performance Analysis
- Individual player statistics
- Performance comparisons
- Form analysis

## 🛠 Tech Stack
- **Programming Language:** Python
- **Libraries & Frameworks:** Pandas, NumPy, Scikit-learn, Flask, Streamlit
- **Data Sources:** Historical match data, player statistics APIs
- **Deployment:** Flask API, Streamlit for visualization

## 📦 Quick Setup
1. **Clone the repository**
   ```bash
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
   pip install -r requirements.txt
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

## 📜 Requirements
The following dependencies are listed in `requirements.txt`:
```txt
anthropic>=0.45.2
numpy>=2.2.2
openai>=1.61.1
openpyxl>=3.1.5
pandas>=2.2.3
plotly-express>=0.4.1
plotly>=6.0.0
psycopg2-binary>=2.9.10
requests>=2.32.3
scikit-learn>=1.6.1
streamlit>=1.42.0
trafilatura>=2.0.0
twilio>=9.4.4
xgboost>=2.1.4
python-dotenv>=1.0.1
```

## 🏆 Future Enhancements
- Integration with live match data sources.
- Advanced deep learning models for better accuracy.
- Web-based interactive dashboard.

## 📊 Optional Database Setup
To use PostgreSQL database:
1. Install PostgreSQL on your system
2. Create a new database
3. Add database URL to your `.env` file:
   ```
   DATABASE_URL=postgresql://username:password@localhost:5432/database_name
   ```

## 🤝 Contributing
Contributions are welcome! Feel free to fork the repo and submit a pull request.

## 📜 License
This project is licensed under the MIT License.

---
### 🔗 Connect with Me
- **GitHub:** [NoorainRaza23](https://github.com/NoorainRaza23)
- **LinkedIn:** [Noorain Raza](https://www.linkedin.com/in/noorainraza/)
