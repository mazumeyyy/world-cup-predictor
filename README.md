# 2026 World Cup Predictor: Machine Learning Edition

A predictive data science model for the FIFA World Cup using historical international match data to predict tournament outcomes.

## How It Works

The model predicts match outcomes (Win/Loss/Draw) based on:
- **Historical Pedigree:** Over 40,000 international matches since 1872.
- **Tournament Context:** Performance isolated to major competitive tournaments (World Cup, Euros, Copa América).
- **Penalty Adjustments:** Incorporates definitive winners from historical penalty shootouts.

## Technologies
- **Python 3.x**
- **Pandas** (Data manipulation & cleaning)
- **Scikit-learn** (Machine learning classification & evaluation)
- **Streamlit** (Interactive web app deployment)

## Project Structure
world-cup-predictor/
├── data_preprocessing.py   # Cleans and filters historical data
├── results.csv             # Raw historical match dataset
├── shootouts.csv           # Raw penalty shootout dataset
└── README.md               # Project documentation

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/mazumeyyy/world-cup-predictor.git](https://github.com/mazumeyyy/world-cup-predictor.git)
   cd world-cup-predictor


