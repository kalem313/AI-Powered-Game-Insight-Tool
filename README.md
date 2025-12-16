# Explainable Game Analytics: Counterfactual AI for User Interest Prediction

An end-to-end Machine Learning pipeline that predicts user interest in video games and provides **Counterfactual Explanations**. Unlike standard "black-box" models, this tool generates actionable suggestions to flip a "Not Interested" prediction into a "Positive" one.

## Intro
Most recommendation systems tell you *what* a user likes. This project explains **why** they might not like it and provides a strategic roadmap to change that outcome. 

By analyzing model coefficients and feature importance, the system suggests:
- **Numerical Targets:** Specific Metascore/User Score thresholds needed to capture interest.
- **Categorical Shifts:** Strategic genre alignments.
- **NLP Optimization:** High-impact keywords to include in game descriptions to better resonate with specific user segments.

## Tech
- **Language:** Python
- **Data Science:** Pandas, NumPy, Scikit-Learn
- **NLP:** TF-IDF Vectorization for textual game descriptions
- **ML Techniques:** Binary Classification (Logistic Regression), Column Transformers, Pipelines

## Dataset Features
The model processes multi-modal data including:
- **Numerical:** Metascores, User Scores, and Rating Counts.
- **Categorical:** Genres, Platforms, Ratings (ESRB), and Developers.
- **Textual:** Raw game descriptions processed via NLP.

## How It Works
1. **Data Preprocessing:** Uses a `ColumnTransformer` to handle numerical scaling, one-hot encoding for categories, and TF-IDF for text simultaneously.
2. **Predictive Modeling:** A trained Logistic Regression model estimates the probability of user interest.
3. **Counterfactual Engine:** If a prediction is negative, the `predict_and_suggest_counterfactual` function:
    - Identifies the features with the highest negative impact.
    - Scans for features (like specific keywords or genres) with the highest positive weights for that specific user.
    - Generates a human-readable strategy report.

## Output
```json
{
    "UserID": 814,
    "GameID": 1300486997,
    "Prediction": "Not Interested",
    "Interest_Probability": 0.34,
    "Counterfactual_Suggestion": "To increase interest, target a Metascore of 85+. Additionally, revise the game description to include high-impact keywords like 'Tactical', 'Immersive', and 'Open-World'."
}
