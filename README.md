# Movie Success Analysis & Recommendation System

A comprehensive data science project analyzing factors that contribute to movie success, featuring predictive modeling for performance classification and a content-based recommendation system.

## Overview

This project explores movie industry data to understand what makes a movie successful. Through exploratory data analysis, feature engineering, and machine learning, we develop models to predict movie performance and recommend similar movies based on content features.

**Key Objectives:**
- Analyze relationships between budget, revenue, ratings, and movie success
- Engineer meaningful features including director and actor success scores
- Build classification models to predict movie performance categories
- Develop a content-based movie recommendation system
- Discover hidden patterns through association rule mining

## Repository Structure

```
.
├── data/
│   ├── raw/
│   │   └── movies.csv              # Original movie dataset
│   └── processed/
│       └── processed-movie.csv     # Fully processed dataset with engineered features
├── notebooks/
│   ├── eda-analysis.ipynb          # Exploratory data analysis and initial cleaning
│   ├── feature-engineering.ipynb   # Advanced feature creation
│   ├── association-mining.ipynb    # Pattern discovery using Apriori algorithm
│   └── models.ipynb                # Predictive modeling and recommendation system
└── README.md
```

## Project Workflow

The analysis follows a structured data science pipeline:

**1. Preprocessing & EDA** → **2. Feature Engineering** → **3. Association Mining** → **4. Predictive Modeling**

This sequence ensures data quality, creates meaningful features, discovers patterns, and finally builds predictive models.

## Notebooks Overview

### 1. [eda-analysis.ipynb](notebooks/eda-analysis.ipynb) - Preprocessing & Exploratory Data Analysis
**Purpose:** Initial data exploration, cleaning, and basic feature engineering.

**Key Operations:**
- Load and inspect the raw movies dataset (7,662 movies from 1980-2020)
- Handle missing values and remove duplicates
- Clean budget and gross revenue columns (remove currency symbols, convert to numeric)
- Create foundational features:
  - `profit_ratio`: Ratio of gross revenue to budget
  - `performance_class`: Categorizes movies (Flop, Average, Hit, Super Hit, Blockbuster, All-Time Blockbuster)
  - `score_cat`: Rating categories (Excellent, Very Good, Good, Average, Poor)
  - `budget_cat`: Budget tiers (Low, Mid, High Budget)
- Output: `processed.csv`

**Key Insight:** Dataset shows significant class imbalance with Flop movies dominating (3,607 out of 7,662 movies).

### 2. [feature-engineering.ipynb](notebooks/feature-engineering.ipynb) - Advanced Feature Creation
**Purpose:** Engineer sophisticated features to capture director and actor influence on movie success.

**Key Operations:**
- Calculate `director_success_score` based on weighted performance of their movies
- Calculate `actor_success_score` using the same methodology
- Categorize directors and actors into performance tiers:
  - **Directors:** Legendary (≥30), High-Performer (≥15), Medium (≥5), Low (<5)
  - **Actors:** Legendary (≥40), High-Performer (≥20), Mid (≥8), Low (<8)
- Output: Enhanced `processed.csv` with success scores and categories

**Weighting System:**
- All-Time Blockbuster: 15 points
- Blockbuster: 5 points
- Super Hit: 3 points
- Hit: 2 points
- Average: 1 point
- Flop: 0 points

### 3. [association-mining.ipynb](notebooks/association-mining.ipynb) - Pattern Discovery
**Purpose:** Discover hidden patterns and relationships between movie attributes using association rule mining.

**Key Operations:**
- Apply Apriori algorithm using the `efficient_apriori` library
- Analyze features: genre, performance_class, score_cat, budget_cat, director_category, actor_category, company, star, director
- Extract association rules with minimum support (0.005) and confidence (0.4)
- Visualize relationships through Support vs. Confidence and Support vs. Lift plots
- Identify top rules by lift metric

**Key Insights:**
- Discover which genre combinations correlate with success
- Identify budget-performance relationships
- Find successful director-actor pairings

### 4. [models.ipynb](notebooks/models.ipynb) - Predictive Modeling & Recommendation
**Purpose:** Build classification models to predict movie performance and develop a recommendation system.

**Key Operations:**

**Data Balancing:**
- Original class distribution (highly imbalanced):
  - Flop: 3,607 movies (47%)
  - Hit: 1,452 movies (19%)
  - Super Hit: 883 movies (12%)
  - Average: 738 movies (10%)
  - All-Time Blockbuster: 502 movies (7%)
  - Blockbuster: 480 movies (6%)
- Merged into balanced categories:
  - Low/Flop: 4,345 movies (Flop + Average)
  - Hit/Success: 2,335 movies (Hit + Super Hit)
  - Big Hit: 982 movies (Blockbuster + All-Time Blockbuster)

**Feature Preprocessing:**
- One-Hot Encoding for categorical features: `budget_cat`, `score_cat`, `genre`
- StandardScaler for numerical features: `director_success_score`, `actor_success_score`, `runtime`

**Classification Models:**

1. **Decision Tree Classifier**
   - Hyperparameter tuning for `max_depth` (tested depths 1-10)
   - Best depth: 8
   - Test Accuracy: **71.10%**
   - Performance by class:
     - Big Hit: Precision 0.76, Recall 0.38, F1-Score 0.51
     - Hit/Success: Precision 0.52, Recall 0.77, F1-Score 0.62
     - Low/Flop: Precision 0.85, Recall 0.74, F1-Score 0.79

2. **Random Forest Classifier**
   - 300 estimators, max_depth=10, class_weight='balanced'
   - Test Accuracy: **71%**
   - Performance by class:
     - Big Hit: Precision 0.48, Recall 0.71, F1-Score 0.58
     - Hit/Success: Precision 0.55, Recall 0.82, F1-Score 0.66
     - Low/Flop: Precision 1.00, Recall 0.65, F1-Score 0.79

**Recommendation System:**
- Content-based recommender using cosine similarity
- Features: One-hot encoded genre, director category, and actor category
- Successfully recommends similar movies based on content attributes

## Model Performance Analysis

### Why Accuracy Alone is Misleading

While both models achieve approximately **71% accuracy**, this metric alone doesn't tell the full story due to **severe class imbalance** in the dataset.

**The Problem:**
- The dataset is dominated by Flop movies (47% of all movies)
- After merging, Low/Flop still represents 57% of the data
- A naive model that always predicts "Low/Flop" would achieve 57% accuracy without learning anything meaningful

**Why We Need Precision, Recall, and F1-Score:**

1. **Precision** answers: "Of all movies we predicted as Big Hits, how many actually were Big Hits?"
   - Random Forest achieves only 48% precision for Big Hits, meaning many false positives

2. **Recall** answers: "Of all actual Big Hits, how many did we correctly identify?"
   - Random Forest achieves 71% recall for Big Hits, catching most successful movies but with many false alarms

3. **F1-Score** balances precision and recall:
   - Big Hit F1-Score: 0.58 (moderate performance)
   - Hit/Success F1-Score: 0.66 (better performance)
   - Low/Flop F1-Score: 0.79 (best performance, but this is the majority class)

**Key Takeaway:** The models perform reasonably well on the majority class (Low/Flop) but struggle with minority classes (Big Hit). This is expected given the imbalance and suggests that **additional features** and **better balancing techniques** could improve performance.

## Installation & Usage

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib seaborn efficient-apriori
```

### Running the Analysis

Execute notebooks in the following order to replicate the complete workflow:

1. **Preprocessing & EDA:**
   ```bash
   jupyter notebook notebooks/eda-analysis.ipynb
   ```

2. **Feature Engineering:**
   ```bash
   jupyter notebook notebooks/feature-engineering.ipynb
   ```

3. **Association Mining:**
   ```bash
   jupyter notebook notebooks/association-mining.ipynb
   ```

4. **Predictive Modeling & Recommendations:**
   ```bash
   jupyter notebook notebooks/models.ipynb
   ```

## Key Features & Insights

### Performance Classification
Movies are categorized into six performance tiers based on profit ratio:
- **Flop:** < 1x return on investment (47% of dataset)
- **Average:** 1x - 1.5x ROI (10% of dataset)
- **Hit:** 1.5x - 3x ROI (19% of dataset)
- **Super Hit:** 3x - 5x ROI (12% of dataset)
- **Blockbuster:** 5x - 8x ROI (6% of dataset)
- **All-Time Blockbuster:** > 8x ROI (7% of dataset)

### Success Scoring System
Directors and actors are evaluated based on their historical performance, with weighted scores reflecting the quality of movies they've been involved in. This creates a reputation metric that can predict future movie success.

### Recommendation Engine
The content-based recommender suggests similar movies by analyzing:
- Genre similarity
- Director tier alignment
- Actor tier alignment

Example recommendations:
- **Inception** → Indiana Jones, Blade Runner, The Terminator
- **Titanic** → Rocky III, Staying Alive, The Natural

### Pattern Discovery
Association rule mining reveals interesting relationships, such as:
- Genre combinations that tend to perform well
- Budget-performance correlations
- Director-actor pairings that lead to success

## Dataset

The project uses a comprehensive movies dataset containing:
- **7,662 movies** spanning 1980-2020
- Movie titles and release information
- Budget and gross revenue figures
- IMDb ratings and scores
- Director and cast information
- Genre classifications
- Production company details
- Runtime information

**Note:** The raw dataset should be placed in `data/raw/movies.csv`. The notebooks expect this path structure.

## Technologies Used

- **Python 3.x**
- **Data Analysis:** pandas, numpy
- **Machine Learning:** scikit-learn
- **Visualization:** matplotlib, seaborn
- **Association Mining:** efficient-apriori
- **Environment:** Jupyter Notebook

## Results & Applications

This analysis can be used for:
- **Investment Decisions:** Predict movie performance before production (71% accuracy)
- **Casting Recommendations:** Identify optimal director-actor combinations
- **Content Strategy:** Understand what attributes drive success
- **Personalization:** Recommend movies to users based on content preferences
- **Market Analysis:** Discover trends and patterns in the movie industry

## Future Scope & Enhancements

To improve model performance and expand capabilities, consider:

### 1. Enhanced Actor Features
- **Actor Popularity Score:** Incorporate social media followers, award wins, and recent box office performance
- **Actor-Genre Affinity:** Track which genres specific actors excel in
- **Star Power Index:** Combine multiple metrics (awards, nominations, previous hits) into a comprehensive popularity score

### 2. Additional Features
- **Marketing Budget:** Advertising and promotional spending data
- **Release Timing:** Season, holidays, competition analysis
- **Franchise Indicator:** Whether the movie is part of a series
- **Critic Reviews:** Rotten Tomatoes scores, Metacritic ratings
- **Social Media Buzz:** Pre-release sentiment analysis

### 3. Advanced Modeling Techniques
- **SMOTE (Synthetic Minority Over-sampling):** Better handle class imbalance
- **Ensemble Methods:** Combine multiple models (XGBoost, LightGBM)
- **Deep Learning:** Neural networks for complex pattern recognition
- **Time Series Analysis:** Incorporate temporal trends in movie success

### 4. Expanded Recommendation System
- **Collaborative Filtering:** User-based recommendations
- **Hybrid Approach:** Combine content-based and collaborative filtering
- **Personalized Recommendations:** Factor in user preferences and viewing history

### 5. Real-Time Prediction
- **API Development:** Deploy models as REST APIs for real-time predictions
- **Dashboard:** Interactive visualization of predictions and insights

## License

This project is available for educational and research purposes.

---

**Note:** Some notebooks may reference Kaggle input paths (e.g., `/kaggle/input/`). Update these paths to match your local directory structure when running locally.
