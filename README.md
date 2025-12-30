# Credit Risk Default Prediction

## Objective
Build a classification model to estimate **probability of default (PD)** for loan applicants
and evaluate performance in an **imbalanced credit risk setting**.  
The project focuses on ranking risk, threshold tuning, and interpretability rather than raw accuracy.

## Dataset
- ~15,000 loan applicants
- Target: `target`
  - `1` = Default
  - `0` = No Default
- Default rate: ~7–8%

### Feature Groups
**Numeric**
- Age
- Monthly income
- Debt ratio
- Credit utilization
- Transaction count (30d)
- Average transaction amount
- Last payment delay (days)
- Internal credit score

**Categorical**
- Employment type
- Education level
- Region
- Device type

## Data Preparation
- Missing value imputation (median for numeric, mode for categorical)
- Standard scaling for numeric features
- One-hot encoding for categorical features
- All preprocessing handled within a `ColumnTransformer` to prevent data leakage

## Exploratory Analysis
- Strong class imbalance identified
- Credit utilization, payment delays, and transaction behavior show strong
  relationships with default risk
- Internal credit score is negatively correlated with default

## Modeling
**Model:** Logistic Regression  
**Why:** Interpretable, production-friendly, strong baseline for credit risk

### Training
- Stratified train/test split (70/30)
- Class imbalance handled using `class_weight="balanced"`
- Cross-validation with ROC AUC optimization

### Performance
- **ROC AUC:** ~0.81–0.82  
- **PR AUC:** ~0.69  

These scores indicate strong ranking ability in a realistic, imbalanced credit dataset.

## Threshold Optimization
- Classification threshold tuned using the Precision–Recall curve
- Selected threshold prioritizes **recall for defaulters**
- Reflects real-world tradeoff between approval rates and credit losses

## Model Explainability
- Permutation importance used to assess feature impact
- Most influential features:
  - Transaction count (30d)
  - Credit utilization
  - Payment delay
- Behavioral and financial variables dominate demographic features

## Error Analysis
- False negatives represent the highest financial risk
- False positives reflect conservative rejections
- Threshold tuning used to align model behavior with business risk tolerance

## Deployment
- Final model serialized using `joblib`
- Can be integrated into a scoring pipeline to produce real-time PD estimates
- Decision thresholds can be adjusted dynamically based on risk appetite

## Key Takeaways
- Credit risk modeling requires metrics beyond accuracy
- ROC AUC and PR AUC provide complementary insight in imbalanced data
- Threshold selection materially impacts business outcomes
- Interpretable models remain competitive when paired with strong preprocessing

## Tools
- Python
- Pandas
- NumPy
- Scikit-learn
-  Matplotlib,
-  SciPy

## Project Structure
