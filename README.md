# Banking Customer Loyalty Analysis
> End-to-end data analysis project exploring what drives customer loyalty in a retail banking dataset

---

## Project Overview

This project analyzes **2,940 banking customers** to understand what separates loyalty tiers, how financial behavior differs across customer segments, and whether loyalty classification can be predicted from financial data alone.

The project follows a complete data analysis workflow:

```
Data Cleaning → Exploratory Data Analysis → Feature Engineering → Machine Learning → Power BI Dashboard
```

**Tools Used:** Python · PostgreSQL · Power BI  
**Libraries:** Pandas · NumPy · Matplotlib · Seaborn · Scikit-learn · XGBoost · SHAP · Imbalanced-learn

---

## Repository Structure

```
banking-customer-loyalty-analysis/
│
├── new_banking.csv                        # Original cleaned dataset
├── banking_dashboard.csv                  # Processed dataset after feature engineering — used for Power BI
├── banking_customer_analysis.ipynb        # EDA, Feature Engineering and ML notebook
├── banking_customer_dashboard.pbix        # Power BI dashboard file
├── banking_dashboard_report.pdf           # Dashboard export — all 5 pages
└── README.md
```

---

## Dataset

| Column | Description |
|--------|-------------|
| loyalty_classification | Customer loyalty tier — Gold, Jade, Platinum, Silver |
| estimated_income | Annual income estimate |
| bank_loans | Total bank loan balance |
| bank_deposits | Total deposit balance |
| superannuation_savings | Retirement savings balance |
| credit_card_balance | Outstanding credit card balance |
| checking_accounts | Checking account balance |
| saving_accounts | Savings account balance |
| foreign_currency_account | Foreign currency holdings |
| business_lending | Business loan balance |
| fee_structure | Banking fee tier — High, Mid, Low |
| banking_relation | Relationship type — Commercial, Institutional, Private Bank, Retail |
| risk_weighting | Internal risk score 1–5 |
| nationality | Customer nationality |
| gender | Customer gender |
| age | Customer age |
| tenure_year | Year customer joined the bank |
| properties_owned | Number of properties owned |
| amount_of_credit_cards | Number of credit cards held |

**Target Variable:** `loyalty_classification` — 4 classes (Gold, Jade, Platinum, Silver)

**Class Distribution:**
| Tier | Count | Percentage |
|------|-------|------------|
| Jade | 1,304 | 44.4% |
| Silver | 754 | 25.6% |
| Gold | 574 | 19.5% |
| Platinum | 308 | 10.5% |

---

## Project Workflow

### 1. Data Cleaning
- Identified and handled missing values across 19 columns
- Decoded numerical encoded columns — gender, banking relation, risk weighting, properties owned
- Extracted `tenure_year` from raw `joined_bank` date column
- Removed non-predictive identifier columns — client_id, name, banking_contact, location_id

### 2. Exploratory Data Analysis

EDA was structured in four layers:

**Univariate Analysis**
- Numerical columns — distribution, outliers, skewness
- Categorical columns — value counts, frequency distributions
- Key finding — Jade tier contains 44% of all customers, Platinum only 10.5%

**Bivariate — Categorical vs Numerical**
Three anchor columns used — loyalty_classification, risk_label, banking_relation
- Income shows similar averages across loyalty tiers but Private Bank customers show the widest income spread — high net worth outliers hidden in the average
- Commercial customers have highest bank loans and checking account balances — consistent with business usage
- Institutional and Private Bank customers hold more foreign currency — consistent with sophisticated investor profile
- Risk weighting shows strong correlation with income at 0.7 — the strongest relationship found in the dataset

**Bivariate — Categorical vs Categorical**
- Gender, nationality, risk label, properties — none of these showed meaningful differences across loyalty tiers
- Fee structure showed a small but notable pattern — Platinum customers concentrated more in Mid fee than High fee
- Banking relation showed Gold tier has highest Private Bank proportion at 48%

**Correlation Heatmap**
- bank_deposits, saving_accounts, checking_accounts strongly correlated at 0.8 — same customer behavior expressed in three columns
- age, properties_owned, amount_of_credit_cards showed near zero correlation with all other columns — low signal for analysis and modelling

### 3. Feature Engineering

Four new columns created to capture financial health at customer level:

```python
df['total_assets'] = (df['bank_deposits'] + df['checking_accounts'] + 
                      df['saving_accounts'] + df['foreign_currency_account'] + 
                      df['superannuation_savings'])

df['total_liabilities'] = (df['credit_card_balance'] + df['bank_loans'] + 
                           df['business_lending'])

df['net_worth'] = df['total_assets'] - df['total_liabilities']

df['debt_to_income'] = df['total_liabilities'] / df['estimated_income']
```

**Key finding from engineered features:**
- Average net worth across all customers is **negative at -185,000**
- Over 60% of customers have total liabilities exceeding total assets
- Debt-to-income ratio averages 11x annual income across all tiers — consistent across all loyalty segments with no meaningful difference

### 4. Machine Learning

**Problem:** Multi-class classification — predict loyalty_classification (4 classes)

**Preprocessing pipeline:**
- Train-test split 80/20 with StratifiedKFold to handle class imbalance
- StandardScaler on numerical columns — fit on train, transform on test only
- OneHotEncoder on categorical columns — nationality, fee_structure, banking_relation
- LabelEncoder on target variable

**Models tested:**

| Model | CV Accuracy | Test Accuracy | Notes |
|-------|-------------|---------------|-------|
| Logistic Regression | 43.7% | 43.5% | Predicted almost only Jade — degenerate |
| Random Forest | 43.1% | 40.8% | Ignored minority classes |
| XGBoost | 38.7% | 38.9% | Most balanced across all four classes |

**Approaches tried to improve accuracy:**

| Fix Applied | Result |
|-------------|--------|
| Class weights balanced | Logistic Regression worsened, XGBoost slightly improved spread |
| Removed risk_weighting | No significant improvement |
| SMOTE oversampling | CV jumped to 69% but Test dropped to 34% — severe overfitting |
| RandomizedSearchCV tuning | Best CV 44.1% but model collapsed to predicting only Jade |

**Final model:** XGBoost without modifications — 39% test accuracy with reasonable spread across all four classes

**Feature importance top findings:**
- fee_structure_Low — most important feature at 0.055
- saving_accounts — 0.042
- credit_card_balance — 0.041
- Engineered features total_liabilities, total_assets, net_worth all appeared in top 15

---

## Key Findings

### What we found

**01 — Fee structure is the clearest difference between loyalty tiers**
High fee customers are most concentrated in Gold tier. Platinum customers surprisingly show the highest proportion of Mid fee customers — top loyalty customers are not necessarily on the highest fee plans.

**02 — Jade tier is the bank's largest underengaged segment**
1,304 customers (44.4%) are in Jade tier with similar financial profiles to higher tiers. No single financial metric clearly separates Jade from Platinum customers.

**03 — 60% of customers owe more than they own**
Total liabilities exceed total assets for 1,782 customers. Average net worth is negative at -185K. This is normal banking behavior — active borrowers generate interest revenue.

**04 — Risk level does not differentiate loyalty tiers**
69% of customers are Low or Very Low risk across ALL loyalty tiers equally. Despite 60% having negative net worth, the bank's risk portfolio is healthy.

**05 — Commercial customers earn more than Private Bank on average**
Average income: Commercial 180K vs Private Bank 168K. However Private Bank shows the widest income spread — the average hides significant high net worth outliers within Private Bank segment.

**06 — Loyalty classification cannot be predicted from financial snapshots alone**
All three ML models confirmed that financial data at a single point in time is insufficient. Behavioral data — transaction frequency, product usage, engagement history — is needed for accurate loyalty prediction.

### Where the model struggled and why

This is an honest finding worth explaining. The model achieved 39% accuracy against a 25% random baseline — meaningful improvement but not production-ready.

The root cause is not model quality but data quality for this specific problem. EDA revealed that most financial columns show nearly identical distributions across all four loyalty tiers. The model cannot learn to separate classes that look the same in the data.

Loyalty is built over time through customer behavior — how often they transact, which products they use, how they engage with the bank. A financial snapshot at one point in time does not capture this behavioral history. The model is the correct tool — the dataset is simply missing the variables that actually drive loyalty.

---

## Power BI Dashboard

Five page interactive dashboard built in Power BI Desktop.

| Page | Focus |
|------|-------|
| Customer Overview | Demographics, loyalty distribution, nationality, gender |
| Financial Health | Income, assets vs liabilities, net worth, DTI by loyalty tier |
| Loyalty & Risk Analysis | Risk distribution, fee structure, deposits vs loans, metrics matrix |
| Banking Relationship | Behavior by relation type, loyalty distribution, risk by relation |
| Executive Summary | Key findings and business recommendations |

**Dashboard highlights:**
- Consistent color coding across all pages — Jade green, Silver grey, Gold amber, Platinum purple
- 100% stacked bar charts for proportional comparison across unequal group sizes
- Assets vs liabilities centerpiece visual — tells the net worth story without negative numbers
- Plain language findings accessible to non-technical business stakeholders

---

## Business Recommendations

**01 — Focus on Jade customers**
Nearly half of all customers are one step away from being more valuable to the bank. A targeted loyalty upgrade campaign focusing on product engagement could meaningfully move high potential Jade customers to Gold or Platinum tier.

**02 — Review fee structure alignment**
Top tier customers are not on the highest fee plans. If Platinum benefits do not justify the cost of higher fee tiers, the fee structure may be discouraging loyalty progression. Reviewing the value proposition of High fee structure could improve both revenue and loyalty tier movement.

**03 — Invest in behavioral data collection**
Financial snapshots alone cannot predict loyalty with useful accuracy. Collecting transaction frequency, product usage history, mobile banking engagement and customer service interaction data would significantly improve both customer understanding and loyalty prediction capability.

---

## Honest Project Reflection

This project was built independently alongside a YouTube tutorial — the EDA, ML approach, feature engineering and dashboard design decisions were made from scratch rather than following the tutorial step by step.

**What went well:**
- Structured EDA with three anchor columns gave systematic findings rather than random chart generation
- Feature engineering created meaningful derived columns — net worth finding emerged from this
- Honest evaluation of ML results — rather than overfitting to show high accuracy, the model limitations were acknowledged and explained
- Dashboard tells a business story across five pages rather than showing generic charts

**What was challenging:**
- Class imbalance in loyalty_classification was a persistent problem — Jade at 44% caused most models to default to predicting Jade for everything
- SMOTE created severe overfitting gap between CV and test accuracy — learned that synthetic oversampling does not help when the underlying features cannot separate classes
- Power BI formatting inconsistencies required significant troubleshooting — background color layering in card visuals specifically

**What would be done differently:**
- Start with a clearer problem statement before beginning EDA
- Collect or engineer behavioral features — transaction count, product count per customer, months since last product added — before attempting ML
- Use SHAP values on a model trained without class imbalance fixes to get cleaner feature importance

---

## How to Run

**Python notebook:**
```
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap imbalanced-learn
```
Open `banking_customer_analysis.ipynb` in Jupyter Notebook or JupyterLab and run all cells in order.

**Power BI dashboard:**
Open `banking_customer_dashboard.pbix` in Power BI Desktop. Data is embedded — no reconnection needed.

---

## Connect

If you found this project useful or have feedback, feel free to connect on LinkedIn or raise an issue on this repository.
