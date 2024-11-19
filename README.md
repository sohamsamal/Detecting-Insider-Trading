# Insider Trading Detection with Machine Learning

This project aims to leverage machine learning and big data to detect insider trading by identifying anomalies in trading patterns. By analyzing financial data and applying sophisticated algorithms, the project highlights suspicious transactions for further investigation.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Problem](#problem)
3. [Methods](#methods)
4. [Results & Discussion](#results--discussion)
5. [Next Steps](#next-steps)
6. [References](#references)

---

## Introduction

Insider trading, where individuals with confidential company information trade stocks for personal gain, poses a significant threat to market integrity. By utilizing machine learning techniques, we aim to detect illegal activities through anomalies in trading patterns. Previous research has applied methods like K-means clustering and Statistically Validated Networks (SVN):

- **Esen et al. (2019)** analyzed 1M+ transactions, including 60K insiders, over 7 years, revealing that outlier transactions earned larger abnormal returns.
- **Mazzarisi et al. (2024)** applied an event-based approach and SVN to detect insider trading in Italian markets.

For this study, we use the **S&P500 Insider Trading dataset** (2016–2017) from Kaggle, which includes:
- Insider details (name, position)
- Company details (name, ticker)
- Transaction details (date, type, shares traded, price, total value, ownership)

---

## Problem

Insider trading distorts market fairness, undermining investor confidence. Detecting and mitigating it is challenging due to its secretive nature and subtle manipulation tactics. Traditional manual audits are time-intensive and reactive. This project seeks to create a data-driven, efficient approach to identify and prevent insider trading.

---

## Methods

### Data Preprocessing
- **Normalization**: Features like trading volume and volatility are scaled using `StandardScaler()` from scikit-learn.
- **Time-Series Transformation**: Insider trading requires analyzing trades before and after events. We use Pandas `.shift()` to create 14-day windows (7 days pre- and post-trade) and compute metrics like average price, trading volume, and volatility for each period.
- **Resampling**: Since insider trading is rare, we use SMOTE to address class imbalance.

### Algorithms
1. **Random Forest**:
   - Effective against overfitting and high-dimensional data.
   - Provides feature importance scores for interpretability.
   - Trained on features like pre- and post-trade price changes, using 100 trees for enhanced accuracy.
2. **Neural Networks (Future Work)**:
   - Plan to use LSTM models to detect temporal patterns.
3. **XGBoost (Future Work)**:
   - Suitable for handling uneven data distribution.

### Progress
- Completed **Time-Series Transformation** to create structured data.
- Developed and evaluated a **Random Forest** model, achieving promising results.

---

## Results & Discussion

### Metrics Used
1. **F1-Score**: Balances precision and recall, crucial for imbalanced datasets.
2. **Cross-Validation Scores**: Evaluates stability and generalization across dataset splits.
3. **Feature Importance**: Highlights key factors influencing predictions, such as post-trade averages, pre-trade averages, and average volume.

### Performance
- Average accuracy: **0.9182** (trained on a balanced 30% insider/non-insider split, tested on 70%).
- Correctly predicted **44 non-insider trades out of 50** in the test set.
- Feature importance analysis revealed the most significant predictors of insider trading.

While the Random Forest model reduced overfitting and achieved high precision, expanding the dataset and refining feature selection can further improve results.

---

## Next Steps

1. **Enhance Feature Selection**:
   - Focus on relevance and interpretability for insider trading patterns.
2. **Increase Dataset Size**:
   - Mitigate overfitting and improve robustness.
3. **Explore Advanced Models**:
   - Utilize LSTMs to capture event-driven patterns and temporal dependencies.
   - Calibrate models to minimize false positives while maintaining accuracy.

---

## References

1. Esen, M. F., Bilgic, E., & Basdas, U. (2019). *How to detect illegal corporate insider trading? A data mining approach for detecting suspicious insider transactions.* Intelligent Systems in Accounting, Finance and Management, 26(2), 60/70. [https://doi.org/10.1002/isaf.1446](https://doi.org/10.1002/isaf.1446)
2. Mazzarisi, P., Camacho-Collados, I., & Esposito, G. (2024). *Detecting insider trading using statistically validated networks.* SSRN. [https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4294752](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4294752)

---

## Thank You!
