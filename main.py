import streamlit as st
import pandas as pd

# ------ PART 1 ------



# Display text
st.title('ML Insider Trading Project: Fall 2024')
st.subheader('Atharva Beesan, Raksha Govind, Sneha Jaiswal, Michelle Liang, Soham Samal')

# * optional kwarg unsafe_allow_html = True
# Media
st.image('./insider-trading.png')

st.header('Proposal')
st.subheader('Introduction')
st.write('Insider trading, where individuals with confidential company information trade stocks for personal gain, poses a major threat to market integrity. With machine learning and big data, we can detect illegal activities by identifying anomalies in trading patterns. By highlighting these anomalies, we can flag suspicious transactions for further investigation. Previous research applied techniques like K-means clustering and statistically validated networks. Esen et al. (2019) analyzed 1M+ transactions, including 60K insiders, over 7 years to detect outliers. The study found that outlier transactions earned larger abnormal returns using traditional event study methods. Mazzarisi et al. (2024) used two methods to detect insider trading in Italian markets: an event-based approach and Statistically Validated Networks to identify investor groups with trading discontinuities around market-sensitive events. We will use the S&P500 Insider Trading dataset from Kaggle, covering insider trading activity from 2016-2017, including transaction details such as date, insider name/position, company name/ticker, transaction type, shares traded, price, total value, and ownership.')
st.subheader('Problem')
st.write('Insider trading is a significant issue in financial markets, potentially distorting market fairness and discouraging investor confidence. Detecting and mitigating insider trading can be a complicated task due to the secret nature of these trades and the subtle ways in which these traders are able to manipulate stock prices. Manual audits and reactive regulatory actions are commonly used to detect insider trading. Our primary motivation is to use a data-driven approach to come up with a more efficient way to catch these crimes.')
st.subheader('Methods')
st.write('For insider trading detection, proper data preprocessing is crucial since stock prices, financial ratios, and stock ownership operate on different scales. One method we will use is normalization, scaling factors like trading volume and volatility consistently across the dataset. We’ll use StandardScaler() from scikit-learn to normalize features with a mean of 0 and variance of 1. Time-Series Transformation is also needed, as insider trading requires analyzing trades before and after events. Pandas .shift() helps convert data to a time-series format. Since insider trading events are rare, we will resample using imbalanced-learns SMOTE. We have selected three key algorithms. Random Forest classifies trades as suspicious or normal using decision trees to evaluate factors like trade volume and timing. Neural Networks can detect insider trading by learning temporal patterns from trade data - we plan to use an LSTM model. Finally, XGBoost handles uneven data and finds patterns using multiple decision trees, combining models to enhance prediction accuracy.')
st.write('At this midterm point we have done Time-Series Transformation and have completed working on our Random Forest. The reason we chose Time-Series Transformation is because we needed to analyze trades before and after certain events took place. We create 14-day windows of stock data for 7 days before and after the trade. Key features like average price, trading volume and volatility are computed separately for pre and post trade periods for each of the windows. This helps quantify changes between trades. Each window is labeled as “insider” or “non-insider” which creates structured data ready for ML classification. The reason we chose Random Forest is because it is helpful against overfitting, handles high-dimensional data well and provides feature importance scores that help us understand how impactful each feature is. This worked by creating an ensemble of decision trees, each of which make predictions and then aggregating all of these predictions into a final classification. The data was split into training and test sets and the model was trained on features like pre and post trade price changes. By using 100 trees in the forest we improved accuracy by averaging multiple outcomes. We then evaluated the model’s performance with various metrics.')
st.subheader('Results & Discussion')
st.write('We had used 3 key metrics to analyze our model’s performance and areas of improvement. The first was f1-score, which is the harmonic mean of precision and recall. This allows us to have a more balanced measure of accuracy, especially for imbalanced datasets. In our case, the number of insider trades is minute compared to the number of normal trades, and thus, f1-score is perfect for measurement here. The second is the cross-validation scores computed with the cross_val_score function assess model stability and generalization across different splits of the dataset. The average accuracy across folds gives a more robust measure than a single train-test split. Finally,  The feature importance from the RandomForestClassifier model gives insight into which features the model deems most influential for making predictions. This can help interpret the model and understand which stock features correlate most strongly with insider trading.The model’s average accuracy rate was 0.9182 when trained with an equal 30% of insider/non-insider trades of our dataset and tested with 70%, which helped reduce bias and improve generalization in model training.')
st.image('./metric-analysis.png')
st.write('The model achieved high precision, recall, f1-score, and support scores, with an accuracy of ~92% post-training, correctly predicting 44 non-insider trades from 50 non-insider windows. While random forests can often be seen as a “black box”, the feature importance histogram shows that the three most significant features in predicting a trade are post trade averages, pre-trade averages, and average volume within the window. Through using random forest in this implementation, our model was able to decrease overfitting, thus improving prediction accuracy, however could be improved on with a larger training dataset.')
st.image('./feature-selection.png')
st.write('Based on the current results of our model, we are looking for our next steps to refine feature selection. This will hopefully enhance the relevance and interpretability to insider trading patterns. We also could hopefully increase the dataset size to help mitigate overfitting and improve robustness. In terms of the model itself, we’re looking to explore other more complex models, such as LSTMs which could allow us to capture more complex patterns as insider trading is often event-driven. We would need to calibrate the model to ensure that our insider trade predictions are not only reliable but are also able to minimize false positives while still maintaining a high accuracy.')
st.subheader('References')
st.write('Esen, M. F., Bilgic, E., & Basdas, U. (2019). How to detect illegal corporate insider trading? A data mining approach for detecting suspicious insider transactions. Intelligent Systems in Accounting, Finance and Management, 26(2), 60/70. https://doi.org/10.1002/isaf.1446')
st.write('Mazzarisi, P., Camacho-Collados, I., & Esposito, G. (2024). Detecting insider trading using statistically validated networks. SSRN. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4294752')
st.header('Gantt Chart')
google_sheets_link = "https://docs.google.com/spreadsheets/d/1jS2X4-41KPCm83fLR_SH6gVN88JeM12aB_Xpcihj-Bk/edit?usp=sharing"

st.markdown(
    f'<iframe src="{google_sheets_link}" width="800" height="800"></iframe>',
    unsafe_allow_html=True,
)

st.header('Contribution Table')
contribution_data = {
    'Team Member': ['Atharva Beesen', 'Raksha Govind', 'Sneha Jaiswal', 'Michelle Liang', 'Soham Samal'],
    'Contribution': ['Data Preprocessing', 'Analysis of Algorithm Model, Github Pages', 
                     'Data Preprocessing and ML Algorithms Methods', 'Next Steps', 
                     'Algorithm Implementation, Quantititative Metrics']
}
contribution_df = pd.DataFrame(contribution_data)
contribution_df.set_index('Team Member', inplace=True)
st.write(contribution_df)
