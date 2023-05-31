# Machine-learning-model-for-CA_2
Ireland Housing Construction Cost
rom statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
from prophet import Prophet
import matplotlib.pyplot as plt
import pandas as pd

# List of valid months
valid_months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

# Reshape the data into a two-column format
df_melted = df.melt(id_vars='Year', var_name='Month', value_name='Cost Index')

# Filter out rows where Month is not in valid_months
df_melted = df_melted[df_melted['Month'].isin(valid_months)]

# Now create the Date column
df_melted['Date'] = pd.to_datetime(df_melted['Year'].astype(str) + ' ' + df_melted['Month'], format='%Y %B')
df_melted = df_melted.drop(['Year', 'Month'], axis=1).sort_values('Date')

# ARIMA Model
# Split the dataset into training and test datasets (70-30 split)
train_size = int(df_melted.shape[0] * 0.7)
train, test = df_melted['Cost Index'][:train_size], df_melted['Cost Index'][train_size:]

# Fit the ARIMA model
model_arima = ARIMA(train, order=(5,1,0))
model_fit = model_arima.fit()

# Make predictions using the test dataset
predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)

# Evaluate the model
rmse = sqrt(mean_squared_error(test, predictions))
print('ARIMA RMSE: %.3f' % rmse)

# Prophet Model
# Prepare the data for the Prophet model
df_prophet = df_melted.rename(columns={'Date': 'ds', 'Cost Index': 'y'})

# Fit the Prophet model
model_prophet = Prophet()
model_prophet.fit(df_prophet)

# Make predictions for the next 5 years
future = model_prophet.make_future_dataframe(periods=60, freq='M')
forecast = model_prophet.predict(future)

# Plot the forecast
model_prophet.plot(forecast)
plt.show()
ARIMA RMSE: 2.320
07:05:35 - cmdstanpy - INFO - Chain [1] done processing
INFO:cmdstanpy:Chain [1] done processing
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
import string
import re

# download necessary NLTK data
nltk.download(['punkt', 'wordnet', 'stopwords'])

# init lemmatizer
lemmatizer = WordNetLemmatizer()
stop = stopwords.words('english')

def clean_text(text):
    # lowercasing
    text = text.lower()
    
    # remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # remove digits
    text = re.sub(r'\d+', '', text)
    
    # tokenize text
    words = word_tokenize(text)
    
    # remove stopwords and lemmatize each word
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop]
    
    return " ".join(words)

# load the dataset
data = pd.read_csv('/content/sample_data/house/Hotel_Reviews.csv')

# handle possible NaN values before combining
data['Negative_Review'] = data['Negative_Review'].fillna('')
data['Positive_Review'] = data['Positive_Review'].fillna('')

# combine positive and negative reviews
data['review'] = data['Negative_Review'] + data['Positive_Review']

# create labels: 1 for positive sentiment, 0 for negative sentiment
data['is_good_review'] = data['Reviewer_Score'].apply(lambda x : 1 if x > 5 else 0)

# clean text data
data['review'] = data['review'].apply(clean_text)

# split dataset
X = data.review
y = data.is_good_review
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)

# vectorize text data
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# train a Naive Bayes model
nb = MultinomialNB()
nb.fit(X_train, y_train)

# make predictions
predictions = nb.predict(X_test)

# print results
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
 Table or graphics should be provided to illustrate the similarities and contrast of the Machine Learning modelling outcomes based on the scoring metric used for the analysis of the above-mentioned scenario. Discuss and elaborate your understanding clearly

from sklearn.linear_model import LassoCV
​
# Train a Lasso Regression model for comparison
lasso_cv = LassoCV(alphas=[1e-3, 1e-2, 1e-1, 1], cv=5)
lasso_cv.fit(X_train, y_train)
​
y_pred_lasso = lasso_cv.predict(X_test)
​
# Calculate metrics for Lasso model
lasso_mse = mean_squared_error(y_test, y_pred_lasso)
lasso_r2 = r2_score(y_test, y_pred_lasso)
​
# Create a data frame for model comparison
model_comparison = pd.DataFrame({
    'Model': ['Ridge', 'Lasso'],
    'Best alpha': [grid_search.best_params_['ridge__alphas'], lasso_cv.alpha_],
    'MSE': [mean_squared_error(y_test, y_pred), lasso_mse],
    'R2 Score': [r2_score(y_test, y_pred), lasso_r2],
})
​
print(model_comparison)
​
   Model        Best alpha       MSE  R2 Score
0  Ridge  (0.1, 1.0, 10.0)  3.817695  0.896930
1  Lasso               1.0  4.036512  0.891023
