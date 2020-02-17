import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as sm
import scipy.stats as stats


from sklearn.model_selection import train_test_split # required to split the data
from sklearn.feature_selection import SelectKBest # to select best feature
from sklearn.feature_selection import chi2 # for feature Selection
#from sklearn.metrics import confusion_matrix # to extract confusion matrix
from sklearn.metrics import accuracy_score # to get the scores
from sklearn.linear_model import LogisticRegression # for logistic regression
from sklearn.metrics import r2_score


# Load the csv to a dataframe
df = pd.read_csv(r'C:\Users\Dell\Desktop\Assignments\Adv Stats\weather-dataset-rattle-package\weatherAUS.csv')

print(df.head())

# Drop columns with substantial missing data
df = df.drop(columns=['RISK_MM'],axis=1)
df = df.drop(columns=['Date', 'Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm'],axis=1)
df = df.drop(columns=['Sunshine', 'Evaporation', 'Cloud3pm', 'Cloud9am'],axis=1)
print(df.head())
print(df.isnull().sum())

# Turning target variable into bool
df['RainToday'].replace({'No': 0, 'Yes': 1},inplace = True)
df['RainTomorrow'].replace({'No': 0, 'Yes': 1},inplace = True)
print(df.head())

print(df.describe())

# Replace missing data with the mean.
df.MinTemp.fillna(df.MinTemp.mean(),inplace=True)
df.MaxTemp.fillna(df.MaxTemp.mean(),inplace=True)
df.Rainfall.fillna(df.Rainfall.mean(),inplace=True)
df.WindSpeed9am.fillna(df.WindSpeed9am.mean(),inplace=True)
df.WindSpeed3pm.fillna(df.WindSpeed3pm.mean(),inplace=True)
df.Humidity9am.fillna(df.Humidity9am.mean(),inplace=True)
df.Temp9am.fillna(df.Temp9am.mean(),inplace=True)
df.Temp3pm.fillna(df.Temp3pm.mean(),inplace=True)
df.RainToday.fillna(df.RainToday.mean(),inplace=True)
df.Humidity3pm.fillna(df.Humidity3pm.mean(),inplace=True)
df.WindGustSpeed.fillna(df.WindGustSpeed.mean(),inplace=True)
df.Pressure9am.fillna(df.Pressure9am.mean(),inplace=True)
df.Pressure3pm.fillna(df.Pressure3pm.mean(),inplace=True)


#replace negative values with 0
df.clip(lower=0)
df[df < 0] = 0

# From selectkbest find the 5 best score and plot it
X = df.loc[:,df.columns!='RainTomorrow']
y = df[['RainTomorrow']]
selector = SelectKBest(chi2, k=5)
selector.fit(X, y)
#X_new = selector.transform(X)
#scores = selector.scores_
print(X.columns[selector.get_support(indices=True)]) #top 5 columns


# Select the selected columns for testing and training
X = df.loc[:, ['Rainfall', 'Humidity3pm', 'Humidity9am', 'WindGustSpeed', 'Temp3pm']].shift(-1).iloc[:-1].values
y = df.iloc[:-1, -1:].values.astype('int')

# Logistic Regression
# Split the data to appropriate testing sample size
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33)

# Generating R2 values
X_train_df = pd.DataFrame(X_train, columns= ['Rainfall', 'Humidity3pm', 'Humidity9am', 'WindGustSpeed', 'Temp3pm'])
y_train_df = pd.DataFrame(y_train, columns=['RainTomorrow'])
#X_train_df['intercept'] = 1.0
model = sm.logit(formula = 'y_train_df ~ X_train_df', data = X_train_df)
result = model.fit()
print(result.summary())

# Test the accuracy
model_lr = LogisticRegression(random_state=0)
model_lr.fit(X_train,y_train)
prediction_lr = model_lr.predict(X_test)
print(prediction_lr)
score = accuracy_score(y_test,prediction_lr)
print('Accuracy - Logistic Regression:',score)

# Calculating R2 for adjusted R2
prediction_lr = model_lr.predict(X_train)
R2 = r2_score(y_train,prediction_lr)
print('Sklearn R2_Score', R2)
n = y_test.size
p = 5
Adj_R2 = 1-(1-R2)*(n-1)/(n-p-1)
print('Calculated Adj R2', Adj_R2)

# Calculating f statistic
F = np.var(X_train) / np.var(y_train)
df1 = len(X_train) - 1
df2 = len(y_train) - 1
alpha = 0.05
p_value = stats.f.sf(F, df1, df2)  # survival function or 1-cdf
print('p-value of the F statistic',p_value)
if(p_value<alpha):
    print('We reject the null hypothesis')
else:
    print('We fail to reject the null hypothesis')


# # Making the Confusion Matrix
# cm = confusion_matrix(y_test, prediction_lr)
#
# # Ploting the Confusion Matrix
# f, ax = plt.subplots(figsize = (3,3))
# sns.heatmap(cm,annot=True,linecolor="red",fmt=".0f",ax=ax)
# plt.xlabel("Predictions")
# plt.ylabel("Test Values")
# plt.show()