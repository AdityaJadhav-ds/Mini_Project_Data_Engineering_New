# Importing important libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pymysql
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler 
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')





# Making connection to MySQL
conn = pymysql.connect(
    host = 'localhost',
    user = 'root', # Give your own username
    password = 'pratik', # Give your own DB password
    database = 'loan_db' # Give your own DB name
)




# Importing the data 
df_applicant = pd.read_sql("SELECT * FROM applicant_info", conn)
df_financial = pd.read_sql("SELECT * FROM financial_info", conn)
df_loan = pd.read_sql("SELECT * FROM loan_info", conn)



# Getting an idea about the data
df_applicant.head()






# Getting an idea about the data
df_financial.head()





# Getting an idea about the data
df_loan.head()





# Closing the connection 
conn.close()





# Merging the data
df = df_applicant.merge(df_financial, on = 'Loan_ID', how = 'inner')
df = df.merge(df_loan, on = 'Loan_ID', how = 'inner')
df.head()




# Merging the data
df = df_applicant.merge(df_financial, on = 'Loan_ID', how = 'inner')
df = df.merge(df_loan, on = 'Loan_ID', how = 'inner')
df.head()






# Getting an idea about the duplicate rows in the dataset
df.duplicated().sum()




# 0 means that there are no duplicates in the data





# Getting an idea about the NULL values in the dataset 
df.isnull().sum().any()





# True means that there are NULL values in the dataset 





# Getting an idea about the shape of the data
print('Shape :', df.shape)
print('Rows :', df.shape[0])
print('Columns :', df.shape[1])





# Getting an idea about column wise sum of NULL values
df.isnull().sum()   





# Dealing with NULL values 
for col in df.columns:
    if df[col].dtype in [np.int64, np.float64]:
        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])





# Getting an idea about column wise sum of NULL values
df.isnull().sum()





# Getting an idea whether there are any missing values in the data
df.isnull().sum().any()



# Dealing with categorical columns 

cat_cols = df.select_dtypes(include=['object']).columns
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le





# Getting an idea about the dtypes 
df.info()



# Getting the proportion of values in Loan_Status
df['Loan_Status'].value_counts()






# Pie Chart for the proportion of Loan Status 
plt.figure(figsize=(6,6))
status_counts = df['Loan_Status'].value_counts()
plt.pie(
    status_counts, 
    labels=status_counts.index, 
    autopct='%.3f%%', 
    startangle=120, 
    colors=['Green', 'Red']  # Green for 0, Red for 1
)
plt.title("Loan Status Proportion")
plt.axis('equal')  # Equal aspect ratio for a perfect circle
plt.legend(['Yes', 'No'])
plt.show()





# Getting the proportion of values in Loan_Status
df['Gender'].value_counts()




# 0 represents Female and 1 represents Male





# Pie Chart for the proportion of Gender
plt.figure(figsize=(6,6))
gender_counts = df['Gender'].value_counts()
plt.pie(
    gender_counts, 
    labels=gender_counts.index, 
    autopct='%.3f%%', 
    startangle=120, 
    colors=['Green', 'Red']  # Green for 0, Red for 1
)
plt.title("Gender Proportion")
plt.axis('equal')  # Equal aspect ratio for a perfect circle
plt.legend(['Male', 'Female'])
plt.show()





# Getting the proportion of values in Loan_Status
df['Married'].value_counts()


# 0 represents "No" and 1 represents "Yes"




# Pie Chart for the proportion of Married Status
plt.figure(figsize=(6,6))
married_counts = df['Married'].value_counts()
plt.pie(
    married_counts, 
    labels=married_counts.index, 
    autopct='%.3f%%', 
    startangle=120, 
    colors=['Green', 'Red']  # Green for 0, Red for 1
)
plt.title("Married Proportion")
plt.axis('equal')  # Equal aspect ratio for a perfect circle
plt.legend(['Yes', 'No'])
plt.show()




# Getting the proportion of values in Education
df['Education'].value_counts()




# 0 represents "Graduate" and 1 represents "Not Graduate"




# Pie Chart for the proportion of Education Level
plt.figure(figsize=(9,6))
education_counts = df['Education'].value_counts()
plt.pie(
    education_counts, 
    labels=education_counts.index, 
    autopct='%.3f%%', 
    startangle=120, 
    colors=['Green', 'Red']  # Green for 0, Red for 1
)
plt.title("Education Proportion")
plt.axis('equal')  # Equal aspect ratio for a perfect circle
plt.legend(['Graduate', 'Not Graduate'])
plt.show()




# Getting the proportion of values in Self_Employed
df['Self_Employed'].value_counts()





# 0 represents no and 1 represents yes




# Pie Chart for the proportion of Self_Employed
plt.figure(figsize=(6,6))
self_employed_counts = df['Self_Employed'].value_counts()
plt.pie(
    self_employed_counts, 
    labels=self_employed_counts.index, 
    autopct='%.3f%%', 
    startangle=120, 
    colors=['Green', 'Red']  # Green for 0, Red for 1
)
plt.title("Self_Employed Proportion")
plt.axis('equal')  # Equal aspect ratio for a perfect circle
plt.legend(['No', 'Yes'])
plt.show()




# Getting the column names 
print(df.columns)




# What are the unique values in each column of the following columns -> 
# 'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status'
cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']
for col in cols:
    print('Unique values in', col, 'are :\n', df[col].unique())






# Statistical Analysis -> 
df.describe()





'''
Conclusions -> 
(1) In "ApplicantIncome", "CoapplicantIncome", "LoanAmount" column, mean is significantly greater than median, 
indicating that there are outliers in right tail.
(2) In "LoanAmountTerm", "CreditHistory", mean is less than median indicating that there are outliers in left tail
'''




# Which gender group has higher average loan amount ?
df.groupby('Gender')['LoanAmount'].mean()

# Male (1) has higher average loan amount than Female (0)



# What is the most common loan term ?
df['Loan_Amount_Term'].mode()[0]


# 360 months / 30 years is the most common loan term



# How many applicants have income higher than 10,000 ?
df[df['ApplicantIncome'] > 10000]['Loan_ID'].count()



# What is the maximum and minimum applicant income ?
print('Minimum Income :', df['ApplicantIncome'].min())
print('Maximum Income :', df['ApplicantIncome'].max())



# Which marital status group has higher loan approval rate ?
df.groupby('Married')['Loan_Status'].mean()



# Not Married (1) are having higher loan approval rate. 



# Which property area has higher loan approval rate ?
df.groupby('Property_Area')['Loan_Status'].mean()



# Property Area with code 1 has higher approval rate.




# What is the average coaaplicant income among approved loans ?
df[df['Loan_Status'] == 1]['CoapplicantIncome'].mean()



# Getting the columns 
print(df.columns)





# Perform Correlation Analysis among all the numerical columns 
plt.figure(figsize = (7, 3))
cm = df[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']].corr()
sns.heatmap(cm, annot = True, cmap = 'Blues')
plt.show()





'''
Conclusion -> 
(1) "Loan Amount" and "Applicant Income" are showin moderate positive correlationship.
(2) Other than that, almost all the columns are uncorrelated w.r.t. each other.
'''





# Getting the columns 
print(df.columns)




# Average Credit History by Education 
avg_credit_by_education = df.groupby("Education")["Credit_History"].mean()

# Plot bar chart
plt.figure(figsize=(7,3))
avg_credit_by_education.plot(kind="bar", color=["skyblue", "salmon"])
plt.title("Average Credit History by Education")
plt.xlabel("Education")
plt.ylabel("Average Credit History")
plt.xticks(rotation=0)
plt.show()


# Graduate (0) has higher average than Not Graduate (1)




# "Average Applicant Income" by "Loan_Amount_Term"
plt.figure(figsize=(7,6))
df.groupby('Loan_Amount_Term')['ApplicantIncome'].mean().plot(kind='bar',color=['skyblue','salmon'])
plt.title("Average Applicant_Income by Loan_Amount_Term")
plt.xlabel("Loan Amount Term")
plt.ylabel("Applicant Income")
plt.show()




# "Average Coapplicant Income" by "Loan_Amount_Term"
plt.figure(figsize=(7,6))
df.groupby('Loan_Amount_Term')['CoapplicantIncome'].mean().plot(kind='bar',color=['skyblue','salmon'])
plt.title("Average Coapplicant Income by Loan_Amount_Term")
plt.xlabel("Loan Amount Term")
plt.ylabel("CoapplicantIncome")
plt.show()





# Average Loan Amount by Self Employed 
plt.figure(figsize=(5,3))
df.groupby('Self_Employed')['LoanAmount'].mean().plot(kind='bar',color=['blue','red'])
plt.title("Average loan amount by self employed")
plt.xlabel("Self Employed")
plt.ylabel("Loan Amount")
plt.show()




# Self Employed People have higher average loan amount 



'''
Since, it is classification model, we will try for -> 
(1) Logistic Regression
(2) Decision Tree
(3) Random Forest
'''




# Seperating the data into training and testing
X = df.drop(columns=["Loan_Status", "Loan_ID"])
y = df["Loan_Status"]

# Performing train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 5)





# Logistic Regression Model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)




# Taking predictions from the model 
y_train_pred = model.predict(X_train)
y_pred = model.predict(X_test)




# Model Evaluation 
from sklearn import metrics
print('Training Accuracy :', np.round(metrics.accuracy_score(y_train, y_train_pred), 3))
print('Training Precision :', np.round(metrics.precision_score(y_train, y_train_pred), 3))
print('Training Recall :', np.round(metrics.recall_score(y_train, y_train_pred), 3))
print('Training F1 Score :', np.round(metrics.f1_score(y_train, y_train_pred), 3))

print('\nTesting Accuracy :', np.round(metrics.accuracy_score(y_test, y_pred), 3))
print('Testing Precision :', np.round(metrics.precision_score(y_test, y_pred), 3))
print('Testing Recall :', np.round(metrics.recall_score(y_test, y_pred), 3))
print('Testing F1 Score :', np.round(metrics.f1_score(y_test, y_pred), 3))




'''
Training and Testing Precision is above 0.75 and within 5% limit of each other.
'''





# Decision Tree Model
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)




# Taking predictions from the model 
y_train_pred = model.predict(X_train)
y_pred = model.predict(X_test)





# Model Evaluation 
from sklearn import metrics
print('Training Accuracy :', np.round(metrics.accuracy_score(y_train, y_train_pred), 3))
print('Training Precision :', np.round(metrics.precision_score(y_train, y_train_pred), 3))
print('Training Recall :', np.round(metrics.recall_score(y_train, y_train_pred), 3))
print('Training F1 Score :', np.round(metrics.f1_score(y_train, y_train_pred), 3))

print('\nTesting Accuracy :', np.round(metrics.accuracy_score(y_test, y_pred), 3))
print('Testing Precision :', np.round(metrics.precision_score(y_test, y_pred), 3))
print('Testing Recall :', np.round(metrics.recall_score(y_test, y_pred), 3))
print('Testing F1 Score :', np.round(metrics.f1_score(y_test, y_pred), 3))





# Creating the parameter dictionary 
from sklearn.model_selection import GridSearchCV
params = {
    'criterion' : ['gini', 'entropy', 'log_loss'],
    'max_depth' : [1, 2, 3, 4, 5],
    'min_samples_split' : [2, 5, 8, 11, 14, 17, 20],
    'min_samples_leaf' : [1, 4, 7, 10, 13, 16, 19],
    'min_impurity_decrease' : [0.00001, 0.0001, 0.001, 0.01, 0.1]
}





# Fitting the GridSearchCV()
model = DecisionTreeClassifier()
grid_search = GridSearchCV(estimator = model, param_grid = params, cv = 5, verbose = 1)

# Fitting the data
grid_search.fit(X_train, y_train)



# Getting the best parameters
print('Best Parameters :\n', grid_search.best_params_)




# Creating the optimal model 
model = DecisionTreeClassifier(criterion = 'gini', max_depth = 1, min_impurity_decrease = 1e-05, 
                               min_samples_leaf = 1, min_samples_split = 2)

# Fitting the data
model.fit(X_train, y_train)




# Taking predictions from the model 
y_train_pred = model.predict(X_train)
y_pred = model.predict(X_test)





# Model Evaluation 
from sklearn import metrics
print('Training Accuracy :', np.round(metrics.accuracy_score(y_train, y_train_pred), 3))
print('Training Precision :', np.round(metrics.precision_score(y_train, y_train_pred), 3))
print('Training Recall :', np.round(metrics.recall_score(y_train, y_train_pred), 3))
print('Training F1 Score :', np.round(metrics.f1_score(y_train, y_train_pred), 3))

print('\nTesting Accuracy :', np.round(metrics.accuracy_score(y_test, y_pred), 3))
print('Testing Precision :', np.round(metrics.precision_score(y_test, y_pred), 3))
print('Testing Recall :', np.round(metrics.recall_score(y_test, y_pred), 3))
print('Testing F1 Score :', np.round(metrics.f1_score(y_test, y_pred), 3))



# Scaling and retraining
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Retrain model with scaled data
model.fit(X_train_scaled, y_train)


# Taking predictions from the model 
y_train_pred = model.predict(X_train)
y_pred = model.predict(X_test)


# Model Evaluation 
from sklearn import metrics
print('Training Accuracy :', np.round(metrics.accuracy_score(y_train, y_train_pred), 3))
print('Training Precision :', np.round(metrics.precision_score(y_train, y_train_pred), 3))
print('Training Recall :', np.round(metrics.recall_score(y_train, y_train_pred), 3))
print('Training F1 Score :', np.round(metrics.f1_score(y_train, y_train_pred), 3))

print('\nTesting Accuracy :', np.round(metrics.accuracy_score(y_test, y_pred), 3))
print('Testing Precision :', np.round(metrics.precision_score(y_test, y_pred), 3))
print('Testing Recall :', np.round(metrics.recall_score(y_test, y_pred), 3))
print('Testing F1 Score :', np.round(metrics.f1_score(y_test, y_pred), 3))



import pickle
with open('loan_approval_model.pkl', 'wb') as f:
    pickle.dump((model, scaler, encoders), f)

