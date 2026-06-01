import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

#----------------------------------------------------------------------#
print("----------Retail_Fraud_Detection-----------")

# Loading the dataset
df = pd.read_csv("retail_fraud_detection_100k.csv")

# Printing the first 3 rows
print("--- FIRST 3 ROWS ---")
print(df.head(3))

# Printing the last 3 rows
print("\n--- LAST 3 ROWS ---")
print(df.tail(3))

# Listing all column names
print("Random sample of data")
print(df.sample(5))

# Setting pandas options to display all columns and wide text
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("Full view of the first 5 rows:")
print(df.head(5))

# Checking the total number of elements (cells)
total_cells = df.size
print("Total number of data points (cells):", total_cells)

# Checking the index of the DataFrame
print("DataFrame Index Info:")
print(df.index)

# Checking the dimensions of the DataFrame
print("Total Dimensions (Rank):", df.ndim)

# Getting data types directly
print("\n--- Initial Column Data Types ---")
print(df.dtypes)

# Checking memory usage per column
print("Memory usage per column (in bytes):",df.memory_usage())

# Checking the axes of the DataFrame
print("DataFrame Axes:")
print(df.axes)

# Counting non-null values per column
print("Non-Null counts for each column:")
print(df.count())


# print("\n--- Processing Datetime & Hour ---")

# 1. Converting string to datetime object
df['transaction_timestamp'] = pd.to_datetime(df['transaction_timestamp'])
print("Timestamp New Dtype:", df['transaction_timestamp'].dtype)

# 2. Extracting Hour (0 to 23)
df['transaction_hour'] = df['transaction_timestamp'].dt.hour
print("Hour Column Dtype:", df['transaction_hour'].dtype)

# 3. Final Check: Do we see hours correctly?
print("\n--- Sample Verification ---")
print(df[['transaction_timestamp', 'transaction_hour']].head(5))


# ----------------------------------------------------------------------#

print("\n 1: Checking Missing (Null) Values ---")
print(df.isnull().sum())

print("\n 2: Statistical Summary of Numerical Data ---")
print(df.describe())

print("\n 3: Fraud Target Column Analysis ---")
print("Total Counts:")
print(df['fraud_flag'].value_counts())

print("\n 4: Fraud Percentage (%):")
print(df['fraud_flag'].value_counts(normalize=True) * 100)

print("\n 5: Data Cleaning & Column Inspection ---")
initial_shape = df.shape
df = df.drop_duplicates()
print(f"After: {df.shape[0]} (Before: {initial_shape[0]})")

print("\nUnique Payment Methods:",df['payment_method'].unique())

print("\nUnique Fraud Risk Categories:",df['fraud_risk'].unique())

print("\nFraud Risk Categories:",df['fraud_risk'].value_counts())

print("\nFraud Risk Categories % :",df['fraud_risk'].value_counts(normalize=True) * 100)

print("\nAvg. Transaction Amount (Fraud vs Normal):",df.groupby('fraud_flag')['transaction_amount'].mean())

payment_fraud_matrix = pd.crosstab(df['payment_method'], df['fraud_flag'])
print(payment_fraud_matrix)
print(pd.crosstab(df['payment_method'], df['fraud_flag'], normalize='index') * 100)

print("Device Type vs Fraud Percentage (%):")
print(pd.crosstab(df['device_type'], df['fraud_flag'], normalize='index') * 100)

print("\nInternational Transactions vs Fraud Percentage (%):")
print(pd.crosstab(df['is_international'], df['fraud_flag'], normalize='index') * 100)

fraud_Time = df[df['fraud_flag'] == 1]
print("\nTop 5 Hours with Highest Fraud Counts:",fraud_Time['transaction_hour'].value_counts().head(5))

print("\nUnique Transaction hour",df['transaction_hour'].unique())

print("\nCount Transaction hour",fraud_Time['transaction_hour'].value_counts())

print(df['transaction_timestamp'].head(5))

df['transaction_day_of_week'] = df['transaction_timestamp'].dt.dayofweek
print("\nUnique Days of Week (0=Mon, 6=Sun):", df['transaction_day_of_week'].unique())

threshold_amount = df['transaction_amount'].quantile(0.75)
df['is_high_value'] = (df['transaction_amount'] > threshold_amount).astype(int)

print(f"High-Value Transaction Threshold: {threshold_amount}")
print("High-Value vs Fraud Cross-tab:")
print(pd.crosstab(df['is_high_value'], df['fraud_flag'], normalize='index') * 100)

df['is_night_transaction'] = df['transaction_hour'].apply(lambda x: 1 if x in [23, 0, 1, 2, 3, 4, 5] else 0)

print("\nNight Transactions vs Fraud Percentage (%):")
print(pd.crosstab(df['is_night_transaction'], df['fraud_flag'], normalize='index') * 100)
print("\nAfter add new column:")
print(df.shape)

X = df.drop(columns=['fraud_flag', 'transaction_id', 'customer_id', 'transaction_timestamp', 'payment_method', 'device_type', 'location', 'merchant_category', 'fraud_risk'])
y = df['fraud_flag']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training Data Size: {X_train.shape[0]} rows")
print(f"Testing Data Size: {X_test.shape[0]} rows")

print("\n (Training in progress)...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Model Training Completed! 🎉")

y_pred = model.predict(X_test)

print("\n--- MODEL PERFORMANCE REPORT ---")
print("Accuracy Score:", accuracy_score(y_test, y_pred) * 100, "%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)
print(importance)

#______________________________________________________________________________#

# GRAPH 1: Pie Chart (Fraud vs Normal Transactions Percentage)
fraud_counts = df['fraud_flag'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(fraud_counts, labels=['Normal (0)', 'Fraud (1)'], autopct='%1.1f%%', colors=['skyblue', 'red'], startangle=90)
plt.title('Fraud vs Normal Transactions (%)')
plt.savefig('fraud_percentage_pie.png', bbox_inches='tight')
print("\nGraph 1 (Pie Chart) saved successfully as 'fraud_percentage_pie.png'!")
plt.show()

# GRAPH 2: Stacked Bar Chart (Payment Method with Fraud Breakdown)
payment_fraud_matrix = pd.crosstab(df['payment_method'], df['fraud_flag'])
payment_fraud_matrix.plot(kind='bar', stacked=True, color=['skyblue', 'salmon'], edgecolor='black', figsize=(8, 5))
plt.title('Total Transactions - Payment Method (Fraud vs Normal)')
plt.xlabel('Payment Method')
plt.ylabel('Transactions Count')
plt.xticks(rotation=45)
plt.legend(['Normal (0)', 'Fraud (1)'])
plt.savefig('payment_method_transactions_stacked_bar.png', bbox_inches='tight')
print("Graph 2 (Stacked Bar) saved successfully as 'payment_method_transactions_stacked_bar.png'!")
plt.show()

# GRAPH 3: Stacked Bar Chart (Merchant Category with Fraud Breakdown)
merchant_fraud_matrix = pd.crosstab(df['merchant_category'], df['fraud_flag'])
merchant_fraud_matrix.plot(kind='bar', stacked=True, color=['lightskyblue', 'orange'], edgecolor='black', figsize=(9, 5))
plt.title('Ekun Transactions - Merchant Category Nusar (Fraud vs Normal)')
plt.xlabel('Merchant Category')
plt.ylabel('Transactions Count')
plt.xticks(rotation=45)
plt.legend(['Normal (0)', 'Fraud (1)'])
plt.savefig('merchant_category_transactions_stacked_bar.png', bbox_inches='tight')
print("Graph 3 (Stacked Bar) saved successfully as 'merchant_category_transactions_stacked_bar.png'!")
plt.show()

