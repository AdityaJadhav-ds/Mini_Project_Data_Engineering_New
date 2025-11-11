import pandas as pd
import pymysql
import numpy as np

# ========== 1. Load JSON ==========
df_applicant = pd.read_json("applicant_info.json", lines=True)
df_financial = pd.read_json("financial_info.json", lines=True)
df_loan = pd.read_json("loan_info.json", lines=True)

print("✅ JSON files loaded")
print("Applicant columns:", df_applicant.columns.tolist())
print("Financial columns:", df_financial.columns.tolist())
print("Loan columns:", df_loan.columns.tolist())

# ========== 2. Preprocess (NaN → None) ==========
def clean_df(df):
    df = df.replace({np.nan: None})
    return df

df_applicant = clean_df(df_applicant)
df_financial = clean_df(df_financial)
df_loan = clean_df(df_loan)

# ========== 3. Connect to MySQL (NO DB selected yet) ==========
conn = pymysql.connect(
    host="localhost",
    user="root",        # 🔧 change if needed
    password="pratik", # 🔧 change if needed
    autocommit=True
)
cursor = conn.cursor()
print("✅ Connected to MySQL server")

# ========== 4. Create Database & Use ==========
cursor.execute("CREATE DATABASE IF NOT EXISTS loan_db")
cursor.execute("USE loan_db")

# ========== 5. Create Tables ==========
cursor.execute("""
CREATE TABLE IF NOT EXISTS applicant_info (
    Loan_ID VARCHAR(50) PRIMARY KEY,
    Gender VARCHAR(10),
    Married VARCHAR(10),
    Dependents VARCHAR(10),
    Education VARCHAR(20),
    Self_Employed VARCHAR(10)
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS financial_info (
    Loan_ID VARCHAR(50) PRIMARY KEY,
    ApplicantIncome FLOAT,
    CoapplicantIncome FLOAT,
    LoanAmount FLOAT,
    Loan_Amount_Term FLOAT,
    Credit_History FLOAT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS loan_info (
    Loan_ID VARCHAR(50) PRIMARY KEY,
    Property_Area VARCHAR(20),
    Loan_Status VARCHAR(10)
)
""")
print("✅ Tables created (if not exist)")

# ========== 6. Insert Data ==========
def insert_data(df, table_name):
    df = df.where(pd.notnull(df), None)  # replace NaN → None
    cols = ",".join([f"`{col}`" for col in df.columns])
    placeholders = ",".join(["%s"] * len(df.columns))
    insert_sql = f"REPLACE INTO `{table_name}` ({cols}) VALUES ({placeholders})"

    clean_data = []
    for row in df.itertuples(index=False, name=None):
        clean_row = [None if (isinstance(val, float) and np.isnan(val)) else val for val in row]
        clean_data.append(tuple(clean_row))

    cursor.executemany(insert_sql, clean_data)
    conn.commit()
    print(f"✅ Inserted {len(df)} rows into {table_name}")

insert_data(df_applicant, "applicant_info")
insert_data(df_financial, "financial_info")
insert_data(df_loan, "loan_info")

cursor.close()
conn.close()
print("🎉 All data successfully loaded into MySQL!")
