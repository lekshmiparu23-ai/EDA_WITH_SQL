import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

# Ensure directory exists
os.makedirs('sample_data', exist_ok=True)

random.seed(42)
np.random.seed(42)
n = 300

ages = np.random.randint(18, 76, n).tolist()
for i in random.sample(range(n), 8):
    ages[i] = random.randint(86, 100)

incomes = np.round(np.random.uniform(15000, 150000, n), 2).tolist()
for i in random.sample(range(n), 6):
    incomes[i] = round(random.uniform(200000, 350000), 2)

df = pd.DataFrame({
    'CustomerID': [f"C{str(i).zfill(3)}" for i in range(1, n+1)],
    'Age': ages,
    'Gender': random.choices(['Male','Female','Other'], weights=[48,48,4], k=n),
    'Income': incomes,
    'Education': random.choices(['Graduate','Post-Graduate','High School','Diploma'], k=n),
    'City': random.choices(['Mumbai','Delhi','Chennai','Bangalore','Hyderabad','Kolkata','Pune','Ahmedabad'], k=n),
    'PurchaseAmount': np.round(np.random.uniform(100, 50000, n), 2),
    'ProductCategory': random.choices(['Electronics','Clothing','Food','Books','Sports'], k=n),
    'Rating': np.random.randint(1, 6, n),
    'ChurnStatus': random.choices([0, 1], weights=[70, 30], k=n),
    'JoinDate': [(datetime(2019,1,1)+timedelta(days=random.randint(0,1826))).strftime('%Y-%m-%d') for _ in range(n)],
    'LastPurchaseDate': [(datetime(2023,1,1)+timedelta(days=random.randint(0,730))).strftime('%Y-%m-%d') for _ in range(n)],
})

for col, idx in [('Age', random.sample(range(n),5)),
                 ('Income', random.sample(range(n),5)),
                 ('Rating', random.sample(range(n),5))]:
    for i in idx:
        df.at[i, col] = None

df.to_csv('sample_data/sample.csv', index=False)
print("Generated sample_data/sample.csv")
