import pandas as pd

# Load your dataset into a DataFrame
dataset = pd.read_csv(r"C:\Users\PC\OneDrive\Desktop\MSc Data Analytics\Predictive Analytics and Machine Learning using Python\8 Assignments\EDA\US  E-commerce records 2020.csv", encoding='ISO-8859-1')

# Replace 'column1' and 'column2' with the names of the columns you want to analyze
column1 = dataset['Quantity']
column2 = dataset['Profit']

# Calculate Pearson's correlation coefficient
pearson_corr = column1.corr(column2, method='pearson')
print(f"Pearson's correlation coefficient: {pearson_corr}")


