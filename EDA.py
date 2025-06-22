import pandas as pd

# Load your dataset into a DataFrame
dataset = pd.read_csv(r"C:\Users\PC\OneDrive\Desktop\MSc Data Analytics\Predictive Analytics and Machine Learning using Python\8 Assignments\EDA\US  E-commerce records 2020.csv", encoding='ISO-8859-1')

# Use describe() to get descriptive statistics for numerical columns
statistics = dataset.describe()

# Specify a new file path (replace 'Your_Documents_Path' with an actual path)
new_file_path = r'Your_Documents_Path\Statistics_Descriptive.xlsx'

# Save the descriptive statistics to an Excel file
statistics.to_excel(new_file_path)

print(f"Descriptive statistics saved to Excel file at {new_file_path}.")
