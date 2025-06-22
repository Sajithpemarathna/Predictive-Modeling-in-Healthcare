import pandas as pd
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv(r"C:\Users\PC\OneDrive\Desktop\MSc Data Analytics\Predictive Analytics and Machine Learning using Python\8 Assignments\EDA\Weekly Aggregated Sales.csv")
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
df.set_index('Date', inplace=True)

# Weekly aggregation
weekly_sales = df.resample('W').sum()

# Adding Week Number and Date Range
weekly_sales['Week_Number'] = weekly_sales.index.isocalendar().week
weekly_sales['Start_Date'] = weekly_sales.index - pd.to_timedelta(weekly_sales.index.weekday, unit='d')
weekly_sales['End_Date'] = weekly_sales['Start_Date'] + pd.DateOffset(days=6)
weekly_sales['Week_Range'] = weekly_sales['Start_Date'].dt.strftime('%Y-%m-%d') + ' to ' + weekly_sales['End_Date'].dt.strftime('%Y-%m-%d')

# Outlier removal using Interquartile Range
Q1 = weekly_sales['Sales ($)'].quantile(0.25)
Q3 = weekly_sales['Sales ($)'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

filtered_sales = weekly_sales[(weekly_sales['Sales ($)'] >= lower_bound) & (weekly_sales['Sales ($)'] <= upper_bound)]

# Exporting the table to Excel
export_table = filtered_sales[['Week_Number', 'Week_Range', 'Sales ($)']]
export_table.to_excel(r"C:\Users\PC\OneDrive\Desktop\Weekly_Sales_Summary.xlsx", index=False)

# Plotting histogram
plt.figure(figsize=(10, 6))
plt.bar(filtered_sales['Week_Number'].astype(str), filtered_sales['Sales ($)'], edgecolor='black')
plt.title('Weekly Sales Amount by Week Number')
plt.xlabel('Week Number')
plt.ylabel('Aggregated Sales ($)')
plt.xticks(rotation=45, ha="right")  # Rotate the x-axis labels for better readability
plt.tight_layout()  # Adjust layout to fit the x-axis labels
plt.show()
