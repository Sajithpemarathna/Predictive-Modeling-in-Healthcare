import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
df = pd.read_csv(r"C:\Users\PC\OneDrive\Desktop\MSc Data Analytics\Predictive Analytics and Machine Learning using Python\8 Assignments\EDA\Weekly Aggregated Sales.csv")
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

# Calculating skewness of sales data
sales_skewness = df['Sales ($)'].skew()

# Plotting density plot
plt.figure(figsize=(10, 6))
sns.kdeplot(df['Sales ($)'], fill=True)
plt.title('Sales Data Density Plot')
plt.xlabel('Sales ($)')
plt.ylabel('Density')

# Displaying the skewness value
plt.text(0.95, 0.95, f'Skewness: {sales_skewness:.2f}', horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

plt.show()
