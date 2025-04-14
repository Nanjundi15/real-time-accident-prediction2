
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ====== Load the Preprocessed Dataset ======
# If the CSV file is inside "real time" folder and the script is in "major project"
file_path = "D:/cyber security/major project/real time/Project/Road_Accident_Data_Reduced.csv"

accident_df = pd.read_csv(file_path)



# ====== Convert DateTime Column to Datetime Format ======
accident_df['Datetime'] = pd.to_datetime(accident_df['Accident Date'] + ' ' + accident_df['Time'])

# ====== Set Plot Style ======
sns.set_style("whitegrid")

# ====== 1. Accident Severity vs. Weather Conditions ======
plt.figure(figsize=(12, 6))
severity_weather = accident_df.groupby(['Weather_Conditions_Num', 'Accident_Severity_Num']).size().unstack(fill_value=0)
severity_weather.plot(kind='bar', stacked=True, colormap='viridis', figsize=(14, 7))
plt.title('Accident Severity by Weather Conditions')
plt.xlabel('Weather Conditions (0=Clear, 1=Rain, 2=High Winds, 3=Snow, 4=Fog, 5=Other, 6=Unknown)')
plt.ylabel('Number of Accidents')
plt.legend(title='Severity (3=Fatal, 2=Serious, 1=Slight)')
plt.show()

# ====== 2. Scatter Plot of Accident Locations ======
plt.figure(figsize=(12, 8))
sns.scatterplot(x='Longitude', y='Latitude', hue='Accident_Severity_Num', size='Speed_limit', 
                data=accident_df, palette='coolwarm', sizes=(20, 200), alpha=0.7)
plt.title('Scatter Plot of Accident Locations')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(title='Severity')
plt.show()

# ====== 3. Accident Frequency by Time ======
plt.figure(figsize=(14, 7))
accident_df['Datetime'].dt.date.value_counts().sort_index().plot(kind='line', color='blue', marker='o')
plt.title('Accident Frequency Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Accidents')
plt.grid(True)
plt.show()
