import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
data = pd.read_csv('./records/epoch_data.csv')

# Group the data by epoch and calculate quintiles based on fitness
data['quintile'] = data.groupby('epoch')['fitness'].transform(lambda x: pd.qcut(x, 5, labels=False))

# Calculate the average fitness for each epoch and quintile
quintile_avg = data.groupby(['epoch', 'quintile'])['fitness'].mean().reset_index()

# Create a line chart
plt.figure(figsize=(10, 6))
for quintile in range(5):
    quintile_data = quintile_avg[quintile_avg['quintile'] == quintile]
    plt.plot(quintile_data['epoch'], quintile_data['fitness'], label=f'Quintile {quintile+1}')

# Print the lowest 10 entries by fitness
lowest_10 = data.nsmallest(10, 'fitness')[['name', 'fitness', 'epoch']]
print("Lowest 10 entries by fitness:")
print(lowest_10)

plt.xlabel('Epoch')
plt.ylabel('Average Fitness')
plt.title('Average Fitness for Epoch Quintiles Over Time')
plt.legend()
plt.grid(True)
plt.show()