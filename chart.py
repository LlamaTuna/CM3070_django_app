import matplotlib.pyplot as plt

# Data for the first iteration
features = ['User Interface', 'Facial Recognition', 'Motion Detection', 'Alert Descriptions']
responses = [3, 4, 5, 5]

# Plotting the bar chart for the first iteration
plt.figure(figsize=(10, 5))
plt.bar(features, responses, color='skyblue')
plt.ylim(0, 5)  # Limiting y-axis to 0-5 since there are 5 users
plt.title('First Iteration Testing Results')
plt.xlabel('Features')
plt.ylabel('Number of Users Responding Positively')

# Save the figure to a file
plt.savefig('first_iteration_results.png')

# Data for the second iteration
features = ['Person Detected', 'Other Objects Detected']
responses = [5, 'Mixed Results']

# Plotting the bar chart for the second iteration
plt.figure(figsize=(10, 5))
plt.bar(features, [5, 2.5], color='lightgreen')  # 2.5 represents mixed results in a generalized way
plt.ylim(0, 5)
plt.title('Second Iteration Testing Results')
plt.xlabel('Features')
plt.ylabel('Number of Users Responding Positively')

# Save the figure to a file
plt.savefig('second_iteration_results.png')

print("Charts have been saved as 'first_iteration_results.png' and 'second_iteration_results.png'.")
