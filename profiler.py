import matplotlib.pyplot as plt
import csv

# Load data from CSV
ncalls = []
cumtimes = []
tottimes = []
function_names = []

with open('camera_app/profiler.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        # Skip rows with invalid data for 'ncalls' and convert to integer
        if '/' in row['ncalls']:
            continue

        ncalls_value = int(row['ncalls'])
        cumtime_value = float(row['cumtime'])
        tottime_value = float(row['tottime'])
        function_name = row['filename:lineno(function)']

        # Filter data
        if ncalls_value > 5 and cumtime_value > 0.001:
            ncalls.append(ncalls_value)
            cumtimes.append(cumtime_value)
            tottimes.append(tottime_value)
            function_names.append(function_name)

# Adjust bubble size
scaling_factor = 50000  # Further increase the scaling factor for better visibility
sizes = [t**0.5 * scaling_factor for t in tottimes]

# Create scatter plot without logarithmic scale
plt.figure(figsize=(12, 8))
plt.scatter(ncalls, cumtimes, s=sizes, alpha=0.5, color='skyblue')

# Annotate key points
for i, function in enumerate(function_names):
    if cumtimes[i] == max(cumtimes) or tottimes[i] == max(tottimes):
        plt.text(ncalls[i], cumtimes[i], function, fontsize=8)

plt.xlabel('Number of Calls (ncalls)')
plt.ylabel('Cumulative Time (cumtime)')
plt.title('Function Profiling: Calls vs. Cumulative Time with Total Time as Bubble Size')
plt.grid(True)
plt.tight_layout()

# Save the plot
plt.savefig('profiler_bubble_plot_updated.png')
