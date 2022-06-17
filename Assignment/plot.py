import matplotlib.pyplot as plt
import pandas as pd


data_path = "data"
ymax = 8500

# iterate over all files in condition group
for i in range(1, 24):
    my_file = data_path + '/condition/condition_' + str(i) + '.csv'   
    df_temp = pd.read_csv(my_file)
    
    # plot full activity time series
    my_alpha=0.25
    fig, ax = plt.subplots(figsize=(18,6))
    ax.scatter(df_temp.timestamp, df_temp.activity , alpha=my_alpha)
    ax.xaxis.set_major_locator(plt.MaxNLocator(20)) # reduce number of x-axis labels
    ax.set_ylim(0, ymax)
    plt.title(my_file)
    plt.xticks(rotation=90)
    plt.grid()
    ax.legend(loc='upper left')
    plt.show()

for i in range(1, 33):
    my_file = data_path + '/control/control_' + str(i) + '.csv'   
    df_temp = pd.read_csv(my_file)
    
    # plot full activity time series
    my_alpha=0.25
    fig, ax = plt.subplots(figsize=(18,6))
    ax.scatter(df_temp.timestamp, df_temp.activity , alpha=my_alpha)
    ax.xaxis.set_major_locator(plt.MaxNLocator(20)) # reduce number of x-axis labels
    ax.set_ylim(0, ymax)
    plt.title(my_file)
    plt.xticks(rotation=90)
    plt.grid()
    ax.legend(loc='upper left')
    plt.show()