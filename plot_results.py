import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas as pd

def read_results(file='./valid/results.csv'):
    return pd.read_csv(file)

def string_to_list(string):
    # string: '[2.11, 3.11]'
    list_strings = string[1:-1].split(', ')
    list_float = [float(x) for x in list_strings]
    return list_float


if __name__ == '__main__':
    # collect data
    if len(sys.argv) > 2:
        results_df = read_results(sys.argv[2])
    else:
        results_df = read_results()

    run_names = []
    data = []
    for col in sorted(results_df.columns):
        vals_string = results_df[col].values[0]
        vals = string_to_list(vals_string)
        run_names.append(col)
        data.append(vals)

    # make fig
    fig = plt.figure()
    ax = fig.add_subplot(111)
    bp = ax.boxplot(data, vert=0)
    ax.set_yticklabels(run_names)
    ax.set_xlabel('MAE')
    # ax.set_title('Comparing models with different train/val splits')

    if len(sys.argv) > 1:
        name = sys.argv[1]
    else:
        name = 'comparison'
    plt.savefig('./figs/'+name+'.png', bbox_inches="tight")
    # show plot
    # plt.show()
