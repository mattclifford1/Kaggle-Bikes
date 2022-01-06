import matplotlib.pyplot as plt
import numpy as np
from utils import read_results, string_to_list




if __name__ == '__main__':
    # collect data
    results_df = read_results()
    run_names = []
    data = []
    for col in results_df.columns:
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
    ax.set_title('Comparing models with different train/val splits')
    plt.savefig('./figs/modelcomparision.png', bbox_inches="tight")
    # show plot
    # plt.show()
