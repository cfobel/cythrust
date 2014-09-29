import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cythrust.si_prefix import si_format  # SI engineering formatting


def plot_runtime_comparison(runtimes):
    '''
    Plot a bar plot comparison of runtimes in a `pandas.DataFrame` with the
    following columns:

     - `N`: Element count.
     - `alg`: Algorithm label.
     - `runtime`: Runtime in seconds.
    '''
    algs = runtimes['alg'].unique()
    N_algs = len(algs)

    fig = plt.figure(figsize=(10, 3))
    axis = fig.add_subplot(111)
    color_cycle = axis._get_lines.color_cycle
    group_width = N_algs + 1

    for i, alg in enumerate(algs):
        results = runtimes[runtimes['alg'] == alg]
        N_results = len(results['runtime'])
        axis.bar(group_width * np.arange(N_results) + i,
                 results['runtime'], color=color_cycle.next(), label=alg)
    axis.legend(loc='upper left')
    axis.set_xticks(group_width * np.arange(N_results) +
                    (group_width - 1) / 2.)
    axis.set_xticklabels([si_format(N) for N in results['N']])
    axis.set_yticklabels(['%ss' % si_format(t) for t in axis.get_yticks()])
    axis.set_xlabel('`N`')
    axis.set_ylabel('Runtime\n(lower=better)')
