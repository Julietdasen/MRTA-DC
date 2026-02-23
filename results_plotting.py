import glob
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import ttest_rel
from natsort import natsorted

# plot all csv results in a folder and plot all same label in one figure
folder = 'testSet_20A_50T_CONDET'
if not os.path.exists(f'{folder}/metrics'):
    os.mkdir(f'{folder}/metrics')
dfs = []
files = []
metric_specs = [
    ('Success Rate', ['success_rate']),
    ('Makespan', ['makespan']),
    ('Time Cost', ['time_cost']),
    ('Average Waiting Time', ['waiting_time']),
    ('Sum Traveling Distance', ['travel_dist']),
    # Backward-compatible: prefer utilization_exec, fallback to legacy efficiency.
    ('Efficiency', ['utilization_exec', 'efficiency']),
]
optional_metric_specs = [
    ('Utilization Wait', ['utilization_wait']),
    ('Utilization Travel', ['utilization_travel']),
]
methods = natsorted(glob.glob(f'{folder}/*.csv'), key=lambda y: y.lower())
for file in methods:
    if file.endswith('.csv'):
        files.append(file.split('/')[1].replace('.csv', ''))
        dfs.append(pd.read_csv(file))

def get_metric_series(df, candidates):
    for col in candidates:
        if col in df.columns:
            return df[col]
    return None


p_metrics = pd.DataFrame(columns=['Method'] + files)
for metric_name, candidates in metric_specs:
    for i, df_i in enumerate(dfs):
        p = dict()
        series_i = get_metric_series(df_i, candidates)
        for j, df_j in enumerate(dfs):
            if df_i is not df_j:
                series_j = get_metric_series(df_j, candidates)
                if series_i is None or series_j is None:
                    p[files[j]] = 'NA, NA'
                else:
                    result = ttest_rel(series_i.values, series_j.values)
                    p[files[j]] = np.format_float_scientific(result.statistic, 2) + ', ' + np.format_float_scientific(result.pvalue, 2)
            else:
                p[files[j]] = '0, 0'
        p['Method'] = files[i] + ' ' + metric_name
        p = pd.DataFrame(p, index=[files[i]])
        p_metrics = pd.concat([p_metrics, p])
p_metrics.to_csv(f'{folder}/metrics/p_metrics.csv', index=False)


metrics_csv = pd.DataFrame(columns=['Method'] + [m[0] for m in metric_specs] + [m[0] for m in optional_metric_specs])

for i, df in enumerate(dfs):
    metrics = dict()
    for metric_name, candidates in metric_specs + optional_metric_specs:
        series = get_metric_series(df, candidates)
        if series is None:
            metrics[metric_name] = 'NA'
            continue
        if candidates[0] == 'success_rate':
            metrics[metric_name] = (np.sum(series) / len(series)).round(3).astype('str') + ' (+- ' + np.nanstd(series).round(3).astype('str') + ')'
        else:
            metrics[metric_name] = np.nanmean(series).round(3).astype('str') + ' (+- ' + np.nanstd(series).round(3).astype('str') + ')'
    metrics['Method'] = files[i]
    metrics = pd.DataFrame(metrics, index=[files[i]])
    metrics_csv = pd.concat([metrics_csv, metrics])
metrics_csv.to_csv(f'{folder}/metrics/metrics.csv', index=False)

for metric_name, candidates in metric_specs:
    plt.figure(dpi=300)
    for id, df in enumerate(dfs):
        series = get_metric_series(df, candidates)
        if series is None:
            continue
        plt.plot(series, label=files[id])
    plt.legend()
    plt.title(metric_name)
    plt.savefig(f'{folder}/metrics/{metric_name}.png')
    plt.close()

# plot average results of all csv files in a folder and error bar
for metric_name, candidates in metric_specs + optional_metric_specs:
    plt.figure(dpi=300)
    plotted = False
    for idx, df in enumerate(dfs):
        series = get_metric_series(df, candidates)
        if series is None:
            continue
        plotted = True
        # plot average and error bar
        mean = np.nanmean(series)
        std = np.nanstd(series)
        min_ = np.min(series)
        max_ = np.max(series)
        plt.errorbar(idx, mean, std, fmt='b', lw=3, alpha=0.5)
        plt.errorbar(idx, mean, np.array([[np.round(mean - min_, 4)], [np.round(max_ - mean, 4)]]), fmt='.', lw=1, label=files[idx])
    if not plotted:
        plt.close()
        continue
    plt.legend(fontsize="7")
    plt.xticks([])
    plt.title(metric_name)
    plt.savefig(f'{folder}/metrics/{metric_name} Average.png')
    plt.close()
