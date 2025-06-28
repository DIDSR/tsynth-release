import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
sys.path.append('../')
from utils.eval_utils import dbtex_criteria, calc_froc, calc_correct_boxes, format_arrays, calc_avg_value

# Helper function to process the results and compute TPR
def process_results(file_path, criteria, avg_fpr, use_precision=False, filter_dict=None):
    results = pd.read_csv(file_path)
    # print(results.head())
    if filter_dict is not None:
        for key, value in filter_dict.items():
            results = results[results[key] == value]
    
    for i in ['pred_boxes', 'scores', 'true_boxes']:
        results[i] = list(results[i].apply(lambda x: format_arrays(x)))
    
    results['correct_boxes'] = list(results.apply(lambda row: calc_correct_boxes(
        row['true_boxes'], row['pred_boxes'], criteria=criteria, threshold=0.5), axis=1))
    
    # results = results.rename(columns={i: f'{i}_{title}' for i in ['correct_boxes', 'pred_boxes', 'scores']})
    
    tpr, avg_fp = calc_froc(results[f'correct_boxes'], results[f'scores'], use_precision=use_precision)
    
    # Interpolation
    if avg_fpr is not None:
        tpr_interpolated = interp1d(avg_fp, tpr, kind='linear', bounds_error=False)(avg_fpr) 
        if not use_precision: 
            tpr_interpolated[avg_fpr > avg_fp.max()] = tpr[np.argmax(avg_fp)]
            tpr_interpolated[avg_fpr < avg_fp.min()] = tpr[np.argmin(avg_fp)]
        return tpr_interpolated
    else:
        return tpr, avg_fp


def plot_cone(ax, x, y, label=None, color='b', linewidth=1):
    y_avg = np.mean(y, axis=0)
    y_min = np.min(y, axis=0)
    y_max = np.max(y, axis=0)
    # Plot the average tpr
    ax.plot(x, y_avg, label=label, color=color, linewidth=linewidth)
    # Fill the area between the min and max tpr to create the cone
    ax.fill_between(x, y_min, y_max, color=color, alpha=0.3, label=None)

def plot_results(ax, result_paths, key, key_values, key_labels, colors, avg_fpr):
    # Loop through each subplot (one for each fpr value)
    for key_value, key_label, color in zip(key_values, key_labels, colors):
        tpr_list = []
        for result_path in result_paths:
            # Mixed name processing
            tpr_mixed = process_results(result_path, dbtex_criteria, avg_fpr,
                                        filter_dict={key: key_value})
            tpr_list.append(tpr_mixed)
        
        plot_cone(ax, avg_fpr, tpr_list, color=color, label=key_label, linewidth=1.5)

    # Set grid with improved visibility
    ax.grid(linestyle='--', alpha=0.7, linewidth=0.6)
    ax.set_ylim([0, 1]) 
    ax.set_xlim([min(avg_fpr), max(avg_fpr)])
    # ax.set_xlim([min(avg_fpr), 2.5])
    # Customize tick labels for clarity
    ax.tick_params(axis='both', labelsize=9) 

    # Adjust the layout for better spacing
    ax.set_xlabel('FPR (per Image)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Sensitivity', fontsize=11, fontweight='bold')
    return ax

# Helper functions for plots

def plot_froc_overall(
    root_dir,
    scenario_names,
    labels,
    num_readers=5,
    avg_fpr=np.linspace(0, 5, 100),
    use_precision=False,
    colors=None,
    save_path=None
):
    if colors is None:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd' ,'#bcbd22'] 

    fig, ax = plt.subplots(figsize=(6, 4.5))
    
    for scenario_name, color in zip(scenario_names, colors):
        tpr_list = []
        for i in range(1, num_readers + 1):
            path = f'{root_dir}/{scenario_name}/{i}/result.csv'
            tpr = process_results(path, dbtex_criteria, avg_fpr, use_precision=use_precision)
            tpr_list.append(tpr)
        plot_cone(ax, avg_fpr, tpr_list, label=labels[scenario_name], color=color, linewidth=1.5)
        

    
    ax.set_xlabel('False Positive Rate (per Image)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Sensitivity', fontsize=14, fontweight='bold')
    ax.grid(linestyle='--', alpha=0.7, linewidth=0.6)
    ax.set_ylim([0, 1])
    ax.set_xlim([0, 5])
    ax.tick_params(axis='both', labelsize=9)

    legend = ax.legend(fontsize=11, title_fontsize=12, loc='lower right', frameon=True)
    legend.get_frame().set_edgecolor('black')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
    
def plot_froc_by_density(
    root_dir,
    scenario_names,
    labels,
    num_readers=5,
    bd_values=[1, 2, 3, 4],
    bd_labels=['Fatty', 'Scattered', 'Heterogeneous', 'Dense'],
    avg_fpr=np.linspace(0, 5, 100),
    use_precision=False,
    colors=None,
    save_path=None
):
    if colors is None:
        colors =['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd' ,'#bcbd22'] 

    fig, axs = plt.subplots(2, 2, figsize=(6, 6))

    for ax, bd, bd_label in zip(axs.flatten(), bd_values, bd_labels):
        for scenario_name, color in zip(scenario_names, colors):
            tpr_list = []
            for i in range(1, num_readers + 1):
                path = f'{root_dir}/{scenario_name}/{i}/result.csv'
                tpr = process_results(path, dbtex_criteria, avg_fpr, use_precision=use_precision, filter_dict={'breast_density': bd})
                tpr_list.append(tpr)
            plot_cone(ax, avg_fpr, tpr_list, label=labels[scenario_name], color=color, linewidth=1.5)

        ax.set_title(bd_label, fontsize=10, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.set_xlim([0, 5])
        ax.grid(linestyle='--', alpha=0.7, linewidth=0.6)
        ax.tick_params(axis='both', labelsize=9)

    axs[1, 0].set_xlabel('FPR (per Image)', fontsize=11, fontweight='bold')
    axs[1, 1].set_xlabel('FPR (per Image)', fontsize=11, fontweight='bold')
    axs[0, 0].set_ylabel('Sensitivity', fontsize=11, fontweight='bold')
    axs[1, 0].set_ylabel('Sensitivity', fontsize=11, fontweight='bold')

    legend = axs[1, 1].legend(fontsize=8, title_fontsize=9, loc='lower right', frameon=True)
    legend.get_frame().set_edgecolor('black')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
