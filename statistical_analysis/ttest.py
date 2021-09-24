import csv
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
import seaborn as sns

def load_subjects_cbf(subjects, obs_name1, obs_name2):
    all_values1 = []
    all_values2 = []

    for subject in subjects:
        obs1 = subject.get_observation(obs_name1)
        obs2 = subject.get_observation(obs_name2)

        asl1 = obs1.asl
        asl2 = obs2.asl
        try:
            values1 = asl1.load_csv('roi_mean_cbf.csv')['Mean'].values
            values2 = asl2.load_csv('roi_mean_cbf.csv')['Mean'].values

            all_values1.append(values1)
            all_values2.append(values2)
        except FileNotFoundError:
            print('{}, lack CBF file'.format(subject.name))

    # transpose all_values to shape (nodal, subject)
    all_values1 = np.array(all_values1).T
    all_values2 = np.array(all_values2).T
    return all_values1, all_values2

def load_subjects_gmv(subjects, obs_name1, obs_name2):
    all_values1 = []
    all_values2 = []

    for subject in subjects:
        obs1 = subject.get_observation(obs_name1)
        obs2 = subject.get_observation(obs_name2)

        t11 = obs1.t1
        t12 = obs2.t1

        values1 = t11.load_csv('label/roi_gmv_{}.csv')['Volume'].values
        values2 = t12.load_csv('label/roi_gmv_{}.csv')['Volume'].values

        all_values1.append(values1)
        all_values2.append(values2)

    # transpose all_values to shape (nodal, subject)
    all_values1 = np.array(all_values1).T
    all_values2 = np.array(all_values2).T
    return all_values1, all_values2

def load_subjects_global_metric(subjects, obs_name, mat_name, metric_name, threshold_type, threshold_value):
    metrics = []
    for subject in subjects:
        # dti = subject.get_observation(obs_name).dti #0803添加
        bold = subject.get_observation(obs_name).bold

        # metric = dti.get_global_metric(metric_name, mat_name, threshold_type, threshold_value=threshold_value) #0803添加
        metric = bold.get_global_metric(metric_name, mat_name, threshold_type, threshold_value=threshold_value)

        metrics.append(metric)
    metrics = np.array(metrics)
    return metrics

def load_subjects_nodal_metric(subjects, obs_name1, obs_name2, mat_name,
                               metric_name, threshold_type, threshold_value):
    all_values1 = []
    all_values2 = []

    for subject in subjects:
        obs1 = subject.get_observation(obs_name1)
        obs2 = subject.get_observation(obs_name2)

        bold1 = obs1.bold
        bold2 = obs2.bold

        values1 = bold1.get_nodal_metric(metric_name, mat_name=mat_name,
                                        threshold_type=threshold_type, threshold_value=threshold_value)
        values2 = bold2.get_nodal_metric(metric_name, mat_name=mat_name,
                                        threshold_type=threshold_type, threshold_value=threshold_value)

        all_values1.append(values1)
        all_values2.append(values2)

    # transpose all_values to shape (nodal, subject)
    all_values1 = np.array(all_values1).T
    all_values2 = np.array(all_values2).T
    return all_values1, all_values2

def load_clinical(subjects, obs_name, metric_name):
    metrics = []
    for subject in subjects:
        obs = subject.get_observation(obs_name)
        if metric_name == 'handtremor':
            metrics.append(obs.args['handtremor'])
        elif metric_name == 'CRST_A':
            metrics.append(obs.args['CRSTA_total'])
        elif metric_name == 'CRST_B':
            metrics.append(obs.args['CRST b_total'])
        elif metric_name == 'CRST_C':
            metrics.append(obs.args['CRST C'])
        elif metric_name == 'CRST_TOTAL':
            metrics.append(obs.args['CRST TOTAL'])
    metrics = np.array(metrics)
    return metrics

def save_nodal_ttest(all_values1, all_values2,
                     t1_mask, out_csv_path, out_nii_path):
    i = 1
    t_dict = {}
    with open(out_csv_path, 'w', newline='') as file:
        
        fieldnames = ['ID', 't-value', 'p-value']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for values1, values2 in zip(all_values1, all_values2):
            t, p = ttest_rel(values1, values2)
            writer.writerow({'ID': i, 't-value': t, 'p-value': p})
            t_dict[i] = t
            i += 1
    t1_mask.save_values(t_dict, out_nii_path)

# graph metric ttest
def graph_metric_ttest(subjects, t1_mask,
                        out_csv_path_ske, out_nii_path_ske,
                        obs_name1='base', obs_name2='360d',
                        threshold_type='intensity',
                        threshold_value=0.006):
    metrics = ['neighbor_degree',
                'betweenness_centrality', 'degree',
                'fault_tolerance', 'shortest_path_length',
                'global_efficiency', 'clustering_coefficient',
                'local_efficiency', 'vulnerability']

    for metric in metrics:
        out_csv_path = out_csv_path_ske.format(metric)
        out_nii_path = out_nii_path_ske.format(metric)
        all_values1, all_values2 = load_subjects_nodal_metric(subjects, obs_name1, obs_name2,
                                                              metric, threshold_type, threshold_value)
        save_nodal_ttest(all_values1, all_values2, t1_mask,
                        out_csv_path, out_nii_path)
# GMV ttest
def gmv_ttest(subjects, t1_mask, out_csv_path, out_nii_path, 
                obs_name1='base', obs_name2='360d', threshold_value=0.006):
    all_values1, all_values2 = load_subjects_gmv(subjects, obs_name1, obs_name2)
    save_nodal_ttest(all_values1, all_values2, t1_mask,
                    out_csv_path, out_nii_path)

def cbf_ttest(subjects, t1_mask, out_csv_path, out_nii_path,
            obs_name1='base', obs_name2='360d', threshold_value=0.006):
    all_values1, all_values2 = load_subjects_cbf(subjects, obs_name1, obs_name2)
    save_nodal_ttest(all_values1, all_values2, t1_mask,
                    out_csv_path, out_nii_path)

def load_subjects_surgery_info(subjects, obs_name, info_name):
    infos = []
    for subject in subjects:
        obs = subject.get_observation(obs_name)
        value = float(obs.args[info_name])
        infos.append(value)
    infos = np.array(infos)
    return infos