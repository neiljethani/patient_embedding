import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

import os
import datetime as dt

import arrow


parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, default= '/home/neil.jethani/patient_embedding/vis/PERCENT', help='Directory relative which all output files are stored')
parser.add_argument('--model_dir', type=str, default= '/home/neil.jethani/patient_embedding/models', help='Directory where models are stored')
args = parser.parse_args()
print(args)

#Plotting Fonts
SMALL_SIZE = 12
MEDIUM_SIZE = 16
BIGGER_SIZE = 20

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)

#Collect Statisics: Each Embedding Type, Each Task, Each Percentage 
results = {'task':[], 'model':[], 'percent':[], 'auroc':[], 'auprc':[]}
for task in ['extended_length_of_stay', 'decompensation', 
             'discharge', 'in_hospital_mortality']:
    for model in ['RAW', 'DAE', 'TRANS', 'DFE', 'PCA']:
        for percent in ['1', '5', '10', '25', '50', '100']:
            if percent == '100':
                model_path = os.path.join(args.model_dir, task, model)
            else:
                model_path = os.path.join(args.model_dir, task, model, percent)
            dates = []
            for date in os.listdir(model_path):
                try:
                    dates.append(dt.datetime.strptime(date, '%Y-%m-%d').date())
                except:
                    continue
            model_date = max(dates).strftime('%Y-%m-%d')
            model_path = os.path.join(model_path, str(model_date))
            metric = pd.read_csv(os.path.join(model_path, 'lossfile.csv'))
            test = metric[metric['dataset']=='test'].loc[0:, :]
            train = metric[metric['dataset']=='train'].loc[0:, :]
            
            auroc = test['auroc'].max()
            auprc = test['auprc'].max()
            
            results['task'].append(task)
            results['model'].append(model)
            results['percent'].append(int(percent))
            results['auroc'].append(auroc)
            results['auprc'].append(auprc)

results = pd.DataFrame(results)

for task in ['extended_length_of_stay', 'decompensation', 
             'discharge', 'in_hospital_mortality']:
    fig, axs = plt.subplots(1,2)
    fig.set_size_inches(20, 8)
    fig.suptitle('Effect of Data Availability for Predicting {}'.format(task))
    result = results[results['task']==task]
    result.drop(columns=['task'])
    result.pivot(index='percent', columns='model', values='auroc').plot(ax=axs[0])
    result.pivot(index='percent', columns='model', values='auprc').plot(ax=axs[1])
    axs[0].set_ylabel('AUROC')
    axs[1].set_ylabel('AUPRC')
    axs[0].set_xlabel('Percent of Training Data')
    axs[1].set_xlabel('Percent of Training Data')
    save_dir = os.path.join(args.output_dir, arrow.now().format('YYYY-MM-DD'))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, '{}.png'.format(task)))