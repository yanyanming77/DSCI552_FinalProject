#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 15:07:34 2021

@author: mingyan
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

os.chdir('/Users/mingyan/Desktop/USC/courses/DSCI_552/final_project')
print(os.getcwd())

# read data
data_original = pd.read_csv('dataset/data_data_new.csv')
data = data_original.copy().reset_index(drop = True)
print(data.info()) # 48784 rows, 160 columns


print(data['race'].value_counts())
# 43202 white, 5582 black


###################################################################################################
# separate white and black
data_white = data.loc[data['race'] == 'white',:]
data_black = data.loc[data['race'] == 'black',:]

data_white['dem_female'].value_counts(normalize=True) # 27077 (62.67% )female
data_black['dem_female'].value_counts(normalize=True) # 3686 (66.93% ) female

data_white['risk_score_rank1'] = pd.qcut(data_white['risk_score_t'], 10, labels = False)
data_black['risk_score_rank1'] = pd.qcut(data_black['risk_score_t'], 10, labels = False)

grouped_white = data_white.groupby('risk_score_rank1')['gagne_sum_t'].mean().to_frame().reset_index()
grouped_black = data_black.groupby('risk_score_rank1')['gagne_sum_t'].mean().to_frame().reset_index()

# visualization Fig1
# to demonstrate conditional on algorithms risk score, health disparities exist
sns.set_style('whitegrid')
fig,ax1 = plt.subplots(figsize = (7,5), dpi = 300)
ax1.scatter('risk_score_rank1','gagne_sum_t', data=grouped_white, color='darkorange',s=20,
            label='White')
ax1.plot('risk_score_rank1','gagne_sum_t', data=grouped_white, color='darkorange',linestyle='--',linewidth=0.8,label='')
ax1.scatter('risk_score_rank1','gagne_sum_t', data=grouped_black, color='darkorchid',s=20,
            label='Black')
ax1.plot('risk_score_rank1','gagne_sum_t', data=grouped_black, color='darkorchid',linestyle='--',linewidth=0.8,label='')
ax1.set_xticks(np.arange(-0.5,10.5,1))
ax1.set_xticklabels(np.arange(0,101,10))
ax1.axvline(x=5, linestyle='--', color='lightgrey')
ax1.axvline(x=9.2, linestyle='--', color='grey')
ax1.annotate('Referred for screen', xy=(3,4), xytext=(3,4), fontsize=10)
ax1.annotate('Defaulted into program', xy=(6.5,5), xytext=(6.5,5), fontsize=10)
ax1.set_ylabel('Number of active chronic conditions', fontsize = 12)
ax1.set_xlabel('Percentile of algorithm-predicted risk score', fontsize = 12)
ax1.set_title('Number of chronic illnesses versus \nalgorithm-predicted risk by race', fontsize = 15, fontweight = 'bold')
plt.legend(title='Race')
plt.show()
# at the same level of algorithm-predicted risk score, Blacks have more
# chronic disease burdens than Whites
# at the 97.5% percentile, blacks have average of 6.02 active chronic diseases
# while whites have 4.36 active chronic disease


# calculate the average number of chronic disease at the 97th percentile
quantile_96_98 = data.groupby(['race'])['risk_score_t'].quantile([0.96,0.98]).tolist()
white_96_98 = data_white.loc[(data_white['risk_score_t']<quantile_96_98[3]) & (data_white['risk_score_t']>quantile_96_98[2]),'gagne_sum_t']
black_96_98 = data_black.loc[(data_black['risk_score_t']<quantile_96_98[1]) & (data_black['risk_score_t']>quantile_96_98[0]),'gagne_sum_t']
white_97 = np.round(white_96_98.sum()/len(white_96_98),2) # 4.36
black_97 = np.round(black_96_98.sum()/len(black_96_98),2) # 6.02

###################################################################################################
# Train model to predict total cost at year t
# prepare train and test set
# exclude race, risk_score_t, program_enrolled_t, gagne_sum_t, cost_avoidable_t, 
# bps_mean_t, ghba1c_mean_t, hct_mean_t, cre_mean_t, ldl_mean_t
# only include demographics data, comorbidity variables, biomarker/medication variabels at time t-1, and cost at time t (label)
data_model = data.drop(['race','risk_score_t','program_enrolled_t','gagne_sum_t','cost_avoidable_t','bps_mean_t',
                        'ghba1c_mean_t','hct_mean_t','cre_mean_t','ldl_mean_t'], axis = 1)

data_model_x = data_model.drop(['cost_t'], axis=1)
data_model_y = data_model['cost_t']
# split into train and test set 
x_train, x_test, y_train, y_test = train_test_split(data_model_x, data_model_y, test_size = 0.33, random_state = 42)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
# 32685 in the train set, 16099 in the test set

# train model
group_kfold = GroupKFold(n_splits=10)
group_kfold_generator = group_kfold.split(x_train,y_train,x_train.index)

linear_model = LassoCV(cv = group_kfold_generator, n_alphas = 100, random_state = 42, 
                       max_iter = 10000, fit_intercept = True, normalize = True)
linear_model.fit(x_train,y_train)

linear_model.alpha_ # 0.9338
rmse_val = np.sqrt(mean_squared_error(y_train, linear_model.predict(x_train))) # 15895.25

# evaluate in test set using RMSE
rmse_test = np.sqrt(mean_squared_error(y_test, linear_model.predict(x_test))) # 15764.57
# evaluate in test set using MAE
mae_test = mean_absolute_error(y_test, linear_model.predict(x_test)) # 6668.44
# evaluate in test set using R2
r2_test = r2_score(y_test, linear_model.predict(x_test)) # 0.2155

# make predictions and create predicted total cost at time t
data['predicted_cost_t'] = linear_model.predict(data_model_x)


# add the predicted cost to white and black subset 
data_white = data_white.merge(data[['predicted_cost_t']], how = 'left', left_index = True, right_index = True)
data_black = data_black.merge(data[['predicted_cost_t']], how = 'left', left_index = True, right_index = True)

# plot the predicted cost and percentile of algorithms risk score
grouped_white_cost = data_white.groupby('risk_score_rank1')['predicted_cost_t'].mean().to_frame().reset_index()
grouped_black_cost = data_black.groupby('risk_score_rank1')['predicted_cost_t'].mean().to_frame().reset_index()
grouped_white_cost_real = data_white.groupby('risk_score_rank1')['cost_t'].mean().to_frame().reset_index()
grouped_black_cost_real = data_black.groupby('risk_score_rank1')['cost_t'].mean().to_frame().reset_index()
# 50-60th percentile of the algorithm-predicted risk score, white people spend average of 
#  5916.38$, black people spend 6031.78$, almost the same
# 90th percentile, white people: 

# visualization Fig2
# demonstrate that by measuring the real costs versus predicetd costs, te model is unbiased
sns.set_style('whitegrid')
fig,ax1 = plt.subplots(figsize = (9,7), dpi = 300)
# predicted cost white
ax1.scatter('risk_score_rank1','predicted_cost_t', data=grouped_white_cost, color='darkorange',s=25,
            label='White predicted cost', marker='*')
ax1.plot('risk_score_rank1','predicted_cost_t', data=grouped_white_cost, color='darkorange',linestyle='--',linewidth=0.8,label='')
# predicted cost black
ax1.scatter('risk_score_rank1','predicted_cost_t', data=grouped_black_cost, color='darkorchid',s=25,
            label='Black predicted cost', marker='*')
ax1.plot('risk_score_rank1','predicted_cost_t', data=grouped_black_cost, color='darkorchid',linestyle='--',linewidth=0.8, label='')
# real cost white
ax1.scatter('risk_score_rank1','cost_t', data=grouped_white_cost_real, color='maroon',s=25,
            label='White real cost')
ax1.plot('risk_score_rank1','cost_t', data=grouped_white_cost_real, color='maroon',linestyle='-',linewidth=0.8,label='')
# real cost black
ax1.scatter('risk_score_rank1','cost_t', data=grouped_black_cost_real, color='dodgerblue',s=25,
            label='Black real cost')
ax1.plot('risk_score_rank1','cost_t', data=grouped_black_cost_real, color='dodgerblue',linestyle='-',linewidth=0.8, label='')
ax1.set_xticks(np.arange(-0.5,10.5,1))
ax1.set_xticklabels(np.arange(0,101,10))
ax1.axvline(x=5, linestyle='--', color='lightgrey')
ax1.axvline(x=9.2, linestyle='--', color='grey')
ax1.annotate('Referred for screen', xy=(3,20000), xytext=(3,20000), fontsize=10)
ax1.annotate('Defaulted into program', xy=(6.3,25000), xytext=(6.3,25000), fontsize=10)
ax1.set_ylabel('Mean total medical expenditure', fontsize = 12)
ax1.set_xlabel('Percentile of algorithm-predicted risk score', fontsize = 12)
ax1.set_title('Predicted costs, costs versus algorithm-predicted risk score by race', fontsize = 15, fontweight = 'bold')
plt.legend(title='Race')
plt.show()

# at every level of algorithm predicted risk, Blacks and Whites 
# have (rouhgly) the same cost in the following year, in other words,
# the algorithm's predictions are well calibrated across races

###################################################################################################
# visualization Fig3 (come with Fig2)
# demonstrate at the same chronic illness level, Blacks have less costs, probably 
# due to less access to healthcare, or relationship with the doctor (see page4 of the article)

# data_white['gagne_sum_t'] = data_white['gagne_sum_t'].astype(int)
# data['chronic_group'] = pd.cut(data['gagne_sum_t'],8,labels=None,ordered=True,precision=0)
# data['chronic_group'].value_counts()

# data_white = data_white.merge(data[['chronic_group']], how='left', left_index = True, right_index = True)
# data_black = data_black.merge(data[['chronic_group']], how='left', left_index = True, right_index = True)

# grouped_white_chronic = data_white.groupby('chronic_group')['cost_t'].mean().to_frame().reset_index()
# grouped_black_chronic = data_black.groupby('chronic_group')['cost_t'].mean().to_frame().reset_index()

# grouped_white_chronic['chronic_group'] = grouped_white_chronic['chronic_group'].astype(str)
# grouped_black_chronic['chronic_group'] = grouped_black_chronic['chronic_group'].astype(str)


# sns.set_style('whitegrid')
# fig,ax1 = plt.subplots(figsize = (7,5), dpi = 300)
# ax1.scatter('chronic_group','cost_t', data=grouped_white_chronic, color='darkorange',s=20,
#             label='White')
# ax1.plot('chronic_group','cost_t', data=grouped_white_chronic, color='darkorange',linestyle='--',linewidth=0.8,label='')
# ax1.scatter('chronic_group','cost_t', data=grouped_black_chronic, color='darkorchid',s=20,
#             label='Black')
# ax1.plot('chronic_group','cost_t', data=grouped_black_chronic, color='darkorchid',linestyle='--',linewidth=0.8,label='')
# ax1.set_xticks(np.arange(7))
# # ax1.set_xticklabels(['0-2','2-4','4-6','6-8','8-11','11-13','13-15','15-17'])
# ax1.set_ylabel('Mean total medical expenditure', fontsize = 12)
# ax1.set_xlabel('Number of active chronic illness', fontsize = 12)
# ax1.set_title('Costs versus number of chronic illness by race', fontsize = 15)
# plt.legend(title='Race')
# plt.show()

#########the method described by the author#######
data['chronic_rank'] = data['gagne_sum_t'].rank(method = 'first')
data['chronic_rank_quantile'] = pd.qcut(data['chronic_rank'], 10, labels = False)
black = data[data.race == 'black']
white = data[data.race =='white']

# calculate average cost in each quantile
black_cost = black.groupby('chronic_rank_quantile')['cost_t'].mean().to_frame().reset_index()
white_cost = white.groupby('chronic_rank_quantile')['cost_t'].mean().to_frame().reset_index()

# log transfer
black_cost['log_cost'] = np.log(black_cost['cost_t'])
white_cost['log_cost'] = np.log(white_cost['cost_t'])

sns.set_style('whitegrid')
fig,ax1 = plt.subplots(figsize = (7,5), dpi = 300)
ax1.scatter('chronic_rank_quantile','log_cost', data=white_cost, color='darkorange',s=20,
            label='White')
ax1.plot('chronic_rank_quantile','log_cost', data=white_cost, color='darkorange',linestyle='--',linewidth=0.8,label='')
ax1.scatter('chronic_rank_quantile','log_cost', data=black_cost, color='darkorchid',s=20,
            label='Black')
ax1.plot('chronic_rank_quantile','log_cost', data=black_cost, color='darkorchid',linestyle='--',linewidth=0.8,label='')
ax1.set_xticks(np.arange(-0.5,10.5,1))
ax1.set_xticklabels(np.arange(0,101,10))
ax1.set_ylabel('Mean total medical expenditure (log)', fontsize = 12)
ax1.set_xlabel('Active chronic illness', fontsize = 12)
ax1.set_title('Real costs (log) versus chronic illness by race', fontsize = 15, fontweight='bold')
plt.legend(title='Race')
plt.show()


# add one graph according to the peer review feedback
# chronic illness vs. algorithm-predicted risk score

# calculate average cost in each quantile
black_risk = black.groupby('chronic_rank_quantile')['risk_score_t'].mean().to_frame().reset_index()
white_risk = white.groupby('chronic_rank_quantile')['risk_score_t'].mean().to_frame().reset_index()

sns.set_style('whitegrid')
fig,ax1 = plt.subplots(figsize = (7,5), dpi = 300)
ax1.scatter('chronic_rank_quantile','risk_score_t', data=white_risk, color='maroon',s=20,
            label='White')
ax1.plot('chronic_rank_quantile','risk_score_t', data=white_risk, color='maroon',linestyle='--',linewidth=0.8,label='')
ax1.scatter('chronic_rank_quantile','risk_score_t', data=black_risk, color='dodgerblue',s=20,
            label='Black')
ax1.plot('chronic_rank_quantile','risk_score_t', data=black_risk, color='dodgerblue',linestyle='--',linewidth=0.8,label='')
ax1.set_xticks(np.arange(-0.5,10.5,1))
ax1.set_xticklabels(np.arange(0,101,10))
ax1.set_ylabel('Predicted risk score', fontsize = 12)
ax1.set_xlabel('Active chronic illness', fontsize = 12)
ax1.set_title('Predicted risk score versus chronic illness by race', fontsize = 15, fontweight='bold')
plt.legend(title='Race')
plt.show()
