# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 19:08:56 2021

@author: belen
"""


import random
import numpy as np
import pandas as pd
import multiprocessing
from sklearn.metrics import accuracy_score, mean_absolute_error
import time
import sys
import os 

def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mze = 1 - accuracy_score(y_true, y_pred)
    return mae, mze





if __name__ == "__main__":
    baseClassifiers_list = ["NaiveBayes", "MultilayerPerceptron",
                            "SMO","IBk", "KStar", "AdaBoostM1",
                            "Bagging", "LogitBoost", "J48", "DecisionStump",
                            "LMT", "RandomForest", "REPTree", "PART",
                            "JRip", "Logistic", "ClassificationViaRegression", 
                            "BayesNet"]

    #directories = [dI for dI in os.listdir('./predictions')]
    directories = ['discretized-regression']
    for direc in directories:
        datasets = [dI for dI in os.listdir('./predictions_opt_val/'+direc)]
        results_mae = []
        results_mze = []
        for dataset_name in datasets:
            mae_tot = []
            mze_tot = []
            for validation in range(10,15):
                mae_class = []
                mze_class = []
                C_validation = pd.read_csv('./predictions_opt_val/'+direc+'/'+dataset_name+'/'+np.str(validation)+ '.csv')
                l_validation= C_validation.real
                C_validation = C_validation.drop(['real', 'Unnamed: 0'], axis = 1)
                C_validation.columns = baseClassifiers_list
                
                for base_classifier in C_validation:
                    y_pred = C_validation[base_classifier]
                    mae,mze = evaluate(l_validation, y_pred)
                    mae_class.append(mae)
                    mze_class.append(mze)
                ## mze and mae for each partition    
                mae_tot.append(mae_class)
                mze_tot.append(mze_class)     
                
            ## mean of all the metrics by classifier(media entre las 5 particiones para cada clasificador)
            results_mae.append(np.append(dataset_name,np.mean(mae_tot, axis = 0).round(3))) 
            results_mze.append(np.append(dataset_name,np.mean(mze_tot, axis = 0).round(3)))
            #print(results_mae)
        df_mae = pd.DataFrame(results_mae, columns = ['dataset_name']+baseClassifiers_list)
        #df_mae[baseClassifiers_list] = df_mae[baseClassifiers_list].round(3)
        df_mae.to_excel('./predictions_opt_val/'+direc+'/mae.xlsx')
        df_mze = pd.DataFrame(results_mze, columns = ['dataset_name']+baseClassifiers_list)
        #df_mze[baseClassifiers_list] = df_mze[baseClassifiers_list].round(3)
        df_mze.to_excel('./predictions_opt_val/'+direc+'/mze.xlsx')
        
            
            
            
            
        
    

# =============================================================================
#     C_validation = pd.read_csv('./predictions/'+dataset_name+'/'+validation+ '.csv')
#     l_validation = C_validation.real
#     C_validation = C_validation.drop(['real', 'Unnamed: 0'], axis = 1)
# =============================================================================
