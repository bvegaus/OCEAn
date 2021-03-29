import time
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error
import os
import statistics




def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mze = 1 - accuracy_score(y_true, y_pred)
    return mae, mze


def GC_model(data):
    num_classifiers = 18
    num_clases = len(data.real.unique())

    real = data.real
    distributions = data.drop(columns = 'real')
    y_pred = []
    for index, row in distributions.iterrows():
        weights = []
        for i in range(num_clases):
            weight_class = 1.0
            for j in range(num_classifiers):
                column = 'p'+str(j)+'_'+str(i)
                weight_class*=row[column]
            weights.append(weight_class)
        # print(weights)
        suma = sum(weights)
        #print(suma)
        if suma != 0.0:
            weights = [x/suma for x in weights]
        #print(weights)
        y_pred.append(weights.index(max(weights)))
        #print(y_pred)
        #print("Nueva instancia")
    mae, mze = evaluate(real, y_pred)
    #print("mae  "+ str(mae))
    #print("mze  " + str(mze))
    return mae, mze








def main():
    directories_predictions = ['./predictions_distributions/'+dI for dI in os.listdir('./predictions_distributions') if os.path.isdir(os.path.join('./predictions_distributions',dI))]
    for dir in directories_predictions:
        datasets = [dir+'/'+dI for dI in os.listdir(dir) if os.path.isdir(os.path.join(dir,dI))]
        mae = []
        mae_std = []
        mze = []
        mze_std = []
        for dat in datasets:
            print(dat)
            mae_dat = []
            mze_dat = []
            for i in range(5):
                val = pd.read_csv(dat+'/'+str(i+10)+'.csv', index_col = 0)
                #print(val)


                mae_it, mze_it = GC_model(val)
                mae_dat.append(mae_it)
                mze_dat.append(mze_it)
            mae_std.append(statistics.stdev(mae_dat))
            mze_std.append(statistics.stdev(mze_dat))
            mae.append(sum(mae_dat)/5)
            mze.append(sum(mze_dat)/5)

        print("MAE para el directorio "+dir+' --> '+str(mae))
        print("MAE_std para el directorio " + dir + ' --> ' + str(mae_std))
        print("MZE para el directorio " + dir + ' --> ' + str(mze))
        print("MZE_std para el directorio " + dir + ' --> ' + str(mze_std))
    # res.to_excel('./res_granular_computing.xlsx', float_format = '%,3f')








if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))