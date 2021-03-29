
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error
import time
import os
import statistics




def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mze = 1 - accuracy_score(y_true, y_pred)
    return mae, mze

def simple_voting(df):
    #print(df)
    df = df.apply(pd.Series.value_counts, axis=1).fillna(0.0)
    y_pred = []
    for index, row in df.iterrows():
        y_pred.append(row.idxmax())
    return y_pred





def main():
    directories_predictions = ['./predictions/'+dI for dI in os.listdir('./predictions') if os.path.isdir(os.path.join('./predictions',dI))]
    print(directories_predictions)
    for dir in directories_predictions:
        datasets = [dir+'/'+dI for dI in os.listdir(dir) if os.path.isdir(os.path.join(dir,dI))]
        mae = []
        mae_std = []
        mze = []
        mze_std= []
        for dat in datasets:
            print(dat)
            mae_dat = []
            mze_dat = []
            for i in range(5):
                val = pd.read_csv(dat+'/'+str(i+10)+'.csv', index_col = 0)
                y_true = val.real
                val.drop(columns = 'real', inplace = True)
                y_pred = simple_voting(val)
                mae_it, mze_it = evaluate(y_true, y_pred)
                mae_dat.append(mae_it)
                mze_dat.append(mze_it)

            mae_std.append(statistics.stdev(mae_dat))
            mze_std.append(statistics.stdev(mze_dat))
            mae.append(sum(mae_dat)/5)
            mze.append(sum(mze_dat)/5)

        # res['MAE_'+dir] = mae
        # res['MZE_' + dir] = mze
        print("MAE para el directorio " + dir + ' --> ' + str(mae))
        print("MAE_std para el directorio " + dir + ' --> ' + str(mae_std))
        print("MZE para el directorio " + dir + ' --> ' + str(mze))
        print("MZE_std para el directorio " + dir + ' --> ' + str(mze_std))


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))