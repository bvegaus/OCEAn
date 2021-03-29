import traceback
import weka.core.jvm as jvm
from weka.core.converters import Loader
from weka.classifiers import SingleClassifierEnhancer, Classifier, Evaluation
from weka.core.dataset import Instances
from weka.filters import Filter
import pandas as pd
import numpy as np
import time

import os




def make_distributions(classifier, data, results, column):
    distributions = []
    for index, inst in enumerate(data):
        pred = classifier.classify_instance(inst)
        dist = classifier.distribution_for_instance(inst).round(3)
        # print("actual " + inst.get_string_value(inst.class_index))
        # print("predicted " + str(pred))
        # print("distribution " + np.str(dist))
        distributions.append(dist)

    for i in range(distributions[0].shape[0]):
        column_new = column + "_" + np.str(i)
        ls = [item[i] for item in distributions]
        results[column_new] = ls



def classification(train, opt, validation, num_clases):
        baseClassifiers_list = ["weka.classifiers.bayes.NaiveBayes", "weka.classifiers.functions.MultilayerPerceptron",
                            "weka.classifiers.functions.SMO","weka.classifiers.lazy.IBk", "weka.classifiers.lazy.KStar", "weka.classifiers.meta.AdaBoostM1",
                            "weka.classifiers.meta.Bagging", "weka.classifiers.meta.LogitBoost", "weka.classifiers.trees.J48", "weka.classifiers.trees.DecisionStump",
                            "weka.classifiers.trees.LMT", "weka.classifiers.trees.RandomForest", "weka.classifiers.trees.REPTree", "weka.classifiers.rules.PART",
                            "weka.classifiers.rules.JRip", "weka.classifiers.functions.Logistic", "weka.classifiers.meta.ClassificationViaRegression", 
                            "weka.classifiers.bayes.BayesNet"]
        results_train = pd.DataFrame()
        results_opt = pd.DataFrame()
        results_validation = pd.DataFrame()

        cost_matrix_list =  [
        "[]", 
        "[0]", 
        "[0.0 1.0; 1.0 0.0]", 
        "[0.0 1.0 2.0; 1.0 0.0 1.0; 2.0 1.0 0.0]", 
        "[0.0 1.0 2.0 3.0; 1.0 0.0 1.0 2.0; 2.0 1.0 0.0 1.0; 3.0 2.0 1.0 0.0]",
        "[0.0 1.0 2.0 3.0 4.0; 1.0 0.0 1.0 2.0 3.0; 2.0 1.0 0.0 1.0 2.0; 3.0 2.0 1.0 0.0 1.0; 4.0 3.0 2.0 1.0 0.0]",
        "[0.0 1.0 2.0 3.0 4.0 5.0; 1.0 0.0 1.0 2.0 3.0 4.0; 2.0 1.0 0.0 1.0 2.0 3.0; 3.0 2.0 1.0 0.0 1.0 2.0; 4.0 3.0 2.0 1.0 0.0 1.0; 5.0 4.0 3.0 2.0 1.0 0.0]",
        "[0.0 1.0 2.0 3.0 4.0 5.0 6.0; 1.0 0.0 1.0 2.0 3.0 4.0 5.0; 2.0 1.0 0.0 1.0 2.0 3.0 4.0; 3.0 2.0 1.0 0.0 1.0 2.0 3.0; 4.0 3.0 2.0 1.0 0.0 1.0 2.0; 5.0 4.0 3.0 2.0 1.0 0.0 1.0; 6.0 5.0 4.0 3.0 2.0 1.0 0.0]",
        "[0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0; 1.0 0.0 1.0 2.0 3.0 4.0 5.0 6.0; 2.0 1.0 0.0 1.0 2.0 3.0 4.0 5.0; 3.0 2.0 1.0 0.0 1.0 2.0 3.0 4.0; 4.0 3.0 2.0 1.0 0.0 1.0 2.0 3.0; 5.0 4.0 3.0 2.0 1.0 0.0 1.0 2.0; 6.0 5.0 4.0 3.0 2.0 1.0 0.0 1.0; 7.0 6.0 5.0 4.0 3.0 2.0 1.0 0.0]",
        "[0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0; 1.0 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0; 2.0 1.0 0.0 1.0 2.0 3.0 4.0 5.0 6.0; 3.0 2.0 1.0 0.0 1.0 2.0 3.0 4.0 5.0; 4.0 3.0 2.0 1.0 0.0 1.0 2.0 3.0 4.0; 5.0 4.0 3.0 2.0 1.0 0.0 1.0 2.0 3.0; 6.0 5.0 4.0 3.0 2.0 1.0 0.0 1.0 2.0; 7.0 6.0 5.0 4.0 3.0 2.0 1.0 0.0 1.0; 8.0 7.0 6.0 5.0 4.0 3.0 2.0 1.0 0.0]",
        "[0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0; 1.0 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0; 2.0 1.0 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0; 3.0 2.0 1.0 0.0 1.0 2.0 3.0 4.0 5.0 6.0; 4.0 3.0 2.0 1.0 0.0 1.0 2.0 3.0 4.0 5.0; 5.0 4.0 3.0 2.0 1.0 0.0 1.0 2.0 3.0 4.0; 6.0 5.0 4.0 3.0 2.0 1.0 0.0 1.0 2.0 3.0; 7.0 6.0 5.0 4.0 3.0 2.0 1.0 0.0 1.0 2.0; 8.0 7.0 6.0 5.0 4.0 3.0 2.0 1.0 0.0 1.0; 9.0 8.0 7.0 6.0 5.0 4.0 3.0 2.0 1.0 0.0]"  ]



        real_train = [] # the real label of the dataset
        for i in range(train.num_instances):
            real_train.append(train.get_instance(i).values[(train.num_attributes-1)])
        results_train['real'] = real_train

        real_opt = [] # the real label of the dataset
        for i in range(opt.num_instances):
            real_opt.append(opt.get_instance(i).values[(opt.num_attributes-1)])
        results_opt['real'] = real_opt

        real_validation = [] # the real label of the dataset
        for i in range(validation.num_instances):
            real_validation.append(validation.get_instance(i).values[(validation.num_attributes-1)])
        results_validation['real'] = real_validation

        num = 0
        for clas in baseClassifiers_list:
            column = "p"+np.str(num)    


            classifier = Classifier(classname=clas)
            classifier.build_classifier(train)


            make_distributions(classifier, train, results_train, column)
            make_distributions(classifier, opt, results_opt, column)
            make_distributions(classifier, validation, results_validation, column)


            num = num+1
        return results_train, results_opt, results_validation


def dataset_classifiers(dataset, iteration):

    if not os.path.exists('./predictions_distributions'):
        os.mkdir('./predictions_distributions')

    if not os.path.exists('./predictions_distributions/ordinal-regression/'+ dataset):
        os.mkdir('./predictions_distributions/ordinal-regression/'+ dataset)
        
    #load a dataset
    loader = Loader("weka.core.converters.ArffLoader")
    train = loader.load_file('./data_weka/ordinal-regression/'+ dataset+ "/"+np.str(iteration)+'.arff')
    train.class_is_last()
    opt = loader.load_file('./data_weka/ordinal-regression/'+ dataset+ "/"+np.str(iteration+5)+'.arff')
    opt.class_is_last()
    validation = loader.load_file('./data_weka/ordinal-regression/' + dataset + "/" + np.str(iteration+10) + '.arff')
    validation.class_is_last()

    num_classes = train.class_attribute.num_values
    print("Numero de clases:    ", num_classes)

    results_train, results_opt, results_validation  = classification(train, opt, validation, num_classes)
            


    train_name = "./predictions_distributions/ordinal-regression/"+dataset+"/" +np.str(iteration) +".csv"
    opt_name = "./predictions_distributions/ordinal-regression/"+dataset+"/" +np.str(iteration+5)+".csv"
    validation_name = "./predictions_distributions/ordinal-regression/"+dataset+"/" +np.str(iteration+10)+".csv"

    results_train.to_csv(train_name)
    results_opt.to_csv(opt_name)
    results_validation.to_csv(validation_name)





def main():
    directory = './datasets-orreview/ordinal-regression/'
    datasets = [dI for dI in os.listdir(directory) if os.path.isdir(os.path.join(directory,dI))]
    for dataset in datasets:
        print("Trabajando en el dataset ", dataset)
        for i in range(5):
            dataset_classifiers(dataset,i)




    
if __name__ == "__main__":
    try:
        start_time = time.time()
        jvm.start()
        main()
        print("--- %s seconds ---" % (time.time() - start_time))
    except Exception:
        print(traceback.format_exc())
    finally:
        jvm.stop()
