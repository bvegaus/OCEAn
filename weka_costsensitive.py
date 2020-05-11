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


def classification(train, test, validation, num_clases):
        baseClassifiers_list = ["weka.classifiers.bayes.NaiveBayes", "weka.classifiers.functions.MultilayerPerceptron",
                            "weka.classifiers.functions.SMO","weka.classifiers.lazy.IBk", "weka.classifiers.lazy.KStar", "weka.classifiers.meta.AdaBoostM1",
                            "weka.classifiers.meta.Bagging", "weka.classifiers.meta.LogitBoost", "weka.classifiers.trees.J48", "weka.classifiers.trees.DecisionStump",
                            "weka.classifiers.trees.LMT", "weka.classifiers.trees.RandomForest", "weka.classifiers.trees.REPTree", "weka.classifiers.rules.PART",
                            "weka.classifiers.rules.JRip", "weka.classifiers.functions.Logistic", "weka.classifiers.meta.ClassificationViaRegression", 
                            "weka.classifiers.bayes.BayesNet"]
        results_train = pd.DataFrame()
        results_test = pd.DataFrame()
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

        real_test = [] # the real label of the dataset
        for i in range(test.num_instances):
            real_test.append(test.get_instance(i).values[(test.num_attributes-1)])
        results_test['real'] = real_test

        real_validation = [] # the real label of the dataset
        for i in range(validation.num_instances):
            real_validation.append(validation.get_instance(i).values[(validation.num_attributes-1)])
        results_validation['real'] = real_validation

        num = 0
        for clas in baseClassifiers_list:
            column = "p"+np.str(num)    
            
            #classifier
            classifier = SingleClassifierEnhancer(
                classname="weka.classifiers.meta.CostSensitiveClassifier",
                options=["-cost-matrix", cost_matrix_list[num_clases], "-M", "-S", "1"])
            base = Classifier(classname=clas)
            classifier.classifier = base    
        
        
            predicted_data_train = None
            predicted_data_test = None
            predicted_data_validation = None

            evaluation = Evaluation(train)
            classifier.build_classifier(train)

            # add predictions
            addcls = Filter(
                    classname="weka.filters.supervised.attribute.AddClassification",
                    options=["-classification"])
            
            addcls.set_property("classifier", Classifier.make_copy(classifier))
            addcls.inputformat(train)

            pred_train = addcls.filter(train)

            pred_test = addcls.filter(test)

            pred_validation = addcls.filter(validation)




            if predicted_data_train is None:
                predicted_data_train = Instances.template_instances(pred_train, 0)
            for n in range(pred_train.num_instances):
                predicted_data_train.add_instance(pred_train.get_instance(n))


            if predicted_data_test is None:
                predicted_data_test = Instances.template_instances(pred_test, 0)
            for n in range(pred_test.num_instances):
                predicted_data_test.add_instance(pred_test.get_instance(n))

            if predicted_data_validation is None:
                predicted_data_validation = Instances.template_instances(pred_validation, 0)
            for n in range(pred_validation.num_instances):
                predicted_data_validation.add_instance(pred_validation.get_instance(n))

            preds_train = [] #labels predicted for the classifer trained in the iteration
            preds_test = []
            preds_validation = []
    
            
            for i in range(predicted_data_train.num_instances):
                preds_train.append(predicted_data_train.get_instance(i).values[(predicted_data_train.num_attributes-1)])

            for i in range(predicted_data_test.num_instances):
                preds_test.append(predicted_data_test.get_instance(i).values[(predicted_data_test.num_attributes-1)])        

            for i in range(predicted_data_validation.num_instances):
                preds_validation.append(predicted_data_validation.get_instance(i).values[(predicted_data_validation.num_attributes-1)])     

            results_train[column] = preds_train
            results_test[column] = preds_test
            results_validation[column] = preds_validation

            num = num+1
        return results_train, results_test, results_validation


def dataset_classifiers(dataset, iteration):

    if not os.path.exists('./predictions/'+ dataset):
        os.mkdir('./predictions/'+ dataset)
        
    #load a dataset
    loader = Loader("weka.core.converters.ArffLoader")
    train = loader.load_file('./data_weka/discretized-10bins/'+ dataset+ "/"+np.str(iteration)+'.arff')
    train.class_is_last()
    test = loader.load_file('./data_weka/discretized-10bins/'+ dataset+ "/"+np.str(iteration+5)+'.arff')
    test.class_is_last()
    validation = loader.load_file('./data_weka/discretized-10bins/' + dataset + "/" + np.str(iteration+10) + '.arff')
    validation.class_is_last()

    num_classes = train.class_attribute.num_values
    print("Numero de clases:    ", num_classes)

    results_train, results_test, results_validation  = classification(train, test, validation, num_classes)
            


    train_name = "./predictions/"+dataset+"/" +np.str(iteration) +".csv"
    test_name = "./predictions/"+dataset+"/" +np.str(iteration+5)+".csv"
    validation_name = "./predictions/"+dataset+"/" +np.str(iteration+10)+".csv"

    results_train.to_csv(train_name)
    results_test.to_csv(test_name)
    results_validation.to_csv(validation_name)

    l_train = results_train.real
    l_test = results_test.real
    l_validation = results_validation.real
    accuracy_train_cv = []
    accuracy_test_cv = []
    accuracy_validation_cv = []
    for classifier in range(18):
        acc_train = 0
        acc_test = 0
        acc_val = 0
        for index_row in range(results_train.shape[0]):
            if l_train[index_row] == results_train.iloc[index_row, classifier+1]:
                acc_train = acc_train + 1
        acc_train = acc_train / results_train.shape[0]

        for index_row in range(results_test.shape[0]):
            if l_test[index_row] == results_test.iloc[index_row, classifier+1]:
                acc_test = acc_test + 1
        acc_test = acc_test / results_test.shape[0]
            
        for index_row in range(results_validation.shape[0]):
            if l_validation[index_row] == results_validation.iloc[index_row, classifier+1]:
                acc_val = acc_val + 1
        acc_val = acc_val / results_validation.shape[0]

        accuracy_train_cv.append(acc_train*100)
        accuracy_test_cv.append(acc_test*100)
        accuracy_validation_cv.append(acc_val*100)   
    
    print("Accuracy train:      ",np.mean(accuracy_train_cv))
    print("Accuracy test:      ",np.mean(accuracy_test_cv))
    print("Accuracy validation:      ",np.mean(accuracy_validation_cv))



def main():
    directory = './datasets-orreview/discretized-regression/10bins/'
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
