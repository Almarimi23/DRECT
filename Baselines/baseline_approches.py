
import pandas as pd
import os
import numpy as np
import itertools
import matplotlib.pyplot as plt
import math
import json
import os
import statistics
from statistics import mean
import warnings
warnings.filterwarnings("ignore")


os.chdir("CSDMetalearningRS")

from ML_Models.Model_def import *
from DataPrepare.TopcoderDataSet import *
from sklearn import tree,naive_bayes,svm
from sklearn import metrics
import time
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn import ensemble
import warnings
from sklearn.pipeline import Pipeline
from ML_Models.Model_def import ML_model
from sklearn.cluster import KMeans,MiniBatchKMeans
import pickle
import numpy as np
from ML_Models.Model_def import ML_model
from sklearn.cluster import KMeans,MiniBatchKMeans
import pickle
import numpy as np

warnings.filterwarnings("ignore")

"""## ML Algorithms"""

# Define a class NBBayes that inherits from ML_model
class NBBayes(ML_model):

    # Define the constructor of the class
    def __init__(self):
        # Call the constructor of the parent class
        ML_model.__init__(self)

    # Define the predict method of the class that takes input data X and returns predicted output Y
    def predict(self,X):
        # If input data is empty, return an empty numpy array
        if len(X)==0:
            return np.array([],dtype=np.int)
        # If verbose mode is on, print a message that NBBayes is predicting
        if self.verbose>0:
            print(self.name," NBBayes is predicting")
        # Predict the output Y using the trained model and input data X
        Y=self.model.predict(X)
        # Return the predicted output Y
        return Y

    # Define the trainModel method of the class that takes input data Xtrain and labels ytrain for training the model
    def trainModel(self, Xtrain, ytrain):

        print("training NB Model")
        # Get the current time
        t0=time.time()

        # Create a Naive Bayes classifier with Gaussian distribution
        self.model=naive_bayes.GaussianNB()

        # Train the Naive Bayes model using the input data Xtrain and labels ytrain
        self.model.fit(Xtrain, ytrain)

        # Get the time after training the model
        t1=time.time()

        # #measure training result
        # vpredict=self.predict(dataSet.validateX)
        # #print(vpredict)
        # vpredict=np.array(vpredict>self.threshold,dtype=np.int)
        # #print(vpredict)
        # score=metrics.accuracy_score(dataSet.validateLabel,vpredict)
        # cm=metrics.confusion_matrix(dataSet.validateLabel,vpredict)
        # print("model",self.name,"trainning finished in %ds"%(t1-t0),"validate score=%f"%score,"CM=\n",cm)

        # Print a message that the training of the Naive Bayes model has finished and show the time taken for training
        print("model",self.name,"training finished in %ds"%(t1-t0))

    # Define the findPath method of the class that returns the path of the trained model
    def findPath(self):
        # Define the path of the trained model as the name of the model with a ".pkl" extension
        modelpath=self.name+".pkl"
        # Return the model path
        return modelpath

class DecsionTree(ML_model):
    # Initialize decision tree parameters
    def initParameters(self):
        self.params={
            'criterion':"gini",
            'splitter':"best",
            'max_depth':5,
            'min_samples_split':2,
            'min_samples_leaf':1,
            'max_features':'auto',
        }

    # Initialize decision tree model and parameters
    def __init__(self):
        ML_model.__init__(self)
        self.initParameters()

    # Predict with decision tree model
    def predict(self,X):
        if len(X) == 0:
            return np.array([], dtype=np.int)
        if self.verbose > 0:
            print(self.name," C4.5 is predicting")

        Y = self.model.predict(X)
        return Y

    # Update decision tree parameters
    def updateParameters(self, paras):
        for k in paras:
            self.params[k] = paras[k]

    # Search for best decision tree parameters using grid search
    def searchParameters(self, Xtrain, ytrain):
        print("searching for best parameters")
        try:
            # Define parameter grid for grid search
            selParas=[
                {'criterion':["gini",'entropy']},
                {'splitter':["best",'random']},
                {'max_depth':[i for i in range(3,10)]},
                {'min_samples_split':[i for i in range(2,10)]},
                {'min_samples_leaf':[i for i in range(1,10)]},
                {'max_features':[None,'sqrt','log2']},
            ]

            # Perform grid search for each parameter set in selParas and update parameters accordingly
            for i in range(len(selParas)):
                para=selParas[i]
                model=tree.DecisionTreeClassifier(**self.params)
                gsearch=GridSearchCV(model,para,scoring='accuracy')
                gsearch.fit(Xtrain,ytrain)
                print("best para",gsearch.best_params_)
                self.updateParameters(gsearch.best_params_)

            # Create decision tree model with updated parameters
            self.model=tree.DecisionTreeClassifier(**self.params)
        except:
            self.model=tree.DecisionTreeClassifier()

    # Train decision tree model with training data
    def trainModel(self, Xtrain, ytrain):
        print("training")
        t0=time.time()

        # Search for best parameters using grid search
        self.searchParameters(Xtrain, ytrain)

        # Fit decision tree model to training data with updated parameters
        self.model.fit(Xtrain, ytrain)

    # Find file path for decision tree model
    def findPath(self):
        modelpath=self.name+".pkl"
        return modelpath

# Creating Random Forest Class
class RandForest(ML_model):

    # Initializing Default Parameters
    def initParameters(self):
        self.params={
            "n_estimators":10,
            "criterion":"gini",
            "max_depth":None,
            "min_samples_split":2,
            "min_samples_leaf":1,
            "min_weight_fraction_leaf":0.,
            "max_features":None,
            "max_leaf_nodes":None,
            "min_impurity_decrease":0.,
            "n_jobs":-1
        }

    # Constructor Method
    def __init__(self):
        ML_model.__init__(self) # Inheriting from Parent Class
        self.initParameters()   # Initializing Parameters

    # Predict Method to Predict Classes
    def predict(self,X):
        if  len(X)==0:
            return np.array([],dtype=np.int)
        if self.verbose>0:
            print(self.name," RF is predicting")

        Y=self.model.predict(X)
        return Y

    # Method to Update Model Parameters
    def updateParameters(self,paras):
        for k in paras:
            self.params[k]=paras[k]

    # Method to Search for Best Parameters
    def searchParameters(self,Xtrain, ytrain):
        print("searching for best parameters")

        try:
            # Parameters to be Searched
            selParas=[
              {"n_estimators":[i for i in range(10,101,10)]},
              {"criterion":["gini","entropy"]},
              {"max_depth":[i for i in range(5,12)]},
              {"min_samples_split":[2,5,7,10]},
              {"min_samples_leaf":[1,2,3,4,5]},
              {"max_features":[None,"sqrt","log2"]},
              {"min_impurity_decrease":[i/100.0 for i in range(0,100,5)]}
            ]

            # Iterating through All Parameters
            for i in range(len(selParas)):
                para=selParas[i]
                model=ensemble.RandomForestClassifier(**self.params) # Creating Random Forest Model
                gsearch=GridSearchCV(model,para,scoring=metrics.make_scorer(metrics.accuracy_score)) # Grid Search Cross Validation
                gsearch.fit(Xtrain, ytrain) # Fitting Grid Search to Data
                print("best para",gsearch.best_params_) # Printing Best Parameters found
                self.updateParameters(gsearch.best_params_) # Updating Best Parameters

            self.model=ensemble.RandomForestClassifier(**self.params) # Creating Random Forest with Updated Parameters
        except:
            self.model=ensemble.RandomForestClassifier(**self.params) # Creating Random Forest with Default Parameters

    # Training Method
    def trainModel(self,Xtrain, ytrain):
        print("training")
        t0=time.time()

        self.searchParameters(Xtrain, ytrain) # Searching for Best Parameters

        self.model.fit(Xtrain, ytrain) # Fitting Data to Model
        # t1=time.time()

        # #measure training result
        # vpredict=self.predict(dataSet.validateX)
        # #print(vpredict)
        # vpredict=np.array(vpredict>self.threshold,dtype=np.int)
        # #print(vpredict)
        # score=metrics.accuracy_score(dataSet.validateLabel,vpredict)
        # cm=metrics.confusion_matrix(dataSet.validateLabel,vpredict)
        # print("model",self.name,"trainning finished in %ds"%(t1-t0),"validate score=%f"%score,"CM=\n",cm)

    def findPath(self):
        modelpath= self.name+".pkl"
        return modelpath

class ClusteringModel(ML_model):
    def __init__(self):
        # Call the constructor of the base class
        ML_model.__init__(self)

    def trainCluster(self,X,n_clusters,minibatch=False):
        # Train a KMeans clustering model on the data X with n_clusters clusters
        # If minibatch is True, use the MiniBatchKMeans algorithm instead of KMeans
        if minibatch:
            km = MiniBatchKMeans(n_clusters=n_clusters, verbose=False)
        else:
            km = KMeans(n_clusters=n_clusters,verbose=False)

        km.fit(X) # Fit the model to the data
        self.model=km # Save the trained model

    def predictCluster(self,X):
        # Predict the cluster assignments for the data X using the trained model
        print(self.name+" is clustering %d tasks"%len(X))
        return  self.model.predict(X)

    def findPath(self):
        # Return the path to save the model
        modelPath= self.name+".pkl"
        return modelPath

class CBCModel:
    def __init__(self,learners,clustering):
        self.models={} # dictionary to store learners
        for i in range(len(learners)):
            self.models[i]=learners[i] # assign learner to a cluster
        self.clustering=clustering # clustering model used to predict clusters

    def predict(self,X,clustered=False):
        if clustered:
            cluster_no=self.clustering.predict(X) # predict cluster number using clustering model
            Y=self.models[cluster_no].predict(X) # use classifier for predicted cluster to predict outcome
            return Y

        Y=self.models[0].predict(X) # if not clustered, use first classifier to predict outcome
        return Y

def buildCBC(clusters=0, classifier_choice=0):
    if classifier_choice == 0:
      classifier=NBBayes
    else:
      classifier=DecsionTree
    learners=[]
    if clusters>0:
        for i in range(clusters):
            learner=classifier()
            if classifier_choice == 0:
              learner.name="NB-classifier_{}".format(i) # set name of learner based on its index
            elif classifier_choice == 1:
              classifier.name= "RandomForest_{}".format(i)
            else:
              learner.name = "DecisionTree_{}".format(i)
            learner.loadModel()
            learners.append(learner)
    else:
        classifier=classifier()
        if classifier_choice == 0:
            classifier.name= "NB-classifier_{}".format(0) # set name of classifier as only one is used
        elif classifier_choice == 1:
            classifier.name= "RandomForest_{}".format(0)
        else:
            classifier.name = "DecisionTree_{}".format(0)
        classifier.loadModel()
        learners.append(classifier)

    clustering=ClusteringModel()

    return CBCModel(learners,clustering)

def train(Xtrain, ytrain, cluster_num, classifier_num=0):
    # build classifier for given cluster number
    if classifier_num == 0:
      model=NBBayes()
      model.name="NB-classifier_{}".format(cluster_num)
    elif classifier_num == 1:
      model=RandForest()
      model.name="RandomForest_{}".format(cluster_num)
    else:
      model=DecsionTree()
      model.name="DecisionTree_{}".format(cluster_num)
    #train model
    model.trainModel(Xtrain, ytrain)
    print(Xtrain)
    model.saveModel()
    return

"""## Setup"""

os.chdir("..")
all_data = pd.read_csv('Full_dataset_1.csv')

x = all_data.copy()

all_data.dropna(inplace=True)
# all_data.sort_values("task_created_date", ascending=True, inplace=True)

all_data.columns

features = ['completedchallenges',
       'totalWins',
       'Success_rate (%)',
       'task_recency (in days)',
       'Cosine_similarity_score_task_descriptions_current_past',
       'Cosine_similarity_score_task_skills_current_past',
       'number_interactions',
       'sum_weights_edges',
       'CP_score']

# # remove any feature that is string
# for col in features:
#     if all_data[col].dtype == 'O':
#         features.remove(col)

# features = ["Success_rate (%)"]

features

topk = 10

"""# 10-Fold CV Single Ml"""

df = all_data.copy()
df.head(2)

# arrange df by task_created_date and challenge_id
df.sort_values(by=['task_created_date', 'challengeId'], inplace=True)
df.reset_index(drop=True, inplace=True)
df.head(2)

# divide datset into 10 folds for cross validation
fold_size = int(len(df)/10)
folds = []
for i in range(10):
    if i == 9:
        folds.append(df.iloc[i*fold_size:])
    else:
        folds.append(df.iloc[i*fold_size:(i+1)*fold_size])


from sklearn.model_selection import train_test_split
results = {}
for i in range(5):
    train_indices = list(range(i + 5))  # Indices for training data
    test_index = i + 5  # Index for test data

    train_data = pd.concat([folds[j] for j in train_indices], ignore_index=True)
    test_data = folds[test_index]

    print(f"{i} {train_indices} {test_index}")

    X = train_data[features]
    y = train_data["handle"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    for x in range(3):
        train(X_train, y_train, 0, x)
        model = buildCBC(clusters=0, classifier_choice=x)

        ypred = model.predict(X_test[features])

        # incoporate results to results
        accuracy = metrics.accuracy_score(y_test, ypred)
        precision = metrics.precision_score(y_test, ypred, average='micro')
        recall = metrics.recall_score(y_test, ypred, average='micro')

        # incoporate results to results dictionsry
        results[f"{i}_{x}"] = [i, train_indices, test_index, x, accuracy, precision, recall]

# create a df from results
results2_df = pd.DataFrame.from_dict(results, orient='index', columns=["Iteration","Train_folds","Test_fold","model", "Accuracy", "Precision", "Recall"])

results2_df

results2_df.to_csv("ML_RESULTS2.csv", index=False)

