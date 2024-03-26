
!pip install pymoo==0.5.0

import pandas as pd
import os
import numpy as np
import itertools
from pymoo.operators.selection.rnd import RandomSelection
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.algorithms.moo.moead import MOEAD, ParallelMOEAD
from pymoo.algorithms.moo.age import AGEMOEA
from pymoo.algorithms.moo.ctaea import CTAEA
from pymoo.visualization.scatter import Scatter
from pymoo.visualization.scatter import Scatter
from pymoo.factory import get_crossover, get_mutation, get_sampling, get_reference_directions
from pymoo.core.problem import Problem, ElementwiseProblem
from pymoo.factory import get_performance_indicator
from pymoo.optimize import minimize
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import math
import warnings
warnings.filterwarnings("ignore")

from pymoo.config import Config
Config.show_compile_hint = False

import json
import os
import statistics
from statistics import mean

np.bool = np.bool_

"""## Objective Fucntions"""

def f1(X, df):
    de = []
    for i in range(len(df)):
        de.append(int(X[i]) * (((df['completedchallenges'].iloc[i] / df['totalChallengsJoined'].iloc[i])
                                + (df['totalWins'].iloc[i] // df['completedchallenges'].iloc[i]))
                               * ((1 / df['task_recency (in days)'].iloc[i]) * 100)
                               + df['CP_score'].iloc[i]))

    f1_value = - (sum(de) / len(de)) if len(de) > 0 else 0

    if math.isinf(f1_value):
        f1_value = 0

    return f1_value
def f2(X, df):
    sds = []
    for i in range(len(df)):
        sds.append(int(X[i]) * (df['Cosine_similarity_descriptions'].iloc[i] +
                                df['Cosine_similarity_skills'].iloc[i] +
                                df['Cosine_similarity_score_task_titles_current_past'].iloc[i] +
                                df['Cosine_similarity_score_task_descriptions_current_past'].iloc[i] +
                                df['Cosine_similarity_score_task_skills_current_past'].iloc[i]))

    f2_value = - (sum(sds) / len(sds)) if len(sds) > 0 else 0
    return f2_value

def f3(X, df):
    ad = []
    for i in range(len(df)):
        ad.append(int(X[i]) * (df['activechallenges'].iloc[i]))

    f3_value = (sum(ad) / len(ad)) if len(ad) > 0 else 0
    return f3_value

class MyProblem(ElementwiseProblem):

    def __init__(self,task_df, maxSize, minSize, objectives = [f1,f2,f3]):

        self.task_df = task_df
        self.maxSize = maxSize
        self.minSize = minSize
        self.objectives = objectives
        super().__init__(n_var=len(self.task_df),
                         n_obj=3,
                         n_constr=2,
                         sampling = get_sampling('real_random')

                         )


    def _evaluate(self, X, out, *args, **kwargs):

        # Fitness functions

        obj1 = self.objectives[0](X,self.task_df)
        obj2 = self.objectives[1](X,self.task_df)
        obj3 = self.objectives[2](X,self.task_df)
        g1 = (self.minSize - np.sum(X))
        g2 = (np.sum(X) - self.maxSize)

        out["F"] = [obj1, obj2, obj3]
        out["G"] = [g1, g2]

"""## Data PreProcessing"""

all_data = pd.read_csv('Full_dataset_1.csv')

all_data.dropna(inplace=True)
# all_data.sort_values("challengeId", ascending=True, inplace=True)

all_data.columns

"""## NSGAII Setup"""

pop_size = 30
n_gen = 20
minSize = 1
maxSize = 600000
obj_algo_name = 'NSGA2'
topk = 10

if obj_algo_name ==  'NSGA2':
    obj_algo = NSGA2
if obj_algo_name == 'NSGA3':
    obj_algo = NSGA3
if obj_algo_name == 'UNSGA3':
    obj_algo = UNSGA3
if obj_algo_name == 'AGEMOEA':
    obj_algo = AGEMOEA

# function to calculate pareto front using the f values
def pareto_front(f_values):
  nondominated = []
  dominated = False
  for i,f in enumerate(f_values):
    for f_val in f_values[(i+1):]:
      if f[0] <= f_val[0] and f[1] <= f_val[1] and f[2] <= f_val[2]:
        dominated = True

    if dominated == False:
      nondominated.append(f)
  return nondominated

# function to run the algo with dataset as input for any particular task/ cv set
def run_algo(TASK_DF, verbose):

    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12) #12
    vectorized_problem = MyProblem(TASK_DF , minSize=minSize, maxSize=maxSize)
    algorithm = obj_algo(
                pop_size=pop_size,
                ref_dirs=ref_dirs,
                sampling=get_sampling("bin_random"),
                crossover=get_crossover("bin_one_point"),
                mutation=get_mutation("bin_bitflip", prob=1/vectorized_problem.n_var),
                seed=1 # ?
            )

    res = minimize(vectorized_problem,
               algorithm,
               termination=('n_gen', n_gen),
               verbose=verbose)
    return res

def get_ranking(X_res, data):
    """
    Generate rankings based on the results from the recommendation system.

    Parameters:
    - X_res (array): Array containing recommendation scores for each developer.
    - data (DataFrame): DataFrame containing developer information.

    Returns:
    - DataFrame: DataFrame with rankings based on recommendation scores.
    """
    # Convert DataFrame columns to a list of developer handles
    columns = data['handle'].values.tolist()

    # Create a DataFrame with recommendation scores and developer handles as columns
    results = pd.DataFrame(X_res, columns=columns)

    # Sum the occurrence of recommendations for each developer
    df = results.sum(axis=0).reset_index()
    df.columns = ["devname", "occurances"]

    # Calculate rank based on occurrences using the 'min' method
    df['rank'] = df["occurances"].rank(method='min', ascending=False)

    # Return the DataFrame with 'devname', 'occurrences', and 'rank' columns
    return df

def get_mrr(ranking_main, ranking_cv):
    """
    Calculate the Mean Reciprocal Rank (MRR) for a set of queries.

    Args:
    - ranking_main (DataFrame): DataFrame containing the main ranking data.
    - ranking_cv (DataFrame): DataFrame containing the cross-validation ranking data.

    Returns:
    - float: Mean Reciprocal Rank (MRR) value.
    """
    # Initialize variables for MRR calculation
    mrr_sum = 0
    num_queries = min(len(ranking_main), len(ranking_cv))

    # Iterate over each query in the rankings
    for i in range(num_queries):
        # Get relevant items from cross-validation ranking and ranked items from main ranking
        relevant_items = set(ranking_cv.iloc[i]['devname'])
        ranked_items = ranking_main.iloc[i]['devname']

        # Initialize variables for reciprocal rank calculation
        reciprocal_rank_sum = 0
        found_relevant_item = False

        # Iterate over each ranked item
        for rank, item in enumerate(ranked_items, 1):
            # Check if the item is relevant
            if item in relevant_items:
                # Calculate reciprocal rank and update sum
                reciprocal_rank_sum += 1 / rank
                found_relevant_item = True
                break  # Stop after finding the first relevant item

        # If no relevant item is found, add 0 to the sum
        if not found_relevant_item:
            reciprocal_rank_sum += 0

        # Add reciprocal rank sum to MRR sum for this query
        mrr_sum += reciprocal_rank_sum

    # Calculate overall MRR as the average of MRRs for all queries with relevant items
    mrr = mrr_sum / num_queries if num_queries > 0 else 0

    return mrr

def get_precisionk(ranking_main, ranking_cv, topk):
    """
    Calculate Precision@k for two rankings.

    Args:
    - ranking_main (DataFrame): DataFrame containing the main ranking data.
    - ranking_cv (DataFrame): DataFrame containing the cross-validation ranking data.
    - topk (int): Top-k value for precision calculation.

    Returns:
    - tuple: Precision@k value, recommended developers, actual developers, common developers.
    """
    # Sort the rankings by rank and select the top-k developers
    ranking_main = ranking_main.sort_values(by="rank", ascending=True).head(topk)
    ranking_cv = ranking_cv.sort_values(by="rank", ascending=True).head(topk)

    # Find common developers in the top-k rankings
    common_devs = ranking_main.merge(ranking_cv, on="devname", how="inner")

    # Calculate the number of true positives (top-k developers recommended by the tool that correctly represent the actual developers)
    true_positives = common_devs.shape[0]

    # Calculate false positives (top-k developers recommended by the tool that do not represent the actual developers)
    false_positive = topk - true_positives

    # Calculate Precision@k using the formula (true_positives / (true_positives + false_positive))
    precision = true_positives / (true_positives + false_positive) if (true_positives + false_positive) > 0 else 0

    # Get lists of recommended, actual, and common developers
    recommended = ranking_main['devname'].unique()
    actual = ranking_cv['devname'].unique()
    common = common_devs['devname'].unique()

    # Round precision to two decimal places and return along with other results
    return round(precision, 2), recommended, actual, common


def get_recallk(ranking_main, ranking_cv, topk):
    """
    Calculate Recall@k for two rankings.

    Args:
    - ranking_main (DataFrame): DataFrame containing the main ranking data.
    - ranking_cv (DataFrame): DataFrame containing the cross-validation ranking data.
    - topk (int): Top-k value for recall calculation.

    Returns:
    - float: Recall@k value.
    """
    # Make copies of the input rankings to avoid modifying the original data
    rm = ranking_main.copy()
    rc = ranking_cv.copy()

    # Sort the rankings by rank and select the top-k developers
    ranking_main = ranking_main.sort_values(by="rank", ascending=True).head(topk)
    ranking_cv = ranking_cv.sort_values(by="rank", ascending=True).head(topk)

    # Find common developers in the top-k rankings
    common_devs = ranking_main.merge(ranking_cv, on="devname", how="inner")

    # Calculate the number of true positives (common developers in both rankings)
    true_positives = common_devs.shape[0]

    # Calculate false negatives (actual developers not represented in top-k actual developers)
    false_negatives = len(rm[~rm['devname'].isin(common_devs['devname']) & ~rm['devname'].isin(rc['devname'])])

    # Calculate Recall@k using the formula (true_positives / (true_positives + false_negatives))
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    # Round recall to two decimal places and return
    return round(recall, 2)

def get_metrics(ranking_main, ranking_cv, topk):
    """
    Calculate multiple evaluation metrics based on two rankings.

    Args:
    - ranking_main (DataFrame): DataFrame containing the main ranking data.
    - ranking_cv (DataFrame): DataFrame containing the cross-validation ranking data.
    - topk (int): Top-k value for evaluation metrics.

    Returns:
    - tuple: Contains Mean Reciprocal Rank (MRR), Precision@k, Recall@k,
             Recommended developers, Actual developers, and Common developers.
    """

    # Preprocess rankings to ensure they are in the desired format
    ranking_main = ranking_main.groupby("devname")['occurances'].sum().reset_index()
    ranking_main.columns = ["devname", "occurances"]
    ranking_main['rank'] = ranking_main["occurances"].rank(method='min', ascending=False)

    ranking_cv = ranking_cv.groupby("devname")['occurances'].sum().reset_index()
    ranking_cv.columns = ["devname", "occurances"]
    ranking_cv['rank'] = ranking_cv["occurances"].rank(method='min', ascending=False)

    # Calculate Mean Reciprocal Rank (MRR) using the get_mrr function
    mrr = get_mrr(ranking_main, ranking_cv)

    # Calculate Precision@k, Recall@k, and get lists of recommended, actual, and common developers
    precision, recommended, actual, common = get_precisionk(ranking_main, ranking_cv, topk)
    recall = get_recallk(ranking_main, ranking_cv, topk)

    # Return a tuple containing all calculated metrics and lists
    return mrr, precision, recall, recommended, actual, common

def plot_scatter_plot(pareto, f_values, ref_points):
  fig = plt.figure()
  ax = fig.add_subplot(projection='3d')

  n = 100

  # For each set of style and range settings, plot n random points in the box
  # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
  for m, pts in [('o', pareto), ('^', f_values)]:
      ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], marker=m)

  ax.scatter(ref_points[0], ref_points[1], ref_points[2])

  ax.set_xlabel('X Label')
  ax.set_ylabel('Y Label')
  ax.set_zlabel('Z Label')
  plt.legend(["pareto_front", "F values", "Ref Point"])

  plt.show()

from sklearn.model_selection import KFold

df = all_data.copy()
# Convert 'task_created_date' to datetime
all_data['task_created_date'] = pd.to_datetime(all_data['task_created_date'])

# Identify unique tasks and their associated developers
task_developer_mapping = all_data.groupby('challengeId')['handle'].unique().to_dict()

n_folds = 10

# Create a KFold object with 10 folds
kf = KFold(n_splits=n_folds, shuffle=False, random_state=None)

# save the folds to a list
folds = []

# Iterate over the folds for training and testing
for train_index, test_index in kf.split(task_developer_mapping):
    # Save the train and test indices to the list
    folds.append((train_index, test_index))

# Initialize an empty list to store performance metrics for each iteration
performance_metrics = []
iteration = 1
# Iterate over the folds for training and testing
for i in range(n_folds//2,n_folds):

    train_index = np.concatenate([fold[0] for fold in folds[0:i]])
    test_index = folds[i][1]

    # Use the next fold for testing
    test_tasks = dict(list(task_developer_mapping.items())[j] for j in test_index)

    # Use the previous folds for training
    train_tasks = dict(list(task_developer_mapping.items())[j] for j in train_index)

    # Filter DataFrame based on selected tasks for training
    train_data = pd.concat([df[df['challengeId'] == task_id].sort_values(by='task_created_date') for task_id in train_tasks.keys()])

    # Train the model on the training data
    res_train = run_algo(train_data, verbose=True)

    # Get the recommended developers for the tasks in the trained set
    recommended_devs_train = get_ranking(res_train.X, train_data)

    # Filter DataFrame based on selected tasks for testing
    test_data = pd.concat([df[df['challengeId'] == task_id].sort_values(by='task_created_date') for task_id in test_tasks.keys()])

    # Test the model on the testing data
    res_test = run_algo(test_data, verbose=True)

    # Get the recommended developers for the tasks in the testing set
    recommended_devs_test = get_ranking(res_test.X, test_data)

    # Calculate the performance metrics
    mrr, precision, recall, recommended, actual, common = get_metrics(recommended_devs_train, recommended_devs_test, topk)

    # get the pareto front
    pareto = pareto_front(res_train.F)


    pareto = [list(i) for i in pareto]

    all_ones = [1 for i in range(len(test_data))]

    # get gd, igd, hv
    gd = get_performance_indicator("gd", pf=np.array(pareto))
    igd = get_performance_indicator("igd", pf=np.array(pareto))
    hv = get_performance_indicator("hv", ref_point=np.array([0, 0, f3(all_ones, test_data)]))

    print("Iteration: ", iteration)
    print("train_folds: ",str(0),"-",str(len(folds[0:i])-1))
    print("test_index: ", i)
    # print("recommended Solutions:", res_train.X.astype(int))
    # print("fitness values: ", res_train.F)
    # print("Actual Solutions:", res_test.X.astype(int))
    # print("fitness values: ", res_test.F)
    print("MRR: ", mrr)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("Recommended: ", recommended)
    print("Actual: ", actual)
    print("Common: ", common)
    print("GD: ", gd.do(res_test.F))
    print("IGD: ", igd.do(res_test.F))
    print("HV: ", hv.do(res_test.F))
    print("")
    # plot_scatter_plot(np.array(pareto), res_test.F, [0, 0, f3(all_ones, test_data)])

    performance_metrics.append({
        "Iteration":iteration,
        "Train_Folds":str(0)+"-"+str(len(folds[0:i])-1),
        "Test_Fold": i,
        "MRR": mrr,
        "Actual": actual,
        "Recommended": recommended,
        "Common": common,
        "gd": gd.do(res_test.F),
        "igd": igd.do(res_test.F),
        "hv": hv.do(res_test.F),
        "precision": precision,
        "recall": recall})

    iteration +=1

# After the loop, you can analyze the overall performance using the collected metrics
performance_metrics_df = pd.DataFrame(performance_metrics)

performance_metrics_df

# save the performance metrics to a csv file
performance_metrics_df.to_csv('performance_metrics_v1.csv', index=False)

"""- For each in performance metrics get Metric,Mean,Median,Mode and getting into df and csv"""

# read the performance metrics from the csv file
pm_df = pd.read_csv('performance_metrics_v1.csv')

def calculate_stats(df, column_name):
    mean = df[column_name].mean()
    median = df[column_name].median()
    mode = df[column_name].mode().iloc[0]  # Take the first mode if it exists

    return mean, median, mode

# Example usage:
pm_df = pd.read_csv('performance_metrics_v1.csv')

# Metrics to calculate stats for
metrics = ['gd', 'igd', 'hv', 'MRR', 'precision', 'recall']

# Create a list to store the results
result_list = []

# Calculate stats for each metric and add to the list
for metric in metrics:
    mean, median, mode = calculate_stats(pm_df, metric)
    result_list.append({'Metric': metric, 'Mean': mean, 'Median': median, 'Mode': mode})

# Convert the list to a DataFrame
result_df = pd.DataFrame(result_list, columns=['Metric', 'Mean', 'Median', 'Mode'])

# Display the result DataFrame
print(result_df)

# save the result to a csv file
result_df.to_csv('metrics_stats_v1.csv', index=False)

"""- For specific task id
    - Taskid will be used as text index at each iteration
    - kindly replace challengeid with the specific id that you want to test for
"""

# Initialize an empty list to store performance metrics for each iteration
challenge_metrics = []

challenge_id = "00a217cd-a4ba-4b2b-9788-0c602aa303f3"

# get the task data from challenge id
test_data = all_data[all_data['challengeId'] == challenge_id]
# convert task to a dataframe
test_data = pd.DataFrame(test_data)

iteration = 1

# Iterate over the folds for training and testing
for i in range(5,10):

    train_index = np.concatenate([fold[0] for fold in folds[0:i]])

    # Split tasks into train and test sets
    train_tasks = dict(list(task_developer_mapping.items())[j] for j in train_index)

    # Filter DataFrame based on selected tasks for training
    train_data = pd.concat([df[df['challengeId'] == task_id].sort_values(by='task_created_date') for task_id in train_tasks.keys()])

    # Train the model on the training data
    res_train = run_algo(train_data, verbose=True)

    # Get the recommended developers for the tasks in the trained set
    recommended_devs_train = get_ranking(res_train.X, train_data)

    # Test the model on the testing data
    res_test = run_algo(test_data, verbose=True)

    # Get the recommended developers for the tasks in the testing set
    recommended_devs_test = get_ranking(res_test.X, test_data)

    # Calculate the performance metrics
    mrr, precision, recall, recommended, actual, common = get_metrics(recommended_devs_train, recommended_devs_test, topk)

    # get the pareto front
    pareto = pareto_front(res_train.F)

    pareto = [list(i) for i in pareto]

    all_ones = [1 for i in range(len(test_data))]

    # get gd, igd, hv
    gd = get_performance_indicator("gd", pf=np.array(pareto))
    igd = get_performance_indicator("igd", pf=np.array(pareto))
    hv = get_performance_indicator("hv", ref_point=np.array([0, 0, f3(all_ones, test_data)]))

    print("Iteration: ", iteration)
    print("train_folds: ",str(0),"-",str(len(folds[0:i])-1))
    print("test_index: ", i)
    # print("recommended Solutions:", res_train.X.astype(int))
    # print("fitness values: ", res_train.F)
    # print("Actual Solutions:", res_test.X.astype(int))
    # print("fitness values: ", res_test.F)
    print("MRR: ", mrr)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("Recommended: ", recommended)
    print("Actual: ", actual)
    print("Common: ", common)
    print("GD: ", gd.do(res_test.F))
    print("IGD: ", igd.do(res_test.F))
    print("HV: ", hv.do(res_test.F))
    print("")
    # plot_scatter_plot(np.array(pareto), res_test.F, [0, 0, f3(all_ones, test_data)])

    challenge_metrics.append({
        "ChallengeID":challenge_id,
        "Iteration":iteration,
        "Train_Folds":str(0)+"-"+str(len(folds[0:i])-1),
        "Test_Fold": i,
        "MRR": mrr,
        "Actual": actual,
        "Recommended": recommended,
        "Common": common,
        "gd": gd.do(res_test.F),
        "igd": igd.do(res_test.F),
        "hv": hv.do(res_test.F),
        "precision": precision,
        "recall": recall})

    iteration +=1

# After the loop, you can analyze the overall performance using the collected metrics
challenge_metrics_df = pd.DataFrame(challenge_metrics)

challenge_metrics_df

challenge_metrics_df.to_csv('challenge_metrics_v1.csv', index=False)
