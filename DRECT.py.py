
#!pip install pymoo==0.5.0

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



def f1(X,df):
    de = []
    for i in range(len(df)):
        
        de.append(int(X[i])*(((((df['completedchallenges'][i] / df['totalChallengsJoined'][i])+( df['totalWins'][i] // df['completedchallenges'][i]))* df['Success_rate'][i])*((1 / df['task_recency (in days)'][i])* 100)) + df['CP_score'][i] ))
    f1 = - (sum(de)/ len(de)) if len(de) > 0 else 0
    return f1


def f2(X,df): 
    sds = []
    for i in range(len(df)):
        sds.append(int(X[i])*(df['Cosine_similarity_descriptions'][i] + df['Cosine_similarity_skills'][i] + df['Cosine_similarity_score_task_titles_current_past'][i] + df['Cosine_similarity_score_task_descriptions_current_past'][i] + df['Cosine_similarity_score_task_skills_current_past'][i]))
    f2 = - (sum(sds) / len(sds)) if len(sds) > 0 else 0
    return f2 

def f3(X,df):
    ad = []
    for i in range(len(df)):
        ad.append(int(X[i])*(df['activechallenges'][i]))

    f3 = (sum(ad) / len(ad)) if len(ad) > 0 else 0
    return f3 

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


all_data = pd.read_csv('Dataset.csv')

all_data.columns

all_data.dropna(inplace=True)
all_data.sort_values("task_created_date", ascending=True, inplace=True)

all_data.columns

pop_size = 80
n_gen = 100
minSize = 1 
maxSize = 600000
obj_algo_name = 'NSGA2' 
topK = 10

if obj_algo_name ==  'NSGA2':
    obj_algo = NSGA2
if obj_algo_name == 'NSGA3':
    obj_algo = NSGA3
if obj_algo_name == 'UNSGA3':
    obj_algo = UNSGA3
if obj_algo_name == 'AGEMOEA':
    obj_algo = AGEMOEA


all_data.shape

all_data['challengeId'].unique()

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

# given res.X we get the number of occurances of each dev name in the solutions 
# and get the member rank basis number of occurances
def get_ranking(X_res, data):
  columns = data['handle'].values.tolist()
  results = pd.DataFrame(X_res, columns = columns)
  df = results.sum(axis=0).reset_index()
  df.columns = ["devname", "occurances"]
  df['rank'] = df["occurances"].rank(method='min', ascending=False)
  return df

# function to get the mean rank for most relevant members (identified during the
# iterations with complete set) in ranking for each cv set
def get_mean_rank(ranking_main, ranking_cv):
  # identify developers given rank 1 during the iterations for the complete set 
  # (only for those developers present in the particular cv set)
  relevant = ranking_main.loc[ranking_main['rank']==1, 'devname'].values.tolist()
  
  avg_rank = []
  for r in relevant:
    rank = ranking_cv.loc[ranking_cv['devname']==r, "rank"]
    rank = 1000 if rank.shape[0] == 0 else rank.iloc[0]
    avg_rank.append(rank)

  return np.mean(avg_rank)

def get_precisionk(ranking_main, ranking_cv, topk=5):
  ranking_cv.sort_values(by="rank", ascending=True, inplace=True)
  ranking_cv = ranking_cv.head(topk)
  ranking_main = ranking_main.head(topk)
  # number of relevant items among top k recommendations
  numerator = ranking_cv.loc[ranking_cv['devname'].isin(ranking_main['devname'].values.tolist())].shape[0]
  return numerator/max(len(ranking_cv), len(ranking_main)), ranking_main['devname'].unique(), ranking_cv['devname'].unique(), ranking_cv.loc[ranking_cv['devname'].isin(ranking_main['devname'].values.tolist()), 'devname'].unique()

def get_recallk(ranking_main, ranking_cv, topk=5):
  ranking_cv.sort_values(by="rank", ascending=True, inplace=True)
  ranking_cv = ranking_cv.head(topk)
  ranking_main = ranking_main.head(topk)
  # number of relevant items among top k recommendations
  numerator = ranking_cv.loc[ranking_cv['devname'].isin(ranking_main['devname'].values.tolist())].shape[0]
  denominator = 0
  for dev in ranking_main['devname'].values.tolist():
    if dev in ranking_cv['devname'].values.tolist():
      denominator += 1
  return 0 if denominator is 0 else numerator/denominator

# using the rankings for main set and the cv set
# (only for those developers present in the particular cv set)
def get_metrics(ranking_main, ranking_cv, topk):
  # duplication was observed in devnames so occurances were added and rank was calculated using the sum
  ranking_main = ranking_main.groupby("devname")['occurances'].sum().reset_index()
  ranking_main.columns = ["devname", "occurances"]
  ranking_main['rank'] = ranking_main["occurances"].rank(method='min', ascending=False)
  ranking_cv = ranking_cv.groupby("devname")['occurances'].sum().reset_index()
  ranking_cv.columns = ["devname", "occurances"]
  ranking_cv['rank'] = ranking_cv["occurances"].rank(method='min', ascending=False)
  mean_rank = get_mean_rank(ranking_main, ranking_cv)
  precision, recommended, actual, common = get_precisionk(ranking_main, ranking_cv, topk)
  recall = get_recallk(ranking_main, ranking_cv, topk)
  return mean_rank, precision, recall , recommended, actual, common

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

  #plt.show()

"""
"""
number = 2
jj = 0
metrics = []
last_x_tasks = 3
unique_task_ids = all_data['challengeId'].unique()

for k, STUDIED_TASK in enumerate(unique_task_ids[1:]):
    try:
      if k >= last_x_tasks:
        TASK_DF = all_data[all_data['challengeId'].isin(unique_task_ids[max(0, k-last_x_tasks+1):k+1])]
        unique_devs = TASK_DF['handle'].unique()
        # take means for devs whose name occurs more than once in the set
        TASK_DF = TASK_DF.groupby('handle').mean().reset_index()

        DF = all_data[all_data['challengeId'] ==  STUDIED_TASK].reset_index(drop=True)
        DF = DF[DF['handle'].isin(unique_devs)].reset_index(drop=True)
        unique_handles = DF['handle'].unique()

        result_prev = run_algo(TASK_DF, False)
        pf = pareto_front(result_prev.F)
        pf = [list(x) for x in pf]
        ranking_previous_task = get_ranking(result_prev.X, TASK_DF)

        # if DF.shape[0]==0:
        #   print("\nNo Actual developer found (from the list of developers in the previous tasks)\n")
        DF = DF.groupby('handle').mean().reset_index()

        all_ones= [1 for i in range(len(DF))]
        result = run_algo(DF, False)

        gd = get_performance_indicator("gd", pf=np.array(pf))
        igd = get_performance_indicator("igd", pf=np.array(pf))
        hv = get_performance_indicator("hv", ref_point=np.array([0, 0, f3(all_ones,DF)]))

        metrics.append({"GD": gd.do(result.F), "IGD": igd.do(result.F), "HV": hv.do(result.F)})
        ranking_current_task = get_ranking(result.X, DF)
        mean_rank, precision, recall, recommended, actual, common = get_metrics(ranking_previous_task, ranking_current_task, topK)

        if DF.shape[0]> 0 and precision > 0.5:
          print("\nRecommended Solutions")
          print(result_prev.X.astype(int))

          print("\nFitness values")
          print(result_prev.F)

          print("\nActual Solutions")
          print(result.X.astype(int))

          print("\nFitness values")
          print(result.F)

          print("Metrics")
          print("Recommended Developers: ", recommended)
          print("Actual Developers: ", actual)
          print("Common Developers: ", common)
          print({"GD": gd.do(result.F), "IGD": igd.do(result.F), "HV": hv.do(result.F), "MRR": 1/mean_rank, "Precision@K": precision, "Recall@K": recall})
          print()
          plot_scatter_plot(np.array(pf), result.F, np.array([0, 0, f3(all_ones,DF)]))
          metrics.append({"GD": gd.do(result.F), "IGD": igd.do(result.F), "HV": hv.do(result.F), "MRR": 1/mean_rank, "Precision@K": precision, "Recall@K": recall})
    except Exception as ex:
      # print("Exception {}".format(ex))
      continue
        
metrics = pd.DataFrame(metrics)
metrics.to_csv("task_details_10.csv", index=False)
print(metrics.mean())

