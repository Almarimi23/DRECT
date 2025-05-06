# !pip install pymoo==0.5.0
# !pip install platypus-opt

import pandas as pd
import os
import numpy as np
import itertools
import math
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt

from pymoo.operators.selection.rnd import RandomSelection
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.algorithms.moo.moead import MOEAD, ParallelMOEAD
from pymoo.algorithms.moo.age import AGEMOEA
from pymoo.algorithms.moo.ctaea import CTAEA
from pymoo.visualization.scatter import Scatter
from pymoo.factory import get_crossover, get_mutation, get_sampling, get_reference_directions
from pymoo.core.problem import Problem, ElementwiseProblem
from pymoo.factory import get_performance_indicator
from pymoo.optimize import minimize
from pymoo.config import Config
Config.show_compile_hint = False

from sklearn.model_selection import KFold
from platypus import Problem as PlatypusProblem, Integer, SPEA2 as PlatypusSPEA2
### ------------------ Objective Functions ------------------ ###

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

### ------------------ pymoo Problem Definition ------------------ ###

class MyProblem(ElementwiseProblem):

    def __init__(self, task_df, maxSize, minSize, objectives=[f1, f2, f3]):
        self.task_df = task_df
        self.maxSize = maxSize
        self.minSize = minSize
        self.objectives = objectives
        super().__init__(n_var=len(self.task_df),
                         n_obj=3,
                         n_constr=2,
                         sampling=get_sampling('real_random'))

    def _evaluate(self, X, out, *args, **kwargs):
        obj1 = self.objectives[0](X, self.task_df)
        obj2 = self.objectives[1](X, self.task_df)
        obj3 = self.objectives[2](X, self.task_df)
        g1 = (self.minSize - np.sum(X))
        g2 = (np.sum(X) - self.maxSize)
        out["F"] = [obj1, obj2, obj3]
        out["G"] = [g1, g2]

### ------------------ Data Preprocessing ------------------ ###

all_data = pd.read_csv('Full_dataset_1.csv')
all_data.dropna(inplace=True)
df = all_data.copy()
df.sort_values(by=['task_created_date', 'challengeId'], inplace=True)
df.reset_index(drop=True, inplace=True)

# Divide dataset into 10 folds for cross validation
fold_size = int(len(df) / 10)
folds = []
for i in range(10):
    if i == 9:
        folds.append(df.iloc[i * fold_size:])
    else:
        folds.append(df.iloc[i * fold_size:(i + 1) * fold_size])

### ------------------ Algorithm Selection ------------------ ###
"""
Available options:
- 'NSGA2' (or 'NSGA3', 'UNSGA3', 'AGEMOEA', 'MOEAD') using pymoo
- 'SPEA2' using Platypus
- 'RandomSearch' using custom implementation

Set the variable below to select the algorithm.
"""
obj_algo_name = 'SPEA2'  # Change to 'SPEA2' or 'RandomSearch'

if obj_algo_name not in ["SPEA2", "RandomSearch"]:
    if obj_algo_name == 'NSGA2':
        obj_algo = NSGA2
    elif obj_algo_name == 'NSGA3':
        obj_algo = NSGA3
    elif obj_algo_name == 'UNSGA3':
        obj_algo = UNSGA3
    elif obj_algo_name == 'AGEMOEA':
        obj_algo = AGEMOEA
    elif obj_algo_name == 'MOEAD':
        obj_algo = MOEAD

pop_size = 10
n_gen = 10
minSize = 1
maxSize = 100000
topk = 10

# Platypus-Based Algorithm Runner (for SPEA2)

def run_platypus_algo(TASK_DF, algo_name, verbose):
    n_var = len(TASK_DF)
    problem = PlatypusProblem(n_var, 3, 2)
    problem.types[:] = [Integer(0, 1) for _ in range(n_var)]
    def evaluate(x):
        obj1 = f1(x, TASK_DF)
        obj2 = f2(x, TASK_DF)
        obj3 = f3(x, TASK_DF)
        return [obj1, obj2, obj3], [minSize - sum(x), sum(x) - maxSize]
    problem.function = evaluate
    if algo_name == "SPEA2":
        algorithm = PlatypusSPEA2(problem, population_size=pop_size)
    else:
        raise ValueError("Unknown Platypus algorithm: " + algo_name)
    algorithm.run(n_gen)
    solutions = algorithm.result
    class ResultWrapper:
        pass
    result = ResultWrapper()
    result.X = np.array([sol.variables for sol in solutions])
    result.F = np.array([sol.objectives for sol in solutions])
    return result

# RandomSearch Implementation

def run_random_search(TASK_DF, verbose):
    n_var = len(TASK_DF)
    n_iter = n_gen * pop_size
    problem = MyProblem(TASK_DF, minSize, maxSize)
    X_list = []
    F_list = []
    for i in range(n_iter):
        # Sample a random binary vector of length n_var
        X = np.random.randint(0, 2, size=n_var)
        out = {}
        problem._evaluate(X, out)
        X_list.append(X)
        F_list.append(out["F"])
    class ResultWrapper:
        pass
    result = ResultWrapper()
    result.X = np.array(X_list)
    result.F = np.array(F_list)
    return result

### ------------------ Unified run_algo Function ------------------ ###

def run_algo(TASK_DF, verbose):
    if obj_algo_name == "SPEA2":
        return run_platypus_algo(TASK_DF, obj_algo_name, verbose)
    elif obj_algo_name == "RandomSearch":
        return run_random_search(TASK_DF, verbose)
    else:
        ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)
        vectorized_problem = MyProblem(TASK_DF, minSize=minSize, maxSize=maxSize)
        if obj_algo_name == 'MOEAD':
            algorithm = obj_algo(
                ref_dirs=ref_dirs,
                n_neighbors=20,
                decomposition="pbi",
                prob_neighbor_mating=0.7,
                sampling=get_sampling("bin_random"),
                crossover=get_crossover("bin_one_point"),
                mutation=get_mutation("bin_bitflip"),
                seed=1
            )
        else:
            algorithm = obj_algo(
                pop_size=pop_size,
                sampling=get_sampling("bin_random"),
                crossover=get_crossover("bin_one_point"),
                mutation=get_mutation("bin_bitflip", prob=1/vectorized_problem.n_var),
                ref_dirs=ref_dirs,
                eliminate_duplicates=True,
                seed=1
            )
        res = minimize(vectorized_problem,
                       algorithm,
                       termination=('n_gen', n_gen),
                       verbose=verbose)
        return res

### ------------------ Evaluation Metrics Functions ------------------ ###

def get_ranking(X_res, data):
    columns = data['handle'].values.tolist()
    if X_res.ndim == 3 and X_res.shape[2] == 1:
        X_res = np.squeeze(X_res, axis=2)
    results = pd.DataFrame(X_res, columns=columns)
    df_rank = results.sum(axis=0).reset_index()
    df_rank.columns = ["devname", "occurances"]
    df_rank['rank'] = df_rank["occurances"].rank(method='min', ascending=False)
    return df_rank

def get_mrr(ranking_main, ranking_cv):
    mrr_sum = 0
    num_queries = min(len(ranking_main), len(ranking_cv))
    for i in range(num_queries):
        relevant_items = set(ranking_cv.iloc[i]['devname'])
        ranked_items = ranking_main.iloc[i]['devname']
        reciprocal_rank_sum = 0
        found_relevant_item = False
        for rank, item in enumerate(ranked_items, 1):
            if item in relevant_items:
                reciprocal_rank_sum += 1 / rank
                found_relevant_item = True
                break
        if not found_relevant_item:
            reciprocal_rank_sum += 0
        mrr_sum += reciprocal_rank_sum
    mrr = mrr_sum / num_queries if num_queries > 0 else 0
    return mrr

def get_precisionk(ranking_main, ranking_cv, topk):
    ranking_main = ranking_main.sort_values(by="rank", ascending=True).head(topk)
    ranking_cv = ranking_cv.sort_values(by="rank", ascending=True).head(topk)
    common_devs = ranking_main.merge(ranking_cv, on="devname", how="inner")
    true_positives = common_devs.shape[0]
    false_positive = topk - true_positives
    precision = true_positives / (true_positives + false_positive) if (true_positives + false_positive) > 0 else 0
    recommended = ranking_main['devname'].unique()
    actual = ranking_cv['devname'].unique()
    common = common_devs['devname'].unique()
    return round(precision, 2), recommended, actual, common

def get_recallk(ranking_main, ranking_cv, topk):
    rm = ranking_main.copy()
    rc = ranking_cv.copy()
    ranking_main = ranking_main.sort_values(by="rank", ascending=True).head(topk)
    ranking_cv = ranking_cv.sort_values(by="rank", ascending=True).head(topk)
    common_devs = ranking_main.merge(ranking_cv, on="devname", how="inner")
    true_positives = common_devs.shape[0]
    false_negatives = len(rc[~rc['devname'].isin(common_devs['devname']) & ~rc['devname'].isin(rm['devname'])])
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    return round(recall, 2)

def get_metrics(ranking_main, ranking_cv, topk):
    ranking_main = ranking_main.groupby("devname")['occurances'].sum().reset_index()
    ranking_main.columns = ["devname", "occurances"]
    ranking_main['rank'] = ranking_main["occurances"].rank(method='min', ascending=False)
    ranking_cv = ranking_cv.groupby("devname")['occurances'].sum().reset_index()
    ranking_cv.columns = ["devname", "occurances"]
    ranking_cv['rank'] = ranking_cv["occurances"].rank(method='min', ascending=False)
    mrr = get_mrr(ranking_main, ranking_cv)
    precision, recommended, actual, common = get_precisionk(ranking_main, ranking_cv, topk)
    recall = get_recallk(ranking_main, ranking_cv, topk)
    return mrr, precision, recall, recommended, actual, common

def plot_scatter_plot(pareto, f_values, ref_points):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for m, pts in [('o', pareto), ('^', f_values)]:
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], marker=m)
    ax.scatter(ref_points[0], ref_points[1], ref_points[2])
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.legend(["pareto_front", "F values", "Ref Point"])
    plt.show()

### ------------------ 10-Fold Cross Validation ------------------ ###

performance_metrics = []
iteration = 1

for i in range(5):
    train_indices = list(range(i + 5))
    test_index = i + 5

    train_data = pd.concat([folds[j] for j in train_indices], ignore_index=True)
    test_data = folds[test_index]

    train_data = pd.concat([df[df['challengeId'] == task_id].sort_values(by='task_created_date')
                            for task_id in train_data['challengeId'].unique()])
    res_train = run_algo(train_data, verbose=True)
    recommended_devs_train = get_ranking(res_train.X, train_data)

    test_data = pd.concat([df[df['challengeId'] == task_id].sort_values(by='task_created_date')
                           for task_id in test_data['challengeId'].unique()])
    res_test = run_algo(test_data, verbose=True)
    recommended_devs_test = get_ranking(res_test.X, test_data)

    mrr, precision, recall, recommended, actual, common = get_metrics(recommended_devs_train, recommended_devs_test, topk)

    pareto = [list(i) for i in res_train.F]
    all_ones = [1 for _ in range(len(test_data))]
    gd = get_performance_indicator("gd", pf=np.array(pareto))
    igd = get_performance_indicator("igd", pf=np.array(pareto))
    hv = get_performance_indicator("hv", ref_point=np.array([0, 0, f3(all_ones, test_data)]))

    print("Iteration: ", iteration)
    print("train_folds: ", list(range(i + 5)))
    print("test_index: ", i + 5)
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

    performance_metrics.append({
        "Iteration": iteration,
        "Train_Folds": list(range(i + 5)),
        "Test_Fold": i + 5,
        "MRR": mrr,
        "Actual": actual,
        "Recommended": recommended,
        "Common": common,
        "gd": gd.do(res_test.F),
        "igd": igd.do(res_test.F),
        "hv": hv.do(res_test.F),
        "precision": precision,
        "recall": recall
    })
    iteration += 1

performance_metrics_df = pd.DataFrame(performance_metrics)
performance_metrics_df.to_csv('performance_metrics_v1.csv', index=False)
print(performance_metrics_df)

pm_df = pd.read_csv('performance_metrics_v1.csv')

def calculate_stats(df, column_name):
    mean_val = df[column_name].mean()
    median_val = df[column_name].median()
    mode_val = df[column_name].mode().iloc[0]
    return mean_val, median_val, mode_val

metrics = ['gd', 'igd', 'hv', 'MRR', 'precision', 'recall']
result_list = []
for metric in metrics:
    mean_val, median_val, mode_val = calculate_stats(pm_df, metric)
    result_list.append({'Metric': metric, 'Mean': mean_val, 'Median': median_val, 'Mode': mode_val})

result_df = pd.DataFrame(result_list, columns=['Metric', 'Mean', 'Median', 'Mode'])
print(result_df)
result_df.to_csv('metrics_stats_v1.csv', index=False)
