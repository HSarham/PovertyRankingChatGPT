import os
from sklearn.metrics import matthews_corrcoef, r2_score
from scipy.stats import spearmanr, pearsonr
import numpy as np
from sklearn.preprocessing import binarize
import matplotlib.pyplot as plt
import pandas as pd
import choix
from scipy.stats import rankdata, bootstrap
import networkx as nx

def get_metric_and_confidence_interval(targets, outputs, metric, confidence):

    if np.any(outputs == "random"):
        #targets = targets.repeat(1000)
        rng = np.random.default_rng(seed=0)

        values = []
        CIs = np.empty((0,2))
        SEs = []

        for i in range(1000):
            if targets.dtype == int:
                outputs = rng.choice(np.unique(targets), size=len(targets), replace=True)
            else:
                outputs = rng.random(size=len(targets))

            result = bootstrap((targets, outputs), metric, random_state=0, confidence_level=confidence, paired=True,
                               vectorized=False, method="percentile", n_resamples=10)

            values.append(metric(targets,outputs))
            CIs = np.vstack((CIs,result.confidence_interval))
            SEs.append(result.standard_error)

        value = np.mean(values)
        CI = np.mean(CIs,axis=0)
        SE = np.mean(SEs)

    else:
        result = bootstrap((targets, outputs), metric, random_state=0, confidence_level=confidence, paired=True,
                           vectorized=False, method="percentile", n_resamples=10000)
        value = metric(targets,outputs)
        CI = result.confidence_interval
        SE = result.standard_error

    return value, CI, SE

def print_metric_and_confidence_interval(targets,outputs,metric,confidence=0.95):
    if metric is spearmanr:
        metric = lambda x,y: spearmanr(x,y).statistic
        metric.__name__ = "spearmanr"

    metric_value, interval, standard_error = get_metric_and_confidence_interval(targets, outputs, metric, confidence)

    print(f"\t\t{metric.__name__}: {metric_value:0.2f}\n\t\t\t95% Confidence Interval: [{interval[0]:0.2f} {interval[1]:0.2f}]"
          f"\tStandard Error: {standard_error:0.2f}")

def get_ranking_and_quantile_metrics(targets, outputs="random", qprobs = (0, 0.2, 0.4, 0.6, 0.8)):

    if np.any(outputs != "random"):
        mask = ~np.isnan(outputs)

        if np.sum(mask) != len(outputs):
            targets = targets[mask]
            outputs = outputs[mask]

    print("\tRank Correlation:")
    print_metric_and_confidence_interval(targets, outputs, spearmanr)

    print("\tQuantile prediction:")
    print()
    _, target_qs = get_quantiles_labels(targets,qprobs)
    if np.any(outputs != "random"):
        _, output_qs = get_quantiles_labels(outputs,qprobs)
    else:
        output_qs = "random"

    print_metric_and_confidence_interval(target_qs,output_qs,matthews_corrcoef)

    num_classes = len(np.unique(target_qs))
    for threshold in range(1,num_classes):
        print(f"\t\tDichotomy {threshold}:")
        bin_targets = binarize(np.array(target_qs).reshape(-1,1),threshold=threshold).reshape(-1).astype(int)
        if np.any(outputs != "random"):
            bin_outputs = binarize(np.array(output_qs).reshape(-1,1),threshold=threshold).reshape(-1)
        else:
            bin_outputs = outputs
        print_metric_and_confidence_interval(bin_targets,bin_outputs,matthews_corrcoef)

def get_metric_diff_CI(targets,outputs_1,outputs_2,metric,confidence):
    result = bootstrap((targets, outputs_1, outputs_2), metric, random_state=0, confidence_level=confidence, paired=True,
                       vectorized=False, method="percentile")
    return metric(targets, outputs_1, outputs_2), result.confidence_interval, result.standard_error

def print_metric_diff_CI(targets,outputs_1,outputs_2,metric,confidence=0.95):
    metric_name = metric.__name__
    if metric is spearmanr:
        metric_diff = lambda y, x, z: metric(y, x).statistic - metric(y, z).statistic
    else:
        metric_diff = lambda y, x, z: metric(y, x) - metric(y, z)

    metric_value, interval, standard_error = get_metric_diff_CI(targets,outputs_1,outputs_2,metric_diff,confidence)

    print(f"\t\t{metric_name} difference: {metric_value:0.2f}\n\t\t\t95% Confidence Interval: [{interval.low:0.2f} {interval.high:0.2f}]"
          f"\tStandard Error: {standard_error:0.2f}"
          f'   {"significantly different" if interval.low>0 or interval.high<0 else "not significantly different"}')

def get_metrics_diff_confidence_intervals(targets, outputs_1, outputs_2,
                                          qprobs = (0,0.2,0.4,0.6,0.8)):
    mask = ~np.isnan(outputs_1) & ~np.isnan(outputs_2)
    _, target_qs = get_quantiles_labels(targets[mask],qprobs)
    _, output_1_qs = get_quantiles_labels(outputs_1[mask],qprobs)
    _, output_2_qs = get_quantiles_labels(outputs_2[mask],qprobs)

    print("\tSpearman Rank Correlation")
    print_metric_diff_CI(target_qs, outputs_1[mask], outputs_2[mask], spearmanr)

    print("\tQuantile prediction:")
    print_metric_diff_CI(target_qs, output_1_qs, output_2_qs, matthews_corrcoef)

    num_classes = len(np.unique(target_qs))
    for threshold in range(1, num_classes):
        print(f"\t\tDichotomy {threshold}:")
        bin_targets = binarize(np.array(target_qs).reshape(-1, 1), threshold=threshold).reshape(-1)
        bin_outputs_1 = binarize(np.array(output_1_qs).reshape(-1, 1), threshold=threshold).reshape(-1)
        bin_outputs_2 = binarize(np.array(output_2_qs).reshape(-1, 1), threshold=threshold).reshape(-1)
        print_metric_diff_CI(bin_targets, bin_outputs_1, bin_outputs_2, matthews_corrcoef)

def get_quantiles_labels(values,qprobs):
    quantiles = np.quantile(values,qprobs)
    #print("quantiles:",quantiles)
    q_indices = []
    qs = np.empty(len(values))
    for i in range(len(quantiles)):
        if i == len(quantiles)-1:
            indices = np.where(quantiles[i]<=values)[0]
        else:
            indices = np.where((quantiles[i]<=values) & (values<quantiles[i+1]))[0]
        q_indices.append(indices)
        qs[indices] = i+1
    
    return q_indices,np.array(qs,dtype=int)

def show_scatter_samples(df,wealth_index_column,cnn_reg_column,indices=None,use_ranks=True):
        wealth_indices = df[wealth_index_column]
        chat_ranking_scores = df["ranking"]
        cnn_ranking_scores = df[cnn_reg_column]

        wi_ranks = wealth_indices
        chat_ranks = chat_ranking_scores
        cnn_ranks = cnn_ranking_scores

        if use_ranks:
            wi_ranks = rankdata(wi_ranks)
            chat_ranks = rankdata(chat_ranks)
            cnn_ranks = rankdata(cnn_ranks)

        if np.any(indices):
            wi_ranks = wi_ranks[indices]
            chat_ranks = chat_ranks[indices]
            cnn_ranks = cnn_ranks[indices]
            
        fig, ax = plt.subplots(1,2)
        fig.set_size_inches(12.8,4.8)

        dot_size = 5

        ax[0].scatter(wi_ranks,cnn_ranks,s=dot_size)
        ax[0].set_ylabel("CNN 3x3 rank (prediction)")
        ax[0].set_xlabel("hv271 rank (ground truth)")
        ax[0].set_aspect("equal")
        
        ax[1].scatter(wi_ranks,chat_ranks,s=dot_size)
        ax[1].set_ylabel("ChatGPT rank (prediction)")
        ax[1].set_xlabel("hv271 rank (ground truth)")
        ax[1].set_aspect("equal")

        os.makedirs("output_figures",exist_ok=True)
        plt.savefig("output_figures/dhs_vs_cnn_chat.pdf",bbox_inches="tight")
        plt.show()

        print(f"CNN 3x3 Rho:{spearmanr(wi_ranks, cnn_ranks).statistic:0.2f},\t"
              f"ChatGPT Rho:{spearmanr(wi_ranks, chat_ranks).statistic:0.2f}")

        print(f"CNN 3x3 R^2:{r2_score(wealth_indices, cnn_ranking_scores):0.2f},\t"
              f"CNN 3x3 Pearson (r):{pearsonr(wealth_indices, cnn_ranking_scores).statistic:0.2f}")


def get_winning_list_winning_dict(comp_df,survey_df):
    winning_dict = {file_name:0 for file_name in survey_df["file_name"]}
    winning_list = []
    tie_pairs = [] 
    for i in range(len(comp_df)):
        im1 = comp_df.iloc[i,0]
        im2 = comp_df.iloc[i,1]
        winner = comp_df.iloc[i,2]
        
        if winner == im1:
            winning_list.append((im1,im2))
        elif winner == im2:
            winning_list.append((im2,im1))

        if winner != "tie":
            winning_dict[winner] += 1
        else:
            tie_pairs.append((im1,im2))

    return winning_list, winning_dict, tie_pairs

def find_defying_comparisons(winning_list,rankings_df,column,ax=None,bins=None,xpercentage=False):

    ranking_column = rankings_df.set_index("file_name")[column].copy()

    if not xpercentage:
        ranking_column = pd.Series(data=rankdata(ranking_column.values),index=ranking_column.index)

    column_dict = ranking_column.to_dict()

    value_range_percent = rankings_df[column].max()-rankings_df[column].min()/100

    if not xpercentage:
        value_range_percent = 1

    defying_comparisons = []
    defying_comparisons_diff = []

    all_diffs = []

    for winner,loser in winning_list:
        winner_ranking = column_dict[winner]
        loser_ranking = column_dict[loser]
        all_diffs.append(winner_ranking-loser_ranking)
        if winner_ranking < loser_ranking:
            defying_comparisons.append((winner,loser))
            defying_comparisons_diff.append(winner_ranking-loser_ranking)

    defying_comparisons_diff = np.abs(np.array(defying_comparisons_diff))

    print("Num conflicts:", len(defying_comparisons))
    print("Total rank disagreement:", np.sum(defying_comparisons_diff))
    print("Rank disagreements' mean:", np.mean(defying_comparisons_diff))
    print("Rank disagreements' std:", np.std(defying_comparisons_diff))
    
    if ax:
        ax.hist(defying_comparisons_diff/value_range_percent,bins=bins,weights=np.ones(defying_comparisons_diff.shape)/len(winning_list)*100)
    else:
        plt.hist(defying_comparisons_diff/value_range_percent,bins=bins,weights=np.ones(defying_comparisons_diff.shape)/len(winning_list)*100)
        plt.xlabel("% of Value Range")
        plt.ylabel("% of Valid Comparisons")
        plt.show()
    
    return all_diffs

class SmartRanker:
    def __init__(self,comp_df,survey_df,wi_column,valid_rows=None):
        self.comp_df = comp_df
        self.survey_df = survey_df
        self.wi_column = wi_column
        self.valid_rows = valid_rows
        self.winning_list , self.winning_dict, self.tie_pairs = get_winning_list_winning_dict(self.comp_df,self.survey_df)
        self.params_dict = {}


    def rank(self): #Maystre and Grosslauser (2015)

        if np.any(self.valid_rows):
            survey_df = self.survey_df.loc[self.valid_rows]

        survey_df_files = np.unique(self.survey_df["file_name"].values)

        if np.any(self.valid_rows):
            comp_df_where = self.comp_df["image1"].isin(survey_df_files) & self.comp_df["image2"].isin(survey_df_files)
            self.comp_df=self.comp_df[comp_df_where]

        image_index_map = {v:i for i,v in enumerate(survey_df_files)}
        self.comp_df = self.comp_df.map(lambda s: image_index_map[s] if s!="tie" else s)
        self.survey_df["file_name"] = self.survey_df["file_name"].map(lambda s: image_index_map[s] if s!="tie" else s)
        num_images = len(image_index_map)

        self.winning_list , _, tie_pairs = get_winning_list_winning_dict(self.comp_df,self.survey_df)

        params_dict = {}

        params_dict["ilsr_pairwise"] =choix.ilsr_pairwise(num_images,self.winning_list,alpha=1e-297)
        params_dict["lsr_pairwise"] = choix.lsr_pairwise(num_images,self.winning_list,alpha=1e-306)
        params_dict["rank_centrality"] = choix.rank_centrality(num_images,self.winning_list,alpha=1e-323)

        survey_df = pd.concat([self.survey_df,pd.DataFrame(params_dict)],axis=1)

        self.analyze_comparison_quality(survey_df)

        return survey_df

    def analyze_comparison_quality(self, survey_df):

        fig, ax = plt.subplots(2, 2)
        fig.set_size_inches(12.8, 9.6)

        columns_titles = [{"wealth_index_hv271": "Survey", "rank_centrality": "Rank Centrality"},
                          {"lsr_pairwise": "LSR", "ilsr_pairwise": "I-LSR"}]

        for i, row_columns_titles in reversed(list(enumerate(columns_titles))):
            for j, column in reversed(list(enumerate(row_columns_titles))):
                if i != 0 or j != 0:
                    ax[i][j].set_ylim(0, 3.5)
                print(f"\n****{row_columns_titles[column]}")
                find_defying_comparisons(self.winning_list, survey_df, column, ax[i][j], bins=20)
                ax[i][j].set_title(row_columns_titles[column])

        fig.supylabel("Disagreeing Comparisons (% of Valid Comparisons)")
        fig.supxlabel("Ranking Difference")
        os.makedirs("output_figures", exist_ok=True)
        plt.savefig("output_figures/histogram_rankings.pdf", bbox_inches="tight")
        plt.show()

    def show_num_3_cycles(self):
        print("Getting the number of 3 cycles... (This might take a while)")

        graph = nx.DiGraph(self.winning_list)
        cycles_3 = list(nx.simple_cycles(graph,length_bound=3)) # Get the 3-cycles

        print("Number of 3 cycles:", len(cycles_3))



