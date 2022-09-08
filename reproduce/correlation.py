import json
import numpy as np
from scipy.stats import pearsonr, spearmanr


def system_level_correlation(human, other, metric=None):
    system_human, system_other = [], []
    for system_name in human:
        human_score = np.mean([human[system_name][doc_id] for doc_id in human[system_name]])
        system_human.append(human_score)
        other_score = np.mean([other[system_name][doc_id][metric] if metric else other[system_name][doc_id]
                               for doc_id in human[system_name]])
        system_other.append(other_score)
    return f"Pearson:{pearsonr(system_human, system_other)[0]}", f"Spearman:{spearmanr(system_human, system_other)[0]}"


def summary_level_correlation(human, other, metric=None):
    summary_human, summary_other = {}, {}
    for system_name in human:
        for doc_id in human[system_name]:
            if doc_id not in summary_human:
                summary_human[doc_id] = []
                summary_other[doc_id] = []
            summary_human[doc_id].append(human[system_name][doc_id])
            summary_other[doc_id].append(other[system_name][doc_id][metric] if metric else other[system_name][doc_id])
    corrs_pear, corrs_spear, corrs_kend = [], [], []
    for doc_id in summary_human:
        corrs_pear.append(pearsonr(summary_human[doc_id], summary_other[doc_id])[0])
        corrs_spear.append(spearmanr(summary_human[doc_id], summary_other[doc_id])[0])
    corrs_pear = [corr for corr in corrs_pear if corr is not np.nan]
    corrs_spear = [corr for corr in corrs_spear if corr is not np.nan]
    return f"Pearson:{np.mean(corrs_pear) if len(corrs_pear) else np.nan}", \
           f"Spearman:{np.mean(corrs_spear) if len(corrs_spear) else np.nan}"


def read_other_metircs(data_file="../data/data_other_metrics.json"):
    with open(data_file, 'r') as f:
        data_other_metrics = json.load(f)
    # negate some metircs
    for key in data_other_metrics:
        for metric in data_other_metrics[key]:
            if metric == "dae(token_err)":
                continue
            data_other_metrics[key][metric] = -data_other_metrics[key][metric]
    return data_other_metrics


def read_exteval_metrics(data_file="../data/data_exteval.json"):
    with open(data_file, 'r') as f:
        data_exteval = json.load(f)
    return data_exteval


def get_human_scores(data_file="../data/data_finalized.json"):
    with open(data_file, 'r') as f:
        data = json.load(f)
    human_scores = {}
    for key in data:
        human_scores[key] = {"incorrect_coref": 0, "incomplete_coref": 0, "incorrect_discourse": 0,
                             "incomplete_discourse": 0, "misleading": 0, "overall": 0}
        overall = 0
        if data[key]["misleading1"] == data[key]["misleading2"] == "yes":
            human_scores[key]["misleading"] = 1
            overall += 1

        if data[key]["incorrect_coref"] == "yes":
            human_scores[key]["incorrect_coref"] = 1
            overall += 1

        if data[key]["incomplete_coref"] == "yes":
            human_scores[key]["incomplete_coref"] = 1
            overall += 1

        if data[key]["incorrect_discourse"] == "yes":
            human_scores[key]["incorrect_discourse"] = 1
            overall += 1

        if data[key]["incomplete_discourse"] == "yes":
            human_scores[key]["incomplete_discourse"] = 1
            overall += 1

        # gold_scores["joint"].append(1 if joint else 0)
        human_scores[key]["overall"] = overall
    return human_scores


def system_summary_correlations():
    all_metrics = read_other_metircs()
    all_metrics.update(read_exteval_metrics())
    human_scores = get_human_scores()

    gold_scores = {}
    scores = {}
    for key in human_scores:
        example_id, model = key.split('_', 1)

        for metric in human_scores[key]:
            if metric not in gold_scores:
                gold_scores[metric] = {}
            if model not in gold_scores[metric]:
                gold_scores[metric][model] = {}
            gold_scores[metric][model][example_id] = human_scores[key][metric]

        for metric in all_metrics[key]:
            if metric not in scores:
                scores[metric] = {}
            if model not in scores[metric]:
                scores[metric][model] = {}
            scores[metric][model][example_id] = all_metrics[key][metric]

    print("====System level====")
    for gold in gold_scores:
        for metric in scores:
            print(gold, metric, system_level_correlation(gold_scores[gold], scores[metric], metric=None))
    print("====Summary level====")
    for gold in gold_scores:
        for metric in scores:
            print(gold, metric, summary_level_correlation(gold_scores[gold], scores[metric], metric=None))


def correlations_for_all_examples():
    all_metrics = read_other_metircs()
    all_metrics.update(read_exteval_metrics())
    human_scores = get_human_scores()

    gold_scores = {}
    scores = {}
    for key in human_scores:
        for metric in human_scores[key]:
            if metric not in gold_scores:
                gold_scores[metric] = []
            gold_scores[metric].append(human_scores[key][metric])

        for metric in all_metrics[key]:
            if metric not in scores:
                scores[metric] = []
            scores[metric].append(all_metrics[key][metric])

    for gold in gold_scores:
        for metric in scores:
            print(gold, metric, f"Pearson:{pearsonr(gold_scores[gold], scores[metric])[0]}",
                  f"Spearman:{spearmanr(gold_scores[gold], scores[metric])[0]}")


if __name__ == '__main__':
    print("=====Correlations for all examples=====")
    correlations_for_all_examples()
    print("=====System and Summary level correlations=====")
    system_summary_correlations()