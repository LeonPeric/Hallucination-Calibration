from copy import deepcopy
import numpy as np
import pandas as pd


def get_adaptive_binning(g_series, num_bins=None):
    max_bins = int(np.floor(1 / g_series.iloc[0]))

    if num_bins is None:
        # num all elements divided by the top one
        num_bins = max_bins
    else:
        assert num_bins <= max_bins

    print('Adaptive binning with', num_bins, 'bins')

    prob_per_bin = 1 / num_bins
    bins = []
    cur_bin_items = []
    cur_bin_probas = []
    cur_bin_prob_total = 0.

    for item, generated_prob in zip(list(g_series.index), g_series.to_list()):
        if (cur_bin_prob_total + generated_prob) > prob_per_bin:
            bins.append((deepcopy(cur_bin_items), deepcopy(cur_bin_probas), cur_bin_prob_total))
            cur_bin_prob_total = 0.
            cur_bin_items = []
            cur_bin_probas = []

        cur_bin_items.append(item)
        cur_bin_probas.append(generated_prob)
        cur_bin_prob_total += generated_prob

    if len(cur_bin_items) > 0:
        bins.append((deepcopy(cur_bin_items), deepcopy(cur_bin_probas), cur_bin_prob_total))

    return bins


# miscalibration(df['generated_distribution'], df['true_distribution'], num_bins=3)
def miscalibration(g_series, p_series, num_bins=None):
    bins = get_adaptive_binning(g_series, num_bins=num_bins)
    miscalibration = 0.
    p_dict = p_series.to_dict()
    for (food_list, prob_list, g_proba) in bins:
        B_len = len(food_list)
        # for whole B
        p_proba = np.sum([p_dict[food] for food in food_list])
        print('bin with g_proba / p_proba', g_proba, p_proba)
        p_coarsened = p_proba / B_len
        
        miscalibration += (1/2) * np.sum(np.abs([(p_coarsened - g_prob) for g_prob in prob_list]))

    return miscalibration