import numpy as np
from itertools import combinations
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

shuffle_type_map = {
    'random': 'Random Shuffle',
    'type': 'Cell-type Shuffle',
    'axon': 'Axon-biased Shuffle',
    'continuous': 'Axon-biased Shuffle'
}

shuffle_type_color = {
    'random': 'tab:blue',
    'type': 'tab:orange',
    'axon': 'tab:green',
    'continuous': 'tab:green'
}

facecolor_dict = {
    'random': plt.cm.tab20.colors[1],
    'type': plt.cm.tab20.colors[3],
    'axon': plt.cm.tab20.colors[5],
    'continuous': plt.cm.tab20.colors[5]
}

def get_counts(dists, unique=False, dendrite=False, basal=False):
    '''
    If dendrite is true, then this counts how many dendrites have a repeated sequence (len >= 3).
    If unique is true, this this requires that the sequence contain unique elements.
    '''
    counts = defaultdict(int)
    for cell_id in dists:
        for branch_id in dists[cell_id]:
            if dendrite:
                lcs_scores = [lcs_score for lcs_score, _ in dists[cell_id][branch_id]]
                i = np.argmax(lcs_scores)
                if lcs_scores[i] >= 3:
                    branch1 = dists[cell_id][branch_id][i][1][0]
                    branch2 = dists[cell_id][branch_id][i][1][1]
                    if basal and (branch1.apical or branch2.apical):
                        continue
                    seq = branch1.get_sequence()
                    if unique and len(set(seq)) != len(seq):
                        continue
                    counts[(cell_id, branch_id)] += 1
            else:
                for lcs_score, (branch1, branch2) in dists[cell_id][branch_id]:
                    if lcs_score >= 3:
                        if basal and (branch1.apical or branch2.apical):
                            continue
                        seq = tuple(branch1.get_sequence())
                        if unique and len(set(seq)) != len(seq):
                            continue
                        counts[seq] += 1
    return counts

def get_kcounts(dists, k, unique=False, dendrite=False, basal=False):
    '''
    If dendrite is true, then this counts how many dendrites have a repeated sequence (len >= 3).
    If unique is true, this this requires that the sequence contain unique elements.
    '''
    counts = defaultdict(int)
    for cell_id in dists:
        for branch_id in dists[cell_id]:
            if dendrite:
                lcs_scores = [lcs_score for lcs_score, _ in dists[cell_id][branch_id]]
                i = np.argmax(lcs_scores)
                if lcs_scores[i] >= 3:
                    branch1 = dists[cell_id][branch_id][i][1][0]
                    branch2 = dists[cell_id][branch_id][i][1][1]
                    if basal and (branch1.apical or branch2.apical):
                        continue
                    seq = branch1.get_sequence()
                    if unique and len(set(seq)) != len(seq):
                        continue
                    counts[(cell_id, branch_id)] += 1
            else:
                for lcs_score, (branch1, branch2) in dists[cell_id][branch_id]:
                    if lcs_score >= 3:
                        if basal and (branch1.apical or branch2.apical):
                            continue
                        seq = tuple(branch1.get_sequence())
                        if unique and len(set(seq)) != len(seq):
                            continue
                        for pseq in combinations(seq, k):
                            counts[pseq] += 1
    return counts


def count_histogram(real_count, all_shuffle_count, bins=5, zscore=None, dendrite=False, stats=False):
    fig = plt.figure(figsize=(20,6))
    plt.axvline(real_count, color='k', linestyle='dashed', linewidth=2, label='Real Data')
    if stats:
        print('Real Data:')
        print('Count:', real_count)
    for shuffle_type in all_shuffle_count:
        plt.hist(all_shuffle_count[shuffle_type], bins=bins, facecolor=facecolor_dict[shuffle_type],
                 edgecolor='black', label=shuffle_type_map[shuffle_type], lw=2)

        if stats:
            print('Shuffle Type:', shuffle_type)
            print('Mean:', np.mean(all_shuffle_count[shuffle_type]))
            print('Median:', np.median(all_shuffle_count[shuffle_type]))
            print('Std:', np.std(all_shuffle_count[shuffle_type]))
            print('Z-score:', (real_count-np.mean(all_shuffle_count[shuffle_type]))/np.std(all_shuffle_count[shuffle_type]))

    if dendrite:
        plt.xlabel('Dendrites with Repeated Sequences', fontsize=28)
    else:
        plt.xlabel('Number of Repeated Sequences', fontsize=28)
    plt.xticks(fontsize=28)
    ax = plt.gca()
    ax.tick_params(axis="x", which="minor", length=7, width=2)
    ax.tick_params(axis="x", which="major", length=10, width=4)
    ax.tick_params(axis='y', which='both', left=False, labelleft=False)  # Hide y ticks and labels
    ax.spines['top'].set_linewidth(4)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_linewidth(4)
    if zscore in all_shuffle_count:
        me = np.mean(all_shuffle_count[zscore])
        st = np.std(all_shuffle_count[zscore])

        def count2Zscore(x):
            return (x-me)/st

        def Zscore2count(x):
            return x*st + me
        
        secax = ax.secondary_xaxis('top', functions=(count2Zscore, Zscore2count))
        secax.set_xlabel('Z-score', fontsize=28)
        secax.tick_params(axis="x", which="both", labelsize=28)
        secax.tick_params(axis="x", which="minor", length=7, width=2)
        secax.tick_params(axis="x", which="major", length=10, width=4)

        return fig, (ax, secax)

    return fig, ax

def axonlength_pdf(real_lengths, all_shuffle_lengths, bins, fig=None, ax=None, xlim=None, stats=False):
    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6,3))
        ax.set_xlabel(r'Total Axon Length (mm)', fontsize=36)
    
    ax.hist(real_lengths, bins=bins, histtype='stepfilled', facecolor='lightgrey', edgecolor='k', linewidth=4)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_xscale('log')
    if xlim:
        ax.set_xlim(xlim)
    ax.tick_params(axis='x', labelsize=36)
    ax.tick_params(axis="x", which="minor", length=7, width=2)
    ax.tick_params(axis="x", which="major", length=10, width=4)
    if stats:
        print('Real Data:')
        print('Mean:', np.mean(real_lengths))
        print('Median:', np.median(real_lengths))
        print('Std:', np.std(real_lengths))

    ax2 = ax.twinx()
    ymax = ax.get_ylim()[1]
    shuffle_types = ['continuous', 'axon', 'type', 'random']
    num_trials = 1
    for shuffle_type in shuffle_types:
        if shuffle_type not in all_shuffle_lengths:
            continue
        if type(all_shuffle_lengths[shuffle_type][0]) == list:
            num_trials = len(all_shuffle_lengths[shuffle_type])
            data = [length for trial_lengths in all_shuffle_lengths[shuffle_type] for length in trial_lengths]
        else:
            data = all_shuffle_lengths[shuffle_type]
        ax2.hist(data, bins=bins, histtype='stepfilled', alpha=0.5, label=shuffle_type_map[shuffle_type], 
                facecolor=facecolor_dict[shuffle_type], edgecolor=shuffle_type_color[shuffle_type], linewidth=4)
        if stats:
            print('Shuffle Type:', shuffle_type)
            print('Mean:', np.mean(data))
            print('Median:', np.median(data))
            print('Std:', np.std(data))
    
    
    ax2.spines['left'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.yaxis.set_visible(False)
    ax2.set_yticks([])
    ax2.set_yticklabels([])
    ax2.set_xscale('log')
    if xlim:
        ax2.set_xlim(xlim)
    ax2.tick_params(axis='x', labelsize=36)
    ax2.tick_params(axis="x", which="minor", length=7, width=2)
    ax2.tick_params(axis="x", which="major", length=10, width=4)
    ax2.set_ylim(0, num_trials*ymax)
    return fig, (ax, ax2)

def axonlength_cdf(real_lengths, all_shuffle_lengths, fig=None, ax=None, xlim=None, stats=False):
    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6,3))
    xs = np.sort(real_lengths)
    ys = np.arange(1, len(xs)+1)/len(xs)

    ax.plot(xs, ys*100, color='k', lw=4)
    if stats:
        print('Real Data:')
        print('Mean:', np.mean(xs))
        print('Median:', np.median(xs))
        print('Std:', np.std(xs))
    for shuffle_type in all_shuffle_lengths:
        if type(all_shuffle_lengths[shuffle_type][0]) == list:
            data = [length for trial_lengths in all_shuffle_lengths[shuffle_type] for length in trial_lengths]
            for trial_lengths in all_shuffle_lengths[shuffle_type]:
                xs = np.sort(trial_lengths)
                ys = np.arange(1, len(xs)+1)/len(xs)
                ax.plot(xs, ys*100, color=shuffle_type_color[shuffle_type], lw=1, alpha=0.5)
            xs = np.sort(data)
            ys = np.arange(1, len(xs)+1)/len(xs)
            ax.plot(xs, ys*100, label=shuffle_type_map[shuffle_type], color=shuffle_type_color[shuffle_type], lw=4)
        else:
            xs = np.sort(all_shuffle_lengths[shuffle_type])
            ys = np.arange(1, len(xs)+1)/len(xs)
            ax.plot(xs, ys*100, label=shuffle_type_map[shuffle_type], color=shuffle_type_color[shuffle_type], lw=4)
        if stats:
            print('Shuffle Type:', shuffle_type)
            print('Mean:', np.mean(xs))
            print('Median:', np.median(xs))
            print('Std:', np.std(xs))

    ax.set_xscale('log')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(4)
    ax.spines['bottom'].set_linewidth(4)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_yticks([25,50,75,100])
    ax.set_yticklabels([])
    ax.set_xscale('log')
    if xlim:
        ax.set_xlim(xlim)
    ax.set_xlabel(r'Total Axon Length (mm)', fontsize=36)
    ax.tick_params(axis='both', labelsize=36)
    ax.tick_params(axis="x", which="minor", length=7, width=2)
    ax.tick_params(axis="x", which="major", length=10, width=4)
    ax.tick_params(axis="y", which="major", length=7, width=4)
    return fig, ax

def axonlength_distribution(real_lengths, all_shuffle_lengths, bins, histogram_types=None, xlim=None, stats=False):
    fig, axs = plt.subplots(2, 1, figsize=(9,12))
    if histogram_types is None:
        axonlength_pdf(real_lengths, all_shuffle_lengths, bins, fig=fig, ax=axs[0], xlim=xlim, stats=stats)
    else:
        axonlength_pdf(real_lengths, {t: all_shuffle_lengths[t] for t in histogram_types}, bins, fig=fig, ax=axs[0], xlim=xlim, stats=stats)

    axonlength_cdf(real_lengths, all_shuffle_lengths, fig=fig, ax=axs[1], xlim=xlim)
    return fig, axs

def dists2list(dists):
    dlist = []
    for seq, dists in dists.items():
        dlist.append(np.mean(dists)/len(seq))
    return dlist

def get_dists(dists, as_list=True, basal=False):
    subseq_dists = defaultdict(list)
    for cell_id in dists:
        for branch_id in dists[cell_id]:
            for lcs_score, (branch1, branch2) in dists[cell_id][branch_id]:
                if lcs_score >= 3:
                    if basal and (branch1.apical or branch2.apical):
                        continue
                    subseq_dists[tuple(branch1.get_sequence())].extend([branch1.distance(), branch2.distance()])

    if as_list:
        return dists2list(subseq_dists)
    return subseq_dists

def sequencedist_pdf(real_dists, all_shuffle_dists, bins, fig=None, ax=None, xlim=None, stats=False):
    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(9,6))
        ax.set_xlabel(r'Inter-synapse distance ($\mu$m/synapse)', fontsize=36)

    ax.hist(real_dists, bins=bins, histtype='stepfilled', facecolor='lightgrey', edgecolor='k', linewidth=4)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_xscale('log')
    if xlim:
        ax.set_xlim(xlim)
    ax.tick_params(axis='x', labelsize=36)
    ax.tick_params(axis="x", which="minor", length=7, width=2)
    ax.tick_params(axis="x", which="major", length=10, width=4)
    if stats:
        print('Real Data:')
        print('Mean:', np.mean(real_dists))
        print('Median:', np.median(real_dists))
        print('Std:', np.std(real_dists))

    ax2 = ax.twinx()
    ymax = ax.get_ylim()[1]
    shuffle_types = ['continuous', 'axon', 'type', 'random']
    num_trials = 1
    for shuffle_type in shuffle_types:
        if shuffle_type not in all_shuffle_dists:
            continue
        if type(all_shuffle_dists[shuffle_type][0]) == list:
            num_trials = len(all_shuffle_dists[shuffle_type])
            data = [length for trial_lengths in all_shuffle_dists[shuffle_type] for length in trial_lengths]
        else:
            data = all_shuffle_dists[shuffle_type]
        ax2.hist(data, bins=bins, histtype='stepfilled', alpha=0.5, label=shuffle_type_map[shuffle_type], 
                facecolor=facecolor_dict[shuffle_type], edgecolor=shuffle_type_color[shuffle_type], linewidth=4)
        if stats:
            print('Shuffle Type:', shuffle_type)
            print('Mean:', np.mean(data))
            print('Median:', np.median(data))
            print('Std:', np.std(data))

    ax2.spines['left'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.yaxis.set_visible(False)
    ax2.set_yticks([])
    ax2.set_yticklabels([])
    ax2.set_xscale('log')
    if xlim:
        ax2.set_xlim(xlim)
    ax2.tick_params(axis='x', labelsize=36)
    ax2.tick_params(axis="x", which="minor", length=7, width=2)
    ax2.tick_params(axis="x", which="major", length=10, width=4)
    ax2.set_ylim(0, num_trials*ymax)
    return fig, (ax, ax2)

def sequencedist_cdf(real_dists, all_shuffle_dists, fig=None, ax=None, xlim=None, stats=False):
    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(9,6))
    
    xs = np.sort(real_dists)
    ys = np.arange(1, len(xs)+1)/len(xs)
    ax.plot(xs, ys*100, lw=4, color='k')
    if stats:
        print('Real Data:')
        print('Mean:', np.mean(xs))
        print('Median:', np.median(xs))
        print('Std:', np.std(xs))
    for shuffle_type in all_shuffle_dists:
        if type(all_shuffle_dists[shuffle_type][0]) == list:
            data = [length for trial_lengths in all_shuffle_dists[shuffle_type] for length in trial_lengths]
            for trial_lengths in all_shuffle_dists[shuffle_type]:
                xs = np.sort(trial_lengths)
                ys = np.arange(1, len(xs)+1)/len(xs)
                ax.plot(xs, ys*100, color=shuffle_type_color[shuffle_type], lw=1, alpha=0.5)
            xs = np.sort(data)
            ys = np.arange(1, len(xs)+1)/len(xs)
            ax.plot(xs, ys*100, label=shuffle_type_map[shuffle_type], color=shuffle_type_color[shuffle_type], lw=4)
        else:
            xs = np.sort(all_shuffle_dists[shuffle_type])
            ys = np.arange(1, len(xs)+1)/len(xs)
            ax.plot(xs, ys*100, label=shuffle_type_map[shuffle_type], color=shuffle_type_color[shuffle_type], lw=4)
        if stats:
            print('Shuffle Type:', shuffle_type)
            print('Mean:', np.mean(xs))
            print('Median:', np.median(xs))
            print('Std:', np.std(xs))

    ax.set_xscale('log')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(4)
    ax.spines['left'].set_linewidth(4)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_yticks([25,50,75,100])
    ax.set_yticklabels([])
    ax.set_xscale('log')
    if xlim:
        ax.set_xlim(xlim)
    ax.set_xlabel(r'Inter-synapse distance ($\mu$m/synapse)', fontsize=36)
    ax.tick_params(axis='both', labelsize=36)
    ax.tick_params(axis="x", which="minor", length=7, width=2)
    ax.tick_params(axis="x", which="major", length=10, width=4)
    ax.tick_params(axis="y", which="major", length=7, width=4)
    return fig, ax

def sequencedist_distribution(real_dists, all_shuffle_dists, bins, histogram_types=None, xlim=[2,110], stats=False):
    fig, axs = plt.subplots(2, 1, figsize=(9,12))
    if histogram_types is None:
        sequencedist_pdf(real_dists, all_shuffle_dists, bins, fig=fig, ax=axs[0], xlim=xlim, stats=stats)
    else:
        sequencedist_pdf(real_dists, {t: all_shuffle_dists[t] for t in histogram_types}, bins, fig=fig, ax=axs[0], xlim=xlim, stats=stats)
    
    sequencedist_cdf(real_dists, all_shuffle_dists, fig=fig, ax=axs[1], xlim=xlim)
    return fig, axs