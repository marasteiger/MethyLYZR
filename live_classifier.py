#!/usr/bin/env python3

"""
This script contains functions for methylation analysis and prediction.

Functions:
- FileChangeHandler: Manages file change events in a directory.
- read_bam_files: Streams BAM files into a processing queue.
- calc_denominator: Computes the denominator for Bayesian inference.
- predict_from_fingerprint: Estimates methylation class probabilities for new samples.
- methylation_reader: Processes methylation data from a queue for analysis.
- process_methylation_data: Analyzes methylation data and updates results.
- generate_report: Creates a summary report of the methylation analysis.
- monitor_directory: Watches a directory for new BAM files and processes them.
- aggregate_results: Combines results from multiple analysis threads.
- filter_by_quality: Filters methylation reads based on quality scores.
- save_results: Saves the methylation analysis results to a file.

Note: This script requires the following dependencies: argparse, os, pathlib, time, collections, ctypes, multiprocessing, numpy, pandas, pysam, arrow, natsort, watchdog, minknow_api.
"""

import argparse
import os
import sys

import pathlib
import time
from collections import defaultdict
from ctypes import c_bool
from multiprocessing import Manager, Process, Queue, Value
import signal
from functools import partial

import numpy as np
import pandas as pd
import pyarrow  # pylint: disable=unused-import # Required for Feather file format
import pysam
import arrow
from natsort import natsorted
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
import minknow_api


def signal_handler(signal, frame, stop_analysis): # pylint: disable=W0621 W0613
    """
    Handles the signal and sets the stop_analysis flag to True.

    Args:
        signal (int): The signal received.
        frame (frame): The current stack frame.
        stop_analysis (multiprocessing.Value): A multiprocessing.Value object used to communicate the stop flag.

    Returns:
        None
    """
    with stop_analysis.get_lock():
        stop_analysis.value = True # catch CTRL+C Interrupt to end program execution


def register_handler(stop_analysis):
    """
    Register a signal handler for SIGINT (Ctrl+C) with a stop_analysis parameter.

    Args:
        stop_analysis (bool): A flag indicating whether to stop the analysis.

    Returns:
        None
    """
    global signal_handler # pylint: disable=W0603
    signal_handler = partial(signal_handler, stop_analysis=stop_analysis)
    signal.signal(signal.SIGINT, signal_handler)


class FileChangeHandler(FileSystemEventHandler):
    """
    A class that handles file change events in a directory.
    Attributes:
        file_queue (Queue): A queue to store the paths of the changed files.
    """

    def __init__(self, file_queue, bams_amount, bams_directory_created, filter_param):
        """
        Initializes a FileChangeHandler object.
        Args:
            file_queue (Queue): A queue to store the paths of the changed files.
        """
        self.file_queue = file_queue
        self.bams_amount = bams_amount
        self.bams_directory_created = bams_directory_created
        self.filter_param = filter_param

    def on_moved(self, event):
        """
        Handles the event when a file is moved.
        If the moved file has the extension ".bam.ctmp", the path of the file (without the extension)
        is added to the file_queue.
        Args:
            event (FileSystemEvent): The event object representing the file move event.
        """
        if not event.is_directory and event.src_path.endswith(".bam.ctmp"):
            self.file_queue.put(os.path.splitext(event.src_path)[0])

    def on_created(self, event):
        """
        Handles the event when a file is created.
        If the created file has the extension ".bam", the path of the file is added to the file_queue.
        Args:
            event (FileSystemEvent): The event object representing the file creation event.
        """
        if not event.is_directory and event.src_path.endswith(".bam"):
            if self.filter_param:
                if self.filter_param in event.src_path:
                    self.file_queue.put(event.src_path)
                    with self.bams_amount.get_lock():
                        self.bams_amount.value += 1
            else:
                self.file_queue.put(event.src_path)
                with self.bams_amount.get_lock():
                    self.bams_amount.value += 1
        if event.is_directory and event.src_path.contains("bam_pass"):
            with self.bams_directory_created.get_lock():
                self.bams_directory_created.value = True


def read_bam_files(
    path,
    recursive,
    file_queue,
    bams_count,
    files_finished
):
    """
    Reads BAM files from the specified path and adds them to a file queue.

    Args:
        path (str): The path to the directory containing the BAM files.
        recursive (bool): Flag indicating whether to search for BAM files recursively in subdirectories.
        file_queue (Queue): The queue to which the BAM file paths will be added.
        bams_count (Value): The shared value representing the count of BAM files.
        files_finished (Value): The shared value indicating whether all files have been processed.
    """
    bams_path = pathlib.Path(path)
    if recursive:
        for file in natsorted(bams_path.rglob("*.bam")):
            file_queue.put(str(file))
            with bams_count.get_lock():
                bams_count.value += 1
    else:
        for file in natsorted(bams_path.glob("*.bam")):
            file_queue.put(str(file))
            with bams_count.get_lock():
                bams_count.value += 1
    time.sleep(1)
    with files_finished.get_lock():
        files_finished.value = True


def calc_denominator(log_likelihoods, prior):
    """
    Function to calculate the denominator P(X) for Bayes theorem.

    Parameters:
    log_likelihoods (numpy.ndarray): Pre-calculated log-likelihoods, i.e., P(x|C).
    prior (float): Pre-defined prior class probability, i.e., P(C).

    Returns:
    float: The logarithm of the denominator P(X).
    """
    # using log-sum-exp trick to avoid numerical underflow
    l_k = np.log(prior) + log_likelihoods
    l_max = np.max(
        l_k
    )  # choosing max(l_k) as constant l_max, such that largest term exp(0)=1
    log_denom = l_max + np.log(np.sum(np.exp(l_k - l_max)))
    return log_denom


def predict_from_fingerprint(
    new_x,
    feature_ids,
    centroids,
    weights,
    noise,
    prior,
    read_weights,
    basecount=300,
    reweighting=True,
    basecount_norm=True,
):
    """
    function to predict class probabilities for a new methlyation profile

    Args:
        new_x (_type_): binary vector with observation of methylation events
        feature_ids (_type_): probe identifiers of the observed methylation events
        centroids (_type_): centroid matrix of dimension (no. probes) x (no. classes)
        weights (_type_): weight matrix of dimension (no. probes) x (no. classes)
        noise (_type_): noise value for each call
        prior (_type_): assumed prior class probability
        read_weights (_type_): weight value for each call
        basecount (int, optional): Basecount that is normalized to. Defaults to 300.
        reweighting (bool, optional): reweighting = FALSE should only be used with constant, class independent Weight matrix -> reweighting = TRUE if a class-dependent weight matrix is used. Defaults to True.
        basecount_norm (bool, optional): bool whether normalizing to basecount. Defaults to True.

    Raises:
        ValueError: _description_

    Returns:
        "posterior": class_posteriors
        "log_likelihoods": log_likelihoods_weighted
        "epic_ids": feature_ids
        "read_weights": read_weights
    """

    if (feature_ids is None) | (len(set(feature_ids) & set(centroids.index)) == 0):
        raise ValueError("Probe names in newX do not agree with reference.")

    # Convert newX to a NumPy array if it's a pd Series
    if isinstance(new_x, pd.Series):
        new_x = new_x.values

    p_1 = centroids.loc[
        feature_ids, prior.index
    ].to_numpy()  # subset centroids by selected features and order classes
    p_1 = p_1 - p_1 * 2 * noise[:, None] + noise[:, None]  # add noise terms

    p_0 = np.log(1 - p_1)  # log complement probabilities
    p_1 = np.log(p_1)  # log probabilities

    # weight matrix for probes x classes
    if weights is None:
        # set everything to 1
        weights = np.ones((p_1.shape))
        # include read weights
        if len(read_weights) > 0:
            weights = weights * read_weights[:, None]
    else:
        # subset weight matrix, order classes, and include read weights
        weights = (
            read_weights[:, None]
            * weights.loc[feature_ids, prior.index].to_numpy()
            * np.mean(1 / read_weights)
        )

    # calculating the likelihoods of all classes C_j, i.e, P(x|C_j)
    exp_w = np.exp(-weights)
    log_likelihoods_weighted = np.sum(
        p_1 * new_x[:, None] * exp_w + p_0 * (1 - new_x[:, None]) * exp_w, axis=0
    )  # likelihoods for each class

    log_denominator = calc_denominator(
        log_likelihoods=log_likelihoods_weighted, prior=prior
    )  # calculating denominator P(X) -> one value

    if reweighting and not basecount_norm:
        # if using a class-dependent weight matrix, we need to recalculate the denominator
        likelihood_mat = np.apply_along_axis(
            lambda x: np.sum(
                p_1 * new_x[:, None] * np.exp(-x)[:, None]
                + p_0 * (1 - new_x[:, None]) * np.exp(-x)[:, None],
                axis=0,
            ),
            0,
            weights,
        )
        likelihood_mat = likelihood_mat.transpose()
        log_denominator = np.apply_along_axis(
            lambda x: calc_denominator(log_likelihoods=x, prior=prior),
            1,
            likelihood_mat,
        )
    elif not reweighting and basecount_norm:
        # if using basecount normalization, we rescale the likelihoods and recalculate the denominator
        log_likelihoods_weighted = (
            log_likelihoods_weighted / np.sum(read_weights) * basecount
        )
        log_denominator = calc_denominator(
            log_likelihoods=log_likelihoods_weighted, prior=prior
        )
    elif reweighting and basecount_norm:
        # both of the above combined
        likelihood_mat = np.apply_along_axis(
            lambda x: np.sum(
                p_1 * new_x[:, None] * np.exp(-x)[:, None]
                + p_0 * (1 - new_x[:, None]) * np.exp(-x)[:, None],
                axis=0,
            ),
            0,
            weights,
        )
        likelihood_mat = likelihood_mat.transpose()

        log_likelihood_mat_weighted = likelihood_mat / \
            np.sum(read_weights) * basecount
        log_likelihoods_weighted = (
            log_likelihoods_weighted / np.sum(read_weights) * basecount
        )
        log_denominator = np.apply_along_axis(
            lambda x: calc_denominator(log_likelihoods=x, prior=prior),
            1,
            log_likelihood_mat_weighted,
        )

    class_posteriors = np.exp(
        np.log(prior) + log_likelihoods_weighted - log_denominator
    )  # calculating posterior probabilities by Bayes' Theorem in log space, then exponentiating

    return {
        "posterior": class_posteriors,
        "log_likelihoods": log_likelihoods_weighted,
        "epic_ids": feature_ids,
        "read_weights": read_weights,
    }


def process_cpgs(
    modified_bases,
    aligned_pairs,
    te,
    methylation_dataset,
    methylation_list,
    methylation_count,
    cpg_count,
    reverse=False
):
    """
    Process CpGs from the given data and update the methylation list, methylation count, and CpG count.

    Args:
        modified_bases (dict): Dictionary containing modified bases data.
        aligned_pairs (pd.DataFrame): DataFrame containing aligned pairs data.
        te (pd.DataFrame): DataFrame containing TE data.
        methylation_dataset (list): List containing methylation dataset information.
        methylation_list (list): List to store processed CpG data.
        methylation_count (multiprocessing.Value): Shared value to track the count of processed methylation.
        cpg_count (multiprocessing.Value): Shared value to track the count of processed CpGs.
        reverse (bool, optional): Flag indicating whether to process reverse CpGs. Defaults to False.
    """
    if reverse:
        tz = pd.DataFrame(modified_bases[("C", 1, "m")], columns=[
                          "query_pos", "methylation"],)
        res = (aligned_pairs.join(tz.set_index("query_pos"),
               on="query_pos").dropna().reset_index(drop=True))
        cpgs = te.join(res.set_index("ref_pos"), on="end").dropna()[
            ["epic_id", "methylation"]]
    else:
        tz = pd.DataFrame(modified_bases[("C", 0, "m")], columns=[
                          "query_pos", "methylation"],)
        res = (aligned_pairs.join(tz.set_index("query_pos"),
               on="query_pos").dropna().reset_index(drop=True))
        cpgs = te.join(res.set_index("ref_pos"), on="start").dropna()[
            ["epic_id", "methylation"]]

    if len(cpgs.index) != 0:
        cpgs["methylation"] = cpgs["methylation"] / 255
        cpgs["scores_per_read"] = [len(cpgs.index)] * len(cpgs.index)
        cpgs["binary_methylation"] = (cpgs["methylation"] >= 0.8).astype(int)
        cpgs["read_id"] = [methylation_dataset[7]] * len(cpgs.index)
        cpgs["start_time"] = [methylation_dataset[4]] * len(cpgs.index)
        cpgs["run_id"] = [methylation_dataset[5]] * len(cpgs.index)
        cpgs["QS"] = [methylation_dataset[8]] * len(cpgs.index)
        cpgs["read_length"] = [methylation_dataset[9]] * len(cpgs.index)
        cpgs["map_qs"] = [methylation_dataset[10]] * len(cpgs.index)

        for _idx, cpg in cpgs.iterrows():
            methylation_list.append(cpg.to_list())
        with methylation_count.get_lock():
            methylation_count.value += len(cpgs.index)
        with cpg_count.get_lock():
            cpg_count.value += len(cpgs.index)


def methylation_reader(
    methylation_queue,
    methylation_list,
    methylation_count,
    sites_set,
    finished,
    cpg_count,
    methylation_read_number_analysed,
):
    """
    Reads methylation data from a queue and processes it.

    Args:
        methylation_queue (Queue): The queue containing the methylation data.
        methylation_list (list): The list to store the processed methylation data.
        methylation_count (Value): The shared value to keep track of the total number of methylation data points.
        sites_set (DataFrame): The DataFrame containing the site information.
        cpg_count (Value): The shared value to keep track of the total number of CpGs.
        methylation_read_number_analysed (Value): The shared value to keep track of the number of methylation reads analyzed.
        finished (Value): The shared value to indicate if the methylation reading process is finished.
    """
    while True:
        data = methylation_queue.get()

        if data is None:
            time.sleep(1)
            with finished.get_lock():
                finished.value += 1
            break
        for _idx, methylation_dataset in data.iterrows():
            with methylation_read_number_analysed.get_lock():
                methylation_read_number_analysed.value += 1

            aligned_pairs = pd.DataFrame(methylation_dataset[1], columns=[
                                         "query_pos", "ref_pos"])
            modified_bases = methylation_dataset[2]

            te = sites_set[sites_set["chromosome"] == methylation_dataset[0]]

            process_cpgs(modified_bases, aligned_pairs, te, methylation_dataset,
                         methylation_list, methylation_count, cpg_count, methylation_dataset[3])


def process_methylation_data(
    methylation_count,
    methylation_list,
    meth_qs,
    methylation_upper_bound,
    methylation_lower_bound,
    min_noise,
    centroids,
    weights,
    class_frequency,
    methylation_read_number_analysed,
    cpg_count
):
    """
    Process methylation data and return the analysis results.

    Args:
        methylation_count (multiprocessing.Value): A shared value representing the methylation count.
        methylation_list (list): A list of methylation data.
        meth_qs (int): The minimum quality score for methylation data.
        methylation_upper_bound (float): The upper bound for methylation likelihood.
        methylation_lower_bound (float): The lower bound for methylation likelihood.
        min_noise (float): The minimum noise value.
        centroids (list): A list of centroids for prediction.
        weights (list): A list of weights for prediction.
        class_frequency (list): A list of class frequencies for prediction.
        methylation_read_number_analysed (multiprocessing.Value): A shared value representing the number of methylation reads analyzed.
        cpg_count (multiprocessing.Value): A shared value representing the count of CpG sites.

    Returns:
        list: A list containing the number of methylation reads analyzed, the count of CpG sites, and the top 5 class labels and their corresponding probabilities.
    """
    with methylation_count.get_lock():
        methylation_count.value = 0

    test_sample = pd.DataFrame(
        list(methylation_list),
        columns=[
            [
                "epic_id",
                "methylation",
                "scores_per_read",
                "binary_methylation",
                "read_id",
                "start_time",
                "run_id",
                "QS",
                "read_length",
                "map_qs"
            ]
        ],
    )
    test_sample = test_sample[(test_sample["QS"] >= meth_qs).squeeze()]

    test_sample = test_sample[
        (
            (test_sample["methylation"] >= methylation_upper_bound)
            | (test_sample["methylation"] <= methylation_lower_bound)
        ).squeeze()
    ]  # filter for likelihood >0.8 or <0.2

    noise = 0.5 - abs(test_sample["methylation"].squeeze().to_numpy() - 0.5)
    noise[noise < min_noise] = min_noise

    read_weights = (
        1 / test_sample["scores_per_read"].squeeze().astype(float).to_numpy())

    binary_vec = test_sample["methylation"].squeeze()
    binary_vec = (binary_vec >= 0.5).astype(
        int)  # create binary methylation vector

    prediction_list = predict_from_fingerprint(
        new_x=binary_vec,
        feature_ids=test_sample["epic_id"].squeeze(),
        centroids=centroids,
        weights=weights,
        noise=noise,
        prior=class_frequency,
        read_weights=read_weights,
    )

    class_posteriors = prediction_list["posterior"].sort_values(
        ascending=False)  # sort by probability

    return [str(time.asctime()), methylation_read_number_analysed.value, cpg_count.value] + class_posteriors[0:5].index.to_list() + class_posteriors[0:5].to_list()


def methylation_calling(
    methylation_list,
    methylation_count,
    min_entries,
    centroids,
    weights,
    class_frequency,
    cpg_count,
    methylation_read_number_analysed,
    methylation_results,
    finished,
    methylation_threads,
    meth_qs,
    methylation_upper_bound=0.8,
    methylation_lower_bound=0.2,
    min_noise=0.05,
):
    """
    Perform methylation calling on a given set of methylation data.

    Args:
        methylation_list (list): List of methylation data.
        methylation_count (Value): Shared value for counting methylation entries.
        min_entries (int): Minimum number of entries required to perform methylation calling.
        centroids (ndarray): Centroids for prediction.
        weights (ndarray): Weights for prediction.
        class_frequency (ndarray): Class frequency for prediction.
        cpg_count (Value): Shared value for counting CpGs.
        methylation_read_number_analysed (Value): Shared value for counting methylation reads.
        methylation_results (list): List to store methylation results.
        finished (Value): Shared value for tracking finished threads.
        methylation_threads (int): Number of methylation threads.
        meth_qs (int): Minimum quality score for methylation data.
        methylation_upper_bound (float, optional): Upper bound for methylation likelihood. Defaults to 0.8.
        methylation_lower_bound (float, optional): Lower bound for methylation likelihood. Defaults to 0.2.
        min_noise (float, optional): Minimum noise value. Defaults to 0.05.

    Returns:
        None
    """
    while True:
        if finished.value == methylation_threads:
            if len(methylation_list) != 0:
                methylation_results.append(process_methylation_data(
                    methylation_count,
                    methylation_list,
                    meth_qs,
                    methylation_upper_bound,
                    methylation_lower_bound,
                    min_noise,
                    centroids,
                    weights,
                    class_frequency,
                    methylation_read_number_analysed,
                    cpg_count
                ))
            break
        if methylation_count.value >= min_entries:
            methylation_results.append(process_methylation_data(
                methylation_count,
                methylation_list,
                meth_qs,
                methylation_upper_bound,
                methylation_lower_bound,
                min_noise,
                centroids,
                weights,
                class_frequency,
                methylation_read_number_analysed,
                cpg_count
            ))


def io_handler(
    file_queue,
    methylation_queue,
    methylation_read_number,
    bams_analysed,
    runs
):
    """
    Handles input/output operations for methylation analysis.

    Args:
        file_queue (Queue): A queue containing the paths of BAM files to be processed.
        methylation_queue (Queue): A queue to store the methylation data.
        methylation_read_number (Value): A shared value to keep track of the number of methylation reads.
        bams_analysed (Value): A shared value to keep track of the number of BAM files analyzed.
        bams_finished (Value): A shared value to indicate the number of BAM files that have finished processing.
        files_finished (Value): A shared value to indicate if all files have finished processing.
        methylation_threads (int): The number of threads for methylation analysis.
        io_threads (int): The number of threads for input/output operations.
    """
    while True:
        bamfile = file_queue.get()
        if bamfile is None:
            break
        alignm = defaultdict(list)

        if not os.path.exists(bamfile + '.bai'):
            pysam.index(bamfile)  # generate a bam index if not already there

        bam = pysam.AlignmentFile(bamfile)  # pylint: disable=E1101

        runs[bam.header.as_dict()['RG'][0]['ID']] = bam.header.as_dict()[
            'RG'][0]['DT']

        for alignment in bam:
            if alignment.is_mapped:
                if not alignment.is_secondary | alignment.is_supplementary:
                    if alignment.mapping_quality >= 10:
                        alignm[alignment.qname].append(
                            (
                                alignment.reference_name,
                                alignment.get_aligned_pairs(),
                                alignment.modified_bases,
                                alignment.is_reverse,
                                alignment.get_tag('st'),
                                alignment.get_tag('RG'),
                                alignment.get_tag('rn'),
                                alignment.qname,
                                alignment.get_tag('qs'),
                                alignment.infer_read_length(),
                                alignment.mapping_quality
                            )
                        )
        if alignm:
            methyl_alignments = pd.concat(
                {k: pd.DataFrame(v) for k, v in alignm.items()})
            methyl_alignments.index.names = ['read_id', 'num_alignments']
            methyl_alignments = methyl_alignments[methyl_alignments[2] != {
            }]

            ba = methyl_alignments.reset_index().groupby(
                'read_id')['num_alignments'].nunique()
            bb = ba[ba == 1]
            bc = methyl_alignments.loc[bb.index]

            ti = np.array_split(bc,4)
            for t in ti:
                methylation_queue.put(t)
                with methylation_read_number.get_lock():
                    methylation_read_number.value += len(t.index)
        bam.close()
        with bams_analysed.get_lock():
            bams_analysed.value += 1


def display_methylation_results(methyl_show):
    """
    Displays the methylation results.

    Args:
        methyl_show (list): A list containing the methylation results.

    Returns:
        None
    """
    print('\r' + f'{methyl_show[0]:50s}', end="")
    print("\nReads: " + str(methyl_show[1]) + " CpGs: " + str(methyl_show[2]))
    print("Class" + '\t' + "Posterior Probability")
    for i in range(3, 8):
        print(str(methyl_show[i]) + '\t' + str(methyl_show[i+5]))


def progress(
    finished,
    methylation_threads,
    bams_analysed,
    bams_amount,
    methylation_read_number_analysed,
    methylation_read_number,
    methylation_results,
    offline,
    minknow_results,
):
    """
    Displays the progress of the methylation analysis.

    Args:
        finished (Value): A shared value indicating the number of finished threads.
        methylation_threads (int): The total number of methylation threads.
        bams_analysed (Value): A shared value indicating the number of BAM files analyzed.
        bams_amount (Value): A shared value indicating the total number of BAM files.
        methylation_read_number_analysed (Value): A shared value indicating the number of methylation reads analyzed.
        methylation_read_number (Value): A shared value indicating the total number of methylation reads.
        methylation_results (list): A list containing the methylation analysis results.

    Returns:
        None
    """
    methyl_show = []
    while True:
        if finished.value >= methylation_threads:
            if len(methylation_results) != 0:
                while methyl_show == methylation_results[-1]:
                    pass
                methyl_show = methylation_results[-1]
                display_methylation_results(methyl_show)
            break

        if len(methylation_results) != 0:
            if methyl_show != methylation_results[-1]:
                methyl_show = methylation_results[-1]
                display_methylation_results(methyl_show)
                if not offline:
                    minknow_results.append(['Reads:', methyl_show[1], 'CpGs:', methyl_show[2], methyl_show[3], methyl_show[8], methyl_show[4],
                                           methyl_show[9], methyl_show[5], methyl_show[10], methyl_show[6], methyl_show[11], methyl_show[7], methyl_show[12]])
                print('\r' + 'Bams: ' + str(bams_analysed.value) + '/' + str(bams_amount.value) + ' Reads: ' +
                      str(methylation_read_number_analysed.value) + '/' + str(methylation_read_number.value), end="", file=sys.stderr)
            else:
                print('\r' + 'Bams: ' + str(bams_analysed.value) + '/' + str(bams_amount.value) + ' Reads: ' +
                      str(methylation_read_number_analysed.value) + '/' + str(methylation_read_number.value), end="", file=sys.stderr)
        else:
            print('\r' + 'Bams: ' + str(bams_analysed.value) + '/' + str(bams_amount.value) + ' Reads: ' +
                  str(methylation_read_number_analysed.value) + '/' + str(methylation_read_number.value), end="", file=sys.stderr)


def main(
    min_entries,
    inputs,
    recursive,
    methylation_threads,
    io_threads,
    reference,
    offline,
    sample,
    output,
    meth_qs,
    dev_key,
    filter_param
):
    """
    Main function for running the MethyLYZR analysis.

    Args:
        min_entries (int): Minimum number of entries required for analysis.
        inputs (str): Path to the input files.
        recursive (bool): Flag indicating whether to search for input files recursively.
        methylation_threads (int): Number of threads for methylation analysis.
        io_threads (int): Number of threads for input/output operations.
        reference (str): Reference genome to use for analysis.
        offline (bool): Flag indicating whether to run the analysis offline.
        sample (str): Name of the sample being analyzed.
        output (str): Path to the output directory.
        meth_qs (int): Methylation quality score threshold.
        dev_key (str): Developer API key for MinK connection.
        filter_param (str): Filter parameter for BAM files.

    Returns:
        None
    """
    app_path = str(pathlib.Path(__file__).parent.resolve())
    output_path = output + "/" + sample + "/"
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

    app_path = str(pathlib.Path(__file__).parent.resolve())
    output_path = output + "/" + sample + "/"
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

    file_queue = Queue()  # Queue for storing file paths for processing
    methylation_queue = Queue()  # Queue for storing methylation data for analysis
    manager = Manager()
    # List for storing methylation data for analysis
    methylation_list = manager.list()
    # Shared value for counting methylation reads
    methylation_count = Value("i", 0, lock=True)
    # List for storing the results of the analysis
    methylation_results = manager.list()
    cpg_count = Value("i", 0, lock=True)  # Shared value for counting CpGs
    # Shared value for counting methylation reads
    methylation_read_number = Value("i", 0, lock=True)
    # Shared value for counting methylation reads analysed
    methylation_read_number_analysed = Value("i", 0, lock=True)
    # Shared value for indicating the completion of the analysis
    finished = Value("i", 0, lock=True)
    # Shared value for stopping the analysis
    stop_analysis = Value(c_bool, False, lock=True)
    # Shared value for BAM directory cerated
    bams_directory_created = Value(c_bool, False, lock=True)
    # Shared value for counting the number of BAM files analysed
    bams_analysed = Value("i", 0, lock=True)
    # Shared value for counting the total number of BAM files
    bams_amount = Value("i", 0, lock=True)
    # Shared value for indicating the completion of the analysis
    files_finished = Value(c_bool, False, lock=True)
    runs = manager.dict()  # Dictionary for storing the results of the analysis
    # List for storing the results of the analysis to be displayed in MinKNOW
    minknow_results = manager.list()

    log_file = open(str(output_path) + str(sample) +
                    ".log", "w", encoding="utf-8")

    print(str(time.asctime()) + " Load alignment data")
    log_file.writelines(str(time.asctime()) + " Load alignment data\n")
    dt = np.dtype([("chromosome", str), ("start", np.int32),
                  ("end", np.int32), ("epic_id", str)])

    if reference == "T2T":
        sites_set = pd.read_csv(
            app_path + "/classifier_EPIC_T2T.bed",
            sep="\t",
            index_col=False,
            names=["chromosome", "start", "end", "epic_id"],
            dtype=dt,
        )
    else:
        sites_set = pd.read_csv(
            app_path + "/classifier_EPIC_HG38.bed",
            sep="\t",
            index_col=False,
            names=["chromosome", "start", "end", "epic_id"],
            dtype=dt,
        )
    sites_set['end'] = sites_set['end']-1

    print(str(time.asctime()) + " Load model data.")
    log_file.writelines(str(time.asctime()) + " Load model data.\n")

    betas_mean = pd.read_feather(
        app_path + "/methylyzr_python/model/centroids/mean.full.betas.feather").set_index("index")
    centroid_cpgs = betas_mean.index.to_list()
    # filter input bed file for centroid CpGs only
    sites_set = sites_set[sites_set['epic_id'].isin(centroid_cpgs)]

    w_relief = pd.read_feather(
        app_path + "/methylyzr_python/model/feature_weights/RELIEFF_allHits_k5centroids_normN.feather").set_index("index")
    class_id_full = pd.read_csv(
        app_path + "/methylyzr_python/data/ClassID.full.CPR.txt", index_col=0, sep="\t")
    cf = class_id_full["Class"].value_counts(normalize=True).sort_index()

    if not offline:
        print(str(time.asctime()) + " Set up File-Watcher and MinKNOW-API.")
        log_file.writelines(str(time.asctime()) +
                            " Set up File-Watcher and  and MinKNOW-API.\n")

        if dev_key:
            minknow_manager = minknow_api.manager.Manager(
                developer_api_token=dev_key)
        else:
            minknow_manager = minknow_api.manager.Manager()

        minknow_connection = None
        for i in minknow_manager.flow_cell_positions():
            fcp = i.connect()
            run_info = fcp.protocol.get_run_info()
            sample_id = run_info.user_info.sample_id.value
            # Check if the sample ID ends with '_MethyLYZR' and the run is in the 'running' state
            if run_info.state == 0 and (sample_id.lower().endswith('_methylyzr') or sample_id.lower() == sample.lower()):
                minknow_connection = fcp
                minknow_output_path = run_info.output_path
                # Convert Windows path to WSL path
                if minknow_output_path.startswith('C:\\'):
                    minknow_output_path = minknow_output_path.replace(
                        'C:\\', '/mnt/c/')
                    minknow_output_path = minknow_output_path.replace(
                        '\\', '/')
                elif minknow_output_path.startswith('D:\\'):
                    minknow_output_path = minknow_output_path.replace(
                        'D:\\', '/mnt/d/')
                    minknow_output_path = minknow_output_path.replace(
                        '\\', '/')
                print(str(time.asctime()) + " Found a MinKNOW-Run with Sample-ID: " + sample + " MinKNOW connection established.")
                log_file.writelines(str(time.asctime()) +
                                    " Found a MinKNOW-Run with Sample-ID: " + sample + " MinKNOW connection established.\n")

        if minknow_connection:
            print(str(time.asctime()) + " Waiting for MinKNOW to start sequencing.")
            log_file.writelines(str(time.asctime()) +
                                " Waiting for MinKNOW to start sequencing.\n")
            for i in minknow_connection.acquisition.watch_current_acquisition_run():
                if i.config_summary.purpose == 2:  # 2 is SEQUENCING
                    if i.state == 1:  # 1 is AQUISITION_RUNNING
                        break
            counter = 0
            while not os.path.exists(minknow_output_path+'/bam_pass'):
                if counter >= 60:  # 5 minutes with a sleep of 5 seconds
                    break
                time.sleep(5)
                counter += 1
            if filter_param: # Wait for barcode subdirectory to be created
                while not os.path.exists(minknow_output_path+'/bam_pass/'+filter_param):
                    if counter >= 60:  # 5 minutes with a sleep of 5 seconds
                        break
                    time.sleep(5)
                    counter += 1

            file_path = minknow_output_path
        else:
            print(str(time.asctime()) + " No MinKNOW connection found.")
            log_file.writelines(str(time.asctime()) +
                                " No MinKNOW connection found.\n")
            file_path = inputs

        register_handler(stop_analysis)

        # Wait for bam_fail to be written but not too long if qscore filter is not set
        time.sleep(10)

        event_handler = FileChangeHandler(
            file_queue, bams_amount, bams_directory_created, filter_param)
        observer = Observer()
        observer.schedule(event_handler, file_path, recursive=recursive)
        observer.start()
        if not minknow_connection:
            print(str(time.asctime()) +
                  " Waiting for BAM directory to be created.")
            log_file.writelines(str(time.asctime()) +
                                " Waiting for BAM directory to be created.\n")
            while True:
                if not file_queue.empty():
                    break
                if bams_directory_created.value:
                    break
    else:
        bam_process = Process(
            target=read_bam_files,
            args=(inputs, recursive, file_queue, bams_amount, files_finished),
        )
        bam_process.start()

    print(str(time.asctime()) + " Generate worker threads and start them.")
    log_file.writelines(str(time.asctime()) +
                        " Generate worker threads and start them.\n")

    # Generate worker threads and start them
    io_processes = []
    for i in range(io_threads):
        io_processes.append(
            Process(
                target=io_handler,
                args=(
                    file_queue,
                    methylation_queue,
                    methylation_read_number,
                    bams_analysed,
                    runs
                ),
            )
        )
        io_processes[i].start()

    methylation_processes = []
    for i in range(methylation_threads):
        methylation_processes.append(
            Process(
                target=methylation_reader,
                args=(
                    methylation_queue,
                    methylation_list,
                    methylation_count,
                    sites_set,
                    finished,
                    cpg_count,
                    methylation_read_number_analysed
                ),
            )
        )
        methylation_processes[i].start()

    methylation_calling_thread = Process(
        target=methylation_calling,
        args=(
            methylation_list,
            methylation_count,
            min_entries,
            betas_mean,
            w_relief,
            cf,
            cpg_count,
            methylation_read_number_analysed,
            methylation_results,
            finished,
            methylation_threads,
            meth_qs
        ),
    )
    methylation_calling_thread.start()

    print(str(time.asctime()) + " Start of analysis.")
    log_file.writelines(str(time.asctime()) + " Start of analysis.\n")

    progress_process = Process(
        target=progress,
        args=(
            finished,
            methylation_threads,
            bams_analysed,
            bams_amount,
            methylation_read_number_analysed,
            methylation_read_number,
            methylation_results,
            offline,
            minknow_results,
        )
    )
    progress_process.start()

    while True:
        if not offline:
            time.sleep(1)

            if minknow_connection:
                if minknow_connection.acquisition.current_status().status == 4 or minknow_connection.acquisition.current_status().status == 0:
                    print('\n' + str(time.asctime()) +
                          " MinKNOW acquisition completed.\n")
                    log_file.writelines(
                        '\n' + str(time.asctime()) + " MinKNOW acquisition completed.\n")
                    break
                if len(minknow_results) > 0:
                    t = minknow_results.pop()
                    minknow_connection.log.send_user_message(
                        identifier='MethyLYZR Results',
                        user_message=f"MethyLYZR Results {t[0]:6} {t[1]:7}  {t[2]:5} {t[3]:6} {t[4]:30} {t[5]:.2%}  {t[6]:30} {t[7]:.2%}  {t[8]:30} {t[9]:.2%}  {t[10]:30} {t[11]:.2%}  {t[12]:30} {t[13]:.2%} ",
                        severity=2
                    )
            if stop_analysis.value:
                print('\n' + str(time.asctime()) + " Analysis stopped.\n")
                log_file.writelines(
                    str(time.asctime()) + " Analysis stopped.\n")
                break

        else:
            if not bam_process.is_alive():
                break

    if not offline:
        observer.stop()
        observer.join()
    else:
        bam_process.join()

    for i in range(io_threads):
        file_queue.put(None)

    for i in range(io_threads):
        io_processes[i].join()

    for i in range(methylation_threads):
        methylation_queue.put(None)

    for i in range(methylation_threads):
        methylation_processes[i].join()

    methylation_calling_thread.join()

    print("\n" + str(time.asctime()) + " Write classification results to file.")
    log_file.writelines(str(time.asctime()) +
                        " Write classification results to file.\n")

    pd.DataFrame(
        list(methylation_results),
        columns=[
            "time",
            "reads",
            "cpgs",
            "1st class",
            "2nd class",
            "3rd class",
            "4th class",
            "5th class",
            "1st pred",
            "2nd pred",
            "3rd pred",
            "4th pred",
            "5th pred",
        ],
    ).to_csv(output_path + sample + "_results.csv", sep="\t")

    if methylation_list:
        print(str(time.asctime()) + " Write Methylation data to file.")
        log_file.writelines(str(time.asctime()) +
                            " Write Methylation data to file.")

        methylation = pd.DataFrame(list(methylation_list), columns=[
                                   "epic_id",
                                   "methylation",
                                   "scores_per_read",
                                   "binary_methylation",
                                   "read_id",
                                   "start_time",
                                   "run_id",
                                   "QS",
                                   "read_length",
                                   "map_qs"
                                   ]).sort_values('start_time')
        methylation_runs = []
        if not isinstance(methylation, pd.DataFrame):
            methylation = pd.DataFrame()

        st = None  # Move the definition of st outside of the loop
        tdif = None  # Move the definition of tdif outside of the loop
        for run_id, st in runs.items():
            t1 = arrow.get(methylation[methylation['run_id'] == run_id].sort_values(
                'start_time').iloc[0]['start_time'])
            tdif = np.floor((t1-arrow.get(st)).total_seconds()/3600)*3600
            test = methylation[methylation['run_id'] == run_id]
            test.loc[:, 'start_time'] = test['start_time'].apply(
                lambda a, st=st, tdif=tdif: int((arrow.get(a)-arrow.get(st)).total_seconds()-tdif))  # Use a default argument to pass st and tdif to the lambda function
            methylation_runs.append(test)

        methylation_cpg_data = pd.concat(methylation_runs)

        methylation_cpg_data.sort_values('start_time').reset_index(
            drop=True).to_feather(output_path + sample + '.feather')
    else:
        print('No data retrieved.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--min_entries", type=int, default=1000, help="Minimum number of new CpG entries for methylation classification. Default is 1000."
    )
    parser.add_argument(
        "--methylation_qsore",
        default=9,
        type=int,
        help="Minimum Q-Score of basecalled read to be considered in methylation classification. Default is 9.",
    )
    parser.add_argument(
        "-i",
        "--inputs",
        type=str,
        default=".",
        required=True,
        help="Filepath of BAM files. If MinKNOW is used, the path to the output directory will be used.",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Recursively monitor subdirectories.",
    )
    parser.add_argument(
        "--offline", action="store_true", help="Run MethyLYZR offline. No MinKNOW connection needed."
    )
    parser.add_argument(
        "--methylation_threads",
        default=8,
        type=int,
        help="Number of Threads used for methylation data extraction. Default is 8.",
    )
    parser.add_argument(
        "--io_threads",
        type=int,
        default=2,
        help="Number of Threads used for io-handling. Default is 2.",
    )
    parser.add_argument(
        "--reference",
        choices=["T2T", "HG38"],
        default="T2T",
        help="Reference for array probes (BEF file linking an EPIC id to a genomic position)",
    )
    parser.add_argument(
        "-s", "--sample", type=str, required=True, help="Name of the Sample. Should be the same as Sample ID in MinKNOW."
    )
    parser.add_argument(
        "-o", "--output", type=str, required=True, help="Path to output folder. A Subfolder with the sample name will be created."
    )
    parser.add_argument(
        "--dev_key", type=str, help="Developer Key for MinKNOW API. Only needed if MinKNOW is nor running on the same machine, e.g. Windows Subsystem for Linux (WSL)."
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="String that needs to be present in path. Useful for filtering on barcodes. If barcode filtering is used the filter phrase needs to be like in the MinKNOW output directory ,e.g. barcode01.",
    )

    args = parser.parse_args()

    main(
        args.min_entries,
        args.inputs,
        args.recursive,
        args.methylation_threads,
        args.io_threads,
        args.reference,
        args.offline,
        args.sample,
        args.output,
        args.methylation_qsore,
        args.dev_key,
        args.filter
    )
