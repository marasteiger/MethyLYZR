# Project: MethyLYZR
# Author: Mara Steiger
# Date: February 05, 2024
# Version: 1.0.0
# Description: This script uses a pre-trained Naïve Bayes model for prediction of CNS tumor
# classes from sparse Nanopore methylation profiles.
#
# Contact: steigerm@molgen.mpg.de
# 
# scores_blocked — Blocked matrix implementation of MethyLYZR scoring
# scores_blocked computes the weighted class log-likelihoods and the class-specific denominator matrix used in MethyLYZR, using a memory-efficient blocked matrix multiplication (GEMM) strategy.
# The original implementation constructs the denominator matrix via repeated class-wise recomputation of likelihoods using apply_along_axis, which:
# - Scales as O(N · C²)
# - Creates large temporary broadcast arrays
# - Becomes memory- and CPU-intensive for large N (number of CpGs)
# scores_blocked performs the mathematically equivalent computation using chunked matrix multiplication, avoiding repeated large allocations and Python-level loops.

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os

import sys
import time


### VARIABLES ###

POSTERIOR_THRESHOLD = 0.6
BASECOUNT = 300  # the number of reads we standardize to

### FUNCTIONS ###

def scores_blocked(P1_log, P0_log, newX, W, read_weights, BASECOUNT=300, chunk_N=20_000):
    """
    Computes:
      num[j]      = sum_i A[i,j] * exp(-W[i,j])                 (C,)
      denom[k,j]  = sum_i A[i,j] * exp(-W[i,k])                 (C,C)
    using row-blocked GEMM.

    P1_log, P0_log must already be logs:
      P1_log = log(tilde_mu)          shape (N,C)
      P0_log = log(1 - tilde_mu)      shape (N,C)
    """
    sys.stderr.write(" - Preparing arrays for blocked GEMM...\n")
    sys.stderr.flush()
    t_prep = time.time()

    N, C = P1_log.shape
    assert P0_log.shape == (N, C)
    assert W.shape == (N, C)
    assert newX.shape == (N,)
    assert read_weights.shape == (N,)

    num = np.zeros(C, dtype=np.float64)
    denom = np.zeros((C, C), dtype=np.float64)   # [k, j]

    n_chunks = (N + chunk_N - 1) // chunk_N
    start_time = time.time()

    sys.stderr.write(f" - Starting blocked GEMM: N={N:,} C={C} chunk_N={chunk_N:,} "
                    f"chunks={(N + chunk_N - 1)//chunk_N}\n")
    sys.stderr.flush()

    for idx, s in enumerate(range(0, N, chunk_N), 1):
        e = min(s + chunk_N, N)

        # ---- progress display ----
        progress = idx / n_chunks
        bar_len = 30
        filled = int(bar_len * progress)
        bar = "=" * filled + "-" * (bar_len - filled)
        elapsed = time.time() - start_time

        sys.stdout.write(
            f"\rBlocked GEMM [{bar}] {idx}/{n_chunks} "
            f"({progress*100:5.1f}%) | {elapsed:6.1f}s"
        )
        sys.stdout.flush()
        # --------------------------

        # x: (n,1)
        x = newX[s:e].astype(np.float64, copy=False)[:, None]

        # Make chunk arrays contiguous for fast BLAS
        P1c = np.ascontiguousarray(P1_log[s:e, :], dtype=np.float64)
        P0c = np.ascontiguousarray(P0_log[s:e, :], dtype=np.float64)
        Wc  = np.ascontiguousarray(W[s:e, :],      dtype=np.float64)

        # A: (n,C)
        A = P1c * x + P0c * (1.0 - x)

        # E: (n,C)
        E = np.exp(-Wc)

        # Numerator: (C,)
        num += np.sum(A * E, axis=0)

        # Denominator matrix: (C,C) with [k,j] = sum_i E[i,k]*A[i,j]
        denom += E.T @ A

    sys.stderr.write(f"Done. Total time: {time.time() - t_prep:.1f}s\n")
    sys.stderr.flush()

    R = np.sum(read_weights, dtype=np.float64)
    num = num / R * BASECOUNT
    denom = denom / R * BASECOUNT
    return num, denom


def calc_denominator(log_likelihoods, prior):
    # function to calculate log denominator P(X) for Bayes theorem

    # params:
    # log_likelihoods: pre-calculated log-likelihoods, i.e., P(x|C)
    # prior: pre-defined prior class probability, i.e., P(C)

    # using log-sum-exp trick to avoid numerical underflow
    l_k = np.log(prior) + log_likelihoods
    l_max = np.max(l_k)  # choosing max(l_k) as constant l_max, such that largest term exp(0)=1   # noqa: E501
    log_denom = l_max + np.log(np.sum(np.exp(l_k - l_max)))
    return log_denom


def predict_from_fingerprint(newX, feature_ids, centroids, W, noise, prior, read_weights):
    # function to predict class probabilities for a new methlyation profile

    # params:
    # newX: binary vector with observation of methylation events
    # feature_ids: probe identifiers of the observed methylation events
    # centroids: centroid matrix of dimension (no. probes) x (no. classes)
    # W: weight matrix of dimension (no. probes) x (no. classes)
    # noise: noise value for each call
    # prior:  assumed prior class probability
    # read_weights: weight value for each call

    if (feature_ids is None) | (len(set(feature_ids) & set(centroids.index)) == 0):
        raise ValueError("Probe names in newX do not agree with reference.")

    # Convert newX to a NumPy array if it's a pd Series
    if isinstance(newX, pd.Series):
        newX = newX.values

    log(" - Setting up P0, P1...")
    P1 = centroids.loc[feature_ids, prior.index].to_numpy()  # subset centroids by selected features and order classes
    P1 = P1 - P1 * 2 * noise[:, None] + noise[:, None]  # add noise terms

    P0 = np.log(1 - P1)  # log complement probabilities
    P1 = np.log(P1)  # log probabilities

    # weight matrix for probes x classes
    log(" - Setting up W...")
    if W is None:
        # set everything to 1
        W = np.ones((P1.shape))
        # include read weights
        if len(read_weights) > 0:
            W = W * read_weights[:, None]
    else:
        # subset weight matrix, order classes, and include read weights
        W = read_weights[:, None] * W.loc[feature_ids, prior.index].to_numpy() * np.mean(1 / read_weights)

    # # calculating the likelihoods of all classes C_j, i.e, P(x|C_j)
    # log_likelihoods_weighted = np.sum(P1 * newX[:, None] * np.exp(-W) + P0 * (1 - newX[:, None]) * np.exp(-W), axis=0)  # likelihoods for each class

    # ## REWEIGHTING + BASECOUNT
    # # apply reweighting
    # likelihood_mat = np.apply_along_axis(
    #     lambda x: np.sum(P1 * newX[:, None] * np.exp(-x)[:, None] + P0 * (1 - newX[:, None]) * np.exp(-x)[:, None], axis=0), 0, W
    # )  # recalculate likelihoods with class-specific weights
    # likelihood_mat = likelihood_mat.transpose()
    # # apply basecount
    # log_likelihood_mat_weighted = likelihood_mat / np.sum(read_weights) * BASECOUNT  # normalize to basecount
    # log_likelihoods_weighted = log_likelihoods_weighted / np.sum(read_weights) * BASECOUNT  # normalize to basecount


    log_likelihoods_weighted, log_likelihood_mat_weighted = scores_blocked(
        P1_log=P1,   # already log(tilde_mu)
        P0_log=P0,   # already log(1-tilde_mu)
        newX=newX,
        W=W,
        read_weights=read_weights,
        BASECOUNT=BASECOUNT,
        chunk_N=20_000,
    )

    log_denominator = np.apply_along_axis(
        lambda x: calc_denominator(log_likelihoods=x, prior=prior), 1, log_likelihood_mat_weighted
    )  # calculating class-specific denominator P(X)

    # calculating posterior probabilities by Bayes' Theorem in log space, then exponentiating
    class_posteriors = np.exp(np.log(prior) + log_likelihoods_weighted - log_denominator)

    return {"posterior": class_posteriors, "log_likelihoods": log_likelihoods_weighted, "epic_ids": feature_ids, "read_weights": read_weights}


def predict_sample(sample_id, sample_dir, centroids, W, class_frequency, min_noise, methylation_lower_bound, methylation_upper_bound, downsample=None):
    # function to load sample data, filter, call prediction for given sequencing time

    # params:
    # sample_id: id of Nanopore sample to be tested
    # centroids: centroid matrix of dimension (no. probes) x (no. classes)
    # W: weight matrix of dimension (no. probes) x (no. classes)
    # class_frequency: # assumed prior class probability
    # min_noise: minimum threshold for noise terms applied to methylation centroids
    # methylation_lower_bound: methylation calls with likelihood > methylation_lower_bound are considered
    # methylation_upper_bound: AND methylations call with likelihood < methylation_upper_bound  are considered

    log(f"Sample: {sample_id}", True)

    log(f"Loading data from {sample_dir}\n")

    # methylation matrix of dimension (no. probes) x 9
    test_sample = pd.read_feather(sample_dir)  # read sample
    test_sample = test_sample.loc[test_sample["epic_id"].isin(centroids.index)]  # filter for CpGs in reference centroids

    log("Preprocessing data...")
    log(" - Filtering values...")
    test_sample = test_sample[(test_sample["methylation"] <= methylation_upper_bound) | (test_sample["methylation"] >= methylation_lower_bound)]  # filter for methylation probability >0.8 or <0.2
    test_sample = test_sample[(test_sample["scores_per_read"] <= 10)]  # filter for number of features per read <= 10

    # Downsample to avoid OOM (optional but recommended for large inputs)
    if downsample is not None:
        if downsample <= 0:
            raise ValueError("downsample must be > 0")

        if downsample < len(test_sample):
            log(f" - Downsampling to {downsample} CpG calls...")
            test_sample = test_sample.sample(n=downsample, random_state=42)
        else:
            log(" - Downsample size larger than available CpGs — skipping downsampling.")

    log(" - Calculating noise...")
    # calculate noise values
    noise = 0.5 - abs(test_sample["methylation"].to_numpy() - 0.5)
    noise[noise < min_noise] = min_noise

    log(" - Calculating read weights...\n")
    # calculate read weights
    read_weights = 1 / test_sample["scores_per_read"].to_numpy()

    # create binary methylation vector
    binary_vec = test_sample["methylation"]
    binary_vec = (binary_vec >= 0.5).astype(int)

    log("Predicting classes...")
    # call prediction function to get class probabilities from methylation rates
    prediction_list = predict_from_fingerprint(newX=binary_vec, feature_ids=test_sample["epic_id"], centroids=centroids,
                                               W=W, noise=noise, prior=class_frequency, read_weights=read_weights)

    class_posteriors = prediction_list["posterior"].sort_values(ascending=False)

    return class_posteriors

def log(message, separator=False):
    if separator:
        print("-------------------------------------------------------")
    print(message)
    if separator:
        print("-------------------------------------------------------\n")


def main(input, sample, centroids, weights, priors, output, minNoise, methLowerBound, methUpperBound, flatPrior, uniformWeights, downsample):
    ### PREPROCESSING ###

#     log("""
#  _  _  ____  ____  _  _  _  _  __    _  _  ____  ____
# ( \/ )(  __)(_  _)/ )( \( \/ )(  )  ( \/ )(__  )(  _ \\
# / \/ \ ) _)   )(  ) __ ( )  / / (_/\ )  /  / _/  )   /
# \_)(_/(____) (__) \_)(_/(__/  \____/(__/  (____)(__\_)
# """, True)

    log("M e t h y L Y Z R", True)

    # log("Starting prediction process...")

    log(f"Reading model from {centroids}")
    log(f"                   {weights}")
    log(f"                   {priors}")


    ## loading data ##

    # mean centroids
    betas_mean = pd.read_feather(centroids).set_index("index")

    # RELIEF-based feature weights
    W_RELIEF = pd.read_feather(weights).set_index("index")

    if uniformWeights:
        log("Using uniform feature weights (RELIEF disabled).")
        W_RELIEF = None
    else:
        log("Using RELIEF feature weights.")

    # class prior information from training data (Capper et al. 2021 + METASTASTS REF)
    CF = pd.read_csv(priors, index_col=0).squeeze("columns")

    # optional flat prior
    if flatPrior:
        log("Using flat (uniform) class prior.")
        CF = pd.Series(1.0 / len(CF), index=CF.index)
    else:
        log("Using class priors from file.")

    if downsample is not None:
        log(f"Downsampling to {downsample} CpG calls.")

    ##################

    ### PREDICTION ###

    ## apply prediciton to given sample
    prediction = predict_sample(
        sample_id=sample,
        sample_dir=input,
        centroids=betas_mean,
        W=W_RELIEF,
        class_frequency=CF,
        min_noise=minNoise,
        methylation_lower_bound=methLowerBound,
        methylation_upper_bound=methUpperBound,
        downsample=downsample,
    )

    ##################

    ### OUTPUT ###

    log(f"Saving results to {output} ... \nDone.", True)

    if not os.path.exists(output):
            os.makedirs(output)

    ## write table
    prediction.reset_index().to_csv(output + "/MethyLYZR_" + sample + ".csv", header=["Class", "Posterior Probability"], index=False)

    ## export barplot

    # Select the top five entries
    top5 = prediction[:5]

    # Create the plot
    plt.style.use("seaborn-v0_8-deep")
    plt.bar(top5.index, top5.values, color=["C0" if value > 0.6 else "grey" for value in top5.values])

    plt.ylim(0, 1)
    plt.xlabel("Top 5 classes")
    plt.ylabel("Posterior probability")
    plt.title(sample)
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.gca().spines[["right", "top"]].set_visible(False)
    plt.tick_params(axis="both", which="both", bottom=False, left=False)  # labels along the bottom edge are off

    # Add a horizontal line at y=0.5
    plt.axhline(y=0.6, color="black", linestyle="--")

    # Save the plot
    plt.savefig(output + "/MethyLYZR_" + sample + ".pdf", format="pdf", bbox_inches="tight")
    plt.show()

    ##############


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default=".", required=True, help="Filepath to Feather file")
    parser.add_argument("-s", "--sample", type=str, required=True, help="Name of the Sample")
    parser.add_argument("-c", "--centroids", type=str, default="model/betas_mean.feather", required=False, help="Filepath to the pre-trained class centroids")
    parser.add_argument("-w", "--weights", type=str, default="model/W_RELIEF.feather", required=False, help="Filepath to the pre-trained weights")
    parser.add_argument("-p", "--priors", type=str, default="model/class_priors.csv", required=False, help="Filepath to the pre-trained class priors")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to output folder")
    parser.add_argument("--minNoise", type=float, default=0.05, help="Minimum noise value added to centroids")
    parser.add_argument("--methLowerBound", type=float, default=0.8, help="Lower bound for calling methylated loci")
    parser.add_argument("--methUpperBound", type=float, default=0.2, help="Upper bound for calling unmethylated loci")

    parser.add_argument("--uniformWeights", action="store_true", help="Use uniform feature weights instead of RELIEF weights")
    parser.add_argument("--flatPrior", action="store_true", help="Use uniform (flat) class prior instead of priors file values")
    parser.add_argument("--downsample", type=int, default=None, help="Randomly downsample to N CpG calls before prediction")

    args = parser.parse_args()

    main(args.input, args.sample, args.centroids, args.weights, args.priors, args.output, args.minNoise, args.methLowerBound, args.methUpperBound, args.flatPrior, args.uniformWeights, args.downsample)
