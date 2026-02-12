# MethyLYZR (Lightweight Pipeline)

This update introduces a memory-efficient preprocessing and prediction
workflow for large Nanopore methylation datasets.
It adds two lightweight scripts that preserve the original classifier
behavior while significantly improving scalability and robustness.

The lightweight pipeline consists of:

-   `scripts/bam2feather_light.py` --- streaming BAM → Feather extraction
-   `MethyLYZR_light.py` --- memory-efficient classifier backend

Both scripts are fully compatible with existing MethyLYZR model files.

------------------------------------------------------------------------

# 1. Streaming BAM → Feather extraction

## `bam2feather_light.py`

`bam2feather_light.py` replaces the original `bam2feather.py`
implementation with a streaming architecture that avoids materializing
large alignment objects in memory.

### Key Improvements

### Streaming CpG extraction

-   Processes alignments one read at a time
-   Avoids storing full `get_aligned_pairs()` outputs
-   Immediately discards alignment state after extracting relevant CpG
    calls
-   Reduces peak RAM usage by orders of magnitude on large ONT runs

------------------------------------------------------------------------

### Multiprocessing with bounded memory

-   Producer--consumer architecture
-   Multiple BAM reader workers
-   Single Feather writer process
-   Queue-based backpressure prevents uncontrolled memory growth

------------------------------------------------------------------------

### Correct alignment filtering

-   Explicitly excludes:
    -   Secondary alignments
    -   Supplementary alignments
-   Avoids incorrect filtering caused by ambiguous bitwise checks

------------------------------------------------------------------------

### 5mC + 5hmC support

-   Parses modbase tags for:
    -   5-methylcytosine (5mC)
    -   5-hydroxymethylcytosine (5hmC)
-   Merges modification probabilities per query position
-   Uses the maximum probability when both modifications are present

------------------------------------------------------------------------

### Reduced Feather footprint

Only three columns are written:

-   `epic_id`
-   `methylation`
-   `scores_per_read`

This is the minimal feature set required by the predictor and
significantly reduces output size.

------------------------------------------------------------------------

# 2. Memory-efficient prediction backend

## `MethyLYZR_light.py`

`MethyLYZR_light.py` preserves the original classification logic but
replaces the memory-heavy denominator computation with a blocked matrix
multiplication implementation.

The classification model itself is unchanged.

------------------------------------------------------------------------

## What it computes

For each CpG call:

-   Binary methylation call
-   Noise-adjusted centroid probabilities
-   Read-weighted and RELIEF-weighted log-likelihood terms

For each class:

-   Weighted numerator score
-   Class-specific denominator matrix
-   Posterior probabilities via Bayes rule in log-space

The mathematical formulation remains identical to the original
implementation.

------------------------------------------------------------------------

## Why this rewrite was necessary

The original denominator calculation:

``` python
np.apply_along_axis(...)
```

-   Iterates over classes in Python
-   Allocates large temporary broadcast arrays
-   Scales as O(N · C²)
-   Becomes memory- and CPU-intensive for large N

------------------------------------------------------------------------

## Blocked GEMM implementation

The denominator matrix is computed as:

Eᵀ A

Where:

-   A = per-feature log-likelihood matrix
-   E = exp(-W) weight suppression matrix

The computation is performed in CpG chunks (default 200,000 rows):

-   No Python loops over classes
-   No repeated broadcast allocation
-   Uses optimized BLAS routines
-   Produces numerically equivalent results (floating-point precision
    differences only)

------------------------------------------------------------------------

## Computational Properties

  Component             Original                       Lightweight
  --------------------- ------------------------------ -----------------------
  BAM extraction        Alignment-level accumulation   Streaming per-read
  Denominator scoring   apply_along_axis loop          Blocked GEMM
  Peak memory           High for large N               Bounded by chunk size
  Model behavior        Unchanged                      Unchanged

Arithmetic complexity remains O(N · C²), but constant factors and memory
pressure are substantially reduced.

------------------------------------------------------------------------

# Compatibility

-   Feather output remains fully compatible with the classifier
-   Centroid, weight, and prior model files remain unchanged
-   Posterior probabilities match original implementation within
    floating-point tolerance
-   No changes were made to the classification model or training
    procedure

------------------------------------------------------------------------

# When to use the lightweight pipeline

Use:

-   `bam2feather_light.py` for large ONT runs or limited-RAM systems
-   `MethyLYZR_light.py` for samples with large CpG counts
-   The full pipeline when processing high-coverage or long-read
    datasets

This update enables routine large-scale ONT methylation classification
without requiring high-memory compute nodes while preserving full
methodological consistency with MethyLYZR.
