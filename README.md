
# MethyLYZR


Pre-trained naïve Bayes model for classification of brain tumor samples from sparse methylation data into 91 CNS tumor classes defined by Capper et al. (2018) and 3 metastases classes (i.e., from breast cancer, lung cancer, and melanoma).






## Requirements

### Hardware requirements

`MethyLYZR.py` requires a 'standard' personal computer with minimum 4 GB RAM or a compute server. While `MethyLYZR.py` does not utilize parallel processing and can technically run on systems with a single-core CPU, we recommend a minimum of a dual-core processor to ensure the system remains responsive for multitasking. When working with large datasets, additional memory is advisable. The runtimes stated below are generated using a computer (Apple iMac) with 64 GB RAM, 10 cores@3 GHz.

### Software requirements

#### OS requirements

The code has been tested on the following systems:

- macOS 13.2.1
- Internal Linux distribution

#### Python dependencies

The Python package dependencies are listed in `requirements.txt`.



## Installation Guide

Before proceeding with the installation, make sure that Python 3.5 or higher is installed. The code was developed and tested using Python version 3.7.11.

### Setting Up the Environment

Follow these steps to set up an environment with all required Python packages:

1. Create a virtual environment for isolation of the dependecies
  ``` bash
  python -m venv venv
  ```

2. Activate the environment
  ``` bash
  source venv/bin/activate
  ```

3. Install the required dependencies using the `requirements.txt` file
  ``` bash
  pip install -r requirements.txt
  ```


## Recommended Preprocessing

#### Basecalling of raw Nanopore signals
First, the raw Nanopore signals should be processed into bases using the dorado basecall server from ONT (https://github.com/nanoporetech/dorado/) implemented in the sequencing software MinKnow, using the high-accuracy basecalling model with 5mC modifications (dna_r9.4.1_450bps_modbases_5mc_cg_hac.cfg). The obtained reads should be mapped to human reference genome GRCH38.p13 e.g.using `minimap2` and saved into BAM files. Information on modified bases should use the MM and ML tags defined in the Sequence Alignment/Map Optional Fields Specification.

#### Processing of BAM files (bam2feather.py)
The extraction of the methylation values from the BAM files into the required file format can be done using the provided Python script `bam2feather.py`. In short, the script first filters for primary alignments with a minimal mapping quality of 10, as reported by minimap2. Positions are further filtered on loci corresponding to a genomic position on the Illumina Infinium Human Methylation 450K BeadChip. The per read and CpG methylation probability is calculated using the SAM MM and ML tags. The output is written to disk as a feather file in the format required for MethyLYZR prediction (see below).

Note: When intending to filter methylation calls by the start times of reads, it is crucial to verify the accuracy of the read start times as provided by the basecaller. We have observed that discrepancies in software versions can lead to inaccuracies in the reconstructed timing of base calls. Therefore, we strongly advise a thorough validation of read start times against your basecalling software's output to ensure the reliability of methylation timing analysis. 

###### Input arguments for bam2feather.py

| Parameter | Description |
| ------ | ------ |
| -i, --inputs | directory with BAM files (required)  |
| -s, --sample | sample (file) name (required)  |
| -o, --output | output directory (required) |
| --sites | file with CpG-Sites-Annotation in bed format (required) |
| -r, --recursive | recursive monitor subdirectories (default: False, action: store_true) |
| --io_threads | number of threads used for io-handling (default: 2) |
| --methylation_threads | number of threads used for methylation data extraction (default: 4) |
| --filter | string (present in the path) for filtering on barcodes |


###### An example run of the bam2feather.py script:
```bash
bam2feather.py -i input_dir_bams -s sample1 -r --sites data/EPIC_CpGannotation.bed -o output_dir --methylation_threads 24 --io_threads 8
```

###### Output Feather table format


| Column | Description |
| ------ | ------ |
| epic_id | EPIC array probe ID associated with CpG site  |
| methylation | methylation probability [0,1]  |
| scores_per_read | number of measurements (covered by reference) on same read |
| read_no | read number |
| binary_methylation | 0 if methylation < 0.8, else 1 |
| read_id | read identifier |
| pct_identity | percent read identity to reference  |
| start_time | time in seconds after start of sequencing |
| run_id | sequencing run identifier |



## Tumor class prediction (MethyLYZR.py)

The Python script `MethyLYZR.py` is used to apply the pre-trained Naïve Bayes model for prediction of CNS tumor class to a given Nanopore sample.

```bash
python MethyLYZR.py -i sampleX.feather -s sampleX -o results
```

#### Pre-trained model
Components making up the trained Naïve Bayes model are deposited in `model/`:
- centroids (`betas_mean.feather`)
    - mean methylation profiles per class
- feature weights (`W_RELIEF.feather`)
    - class-specific weights derived from ReliefF-based method
- class priors (`class_priors.csv`)
    - prior probabilities per class derived from relative frequencies in training data

This model encompasses 91 CNS + 3 metastasis classes.

#### Input arguments for MethyLYZR.py

| Parameter | Description |
| ------ | ------ |
| -i, --inputs | directory with BAM files (required)  |
| -s, --sample | sample (file) name (required)  |
| -o, --output | output directory (required) |
| -c, --centroids | directory of pre-trained class centroids (default: 'model/betas_mean.feather')  |
| -w, --weights | directory of pre-trained class centroids (default: 'model/W_RELIEF.feather')  |
| -p, --priors | directory of pre-trained class priors (default: 'model/class_priors.csv')  |
| --minNoise | minimum noise value added to centroids (default: 0.05) |
| --methLowerBound | lower bound for calling methylated loci (default: 0.8) |
| --methUpperBound | upper bound for calling unmethylated loci (default: 0.2) |





#### What it does
1.  reads the pre-trained model (centroids + feature weights + class priors)
2.  loads sample from corresponding feather file
3.  preprocesses the methylation measurements
    - filtering for (methylation probabilty < methUpperBound) OR (methylation probabilty > methLowerBound)
    - filtering for number of features per read <= 10
    - calculating noise
    - calculating read weights
    - binarize methylation probabilities
4.  predicts posterior probabilities for all 94 classes using Naïve Bayesian framework
5.  outputs
    - a ranked list with the posterior probabilites for all classes as a CSV
    - a barplot showing the posterior probabilities of the top 5 hits as a PDF

#### Posterior probabilities

The probabilities of predictions output by `MethyLYZR.py` range from 0 to 1 and reflect the certainty of the prediction.

- Scores > 0.6: Predictions with probabilities above 0.6 can be regarded as high-certainty predictions.

- Scores <= 0.6: Predictions with scores below 0.5 are considered to have lower certainty. These results suggest that the model has lower confidence in the prediction, and they should be treated accordingly.



### Demo


This demo illustrates how to use MethyLYZR to process sample data and generate output files. We use `IEG14_4d93b1bb47f2b6315c552126160915afc2a89ca7.feather` as input to produce a CSV table of the predictions and a PDF with visualized top 5 classes.


0. Preprocess the input data using `bam2feather.py`
  ```bash
  python bam2feather.py -i IEG14_HG38/ -s IEG14_4d93b1bb47f2b6315c552126160915afc2a89ca7 -r --sites data/EPIC_CpGannotation.bed -o demo_outputs --methylation_threads 24 --io_threads 8

  ```

Follow these steps to replicate the demo:

1.  Run `methyLYZR.py` on the preprocessed data for test sample IEG14 (59.7 seconds)

  ```bash
  python methyLYZR.py -i data/ONT_15min_runs/IEG14_4d93b1bb47f2b6315c552126160915afc2a89ca7.feather -s IEG14 -o results \
  -c model/betas_mean.feather -w model/W_RELIEF.feather -p model/class_priors.csv --minNoise 0.05 --methLowerBound 0.8 --methUpperBound 0.2
  ```

  This will print the following log to the command line
  ```
  -------------------------------------------------------
    M e t h y L Y Z R
  -------------------------------------------------------

  Reading model from model/betas_mean.feather
                       model/W_RELIEF.feather
                       model/class_priors.csv
  -------------------------------------------------------
  Sample: IEG14
  -------------------------------------------------------

  Loading data from data/ONT_15min_runs/IEG14_4d93b1bb47f2b6315c552126160915afc2a89ca7.feather

  Preprocessing data...
   - Filtering values...
   - Calculating noise...
   - Calculating read weights...

  Predicting classes...

  Saving results to results/MethyLYZR_IEG14.csv
                      results/MethyLYZR_IEG14.pdf
  -------------------------------------------------------
  Done.
  -------------------------------------------------------

  ```

2. Look at output generated by this demo.

  - Ranked table with posteriors of all classes

    <img src="https://github.com/user-attachments/assets/d6186c36-8edb-4aac-bf3f-12337544d257" alt="IEG14 Head Output Table" width="400"/>
  - Barplot with top 5 predicted classes

    <img src="https://github.com/user-attachments/assets/d87f4780-813b-4ee5-b981-633f3593dd2e" alt="IEG14 Output Barplot" width="400"/>




## License


All source code and binary files are available under the following provisions:

Any copyright or patent right is owned by and proprietary material of the Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. ("MPG").
MPG makes no representations or warranties of any kind concerning the source code, neither express nor implied, and the absence of any legal or actual defects, whether discoverable or not.

The source code and binary files available are subject to a non-exclusive, revocable, non-transferable, and limited right to use for the exclusive purpose of undertaking academic or not-for-profit research, which is granted hereby.
Use of the Code or any part thereof for commercial or clinical purposes and alterations are strictly prohibited in the absence of a Commercial License Agreement from MPG (Contact: info@max-planck-innovation.de).

Download and/or use of the source code and binary files under the aforementioned research license comes with the acceptance of the license agreement provided with the download.
Any further distribution is prohibited.
