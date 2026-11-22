# GGDNIR

Research code for **Good Generations Do Not Imply Informative Representations: Evaluating Language-Model Latents in Molecular Diffusion** by **Ali Forooghi, Luis Rueda, and Alioune Ngom**, School of Computer Science, University of Windsor.

This implementation compares molecular representations from ChemBERTa, Llama 3.2 1B, and a dataset-specific encoder trained from random initialization. It connects frozen encoder states to a latent diffusion model and an autoregressive SELFIES decoder through fixed PCA or learned projections. Representation diagnostics, reconstruction behavior, and generated-molecule quality provide separate evidence about the resulting systems.

## Contents

- [Installation](#installation)
- [Data preparation](#data-preparation)
- [Training and generation](#training-and-generation)
- [Encoder and interface comparisons](#encoder-and-interface-comparisons)
- [Evaluation and diagnostics](#evaluation-and-diagnostics)
- [GuacaMol and MOSES](#guacamol-and-moses)
- [Reproducibility](#reproducibility)
- [Citation](#citation)
- [License and contact](#license-and-contact)

## Release contents

This repository distributes Python source, dependency specifications, citation metadata, and the MIT license. Datasets, pretrained weights, embedding caches, experiment outputs, notebooks, manuscript files, development tests, and shell runners are excluded. The commands below run directly from the repository root without a separate scripts directory. Data and output directories are created locally when needed.

| Path | Purpose |
| --- | --- |
| `molecule_diffusion/` | Encoders, latent interfaces, diffusion, SELFIES decoding, training, sampling, and evaluation |
| `requirements.txt` | Core dependencies |
| `requirements-benchmarks.txt` | Additional dependencies for optional benchmark evaluation |
| `CITATION.cff` | Machine-readable software citation |
| `LICENSE` | MIT license for this code |

## Installation

The release is checked with Python 3.12 on Linux. A CUDA-capable GPU is recommended for training; the commands also accept `--device cpu` for small runs. Full token-level embedding caches can occupy tens to hundreds of gigabytes, depending on the encoder and dataset.

```bash
git clone https://github.com/FlockOfBird/GGDNIR.git
cd GGDNIR
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m molecule_diffusion --help
```

Install a PyTorch build appropriate for your CUDA environment when using a GPU. Encoder weights and tokenizers are downloaded through Hugging Face on first use. The Llama experiments require access to `meta-llama/Llama-3.2-1B` and authentication with an authorized Hugging Face account. Keep credentials in your local environment.

## Data preparation

Obtain ZINC250k independently. The preprocessing command expects a CSV with these case-sensitive columns:

| Column | Meaning |
| --- | --- |
| `smiles` | Molecular SMILES string |
| `logP` | Octanol–water partition coefficient |
| `qed` | Quantitative estimate of drug-likeness |
| `SAS` | Synthetic-accessibility score |

Save the source CSV locally as `data/250k_rndm_zinc_drugs.csv`, or supply its location explicitly:

```bash
python -m molecule_diffusion.preprocess_zinc \
  --input-path data/250k_rndm_zinc_drugs.csv \
  --output-path data/250k_zinc_clean.csv
```

Preprocessing canonicalizes and validates SMILES, removes duplicate canonical molecules, filters failed SELFIES conversions, and adds SELFIES and molecular-weight columns. Generative embedding preparation needs a `smiles` column; property-regressor training additionally requires its selected property column. Properties are not required as generation targets for the unguided workflow below.

The default ZINC partition uses approximately 80% training, 10% validation, and 10% test data with scaffold grouping and data seed 42. PCA, latent standardization, and the SELFIES vocabulary are fitted on training data. Preserve the same input file, row order, and preprocessing when comparing conditions.

## Training and generation

The following Bash example runs one ChemBERTa condition with token-wise PCA to 256 dimensions. Define the shared arguments once in the same shell. Keep them consistent across all stages of the run.

```bash
common=(
  --encoder-name DeepChem/ChemBERTa-100M-MLM
  --dataset-path data/250k_zinc_clean.csv
  --generative-dataset-path data/250k_zinc_clean.csv
  --artifact-dir artifacts/zinc/chemberta_token_pca_seed101
  --latent-representation token_sequence
  --projection-mode pca
  --latent-projection-dim 256
  --latent-projection-fit-tokens 50000
  --max-length 160
  --seed 42
  --device cuda:0
)

# 1. Encode the scaffold splits and fit the latent interface.
python -m molecule_diffusion prepare-generative-embeddings \
  "${common[@]}" --batch-size 32

# 2. Train the decoder and diffusion model with synchronized checkpoints.
python -m molecule_diffusion train-diffusion \
  "${common[@]}" --joint-decoder-diffusion \
  --training-seed 101 --epochs 40 --batch-size 64

# 3. Generate 5,000 unguided attempts and calculate per-run metrics.
python -m molecule_diffusion sample \
  "${common[@]}" --num-samples 5000 --sample-batch-size 64 \
  --sampling-seed 10101 --experiment-name unguided_seed10101 \
  --guidance-scale 0
```

The diffusion model uses eight residual MLP-Mixer blocks, hidden width 6144, token-mixing width 256, and 500 DDPM steps with a linear noise schedule. The SELFIES decoder has four Transformer layers. The default denoiser and decoder learning rates are `1e-4`; the learned projection rate is `1e-5`. The reconstruction and variance-penalty weights are `1.0` and `1e-3`.

Training saves checkpoints and run configuration under the chosen artifact directory. Sampling writes:

```text
<artifact-dir>/
  checkpoints/
  run_config.json
  metrics.json
  experiments/sampling/unguided_seed10101/
    samples.csv
    experiment.json
    metrics.json
    plots/
```

The sample CSV includes a `SMILES` column. Generation failures remain part of the requested attempts. Use a fresh artifact directory for each encoder, interface, and training seed. Use a distinct `--experiment-name` for each repeated sampling run.

## Encoder and interface comparisons

The ZINC250k manuscript comparison uses each of the three encoders with each of these interfaces:

| Interface | `--latent-representation` | `--projection-mode` |
| --- | --- | --- |
| Token-wise PCA-256 | `token_sequence` | `pca` |
| Learned token projection to 256 dimensions | `token_sequence` | `learned` |
| Mean-pooled PCA-256 | `mean_pooled` | `pca` |

Change the corresponding entries in `common` and choose a new artifact directory before repeating the stages above. For learned projection, the encoder cache retains the original width and `train-diffusion` jointly trains the projector, decoder, and denoiser automatically. For fixed PCA, retain `--joint-decoder-diffusion` to use the joint training path. Mean pooling produces a single latent vector per molecule before projection.

For Llama, set `--encoder-name meta-llama/Llama-3.2-1B`. To create the dataset-specific Trained Encoder, first train a randomly initialized ChemBERTa architecture on the ZINC training partition:

```bash
python -m molecule_diffusion.encoder_adaptation \
  --source-model DeepChem/ChemBERTa-100M-MLM \
  --output-dir artifacts/encoders/zinc_trained/model \
  --dataset-path data/250k_zinc_clean.csv \
  --data-seed 42 --training-seed 101 --objective mlm --scratch \
  --max-length 160 --epochs 3 --batch-size 8 \
  --accumulation-steps 4 --learning-rate 1e-4 --device cuda:0
```

Then set `--encoder-name artifacts/encoders/zinc_trained/model` in `common` and repeat the downstream stages. The encoder is frozen during downstream training. Train a separate scratch encoder for each dataset.

The package also supports a learned mean-pooled interface, used for the GuacaMol extension, and optional property-regressor guidance. The manuscript generation workflow uses `--guidance-scale 0`. Property and guidance regressors are separate stages (`train-property-regressor` and `train-guidance-regressor`) and are unnecessary for unguided sampling. Inspect each command with `--help` before extending the workflow.

## Evaluation and diagnostics

Sampling calculates per-run generation metrics and property plots by default. To reevaluate the saved sample file from the example:

```bash
python -m molecule_diffusion evaluate "${common[@]}" \
  --samples-path artifacts/zinc/chemberta_token_pca_seed101/experiments/sampling/unguided_seed10101/samples.csv
```

Representation diagnostics include pooled-feature spread, chemical-neighborhood consistency, and property probes:

```bash
python -m molecule_diffusion.study_latents \
  --encoder-name DeepChem/ChemBERTa-100M-MLM \
  --dataset-path data/250k_zinc_clean.csv --data-seed 42 \
  --train-count 2000 --val-count 500 --geometry-count 512 \
  --output artifacts/diagnostics/chemberta.json --device cuda:0
```

To inspect reverse-diffusion trajectories after training:

```bash
python -m molecule_diffusion.reverse_trajectory \
  --artifact-dir artifacts/zinc/chemberta_token_pca_seed101 \
  --device cuda:0
```

Additional study aggregation and plotting modules are included for analysis of existing study manifests. The per-run evaluation command and the official benchmark evaluators use different reference and scoring protocols. Compare scores only when their molecule counts, reference collections, canonicalization rules, and denominators match. Teacher-forced token accuracy measures prediction with reference prefixes; free-running molecular reconstruction requires a separate assessment.

## GuacaMol and MOSES

Optional benchmark support uses the following installation sequence. The historical benchmark packages are installed without dependency resolution because they declare legacy dependencies; this project supplies current RDKit and the dependencies used by its evaluation paths.

```bash
python -m pip install -r requirements-benchmarks.txt
python -m pip install --no-deps guacamol==0.5.5 molsets==0.3.1
python -m molecule_diffusion.benchmark_data \
  --benchmark guacamol --output-dir data/benchmarks/guacamol
```

GuacaMol preparation downloads official train/validation/test files, verifies their published MD5 hashes, and creates `pipeline.csv` with `smiles` and `benchmark_split` columns. For this dataset, redefine `common` as follows and repeat the three training/generation stages above, using 10 training epochs and 30,000 generation attempts:

```bash
common=(
  --encoder-name DeepChem/ChemBERTa-100M-MLM
  --dataset-path data/benchmarks/guacamol/pipeline.csv
  --generative-dataset-path data/benchmarks/guacamol/pipeline.csv
  --generative-split-column benchmark_split
  --disable-canonicalize
  --artifact-dir artifacts/guacamol/chemberta_pooled_learned_seed101
  --latent-representation mean_pooled
  --projection-mode learned
  --latent-projection-dim 256
  --max-length 160
  --seed 42
  --device cuda:0
)
```

Use `--skip-evaluation` during benchmark sampling to defer scoring to the official evaluator. Preserve the official split strings without additional canonicalization. For the GuacaMol Trained Encoder, pass the benchmark dataset to `encoder_adaptation` together with `--split-column benchmark_split --disable-canonicalize` and a separate output directory.

```bash
python -m molecule_diffusion.benchmark_evaluation \
  --benchmark guacamol \
  --samples-csv artifacts/guacamol/chemberta_pooled_learned_seed101/experiments/sampling/unguided_seed10101/samples.csv \
  --guacamol-train-file data/benchmarks/guacamol/official/guacamol_v1_train.smiles \
  --output-dir artifacts/guacamol/chemberta_pooled_learned_seed101/benchmark_evaluation \
  --device cuda:0
```

The optional `--goal-directed` mode ranks a fixed generated pool separately for each objective. It is a sample-and-rank baseline. GuacaMol FCD uses a compatibility adapter with the PyTorch checkpoint and can differ numerically from the historical TensorFlow implementation.

For MOSES, use `--benchmark moses` in preparation and evaluation, update the dataset and output paths, and generate at least 30,000 attempts. The preparation module holds out 5% of the official training set for validation; evaluation retains the official training, test, and scaffold-test references. MOSES is an additional supported benchmark beyond the manuscript's ZINC250k and GuacaMol comparison.

## Reproducibility

The examples specify one training seed and one sampling seed. Repeat conditions with separate artifact directories to measure variability. Keep `--seed` fixed for the data/cache split, change `--training-seed` for model initialization and training order, and change `--sampling-seed` for independent generation streams.

Record the repository commit, complete commands, source-data checksum, package versions, encoder checkpoint revision, GPU model, and CUDA version with each experiment. The dependency lists specify compatible ranges rather than a lockfile; save the resolved environment with `python -m pip freeze` for an individual run. Full experiments require independently obtained data and encoder downloads. This code-only release does not bundle the original fitted checkpoints or paper result tables.

For a source syntax check:

```bash
python -m compileall -q molecule_diffusion
```

## Citation

If this code supports your research, please cite the accompanying manuscript:

> Ali Forooghi, Luis Rueda, and Alioune Ngom. *Good Generations Do Not Imply Informative Representations: Evaluating Language-Model Latents in Molecular Diffusion.*

Publication venue and DOI are not specified in this release. For a software citation, use [CITATION.cff](CITATION.cff) and include the commit used for your experiments.

## License and contact

The code is released under the [MIT License](LICENSE). External datasets and encoder checkpoints retain their respective licenses and access terms.

For questions or reproducibility issues, open a [GitHub issue](https://github.com/FlockOfBird/GGDNIR/issues) or contact Ali Forooghi at [foroogh@uwindsor.ca](mailto:foroogh@uwindsor.ca).
