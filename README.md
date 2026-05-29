<p align="center">
  <img src="docs/umutextstats-logo.png" alt="UMUTextStats logo" width="500"/>
</p>

# UMUTextStats Python

Work in progress Python port of **UMUTextStats**, a linguistic feature extraction framework inspired by LIWC and stylometric analysis systems.

The project analyzes text datasets and extracts lexical, morphosyntactic, stylistic, pragmatic, psycholinguistic, social-media and error-related features from an XML configuration file.

---

# Status

This is an active migration/adaptation of UMUTextStats to Python.

Current features include:

- CSV input loading
- XML configuration loading
- Text preprocessing
- Cached pipeline stages
- Common feature computation
- Stanza POS/NER annotation
- Dictionary-based dimensions
- Pattern-based dimensions
- Composite dimensions
- Feature summarization
- Feature ranking
- Sparse/zero-only feature detection
- Aggregate statistics by metadata groups
- CSV/JSON output
- Optional profiling stats

---

# Installation

```bash
pip install -e .
```

---

# CLI Usage

## Analyze

Extract linguistic features from a dataset.

```bash
umutextstats analyze dataset.csv -t tweet -o features.csv
```

With profiling stats:

```bash
umutextstats analyze dataset.csv \
  -t tweet \
  -o features.csv \
  --stats stats.csv
```

Using a custom XML configuration:

```bash
umutextstats analyze dataset.csv \
  -t tweet \
  -c path/to/config.xml \
  -o features.csv
```

Disable cache:

```bash
umutextstats analyze dataset.csv \
  -t tweet \
  -o features.csv \
  --no-cache
```

---

## Summarize

Compute descriptive statistics from extracted features.

```bash
umutextstats summarize features.csv
```

Generate JSON summary:

```bash
umutextstats summarize features.csv \
  -o summary.json
```

Force long/tidy CSV format:

```bash
umutextstats summarize features.csv \
  --format long \
  -o summary.csv
```

Force nested JSON format:

```bash
umutextstats summarize features.csv \
  --format nested \
  -o summary.json
```

---

## Aggregate

Compute grouped statistics using metadata columns.

Example dataset:

| id | text | split | author |
|---|---|---|---|
| 1 | ... | train | alice |
| 2 | ... | test | bob |

Example features:

| id | feature_1 | feature_2 |
|---|---|---|
| 1 | ... | ... |
| 2 | ... | ... |

Aggregate by split:

```bash
umutextstats aggregate \
  features.csv \
  --metadata dataset.csv \
  --group-by split \
  --id-column id \
  -o aggregate_by_split.json
```

Aggregate by author:

```bash
umutextstats aggregate \
  features.csv \
  --metadata dataset.csv \
  --group-by author \
  --id-column id \
  -o aggregate_by_author.json
```

### Inspecting dimensions

Use `inspect` to understand why a dimension matched a text. The command highlights the elements that contributed to the score and is useful for debugging dictionaries, regex patterns, and linguistic rules.

Inspect hashtags:

```bash
echo "Me encanta #Python y #NLP" | umutextstats inspect digital-addressivity-hashtags -
```

Use a custom configuration
```
umutextstats inspect \
    digital-platform-language-jargon \
    tweet.txt \
    --config configs/social_media.yaml
```

---

# Output Formats

Supported output formats:

```text
.csv
.json
```

Outputs include:

- Feature matrices
- Dataset summaries
- Feature rankings
- Aggregate/group statistics


---
## UNIX Pipes and Standard Input
UMUTextStats supports UNIX pipes and standard streams, making it easy to integrate into shell workflows.

### Analyze from standard input

```bash
echo 'tweet
hola mundo
esto es una prueba' \
| umutextstats analyze - -t tweet --only stylometry --no-stanza
```

### Analyze and summarize in a single pipeline

```bash
cat dataset.csv \
| umutextstats analyze - -t tweet \
    --only stylometry \
    --head 100 \
    --no-stanza \
| umutextstats summarize - \
    --rank-by mean \
    --top 10
```

### Combine multiple feature families

```bash
cat dataset.csv \
| umutextstats analyze - -t tweet \
    --only 'stylometry|errors|register' \
| umutextstats summarize - \
    --rank-by nonzero_ratio
```

### Save intermediate features while still using pipes

```bash
cat dataset.csv \
| umutextstats analyze - -t tweet -o features.csv

umutextstats summarize features.csv \
    --rank-by mean \
    --top 20
```

### Quick exploratory analysis

```bash
cat dataset.csv \
| umutextstats analyze - -t tweet \
    --head 20 \
    --only psycholinguistic-processes \
| umutextstats summarize - \
    --sparse-threshold 0.01
```

---

## Cache Management

`umutextstats` includes a local cache system to avoid recomputing
expensive NLP pipelines, embeddings, and feature extraction steps.

The cache is automatically invalidated when relevant processing
parameters change (for example processors, GPU usage, or batch sizes).

---

### Show cache information

Display cache directory, total files, and size:

```bash
umutextstats cache info
```

Example output:

```text
Cache dir: .umutextstats_cache
Exists: True
Files: 182
Size: 4.2 GB
```

---

### List cache files

Show cached files ordered by most recent usage:

```bash
umutextstats cache list
```

Limit the number of displayed files:

```bash
umutextstats cache list --limit 100
```

Example output:

```text
2025-08-12 11:30:42      12.4 MB  stanza/abc123.pkl
2025-08-12 11:28:01      84.1 MB  embeddings/def456.parquet
```

---

### Prune old cache files

Delete cache files older than a given number of days:

```bash
umutextstats cache prune --older-than-days 30
```

Skip confirmation:

```bash
umutextstats cache prune --older-than-days 30 --yes
```

---

### Clear the entire cache

Delete the full cache directory:

```bash
umutextstats cache clear
```

Skip confirmation:

```bash
umutextstats cache clear --yes
```

---

### Custom cache directory

All cache commands support a custom cache directory:

```bash
umutextstats cache info --cache-dir .cache_dev
```

---

---

# Configuration System

UMUTextStats uses a hierarchical configuration system to define linguistic dimensions, feature groups, dictionaries and extraction strategies.

The default configuration is distributed as a YAML file, while legacy XML configurations are still supported for compatibility.

---

## Convert configuration formats

Convert XML configurations to YAML:

```bash
umutextstats config convert \
    src/umutextstats/resources/config/default.xml \
    default.yaml
```

---

## Validate configuration files

Validate a configuration file and detect common issues:

```bash
umutextstats config validate default.yaml
```

Validation checks currently include:

- Duplicated dimension keys
- Unknown dimension classes
- Unknown composite strategies
- Invalid ratio definitions
- Missing required parameters

Example output:

```text
[error] stylometry-words: Duplicated dimension key.
[error] readability: Unknown class: UnknownDimension
```

---

## Inspect the configuration tree

Print the hierarchical dimension tree:

```bash
umutextstats config tree
```

Limit the displayed depth:

```bash
umutextstats config tree --max-depth 2
```

Example output:

```text
└── morphosyntax
    ├── morphosyntax-gender
    │   ├── morphosyntax-gender-feminine
    │   └── morphosyntax-gender-masculine
    └── morphosyntax-number
```

---

## Custom configurations

Use a custom configuration file during analysis:

```bash
umutextstats analyze dataset.csv \
    -t tweet \
    -c custom_config.yaml \
    -o features.csv
```

Both YAML and legacy XML configuration files are supported.

---



### Notes

- Cache keys are parameter-aware.
- Different NLP processor configurations generate different cache entries.
- Cache cleanup is safe and only removes cached artifacts.
- Recommended for large NLP pipelines and embedding workflows.

---

# Feature Analysis

UMUTextStats includes utilities for:

- Descriptive statistics
- Feature ranking
- Sparse feature detection
- Zero-only feature detection

Future analysis modules may include:

- Mutual Information
- Information Gain
- Effect Size
- Discriminative feature analysis
- Stylometric profiling

---

# Development

Run tests with:

```bash
pytest
```

---

# Notes

This project is still under development.

Results, APIs and internal feature implementations may change while the Python version is being aligned with the original UMUTextStats behavior.


---


### Acknowledgements
This work is part of the research project AIInFunds (PDC2021-121112-I00), funded by MCIN/AEI/10.13039/501100011033 and by the European Union NextGenerationEU/PRTR. This work is also part of the research project LaTe4PSP (PID2019-107652RB-I00), funded by MCIN/AEI/10.13039/501100011033. Besides, it was partially supported by Fundacion Séneca -the Regional Agency for Science and Technology of Murcia (Spain)- through project 20963/PI/18. In addition, José Antonio García-Díaz is supported by Banco Santander and the University of Murcia through the Doctorado Industrial programme

### Citation
```
@inproceedings{garcia2022umutextstats,
  title={Umutextstats: A linguistic feature extraction tool for spanish},
  author={Garc{\'\i}a-D{\'\i}az, Jos{\'e} Antonio and Vivancos-Vicente, Pedro Jos{\'e} and Almela, Angela and Valencia-Garc{\'\i}a, Rafael},
  booktitle={Proceedings of the thirteenth language resources and evaluation conference},
  pages={6035--6044},
  year={2022}
}
```