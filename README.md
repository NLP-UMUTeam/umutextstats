# UMUTextStats Python

Work in progress Python port of **UMUTextStats**, a linguistic feature extraction tool similar to LIWC. 

The project analyzes text datasets and generates lexical, morphosyntactic, stylistic, pragmatic, social-media and error-related features from an XML configuration file.

## Status

This is an active migration/adaptation of UMUTextStats to Python.

Current features include:

- CSV input loading
- XML configuration loading
- Text preprocessing
- Cached pipeline stages
- Stanza POS/NER annotation
- Common feature cache
- Dictionary-based dimensions
- Pattern-based dimensions
- Composite dimensions
- CSV/JSON output
- Optional profiling stats


## Installation

```bash
pip install -e .
```

## CLI Usage
### Analyze
```
umutextstats analyze dataset.csv -t tweet -o features.csv
```

With profiling stats:
```
umutextstats analyze dataset.csv -t tweet -o features.csv --stats stats.csv
```

Using a custom XML configuration:
```
umutextstats analyze dataset.csv -t tweet -c path/to/config.xml -o features.csv
```

Disable cache
```
umutextstats analyze dataset.csv -t tweet -o features.csv --no-cache
```


By default, intermediate stages are cached in:
```
.cache/
```

Cached stages include preprocessing, Stanza annotations and common computed features.

### Output
Supported output formats:
```
.csv
.json
```

Example:
```
umutextstats analyze dataset.csv \
  --text-column tweet \
  --output features.csv \
  --stats stats.csv
```


## Development
Run tests with:
```
pytest
```


## Notes
This project is still under development. Results and APIs may change while the Python version is being aligned with the original UMUTextStats behavior.