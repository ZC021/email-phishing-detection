# Email Phishing Detection Baseline

This repository has been reset with a minimal baseline implementation for detecting phishing emails.

## Features

- Extracts simple features from `.eml` files (subject length, body length, URL count, suspicious keywords).
- Trains a logistic regression model on a CSV dataset.
- Predicts whether a single email is phishing.

## Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Train a model:

```bash
python main.py --train data/sample_phishing_dataset.csv
```

Predict using a saved model:

```bash
python main.py --predict path/to/email.eml
```

This is a basic starting point. Additional functionality and a web UI can be built on top of this baseline.
