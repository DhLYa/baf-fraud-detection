# Bank Account Fraud Detection

Fraud detection on the Base dataset of the Bank Account Fraud (BAF) suite. The project
covers EDA, preprocessing, model tuning, evaluation at a fixed false positive rate, a
fairness audit by customer age, and SHAP explanations.

**Headline result:** tuned XGBoost catches **58.8%** of fraud at a 5% false positive rate
(PR-AUC 0.223, ROC-AUC 0.899). Every model wrongly flags legitimate applicants aged 50+
about 3 times as often as younger ones. Removing `customer_age` narrows that gap (FPR ratio
0.32 to 0.43) for under 1 point of recall.

## The BAF dataset suite

BAF is a suite of 6 synthetic bank account opening fraud datasets, published by Feedzai at
NeurIPS 2022. It was built as a realistic benchmark for evaluating new and existing ML
methods on tabular data. It has:

- **Realistic data**: generated from a real-world fraud detection dataset and anonymised.
- **Heavy class imbalance**: about 1.1% of applications are fraudulent.
- **Temporal drift**: a `month` column (0 to 7) gives the time ordering, and the fraud rate
  changes over time.
- **Bias**: protected attributes are included, and the 5 variant datasets add controlled
  types of bias on top of Base.

This project uses the **Base** dataset only.

| | |
|---|---|
| Rows | 1,000,000 |
| Columns | 32: 30 features, plus the `fraud_bool` label and `month` |
| Feature types | 25 numeric, 5 string categoricals (`payment_type`, `employment_status`, `housing_status`, `source`, `device_os`) |
| Fraud cases | 11,029 (1.10%) |
| Protected attributes | `customer_age`, `income`, `employment_status` |
| Paper | [Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets for ML Evaluation](https://arxiv.org/abs/2211.13358) |
| Repository | [feedzai/bank-account-fraud](https://github.com/feedzai/bank-account-fraud) |

### Features

| Group | Columns |
|---|---|
| Applicant | `income`, `customer_age`, `employment_status`, `housing_status`, `credit_risk_score`, `proposed_credit_limit` |
| Address and banking history | `prev_address_months_count`, `current_address_months_count`, `bank_months_count`, `has_other_cards` |
| Contact details | `name_email_similarity`, `email_is_free`, `phone_home_valid`, `phone_mobile_valid` |
| Application | `days_since_request`, `intended_balcon_amount`, `payment_type`, `foreign_request`, `source` |
| Velocity and aggregates | `zip_count_4w`, `velocity_6h`, `velocity_24h`, `velocity_4w`, `bank_branch_count_8w`, `date_of_birth_distinct_emails_4w` |
| Device and session | `device_os`, `session_length_in_minutes`, `keep_alive_session`, `device_distinct_emails_8w`, `device_fraud_count` |
| Time | `month` |

`customer_age` is recorded in decades (10 to 90) and `income` in bands (0.1 to 0.9).
`device_fraud_count` is 0 in every row of Base.

### Missing values

There are no nulls. Missing values are encoded as negative numbers instead:

| Column | Missing code | Rows missing |
|---|---|---|
| `intended_balcon_amount` | any negative value | 742,523 (74.25%) |
| `prev_address_months_count` | -1 | 712,920 (71.29%) |
| `bank_months_count` | -1 | 253,635 (25.36%) |
| `current_address_months_count` | -1 | 4,254 (0.43%) |
| `session_length_in_minutes` | -1 | 2,015 (0.20%) |
| `device_distinct_emails_8w` | -1 | 359 (0.04%) |

The negative values in `credit_risk_score` (14,445 rows, minimum -170) and `velocity_6h`
(44 rows) are genuine values, not missing codes.

### Sample rows

Four rows from `Base.csv`, transposed, with values rounded. The last row is the first
fraudulent row in the file.

| Column | Row 1 | Row 2 | Row 3 | Row 44 |
|---|---|---|---|---|
| fraud_bool | 0 | 0 | 0 | **1** |
| income | 0.3 | 0.8 | 0.8 | 0.9 |
| name_email_similarity | 0.987 | 0.617 | 0.997 | 0.571 |
| prev_address_months_count | -1 | -1 | 9 | -1 |
| current_address_months_count | 25 | 89 | 14 | 157 |
| customer_age | 40 | 20 | 40 | 60 |
| days_since_request | 0.007 | 0.010 | 0.012 | 0.006 |
| intended_balcon_amount | 102.45 | -0.85 | -1.49 | 35.34 |
| payment_type | AA | AD | AB | AA |
| zip_count_4w | 1059 | 1658 | 1095 | 4079 |
| velocity_6h | 13096 | 9223 | 4471 | 8139 |
| velocity_24h | 7851 | 5745 | 5472 | 4312 |
| velocity_4w | 6742 | 5942 | 5993 | 6341 |
| bank_branch_count_8w | 5 | 3 | 15 | 2 |
| date_of_birth_distinct_emails_4w | 5 | 18 | 11 | 17 |
| employment_status | CB | CA | CA | CA |
| credit_risk_score | 163 | 154 | 89 | 131 |
| email_is_free | 1 | 1 | 1 | 1 |
| housing_status | BC | BC | BC | BA |
| phone_home_valid | 0 | 1 | 0 | 1 |
| phone_mobile_valid | 1 | 1 | 1 | 1 |
| bank_months_count | 9 | 2 | 30 | 30 |
| has_other_cards | 0 | 0 | 0 | 0 |
| proposed_credit_limit | 1500 | 1500 | 200 | 1500 |
| foreign_request | 0 | 0 | 0 | 0 |
| source | INTERNET | INTERNET | INTERNET | INTERNET |
| session_length_in_minutes | 16.22 | 3.36 | 22.73 | 4.04 |
| device_os | linux | other | windows | windows |
| keep_alive_session | 1 | 1 | 0 | 1 |
| device_distinct_emails_8w | 1 | 1 | 1 | 1 |
| device_fraud_count | 0 | 0 | 0 | 0 |
| month | 0 | 0 | 0 | 0 |

## Getting the data

The data is not included in this repository. To get it:

1. Download the dataset from
   [Kaggle](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022).
   You need a free Kaggle account. With the Kaggle API set up, you can run:
   ```bash
   kaggle datasets download -d sgpjesus/bank-account-fraud-dataset-neurips-2022
   ```
2. Unzip it and place `Base.csv` at `data/Base.csv`.

The download contains all 6 datasets. Only `Base.csv` is needed here.

## Evaluation approach

### Temporal split

The data is split by `month` to mimic deployment, where a model trained on the past scores
future applications:

| Split | Months | Rows | Fraud cases | Fraud rate |
|---|---|---|---|---|
| Train | 0 to 6 | 903,157 | 9,601 | 1.06% |
| Validation | 7 | 96,843 | 1,428 | 1.47% |

The fraud rate drifts from month to month:

| Month | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| Rows | 132,440 | 127,620 | 136,979 | 150,936 | 127,691 | 119,323 | 108,168 | 96,843 |
| Fraud rate | 1.13% | 0.94% | 0.88% | 0.92% | 1.14% | 1.18% | 1.34% | 1.47% |

Month 7 has the highest fraud rate of any month, so validation is harder than a random split
would be.

Month 7 is used for tuning, for XGBoost early stopping, for setting the threshold and for
reporting results. There is no separate holdout, so the reported figures are optimistically
biased. Re-evaluating on an untouched holdout is the first item under future work.

### Performance: recall at 5% FPR

Accuracy is not useful here. A model that predicts "not fraud" for everyone scores about 99%
accuracy and catches nothing. Following the BAF paper, the main metric is recall at 5%
false positive rate: the share of fraud caught when 5% of legitimate applicants are
flagged. This matches how a bank works, with a fixed review capacity and a cap on how many
genuine customers can be inconvenienced. PR-AUC and ROC-AUC are reported as secondary
metrics.

The decision threshold is set on month 7 so that 5% of its legitimate applicants are
flagged.

### Fairness: predictive equality

Fairness is measured as **predictive equality**: whether legitimate applicants in different
groups are wrongly flagged at the same rate. It is reported as the FPR ratio:

```
FPR ratio = min(group FPRs) / max(group FPRs)
```

A value of 1 means equal false positive rates. It is reported two ways:

- **Under 50 vs 50 and over** (`customer_age` < 50 vs >= 50), the binary split used in the
  BAF paper. Because ages are recorded in decades, the 50s fall in the older group.
- **All ages**: the FPR in each decade bucket, then the lowest divided by the highest. The
  90s bucket has only 3 validation rows, so it is merged into 80+. Small buckets have noisy
  FPRs, so `04` plots each bucket's FPR with a 95% Wilson interval.

## Notebooks

| Notebook | Contents |
|---|---|
| `01_eda.ipynb` | Missing-value codes, feature types, fraud rate by month, feature distributions by class, columns to drop, income bands |
| `02_preprocessing_and_baselines.ipynb` | Missing-value flags, temporal split, dropped columns, the XGBoost and logistic regression feature sets, saved to `data/preprocessed/` |
| `03_hyperparameter_tuning.ipynb` | Baseline models, Optuna random search for both models on month 7, optional MLflow logging, saved models |
| `04_evaluation_and_fairness.ipynb` | Baseline vs tuned comparison, retraining without `customer_age`, FPR by age bucket |
| `05_explainability.ipynb` | Global SHAP (mean \|SHAP\| and beeswarm plots) and local waterfall plots for 4 cases per model |

Shared code (paths, the split, feature engineering, metrics, plotting and MLflow logging)
is in the `baf` package under `src/baf/`, which every notebook imports.

## Preprocessing

| Step | Applied to | Models | Reason |
|---|---|---|---|
| Replace missing codes with NaN and add a `<col>_missing_flag` | The 6 columns above | Both | Keeps "missing" as a signal without treating -1 as a real value |
| Drop | `device_fraud_count`, `session_length_in_minutes_missing_flag`, `device_distinct_emails_8w_missing_flag` | Both | Constant, or flags covering few rows with fraud rates close to the base rate |
| Encode as 0/1 | `source` (INTERNET / TELEAPP) | Both | Only two levels |
| Native categoricals and NaN handling | `payment_type`, `employment_status`, `housing_status`, `device_os` | XGBoost | Trees need no scaling, imputation or encoding |
| Merge rare levels | `payment_type` AE, `employment_status` CG, `housing_status` BF and BG | Logistic regression | Each under 0.2% of rows, merged into the level with the closest fraud rate |
| Band | `proposed_credit_limit` into 5 bands | Logistic regression | Fraud jumps from about 2% to 12-17% above 1,500 |
| Add flag | `income_top_decile` (income >= 0.9) | Logistic regression | Fraud rate nearly doubles in the top income band |
| Impute the median | Address, bank, balcon and session length columns | Logistic regression | `device_distinct_emails_8w` is filled with its mode instead |
| Log transform | `days_since_request`, `zip_count_4w`, `bank_branch_count_8w`, `session_length_in_minutes` | Logistic regression | Long right tails would dominate a single slope |
| One-hot encode | The 4 categoricals, credit limit bands, `device_distinct_emails_8w` | Logistic regression | Most common level is the reference |
| Standardise | All numeric features | Logistic regression | Puts coefficients on a common scale |

Imputation values, the income cutoff and scaling statistics come from the training months
only. The XGBoost set has 33 columns and the logistic regression set has 50.

## Results

All figures are on month 7, at the 5% FPR threshold set on that month. See the note under
[Temporal split](#temporal-split) on optimistic bias.

### Baseline vs tuned

| Model | Recall @ 5% FPR | Precision | PR-AUC | ROC-AUC | FPR < 50 | FPR >= 50 | FPR ratio (age) |
|---|---|---|---|---|---|---|---|
| Logistic regression (baseline) | 54.7% | 14.1% | 0.205 | 0.892 | 3.9% | 12.0% | 0.33 |
| Logistic regression (tuned) | 55.3% | 14.2% | 0.206 | 0.891 | 4.0% | 11.9% | 0.33 |
| XGBoost (baseline) | 50.8% | 13.2% | 0.177 | 0.876 | 3.9% | 12.8% | 0.30 |
| XGBoost (tuned) | **58.8%** | **15.0%** | **0.223** | **0.899** | 3.9% | 12.2% | 0.32 |

- Baselines use `class_weight='balanced'` (logistic regression) and `scale_pos_weight=93.1`,
  the non-fraud to fraud ratio (XGBoost).
- Each model was tuned with 50 trials of seeded Optuna random search, maximising recall at
  5% FPR.
  - Tuned logistic regression (trial 12): elastic net with `C=7.84`, `l1_ratio=0.85`, no
    class weights.
  - Tuned XGBoost (trial 20): `max_depth=6`, `learning_rate=0.029`, `subsample=0.5`,
    `colsample_bytree=0.6`, `min_child_weight=20.1`, `scale_pos_weight=2.24`, with 530
    trees after early stopping.
- Tuning lifted XGBoost recall by 8 points. Logistic regression moved by under 1 point,
  so it was already close to what a linear model gets from these features.

### With and without age

Both tuned models were retrained without `customer_age`, keeping their tuned
hyperparameters.

| Model | Features | Recall @ 5% FPR | PR-AUC | FPR < 50 | FPR >= 50 | FPR ratio (age) | FPR ratio (all ages) |
|---|---|---|---|---|---|---|---|
| Logistic regression (tuned) | All features | 55.3% | 0.206 | 4.0% | 11.9% | 0.33 | 0.036 |
| Logistic regression (tuned) | `customer_age` removed | 54.3% | 0.202 | 4.3% | 9.8% | 0.44 | 0.137 |
| XGBoost (tuned) | All features | 58.8% | 0.223 | 3.9% | 12.2% | 0.32 | 0.051 |
| XGBoost (tuned) | `customer_age` removed | 58.2% | 0.218 | 4.3% | 9.9% | 0.43 | 0.103 |

Removing a protected attribute ("fairness through unawareness") does not guarantee equal
error rates, because other features can act as proxies for age. This comparison measures
how much of the disparity removing age actually fixes, and what it costs in recall:

- Recall drops by under 1 point for both models.
- The FPR for applicants aged 50+ falls from about 12% to about 10%, while under-50s rise
  slightly from about 4.0% to 4.3%. Older applicants are still wrongly flagged more than
  twice as often.
- FPR rises steadily with age. With age included, tuned XGBoost flags 1.4% of legitimate
  applicants in their teens but 28% of those aged 80+; without age, that becomes 1.8% and
  17%. The 80+ bucket has only 71 legitimate validation applicants, so its FPR is noisy.

### Explainability (SHAP)

SHAP values are in log-odds and computed for all 96,843 validation applicants. The 22
one-hot columns of the logistic regression are summed back into their source features, so
both models are explained on the original features.

From the tuned logistic regression:

- The strongest signals are `prev_address_months_count` being missing (mean |SHAP| 0.67),
  `phone_home_valid` (0.45), `device_os` (0.44), `has_other_cards` (0.38) and
  `housing_status` (0.32).
- **Signals for fraud**: low `name_email_similarity`, a free email domain, high
  `credit_risk_score`, older `customer_age`, a Windows device, and one `housing_status`
  level (BA, which has a 3.75% fraud rate in the raw data).
- **Signals for legitimate**: a valid home phone, having other cards, a kept-alive session,
  and a recorded previous address.
- `customer_age` ranks 12th of 34 features (0.17). Its SHAP value rises steadily from -0.44
  log-odds for teenagers to +0.85 at 80, which links the explanation to the fairness result
  above.

From the tuned XGBoost:

- The strongest signals are `device_os` (0.49), `housing_status` (0.44), `phone_home_valid`
  (0.35), `current_address_months_count` (0.30), `keep_alive_session` (0.29) and
  `prev_address_months_count` (0.28).
- Both models agree on most directions: Windows devices, `housing_status` BA, a free email
  domain and low `name_email_similarity` push towards fraud, while a valid home phone, other
  cards and a kept-alive session push away from it.
- The biggest difference is address history. XGBoost uses the month counts directly and
  barely uses the `prev_address_months_count` missing flag (0.06, 24th of 33), because trees
  handle NaN natively.
- `customer_age` ranks 15th of 33 (0.12). Its effect is close to a step: near zero up to the
  40s, then +0.15 at 50 and about +0.6 from 70 up.
- `income` matters more for XGBoost (0.23 vs 0.13) and acts mostly through the top band
  (+0.37 log-odds at 0.9, against 0.01 or less below it).

The XGBoost beeswarm can be compared with the logistic regression one. In a linear model,
each feature's SHAP value depends only on that feature's own value. Vertical spread at a
fixed value in the XGBoost plot therefore shows interactions the tree model learned.

## Repository layout

```
notebooks/          01 to 05, run in order
src/baf/            shared package: config, data, features, metrics, plotting, tracking
pyproject.toml      makes src/baf installable with pip install -e .
data/               Base.csv goes here; preprocessed/ is written by 02 (not committed)
models/             saved models and tuning_summary.json, written by 03 (not committed)
results/shap/       cached SHAP values, written by 05 (not committed)
mlflow.db, mlartifacts/   local MLflow tracking data (not committed)
```

## Setup

The notebooks were run with Python 3.11 and pandas 2.3, NumPy 2.4, scikit-learn 1.9,
XGBoost 3.2, Optuna 4.9, MLflow 3.13, SHAP 0.51, matplotlib 3.11 and pyarrow 23.

From the repository root, install the shared package in editable mode:

```bash
pip install -e .
```

Place `Base.csv` in `data/`, then run the notebooks in order. Each notebook reads what the
previous one wrote.

`03_hyperparameter_tuning` and `04_evaluation_and_fairness` train XGBoost on a GPU
(`XGB_DEVICE = "cuda"`). Set `XGB_DEVICE = "cpu"` if you don't have one. With an RTX 4070
SUPER, `03` takes about 15 minutes and the other notebooks take under 2 minutes each.

`05_explainability` caches SHAP values in `results/shap/` and only recomputes them when
the model file changes.

### Experiment tracking with MLflow

Parameters, metrics, plots and models from `03` and `04` can be logged to a local MLflow
server, under the experiment "BAF account creation fraud detection". Start the server from
the repository root:

```bash
mlflow server --host 127.0.0.1 --port 5000
```

Then open `http://127.0.0.1:5000`.

Tracking is off by default. To turn it on, set `LOG_TO_MLFLOW = True` at the top of `03`
and `04`. With it off, results print inline and no server is needed.

## Limitations

- Only the Base dataset is used. The 5 variants, which add controlled bias, are not
  evaluated.
- The data is synthetic. It is designed to be realistic, but results may not carry over
  to a specific bank's data.
- Fairness is checked for age only. Income and employment status are also protected but weren't analysed
  attributes in the paper.
- A single threshold, set at 5% FPR, is evaluated. A real deployment would pick the
  threshold from review capacity and the cost of fraud.
- Tuning, early stopping, threshold setting and reporting all use month 7, so the reported
  performance is optimistically biased.

## Future work

- Re-compare all models on an untouched holdout to remove the optimistic bias. For example:
  train on months 0 to 5, tune and set the threshold on month 6, and report once on month 7.
- Add more models, including approaches beyond supervised classification, such as
  autoencoder anomaly detection trained on legitimate applications only, using
  reconstruction error as the fraud score.
- Evaluate on the 5 BAF variants to see how bias in the training data changes the
  predictive equality results.
- Apply fairness mitigations, such as group-specific thresholds or reweighting, and
  measure the trade-off between recall and FPR ratio.
- Monitor drift month by month and test retraining strategies.
- Add a cost-based threshold that accounts for fraud losses and review costs.
- Compare with LightGBM and CatBoost, using their native categorical handling.

## Citation

```bibtex
@article{jesusTurningTablesBiased2022,
  title={Turning the {{Tables}}: {{Biased}}, {{Imbalanced}}, {{Dynamic Tabular Datasets}} for {{ML Evaluation}}},
  author={Jesus, S{\'e}rgio and Pombal, Jos{\'e} and Alves, Duarte and Cruz, Andr{\'e} and Saleiro, Pedro and Ribeiro, Rita P. and Gama, Jo{\~a}o and Bizarro, Pedro},
  journal={Advances in Neural Information Processing Systems},
  year={2022}
}
```

## Licence

The BAF datasets are released by Feedzai under their own licence. See the
[dataset repository](https://github.com/feedzai/bank-account-fraud) for terms before reusing
the data.
