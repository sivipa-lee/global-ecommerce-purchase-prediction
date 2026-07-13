# Global E-Commerce Customer Purchase Prediction

An end-to-end machine learning project that predicts whether an e-commerce customer is likely to make a purchase. The work combines customer profiles, browsing sessions, transaction history, marketing campaigns, and geographic data to support more targeted customer engagement.

## Project objective

Build a binary-classification model that identifies customers with a higher likelihood of purchase. This can help marketing teams prioritize audiences, personalize outreach, and use campaign budget more efficiently.

## Dataset

The project uses the **Global E-Commerce Customer Purchase Prediction** competition dataset from Kaggle. It includes six related tables:

| Table | Purpose |
| --- | --- |
| `train` | Customer IDs and the purchase target |
| `customers` | Customer demographics, loyalty, and account information |
| `sessions` | Browsing behaviour and campaign exposure |
| `transactions` | Order history and transaction attributes |
| `marketing_campaigns` | Campaign type, timing, and budget |
| `geo_data` | Regional economic and digital-access indicators |

Data files are not included in this repository. Download them from the [Kaggle competition page](https://www.kaggle.com/competitions/global-e-commerce-customer-purchase-prediction/data) and place them in the same directory as the notebook before running it.

## Approach

1. **Clean and standardize data** — handle missing values, inconsistent categories, date fields, outliers, and noisy columns.
2. **Create customer-level features** — aggregate session and transaction data into behavioural, recency, spending, campaign, device, traffic-source, payment, and shipping features.
3. **Explore the data** — examine target balance and relationships between customer behaviour and purchase likelihood.
4. **Train and compare models** — evaluate CatBoost, LightGBM, XGBoost, an LSTM sequence model, and ensemble approaches using an 80/10/10 train-validation-test split.
5. **Tune and select features** — use Optuna tuning and backward feature elimination to refine tree-based models.

## Results

Tree-based models outperformed the LSTM approach. The strongest validation results were very close across CatBoost, LightGBM, and XGBoost:

| Model | Validation ROC-AUC |
| --- | ---: |
| CatBoost | 0.6224 |
| LightGBM | **0.6225** |
| XGBoost | 0.6218 |
| LSTM | 0.5976 |

The final CatBoost feature-selection run achieved a validation ROC-AUC of **0.6216** and a test ROC-AUC of **0.6191**. These results suggest that aggregated customer behaviour is more predictive than the event sequence alone for this dataset.

## Key takeaways

- Recent engagement, browsing behaviour, transactions, and campaign exposure can be combined into meaningful customer-level signals.
- Gradient-boosted tree models handled the tabular data more effectively than the LSTM sequence model.
- The project evaluates multiple modeling strategies rather than relying on a single algorithm.

## Tech stack

Python, pandas, NumPy, scikit-learn, CatBoost, LightGBM, XGBoost, TensorFlow/Keras, Optuna, Matplotlib, and Seaborn.

## How to run

```bash
pip install -r requirements.txt
jupyter notebook global_ecommerce_purchase_prediction.ipynb
```

## Repository structure

```text
.
├── global_ecommerce_purchase_prediction.ipynb  # Full analysis and modeling workflow
├── requirements.txt                            # Python dependencies
└── README.md                                   # Project overview
```

## Author

Sivipa Lee
