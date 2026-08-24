# Freight Rate Prediction

## What this is

A regression pipeline that predicts the price of a truckload freight shipment from information about the load: pickup and delivery city, distance, equipment type, weight, date, and two market signal columns. Pricing freight loads automatically is a common problem in logistics, since manually quoting every shipment does not scale and tends to be inconsistent from one person to the next. This project builds, tests, and validates a model that does that pricing directly from the data.

## The data

* `train-test.csv`: 48,000 labeled loads used to train the model and validate it internally.
* `validation.csv`: 12,000 unlabeled loads to generate final predictions for.
* `december-chart-inputs.csv`: 31 rows, one for each day in a single month, with every field held fixed except the date. This isolates how the model's predicted rate moves with time alone, independent of route or load details.
* `validation_predictions.csv`: the final output, one predicted rate per load in `validation.csv`.
* `score.py`: checks that the output files are in the correct format and builds a chart from the December predictions.

## Approach

1. Explore the data to understand the target variable and find data quality issues.
2. Clean the issues found: a sign error affecting some weight values, missing values in two columns, and a mismatch between the stated distance and the distance implied by pickup and delivery coordinates.
3. Engineer features, testing choices empirically instead of assuming they help. Pickup and delivery city identity, for example, was tested and then dropped after it made holdout accuracy worse rather than better.
4. Split the data by date rather than at random, training on earlier months and validating on the most recent ones, since the real task is forecasting into months the model has not seen.
5. Compare several models spanning linear, tree ensemble, boosted tree, and neural network families, and select the one that performs best on the holdout rather than defaulting to any one model by reputation.
6. Retrain the selected model on the full dataset and generate the final predictions.

## Result

A tuned gradient boosted tree model was selected after the full comparison. It reaches a mean absolute error of around $115 on the internal holdout, which sits close to the natural price variation found even between otherwise identical shipments, suggesting the model is close to the practical limit of what these features alone can predict.

## Running this

Install the dependencies:

```bash
pip install -r requirements.txt
```

Open `freight_rate_modeling.ipynb` and run every cell from top to bottom. This regenerates `validation_predictions.csv` and fills in `december-chart-inputs.csv`.

Then validate the outputs and build the chart:

```bash
python score.py --predictions validation_predictions.csv --december-predictions december-chart-inputs.csv
```

## Files in this repository

* `freight_rate_modeling.ipynb`: all of the code, from loading the data through the final predictions.
* `score.py`: output format validator and chart generator.
* `train-test.csv`, `validation.csv`, `december-chart-inputs.csv`: the data.
* `validation_predictions.csv`: the final predictions.
* `requirements.txt`: pinned dependencies.
