"""
Simple Model Comparison
=======================

This example uses the ``iris`` dataset and performs binary classifications
using different models. At the end, it compares the performance of the models
using different scoring functions and performs a statistical test to assess
whether the difference in performance is significant.

.. include:: ../../links.inc
"""
# Authors: Nicolás Nieto <n.nieto@fz-juelich.de>
# License: AGPL
# %%
from seaborn import load_dataset
from julearn.models.xgb_cvearlystopping import XGBClassifierCVEarlyStopping
from julearn import run_cross_validation
from julearn.utils import configure_logging

###############################################################################
# Set the logging level to info to see extra information.
configure_logging(level="INFO")

###############################################################################
df_iris = load_dataset("iris")

###############################################################################
# The dataset has three kind of species. We will keep two to perform a binary
# classification.
df_iris = df_iris[df_iris["species"].isin(["versicolor", "virginica"])]

###############################################################################
# As features, we will use the sepal length, width and petal length.
# We will try to predict the species.

X = ["sepal_length", "sepal_width", "petal_length"]
y = "species"
model = XGBClassifierCVEarlyStopping()
scores = run_cross_validation(
    X=X,
    y=y,
    data=df_iris,
    model=model,
    problem_type="classification",
    preprocess="zscore",
)

print(scores["test_score"])
# %%
