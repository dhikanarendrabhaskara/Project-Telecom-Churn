import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: 0.8145086390829605
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=SGDClassifier(alpha=0.001, eta0=0.01, fit_intercept=True, l1_ratio=0.25, learning_rate="constant", loss="hinge", penalty="elasticnet", power_t=100.0)),
    GradientBoostingClassifier(learning_rate=0.1, max_depth=2, max_features=0.2, min_samples_leaf=18, min_samples_split=20, n_estimators=100, subsample=0.4)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
