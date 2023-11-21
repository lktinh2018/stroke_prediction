from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np


from global_vars import *

# load data
df = pd.read_csv(INPUT_CSV_PATH)
dataset = df.to_numpy()

# split data into X and y
X = dataset[:, :-1]
Y = dataset[:, -1]

# split data into train and test sets
seed = 7
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=TEST_RATIO, random_state=seed
)

# fit model no training data
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    # class_weight="balanced",
)
model.fit(X_train, y_train)

# make predictions for test data
y_pred = model.predict(X_test)

predictions = [round(value) for value in y_pred]

# Save to file
np_array = np.array(predictions)
np.save("Outputs_RandomForest.npy", np_array)

# Save gts
np_array = np.array(y_test)
np.save("Gts.npy", np_array)

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
