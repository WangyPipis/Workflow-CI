import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# aktifkan autolog
mlflow.sklearn.autolog()

# load dataset
df = pd.read_csv("titanic_preprocessing.csv")

# pisahkan fitur dan target
X = df.drop("Survived", axis=1)
y = df["Survived"]

# split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# nested=True agar tidak bentrok dengan mlflow run .
with mlflow.start_run(nested=True):

    # model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    # training
    model.fit(X_train, y_train)

    # prediksi
    y_pred = model.predict(X_test)

    # evaluasi
    acc = accuracy_score(y_test, y_pred)

    # log metric manual
    mlflow.log_metric("accuracy", acc)

    print("Accuracy:", acc)