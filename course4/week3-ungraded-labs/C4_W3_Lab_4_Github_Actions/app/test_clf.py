import pickle
from main import clf


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def test_accuracy():

    # Load test data
    with open("data/test_data.pkl", "rb") as file:
        test_data = pickle.load(file)

    # Unpack the tuple
    X_test, y_test = test_data

    # Compute accuracy of classifier
    acc = clf.score(X_test, y_test)

    # Accuracy should be over 90%
    assert acc > 0.9

def test_pipeline_and_scaler():

    # Check if clf is an instance of sklearn.pipeline.Pipeline
    is_Pipeline = isinstance(clf, Pipeline)
    assert is_Pipeline, "The classifier must be an instance of sklearn.pipeline.Pipeline"

    if is_Pipeline:
        # Check if first step of pipeline is an instance of
        # sklearn.preprocessing.StandardScaler
        first_step = [v for v in clf.named_steps.values()][0]
        assert isinstance(first_step, StandardScaler)
