from train import train_model


def test_accuracy_is_good_enough():
    _, accuracy = train_model()
    assert accuracy >= 0.90, f"Accuracy too low: {accuracy}"


def test_prediction_output_is_valid():
    model, _ = train_model()
    prediction = model.predict([[5.1, 3.5, 1.4, 0.2]])
    assert prediction.shape == (1,)
    assert prediction[0] in [0, 1, 2]