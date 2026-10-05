from bot import _prediction_probabilities


def test_prediction_probabilities_sum_to_one():
    a, draw, b = _prediction_probabilities(1.8, 1.2)
    total = a + draw + b
    assert abs(total - 1.0) < 1e-9
    assert all(0.0 <= value <= 1.0 for value in (a, draw, b))


if __name__ == "__main__":
    test_prediction_probabilities_sum_to_one()
    print("Prediction probability test passed.")
