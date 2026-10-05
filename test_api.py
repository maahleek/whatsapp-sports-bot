from bot import _extract_player_records, _prediction_probabilities


def test_prediction_probabilities_sum_to_one():
    a, draw, b = _prediction_probabilities(1.8, 1.2)
    total = a + draw + b
    assert abs(total - 1.0) < 1e-9
    assert all(0.0 <= value <= 1.0 for value in (a, draw, b))


def test_nested_player_search_shape():
    payload = {
        "response": {
            "suggestions": [
                {"name": "Erling Haaland", "teamName": "Manchester City"},
                {"name": "Another Player", "teamName": "Another Club"},
            ]
        }
    }
    players = _extract_player_records(payload)
    assert len(players) == 2
    assert players[0]["name"] == "Erling Haaland"


if __name__ == "__main__":
    test_prediction_probabilities_sum_to_one()
    test_nested_player_search_shape()
    print("API helper tests passed.")
