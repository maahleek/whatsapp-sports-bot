from bot import (
    _extract_player_records,
    _normalize_team_name,
    _prediction_probabilities,
    _prediction_strength,
    _score_projection,
    _remember_matchup,
    _last_matchup,
)


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


def test_team_name_normalization():
    assert _normalize_team_name("Arsenal FC") == "arsenal"
    assert _normalize_team_name("Liverpool F.C.") == "liverpool"


def test_prediction_strength_can_use_season_with_small_recent_sample():
    season = {
        "points_per_game": 2.4,
        "goal_diff_per_game": 0.8,
    }
    recent = {
        "played": 1,
        "points_per_game": 3.0,
        "goal_diff_per_game": 1.0,
    }
    strength, source = _prediction_strength(season, recent)
    assert strength > 0
    assert source == "season + recent form"


def test_poisson_score_projection_is_normalized():
    projection = _score_projection(
        (1.8, 1.0),
        (1.5, 1.2),
        first_is_home=True,
    )
    total = (
        projection["first_win"]
        + projection["draw"]
        + projection["second_win"]
    )
    assert abs(total - 1.0) < 1e-9
    assert len(projection["top_scores"]) == 3
    assert all(item[0] >= 0 and item[1] >= 0 for item in projection["top_scores"])


def test_matchup_context_for_followups():
    user_id = "test-user"
    _remember_matchup(user_id, "Arsenal", "Liverpool")
    assert _last_matchup(user_id) == ("Arsenal", "Liverpool")


if __name__ == "__main__":
    test_prediction_probabilities_sum_to_one()
    test_nested_player_search_shape()
    test_team_name_normalization()
    test_prediction_strength_can_use_season_with_small_recent_sample()
    test_poisson_score_projection_is_normalized()
    test_matchup_context_for_followups()
    print("API helper tests passed.")
