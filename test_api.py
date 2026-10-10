from betting_platforms import get_platform, normalize_platform_name, platform_capability_summary
from bot import (
    _asian_handicap_probabilities,
    _asian_handicap_settlement,
    _betting_market_report,
    _concise_platform_prediction_report,
    _extract_player_records,
    _normalize_team_name,
    _prediction_probabilities,
    _prediction_strength,
    _score_projection,
    _total_line_settlement,
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


def test_betting_market_report_from_score_matrix():
    projection = _score_projection(
        (1.8, 1.0),
        (1.5, 1.2),
        first_is_home=True,
    )
    context = {
        "projection": projection,
        "fixture": {"home": "Arsenal", "away": "Liverpool", "date": "2026-11-01"},
        "first_name": "Arsenal",
        "second_name": "Liverpool",
        "first_is_home": True,
        "venue_note": "Fixture: Arsenal vs Liverpool on 2026-11-01.",
    }
    report = _betting_market_report(context, "all")
    assert "1X2 / Match result:" in report
    assert "Both teams to score:" in report
    assert "Total goals:" in report
    assert "Common Asian handicap lines:" in report
    assert "Most likely exact scores:" in report


def test_asian_handicap_probabilities_sum_to_one():
    projection = _score_projection(
        (1.7, 1.1),
        (1.4, 1.3),
        first_is_home=True,
    )
    win, push, loss = _asian_handicap_probabilities(
        projection["score_matrix"],
        -1.0,
        side="home",
    )
    assert abs((win + push + loss) - 1.0) < 1e-9


def test_quarter_asian_handicap_settlement_is_normalized():
    projection = _score_projection(
        (1.7, 1.1),
        (1.4, 1.3),
        first_is_home=True,
    )
    settlement = _asian_handicap_settlement(
        projection["score_matrix"],
        -0.75,
        side="home",
    )
    assert abs(sum(settlement.values()) - 1.0) < 1e-9
    assert settlement["half_win"] >= 0.0
    assert settlement["half_loss"] >= 0.0


def test_specific_total_line_settlement_is_normalized():
    projection = _score_projection(
        (1.7, 1.1),
        (1.4, 1.3),
        first_is_home=True,
    )
    settlement = _total_line_settlement(
        projection["score_matrix"],
        2.25,
        side="over",
    )
    assert abs(sum(settlement.values()) - 1.0) < 1e-9


def test_specific_total_request_is_concise():
    projection = _score_projection(
        (1.8, 1.0),
        (1.5, 1.2),
        first_is_home=True,
    )
    context = {
        "projection": projection,
        "fixture": {"home": "Arsenal", "away": "Liverpool", "date": "2026-11-01"},
        "first_name": "Arsenal",
        "second_name": "Liverpool",
        "first_is_home": True,
        "venue_note": "Fixture: Arsenal vs Liverpool on 2026-11-01.",
    }
    report = _betting_market_report(context, "Over 2.5")
    assert "Over 2.5" in report
    assert "Over 3.5" not in report


def test_platform_prediction_report_is_whatsapp_concise():
    projection = _score_projection(
        (1.8, 1.0),
        (1.5, 1.2),
        first_is_home=True,
    )
    context = {
        "projection": projection,
        "fixture": {"home": "Arsenal", "away": "Liverpool", "date": "2026-11-01"},
        "first_name": "Arsenal",
        "second_name": "Liverpool",
        "first_is_home": True,
        "venue_note": "Fixture: Arsenal vs Liverpool on 2026-11-01.",
    }
    platform = get_platform("SportyBet")
    assert platform is not None
    report = _concise_platform_prediction_report(context, platform)
    assert "Platform: SportyBet" in report
    assert "Model snapshot:" in report
    assert len(report) < 1500


def test_betting_platform_aliases():
    assert normalize_platform_name("SportyBet") == "sportybet"
    assert normalize_platform_name("sporty bet") == "sportybet"
    assert normalize_platform_name("Bet9ja") == "bet9ja"
    assert normalize_platform_name("1xBet") == "1xbet"


def test_platform_capability_is_honest_about_booking_codes():
    platform = get_platform("SportyBet")
    assert platform is not None
    summary = platform_capability_summary(platform)
    assert "supports booking/share codes" in summary
    assert "no public programmatic booking-code API is configured" in summary


if __name__ == "__main__":
    test_prediction_probabilities_sum_to_one()
    test_nested_player_search_shape()
    test_team_name_normalization()
    test_prediction_strength_can_use_season_with_small_recent_sample()
    test_poisson_score_projection_is_normalized()
    test_matchup_context_for_followups()
    test_betting_market_report_from_score_matrix()
    test_asian_handicap_probabilities_sum_to_one()
    test_quarter_asian_handicap_settlement_is_normalized()
    test_specific_total_line_settlement_is_normalized()
    test_specific_total_request_is_concise()
    test_platform_prediction_report_is_whatsapp_concise()
    test_betting_platform_aliases()
    test_platform_capability_is_honest_about_booking_codes()
    print("API helper tests passed.")
