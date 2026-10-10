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
    _sportybet_selection_model_support,
    _sportybet_context_has_enough_data,
    _sportybet_platform_probability,
    _format_sportybet_analysis_summary,
    _format_sportybet_cached_view,
    _parse_sportybet_booking_request,
    _resolve_sportybet_fixture,
    _sportybet_fixture_suggestions,
    _resolve_sportybet_pick,
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


def test_sportybet_selection_model_support_for_1x2():
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
    probability, label = _sportybet_selection_model_support(
        context,
        {
            "market_id": "1",
            "market_name": "1X2",
            "outcome_id": "1",
            "outcome_name": "Home",
            "specifier": "",
        },
    )
    assert probability is not None
    assert 0.0 <= probability <= 1.0
    assert label == "1X2 home"


def test_sportybet_selection_model_support_for_total():
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
    probability, label = _sportybet_selection_model_support(
        context,
        {
            "market_id": "18",
            "market_name": "Over/Under",
            "outcome_id": "12",
            "outcome_name": "Over",
            "specifier": "total=2.5",
        },
    )
    assert probability is not None
    assert 0.0 <= probability <= 1.0
    assert label == "Over 2.5 Goals"


def test_sportybet_corner_total_is_not_treated_as_goals():
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
    probability, label = _sportybet_selection_model_support(
        context,
        {
            "market_id": "166",
            "market_name": "Corners - Over/Under",
            "outcome_id": "12",
            "outcome_name": "Over 8.5",
            "specifier": "total=8.5",
        },
    )
    assert probability is None
    assert label == "Corners - Over/Under"


def test_sportybet_text_double_chance_home_or_away():
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
    probability, label = _sportybet_selection_model_support(
        context,
        {
            "market_id": "10",
            "market_name": "Double Chance",
            "outcome_id": "",
            "outcome_name": "Home or Away",
            "specifier": "",
        },
    )
    assert probability is not None
    assert 0.0 <= probability <= 1.0
    assert label == "Double Chance 12"


def test_sportybet_team_total_is_modelled_as_team_goals():
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
    probability, label = _sportybet_selection_model_support(
        context,
        {
            "market_id": "24",
            "market_name": "Liverpool Over/Under",
            "outcome_id": "",
            "outcome_name": "Over 0.5",
            "specifier": "",
        },
    )
    assert probability is not None
    assert 0.0 <= probability <= 1.0
    assert label == "Liverpool Over 0.5 Goals"


def test_sportybet_context_rejects_tiny_recent_sample():
    context = {
        "first_season": None,
        "second_season": None,
        "first_recent": {"played": 1},
        "second_recent": {"played": 5},
    }
    assert _sportybet_context_has_enough_data(context) is False


def test_sportybet_context_accepts_season_data():
    context = {
        "first_season": {"played": 8},
        "second_season": {"played": 8},
        "first_recent": None,
        "second_recent": None,
    }
    assert _sportybet_context_has_enough_data(context) is True


def test_sportybet_platform_probability_prefers_feed_value():
    probability, label = _sportybet_platform_probability(
        {"source_probability": 0.72, "odds": 1.40}
    )
    assert probability == 0.72
    assert label == "SportyBet feed probability"


def test_sportybet_platform_probability_falls_back_to_odds():
    probability, label = _sportybet_platform_probability(
        {"source_probability": None, "odds": 2.0}
    )
    assert probability == 0.5
    assert label == "Raw odds-implied chance"


def _sample_cached_sportybet_analysis():
    return {
        "code": "ABC123",
        "selection_count": 3,
        "counts": {
            "higher": 1,
            "moderate": 0,
            "lower": 1,
            "unmodelled": 1,
        },
        "records": [
            {
                "index": 1,
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "market_name": "Over/Under",
                "outcome_name": "Over 2.5",
                "odds": 1.50,
                "platform_probability": 0.67,
                "platform_probability_label": "SportyBet feed probability",
                "model_probability": 0.72,
                "model_label": "Over 2.5 Goals",
                "alignment": "higher model support",
                "gap": 5.0,
            },
            {
                "index": 2,
                "home_team": "Liverpool",
                "away_team": "Everton",
                "market_name": "1X2",
                "outcome_name": "Home",
                "odds": 1.80,
                "platform_probability": 0.56,
                "platform_probability_label": "Raw odds-implied chance",
                "model_probability": 0.44,
                "model_label": "1X2 home",
                "alignment": "lower model support",
                "gap": -12.0,
            },
            {
                "index": 3,
                "home_team": "Man Utd",
                "away_team": "Tottenham",
                "market_name": "Corners - Over/Under",
                "outcome_name": "Over 8.5",
                "odds": 1.36,
                "platform_probability": 0.74,
                "platform_probability_label": "SportyBet feed probability",
                "model_probability": None,
                "model_label": "Corners - Over/Under",
                "alignment": None,
                "gap": None,
                "model_reason": "unsupported_market",
            },
        ],
    }


def test_sportybet_default_summary_is_compact():
    summary = _format_sportybet_analysis_summary(
        _sample_cached_sportybet_analysis()
    )
    assert "SportyBet Code: ABC123" in summary
    assert "SHOW MODELLED" in summary
    assert "SHOW UNMODELLED" in summary
    assert "SHOW ALL" in summary
    assert "Unsupported market types:" in summary
    assert "Corners - Over/Under" in summary
    assert len(summary) < 1500


def test_sportybet_cached_views_filter_without_reanalysis():
    analysis = _sample_cached_sportybet_analysis()
    modelled = _format_sportybet_cached_view(analysis, "modelled")
    unmodelled = _format_sportybet_cached_view(analysis, "unmodelled")

    assert "Selections with independent model comparison: 2" in modelled
    assert "Arsenal vs Chelsea" in modelled
    assert "Man Utd vs Tottenham" not in modelled

    assert "Selections without independent model comparison: 1" in unmodelled
    assert "Man Utd vs Tottenham" in unmodelled
    assert "Loaded from the cached analysis" in unmodelled


def _sample_sportybet_fixture_for_creation():
    return {
        "event_id": "sr:match:123",
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "league": "Premier League",
        "category": "England",
        "start_ms": 9999999999999,
        "match_status": "Not start",
        "markets": [
            {
                "market_id": "1",
                "market_name": "1X2",
                "specifier": None,
                "status": 0,
                "outcomes": [
                    {"outcome_id": "1", "outcome_name": "Home", "odds": 1.80, "is_active": True},
                    {"outcome_id": "2", "outcome_name": "Draw", "odds": 3.50, "is_active": True},
                    {"outcome_id": "3", "outcome_name": "Away", "odds": 4.20, "is_active": True},
                ],
            },
            {
                "market_id": "18",
                "market_name": "Over/Under",
                "specifier": "total=2.5",
                "status": 0,
                "outcomes": [
                    {"outcome_id": "12", "outcome_name": "Over 2.5", "odds": 1.70, "is_active": True},
                    {"outcome_id": "13", "outcome_name": "Under 2.5", "odds": 2.05, "is_active": True},
                ],
            },
            {
                "market_id": "10",
                "market_name": "Double Chance",
                "specifier": None,
                "status": 0,
                "outcomes": [
                    {"outcome_id": "4", "outcome_name": "Home or Draw", "odds": 1.20, "is_active": True},
                    {"outcome_id": "5", "outcome_name": "Draw or Away", "odds": 1.80, "is_active": True},
                    {"outcome_id": "6", "outcome_name": "Home or Away", "odds": 1.25, "is_active": True},
                ],
            },
            {
                "market_id": "29",
                "market_name": "GG/NG",
                "specifier": None,
                "status": 0,
                "outcomes": [
                    {"outcome_id": "74", "outcome_name": "GG", "odds": 1.75, "is_active": True},
                    {"outcome_id": "76", "outcome_name": "NG", "odds": 1.95, "is_active": True},
                ],
            },
        ],
    }


def test_parse_sportybet_booking_creation_request():
    parsed = _parse_sportybet_booking_request(
        "Create SportyBet code:\n"
        "Arsenal vs Chelsea | Over 2.5\n"
        "Liverpool vs Everton | Home"
    )
    assert len(parsed) == 2
    assert parsed[0]["team1"] == "Arsenal"
    assert parsed[0]["pick"] == "Over 2.5"


def test_sportybet_fixture_suggestions_for_unavailable_matchup():
    fixtures = [
        {
            "event_id": "1",
            "home_team": "Arsenal",
            "away_team": "Leeds United",
            "start_ms": 9999999999999,
            "match_status": "Not start",
            "markets": [],
        },
        {
            "event_id": "2",
            "home_team": "Chelsea",
            "away_team": "AFC Bournemouth",
            "start_ms": 9999999999999,
            "match_status": "Not start",
            "markets": [],
        },
    ]
    suggestions = _sportybet_fixture_suggestions(
        fixtures,
        "Arsenal",
        "Chelsea",
    )
    assert "Arsenal vs Leeds United" in suggestions
    assert "Chelsea vs AFC Bournemouth" in suggestions


def test_resolve_sportybet_fixture_for_creation():
    fixture = _sample_sportybet_fixture_for_creation()
    resolved = _resolve_sportybet_fixture(
        [fixture],
        "Arsenal FC",
        "Chelsea",
    )
    assert resolved["event_id"] == "sr:match:123"


def test_resolve_sportybet_over_under_for_creation():
    fixture = _sample_sportybet_fixture_for_creation()
    selection = _resolve_sportybet_pick(fixture, "Over 2.5")
    assert selection["market_id"] == "18"
    assert selection["specifier"] == "total=2.5"
    assert selection["outcome_id"] == "12"
    assert selection["outcome_name"] == "Over 2.5"


def test_resolve_sportybet_double_chance_for_creation():
    fixture = _sample_sportybet_fixture_for_creation()
    selection = _resolve_sportybet_pick(fixture, "Double Chance X2")
    assert selection["market_id"] == "10"
    assert selection["outcome_name"] == "Draw or Away"


def test_resolve_sportybet_btts_for_creation():
    fixture = _sample_sportybet_fixture_for_creation()
    selection = _resolve_sportybet_pick(fixture, "BTTS Yes")
    assert selection["market_id"] == "29"
    assert selection["outcome_name"] == "GG"


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
    assert "experimental read-only lookup" in summary
    assert "automatic wager placement is not enabled" in summary


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
    test_sportybet_selection_model_support_for_1x2()
    test_sportybet_selection_model_support_for_total()
    test_sportybet_corner_total_is_not_treated_as_goals()
    test_sportybet_text_double_chance_home_or_away()
    test_sportybet_team_total_is_modelled_as_team_goals()
    test_sportybet_context_rejects_tiny_recent_sample()
    test_sportybet_context_accepts_season_data()
    test_sportybet_platform_probability_prefers_feed_value()
    test_sportybet_platform_probability_falls_back_to_odds()
    test_sportybet_default_summary_is_compact()
    test_sportybet_cached_views_filter_without_reanalysis()
    test_parse_sportybet_booking_creation_request()
    test_sportybet_fixture_suggestions_for_unavailable_matchup()
    test_resolve_sportybet_fixture_for_creation()
    test_resolve_sportybet_over_under_for_creation()
    test_resolve_sportybet_double_chance_for_creation()
    test_resolve_sportybet_btts_for_creation()
    test_betting_platform_aliases()
    test_platform_capability_is_honest_about_booking_codes()
    print("API helper tests passed.")
