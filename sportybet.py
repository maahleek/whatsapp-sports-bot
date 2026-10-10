from __future__ import annotations

import os
import re
import time
from typing import Any

import requests


SPORTYBET_API_BASE_URL = os.getenv(
    "SPORTYBET_API_BASE_URL",
    "https://www.sportybet.com",
).rstrip("/")
SPORTYBET_REGION = os.getenv("SPORTYBET_REGION", "ng").strip().lower() or "ng"
SPORTYBET_TIMEOUT_SECONDS = float(os.getenv("SPORTYBET_TIMEOUT_SECONDS", "12"))
SPORTYBET_DEFAULT_CREATE_MARKETS = ("1", "10", "18", "29")

_BOOKING_CODE_RE = re.compile(r"^[A-Z0-9]{4,12}$")


class SportyBetLookupError(RuntimeError):
    pass


def normalize_booking_code(value: str) -> str:
    code = re.sub(r"\s+", "", value or "").upper()
    if not _BOOKING_CODE_RE.fullmatch(code):
        raise ValueError(
            "SportyBet booking codes should be 4-12 letters/numbers."
        )
    return code


def _headers() -> dict[str, str]:
    return {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Current-Country": SPORTYBET_REGION.upper(),
        "User-Agent": "whatsapp-sports-bot/2.0",
    }


def fetch_booking(code: str) -> dict[str, Any]:
    """Read an existing SportyBet share/booking code.

    SportyBet does not publish a public developer API. This uses the same
    read-only web endpoint used by its site and may change without notice.
    """
    clean = normalize_booking_code(code)
    url = (
        f"{SPORTYBET_API_BASE_URL}/api/{SPORTYBET_REGION}"
        f"/orders/share/{clean}"
    )

    try:
        response = requests.get(
            url,
            headers=_headers(),
            timeout=SPORTYBET_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        raise SportyBetLookupError(
            "SportyBet could not be reached right now."
        ) from exc

    if response.status_code == 404:
        raise SportyBetLookupError(
            f"SportyBet code {clean} was not found or may have expired."
        )

    try:
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise SportyBetLookupError(
            "SportyBet returned an unexpected response."
        ) from exc

    if not isinstance(payload, dict):
        raise SportyBetLookupError(
            "SportyBet returned an unexpected response format."
        )

    biz_code = payload.get("bizCode")
    data = payload.get("data")
    if biz_code not in (None, 10000) or not isinstance(data, dict):
        message = str(payload.get("message") or "").strip()
        if message:
            raise SportyBetLookupError(
                f"SportyBet could not load that code: {message}"
            )
        raise SportyBetLookupError(
            f"SportyBet code {clean} was not found or may have expired."
        )

    if not data.get("shareCode"):
        raise SportyBetLookupError(
            f"SportyBet code {clean} was not found or may have expired."
        )

    return payload


def _competition_name(event: dict[str, Any]) -> str:
    sport = event.get("sport") or {}
    category = sport.get("category") or {}
    tournament = category.get("tournament") or {}
    return str(tournament.get("name") or category.get("name") or "").strip()


def extract_booking(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize a SportyBet booking payload and intentionally drop user/account IDs."""
    data = payload.get("data") or {}
    normalized: dict[str, Any] = {
        "share_code": str(data.get("shareCode") or "").strip(),
        "share_url": str(data.get("shareURL") or "").strip(),
        "deadline": data.get("deadline"),
        "bet_type": str(data.get("betType") or "").strip(),
        "selections": [],
        "unavailable_outcomes": data.get("unavailableOutcomes") or [],
    }

    selections: list[dict[str, Any]] = []
    for event in data.get("outcomes") or []:
        if not isinstance(event, dict):
            continue

        home = str(event.get("homeTeamName") or "").strip()
        away = str(event.get("awayTeamName") or "").strip()
        event_id = str(event.get("eventId") or "").strip()

        for market in event.get("markets") or []:
            if not isinstance(market, dict):
                continue

            market_name = str(
                market.get("name")
                or market.get("desc")
                or market.get("title")
                or ""
            ).strip()
            market_id = str(market.get("id") or "").strip()
            specifier = str(market.get("specifier") or "").strip()

            for outcome in market.get("outcomes") or []:
                if not isinstance(outcome, dict):
                    continue

                odds_raw = outcome.get("odds")
                try:
                    odds = float(odds_raw) if odds_raw is not None else None
                except (TypeError, ValueError):
                    odds = None

                probability_raw = outcome.get("probability")
                try:
                    source_probability = (
                        float(probability_raw)
                        if probability_raw is not None
                        else None
                    )
                except (TypeError, ValueError):
                    source_probability = None

                selections.append(
                    {
                        "event_id": event_id,
                        "home_team": home,
                        "away_team": away,
                        "competition": _competition_name(event),
                        "kickoff_ms": event.get("estimateStartTime"),
                        "match_status": str(event.get("matchStatus") or "").strip(),
                        "market_id": market_id,
                        "market_name": market_name,
                        "specifier": specifier,
                        "outcome_id": str(outcome.get("id") or "").strip(),
                        "outcome_name": str(outcome.get("desc") or "").strip(),
                        "odds": odds,
                        "source_probability": source_probability,
                        "is_active": bool(outcome.get("isActive", 1)),
                    }
                )

    normalized["selections"] = selections
    return normalized


def _check_payload(payload: Any, *, action: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise SportyBetLookupError(
            f"SportyBet returned an unexpected response while trying to {action}."
        )

    biz_code = payload.get("bizCode")
    if biz_code not in (None, 10000):
        message = str(payload.get("message") or "").strip()
        detail = f": {message}" if message else ""
        raise SportyBetLookupError(
            f"SportyBet rejected the request{detail}"
        )
    return payload


def _safe_json_response(response: requests.Response, *, action: str) -> dict[str, Any]:
    try:
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise SportyBetLookupError(
            f"SportyBet returned an unexpected response while trying to {action}."
        ) from exc
    return _check_payload(payload, action=action)


def _normalize_market(raw: dict[str, Any], event_id: str) -> dict[str, Any]:
    outcomes: list[dict[str, Any]] = []
    for outcome in raw.get("outcomes") or []:
        if not isinstance(outcome, dict):
            continue

        odds_raw = outcome.get("odds")
        try:
            odds = float(odds_raw) if odds_raw is not None else None
        except (TypeError, ValueError):
            odds = None

        probability_raw = outcome.get("probability")
        try:
            probability = (
                float(probability_raw)
                if probability_raw is not None
                else None
            )
        except (TypeError, ValueError):
            probability = None

        outcomes.append(
            {
                "outcome_id": str(outcome.get("id") or "").strip(),
                "outcome_name": str(outcome.get("desc") or "").strip(),
                "odds": odds,
                "probability": probability,
                "is_active": int(outcome.get("isActive", 1) or 0) == 1,
            }
        )

    return {
        "event_id": event_id,
        "market_id": str(raw.get("id") or "").strip(),
        "market_name": str(
            raw.get("desc")
            or raw.get("name")
            or raw.get("title")
            or raw.get("id")
            or ""
        ).strip(),
        "specifier": (
            str(raw.get("specifier")).strip()
            if raw.get("specifier") is not None
            else None
        ),
        "status": int(raw.get("status") or 0),
        "outcomes": outcomes,
    }


def _normalize_fixture(
    raw: dict[str, Any],
    tournament: dict[str, Any] | None = None,
) -> dict[str, Any]:
    event_id = str(raw.get("eventId") or "").strip()
    tournament = tournament or {}
    sport = raw.get("sport") or {}
    category = sport.get("category") or {}

    try:
        start_ms = int(raw.get("estimateStartTime") or 0)
    except (TypeError, ValueError):
        start_ms = 0

    return {
        "event_id": event_id,
        "home_team": str(raw.get("homeTeamName") or "").strip(),
        "away_team": str(raw.get("awayTeamName") or "").strip(),
        "league": str(
            tournament.get("name")
            or (category.get("tournament") or {}).get("name")
            or ""
        ).strip(),
        "category": str(
            tournament.get("categoryName")
            or category.get("name")
            or ""
        ).strip(),
        "start_ms": start_ms,
        "match_status": str(raw.get("matchStatus") or "Not start").strip(),
        "markets": [
            _normalize_market(market, event_id)
            for market in (raw.get("markets") or [])
            if isinstance(market, dict)
        ],
    }


def _team_key(value: str) -> str:
    value = re.sub(r"[^a-z0-9 ]+", " ", (value or "").casefold())
    aliases = {"utd": "united", "st": "saint"}
    tokens = [
        aliases.get(token, token)
        for token in value.split()
        if token not in {"fc", "afc", "cf", "club", "football"}
    ]
    return " ".join(tokens)


def _team_matches(requested: str, actual: str) -> bool:
    req = _team_key(requested)
    act = _team_key(actual)
    if not req or not act:
        return False
    if req == act:
        return True
    if len(req) >= 4 and req in act:
        return True
    if len(act) >= 4 and act in req:
        return True

    req_tokens = set(req.split())
    act_tokens = set(act.split())
    if not req_tokens or not act_tokens:
        return False
    overlap = len(req_tokens & act_tokens)
    required = 1 if min(len(req_tokens), len(act_tokens)) == 1 else 2
    return overlap >= required


def _fixture_matches_pair(
    fixture: dict[str, Any],
    team1: str,
    team2: str,
) -> bool:
    home = str(fixture.get("home_team") or "")
    away = str(fixture.get("away_team") or "")
    return (
        _team_matches(team1, home) and _team_matches(team2, away)
    ) or (
        _team_matches(team1, away) and _team_matches(team2, home)
    )


def fetch_upcoming_fixtures(
    *,
    team_pairs: list[tuple[str, str]] | None = None,
    market_ids: tuple[str, ...] = SPORTYBET_DEFAULT_CREATE_MARKETS,
    timeline_hours: int = 720,
    page_size: int = 100,
    max_pages: int = 20,
) -> list[dict[str, Any]]:
    """Fetch upcoming SportyBet fixtures and current markets.

    This uses SportyBet's undocumented website endpoint. It is read-only.
    When team_pairs are provided, paging stops once every requested matchup
    has been found.
    """
    timeline_hours = max(12, min(int(timeline_hours), 720))
    page_size = max(1, min(int(page_size), 100))
    max_pages = max(1, min(int(max_pages), 20))

    requested_pairs = team_pairs or []
    found_pair_indexes: set[int] = set()
    fixtures: list[dict[str, Any]] = []

    url = (
        f"{SPORTYBET_API_BASE_URL}/api/{SPORTYBET_REGION}"
        "/factsCenter/pcUpcomingEvents"
    )

    for page in range(1, max_pages + 1):
        params = {
            "sportId": "sr:sport:1",
            "marketId": ",".join(dict.fromkeys(market_ids)),
            "pageSize": str(page_size),
            "pageNum": str(page),
            "todayGames": "false",
            "timeline": str(timeline_hours),
            "_t": str(int(time.time() * 1000)),
        }

        try:
            response = requests.get(
                url,
                headers=_headers(),
                params=params,
                timeout=SPORTYBET_TIMEOUT_SECONDS,
            )
        except requests.RequestException as exc:
            raise SportyBetLookupError(
                "SportyBet fixtures could not be reached right now."
            ) from exc

        payload = _safe_json_response(
            response,
            action="load upcoming fixtures",
        )
        data = payload.get("data") or {}
        tournaments = data.get("tournaments") or []

        page_fixtures: list[dict[str, Any]] = []
        for tournament in tournaments:
            if not isinstance(tournament, dict):
                continue
            for event in tournament.get("events") or []:
                if not isinstance(event, dict):
                    continue
                fixture = _normalize_fixture(event, tournament)
                if fixture["event_id"]:
                    page_fixtures.append(fixture)

        fixtures.extend(page_fixtures)

        if requested_pairs:
            for index, (team1, team2) in enumerate(requested_pairs):
                if index in found_pair_indexes:
                    continue
                if any(
                    _fixture_matches_pair(fixture, team1, team2)
                    for fixture in page_fixtures
                ):
                    found_pair_indexes.add(index)

            if len(found_pair_indexes) == len(requested_pairs):
                break

        if not page_fixtures or len(page_fixtures) < page_size:
            break

    return fixtures


def create_booking(
    selections: list[dict[str, Any]],
) -> dict[str, Any]:
    """Create an anonymous, non-staking SportyBet share/booking code."""
    if not selections:
        raise ValueError("Provide at least one SportyBet selection.")
    if len(selections) > 20:
        raise ValueError("SportyBet booking-code creation is limited to 20 selections.")

    payload_selections: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str | None]] = set()

    for selection in selections:
        event_id = str(selection.get("event_id") or "").strip()
        market_id = str(selection.get("market_id") or "").strip()
        outcome_id = str(selection.get("outcome_id") or "").strip()
        specifier_raw = selection.get("specifier")
        specifier = (
            str(specifier_raw).strip()
            if specifier_raw is not None
            else None
        )

        if not event_id or not market_id or not outcome_id:
            raise ValueError(
                "Every SportyBet selection needs an event, market, and outcome."
            )

        key = (event_id, market_id, outcome_id, specifier)
        if key in seen:
            raise ValueError("Duplicate SportyBet selection detected.")
        seen.add(key)

        payload_selections.append(
            {
                "eventId": event_id,
                "marketId": market_id,
                "specifier": specifier,
                "outcomeId": outcome_id,
            }
        )

    url = (
        f"{SPORTYBET_API_BASE_URL}/api/{SPORTYBET_REGION}"
        "/orders/share"
    )

    try:
        response = requests.post(
            url,
            headers=_headers(),
            json={"selections": payload_selections},
            timeout=SPORTYBET_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        raise SportyBetLookupError(
            "SportyBet could not create the booking code right now."
        ) from exc

    payload = _safe_json_response(
        response,
        action="create a booking code",
    )
    data = payload.get("data")
    if not isinstance(data, dict) or not data.get("shareCode"):
        raise SportyBetLookupError(
            "SportyBet did not return a booking code. "
            "A fixture, market, or price may have changed."
        )

    return extract_booking(payload)
