from __future__ import annotations

import os
import re
from typing import Any

import requests


SPORTYBET_API_BASE_URL = os.getenv(
    "SPORTYBET_API_BASE_URL",
    "https://www.sportybet.com",
).rstrip("/")
SPORTYBET_REGION = os.getenv("SPORTYBET_REGION", "ng").strip().lower() or "ng"
SPORTYBET_TIMEOUT_SECONDS = float(os.getenv("SPORTYBET_TIMEOUT_SECONDS", "12"))

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
