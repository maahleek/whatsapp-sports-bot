import json
import math
import os
import random
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from datetime import datetime, timedelta, timezone
from threading import Lock
from functools import lru_cache
from typing import Any

import requests
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, Response
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import create_react_agent
from tavily import TavilyClient
from twilio.request_validator import RequestValidator
from twilio.rest import Client

from betting_platforms import get_platform, platform_capability_summary, platform_market_name
from rag import lookup_betting_term, search_knowledge
from sportybet import (
    SportyBetLookupError,
    create_booking,
    extract_booking,
    fetch_booking,
    fetch_upcoming_fixtures,
    normalize_booking_code,
)

load_dotenv()

HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT_SECONDS", "12"))
SPORTSDB_API_KEY = os.getenv("SPORTSDB_API_KEY", "123")
SPORTSDB_BASE_URL = f"https://www.thesportsdb.com/api/v1/json/{SPORTSDB_API_KEY}"
FOOTBALL_DATA_BASE_URL = "https://api.football-data.org/v4"
TWILIO_WHATSAPP_FROM = os.getenv("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886")
VERIFY_TWILIO_SIGNATURE = os.getenv("VERIFY_TWILIO_SIGNATURE", "false").lower() == "true"
MEMORY_NAMESPACE = os.getenv("MEMORY_NAMESPACE", "v3")

_PROCESSED_MESSAGE_SIDS: set[str] = set()
_PROCESSED_MESSAGE_ORDER: deque[str] = deque()
_PROCESSED_MESSAGE_LOCK = Lock()
_MAX_PROCESSED_MESSAGE_SIDS = 1000

_LAST_MATCHUPS: dict[str, tuple[str, str]] = {}
_MATCHUP_CONTEXT_LOCK = Lock()

_SPORTYBET_ANALYSIS_CACHE: dict[str, dict[str, Any]] = {}
_SPORTYBET_ANALYSIS_CACHE_LOCK = Lock()

_SPORTYBET_SLIP_STATES: dict[str, dict[str, Any]] = {}
_SPORTYBET_SLIP_STATE_LOCK = Lock()

LEAGUE_CODES = {
    "premier league": "PL",
    "english premier league": "PL",
    "epl": "PL",
    "championship": "ELC",
    "efl championship": "ELC",
    "english championship": "ELC",
    "la liga": "PD",
    "spanish la liga": "PD",
    "bundesliga": "BL1",
    "german bundesliga": "BL1",
    "serie a": "SA",
    "italian serie a": "SA",
    "ligue 1": "FL1",
    "french ligue 1": "FL1",
    "eredivisie": "DED",
    "dutch eredivisie": "DED",
    "primeira liga": "PPL",
    "liga portugal": "PPL",
    "portuguese primeira liga": "PPL",
    "campeonato brasileiro serie a": "BSA",
    "campeonato brasileiro série a": "BSA",
    "brasileirao serie a": "BSA",
    "brasileirão serie a": "BSA",
    "brazil serie a": "BSA",
    "champions league": "CL",
    "uefa champions league": "CL",
}

app = FastAPI(
    title="AI-Powered WhatsApp Football Assistant",
    version="2.0.0",
    description="Agentic football assistant with tool calling, live sports data, RAG, memory, and WhatsApp delivery.",
)


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _safe_get_json(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    response = requests.get(
        url,
        headers=headers,
        params=params,
        timeout=HTTP_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise RuntimeError("API returned an unexpected response format.")
    return data


def _current_season_start_year() -> int:
    now = datetime.now(timezone.utc)
    return now.year if now.month >= 7 else now.year - 1


def _league_code(league_name: str) -> str | None:
    return LEAGUE_CODES.get(league_name.strip().lower())


def _team_record(team_name: str) -> tuple[str, str]:
    profile = _team_profile(team_name)
    return profile["id"], profile["name"]


@lru_cache(maxsize=1024)
def _recent_events(team_name: str, limit: int = 5) -> tuple[str, list[dict[str, Any]]]:
    team_id, canonical_name = _team_record(team_name)
    data = _safe_get_json(
        f"{SPORTSDB_BASE_URL}/eventslast.php",
        params={"id": team_id},
    )
    events = data.get("results") or []
    return canonical_name, events[-limit:]


def _form_summary(team_name: str) -> dict[str, float | str | int]:
    canonical_name, events = _recent_events(team_name, 5)
    if not events:
        raise ValueError(f"No recent results found for {canonical_name}.")

    points = 0
    goals_for = 0
    goals_against = 0
    wins = draws = losses = 0

    for event in events:
        home = str(event.get("strHomeTeam", ""))
        away = str(event.get("strAwayTeam", ""))
        home_score = int(event.get("intHomeScore") or 0)
        away_score = int(event.get("intAwayScore") or 0)

        is_home = home.casefold() == canonical_name.casefold()
        gf = home_score if is_home else away_score
        ga = away_score if is_home else home_score

        goals_for += gf
        goals_against += ga

        if gf > ga:
            wins += 1
            points += 3
        elif gf == ga:
            draws += 1
            points += 1
        else:
            losses += 1

    played = len(events)
    return {
        "team": canonical_name,
        "played": played,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "goals_for": goals_for,
        "goals_against": goals_against,
        "goals_for_per_game": goals_for / played,
        "goals_against_per_game": goals_against / played,
        "points_per_game": points / played,
        "goal_diff_per_game": (goals_for - goals_against) / played,
    }


def _prediction_probabilities(
    strength_a: float,
    strength_b: float,
) -> tuple[float, float, float]:
    difference = strength_a - strength_b
    draw_probability = max(
        0.16,
        min(0.30, 0.28 - 0.05 * min(abs(difference), 2.4)),
    )
    decisive_probability = 1.0 - draw_probability
    team_a_share = 1.0 / (1.0 + math.exp(-1.15 * difference))
    team_a_probability = decisive_probability * team_a_share
    team_b_probability = decisive_probability - team_a_probability
    return team_a_probability, draw_probability, team_b_probability


def _football_data_headers() -> dict[str, str]:
    return {"X-Auth-Token": _require_env("FOOTBALL_DATA_KEY")}


def _normalize_team_name(name: str) -> str:
    normalized = re.sub(r"[^a-z0-9 ]+", " ", name.casefold())
    tokens = normalized.split()

    # Handle dotted club suffixes such as F.C., A.F.C., and C.F.,
    # which become separate single-letter tokens after punctuation removal.
    for suffix in (["a", "f", "c"], ["f", "c"], ["c", "f"]):
        if tokens[-len(suffix):] == suffix:
            tokens = tokens[:-len(suffix)]
            break

    tokens = [
        token
        for token in tokens
        if token not in {"fc", "afc", "cf", "club", "football"}
    ]
    return " ".join(tokens)


@lru_cache(maxsize=512)
def _team_profile(team_name: str) -> dict[str, str]:
    data = _safe_get_json(
        f"{SPORTSDB_BASE_URL}/searchteams.php",
        params={"t": team_name},
    )
    teams = data.get("teams") or []
    if not teams:
        raise ValueError(f"Team '{team_name}' not found.")

    team = teams[0]
    return {
        "id": str(team.get("idTeam") or ""),
        "name": str(team.get("strTeam") or team_name),
        "league": str(team.get("strLeague") or ""),
    }


@lru_cache(maxsize=16)
def _competition_standings(code: str) -> list[dict[str, Any]]:
    """Fetch and cache a competition table so batch analysis does not repeat the same request."""
    data = _safe_get_json(
        f"{FOOTBALL_DATA_BASE_URL}/competitions/{code}/standings",
        headers=_football_data_headers(),
    )
    standings = data.get("standings") or []
    return standings[0].get("table", []) if standings else []


def _season_team_summary(team_name: str) -> dict[str, float | int | str] | None:
    profile = _team_profile(team_name)
    code = _league_code(profile["league"])
    if not code:
        return None

    table = _competition_standings(code)
    target = _normalize_team_name(profile["name"])

    for item in table:
        api_name = str(item.get("team", {}).get("name") or "")
        if _normalize_team_name(api_name) != target:
            continue

        played = int(item.get("playedGames") or 0)
        if played <= 0:
            return None

        points = int(item.get("points") or 0)
        goals_for = int(item.get("goalsFor") or 0)
        goals_against = int(item.get("goalsAgainst") or 0)
        goal_difference = int(
            item.get("goalDifference")
            if item.get("goalDifference") is not None
            else goals_for - goals_against
        )
        return {
            "team": profile["name"],
            "league": profile["league"],
            "played": played,
            "points": points,
            "position": int(item.get("position") or 0),
            "goals_for": goals_for,
            "goals_against": goals_against,
            "points_per_game": points / played,
            "goal_diff_per_game": goal_difference / played,
        }

    return None


def _prediction_strength(
    season: dict[str, float | int | str] | None,
    recent: dict[str, float | int | str] | None,
) -> tuple[float, str]:
    if season is not None:
        season_strength = (
            float(season["points_per_game"])
            + 0.18 * float(season["goal_diff_per_game"])
        )
        if recent is not None:
            recent_played = int(recent["played"])
            recent_weight = min(0.35, 0.07 * recent_played)
            recent_strength = (
                float(recent["points_per_game"])
                + 0.20 * float(recent["goal_diff_per_game"])
            )
            combined = (
                (1.0 - recent_weight) * season_strength
                + recent_weight * recent_strength
            )
            return combined, "season + recent form"
        return season_strength, "season performance"

    if recent is not None:
        strength = (
            float(recent["points_per_game"])
            + 0.20 * float(recent["goal_diff_per_game"])
        )
        return strength, "recent form only"

    raise ValueError("No usable performance data was available for this team.")


def _find_upcoming_fixture(team1: str, team2: str) -> dict[str, str] | None:
    """Find an upcoming fixture between two teams and identify the real home side."""
    first = _team_profile(team1)
    second = _team_profile(team2)
    first_name = _normalize_team_name(first["name"])
    second_name = _normalize_team_name(second["name"])

    # Prefer the structured competition schedule when both teams are in the same supported league.
    first_code = _league_code(first["league"])
    second_code = _league_code(second["league"])
    if first_code and first_code == second_code:
        try:
            data = _safe_get_json(
                f"{FOOTBALL_DATA_BASE_URL}/competitions/{first_code}/matches",
                headers=_football_data_headers(),
                params={"status": "SCHEDULED"},
            )
            for match in data.get("matches") or []:
                home = str(match.get("homeTeam", {}).get("name") or "")
                away = str(match.get("awayTeam", {}).get("name") or "")
                if {
                    _normalize_team_name(home),
                    _normalize_team_name(away),
                } != {first_name, second_name}:
                    continue
                return {
                    "home": home,
                    "away": away,
                    "date": str(match.get("utcDate") or "")[:10],
                }
        except Exception:
            pass

    # Fall back to TheSportsDB upcoming events.
    data = _safe_get_json(
        f"{SPORTSDB_BASE_URL}/eventsnext.php",
        params={"id": first["id"]},
    )
    for event in data.get("events") or []:
        home = str(event.get("strHomeTeam") or "")
        away = str(event.get("strAwayTeam") or "")
        if {
            _normalize_team_name(home),
            _normalize_team_name(away),
        } != {first_name, second_name}:
            continue
        return {
            "home": home,
            "away": away,
            "date": str(event.get("dateEvent") or ""),
        }
    return None

def _scoring_rates(
    season: dict[str, float | int | str] | None,
    recent: dict[str, float | int | str] | None,
) -> tuple[float, float] | None:
    """Return blended goals-for and goals-against rates per match."""
    if season is not None:
        played = int(season["played"])
        gf = float(season["goals_for"]) / played
        ga = float(season["goals_against"]) / played
        if recent is not None:
            recent_played = int(recent["played"])
            weight = min(0.30, 0.06 * recent_played)
            gf = (1.0 - weight) * gf + weight * float(recent["goals_for_per_game"])
            ga = (1.0 - weight) * ga + weight * float(recent["goals_against_per_game"])
        return gf, ga

    if recent is not None:
        return (
            float(recent["goals_for_per_game"]),
            float(recent["goals_against_per_game"]),
        )
    return None


def _poisson_probability(goals: int, rate: float) -> float:
    return math.exp(-rate) * (rate ** goals) / math.factorial(goals)


def _score_projection(
    first_rates: tuple[float, float],
    second_rates: tuple[float, float],
    *,
    first_is_home: bool | None,
) -> dict[str, Any]:
    """Create a simple Poisson score projection from scoring/conceding rates."""
    first_for, first_against = first_rates
    second_for, second_against = second_rates

    first_rate = (first_for + second_against) / 2.0
    second_rate = (second_for + first_against) / 2.0

    if first_is_home is True:
        first_rate *= 1.08
        second_rate *= 0.94
    elif first_is_home is False:
        first_rate *= 0.94
        second_rate *= 1.08

    first_rate = min(4.0, max(0.20, first_rate))
    second_rate = min(4.0, max(0.20, second_rate))

    matrix: list[tuple[int, int, float]] = []
    for first_goals in range(7):
        for second_goals in range(7):
            probability = (
                _poisson_probability(first_goals, first_rate)
                * _poisson_probability(second_goals, second_rate)
            )
            matrix.append((first_goals, second_goals, probability))

    total = sum(item[2] for item in matrix) or 1.0
    normalized = [
        (first_goals, second_goals, probability / total)
        for first_goals, second_goals, probability in matrix
    ]
    first_win = sum(p for a, b, p in normalized if a > b)
    draw = sum(p for a, b, p in normalized if a == b)
    second_win = sum(p for a, b, p in normalized if a < b)
    top_scores = sorted(normalized, key=lambda item: item[2], reverse=True)[:3]

    return {
        "first_rate": first_rate,
        "second_rate": second_rate,
        "first_win": first_win,
        "draw": draw,
        "second_win": second_win,
        "top_scores": top_scores,
        "score_matrix": normalized,
    }


def _fixture_order_projection(context: dict[str, Any]) -> tuple[str, str, list[tuple[int, int, float]]]:
    """Return home/away labels and the score matrix in actual fixture order."""
    projection = context.get("projection")
    if projection is None:
        raise ValueError("No score projection is available.")

    matrix = list(projection["score_matrix"])
    fixture = context.get("fixture")
    first_name = str(context["first_name"])
    second_name = str(context["second_name"])
    first_is_home = context.get("first_is_home")

    if fixture is not None and first_is_home is False:
        swapped = [(second_goals, first_goals, p) for first_goals, second_goals, p in matrix]
        return str(fixture["home"]), str(fixture["away"]), swapped

    if fixture is not None:
        return str(fixture["home"]), str(fixture["away"]), matrix

    return first_name, second_name, matrix


def _asian_handicap_probabilities(
    matrix: list[tuple[int, int, float]],
    line: float,
    *,
    side: str,
) -> tuple[float, float, float]:
    """Return full-win, push, and loss probabilities for whole/half Asian lines."""
    win = push = loss = 0.0
    for home_goals, away_goals, probability in matrix:
        goal_difference = home_goals - away_goals
        adjusted = (
            goal_difference + line
            if side == "home"
            else -goal_difference + line
        )
        if adjusted > 0:
            win += probability
        elif adjusted == 0:
            push += probability
        else:
            loss += probability
    return win, push, loss


def _split_quarter_line(line: float) -> tuple[float, float] | None:
    """Split a quarter line into the two adjacent half/whole lines."""
    scaled = round(line * 4)
    if abs(line * 4 - scaled) > 1e-9 or scaled % 2 == 0:
        return None
    lower = math.floor(line * 2) / 2.0
    upper = math.ceil(line * 2) / 2.0
    return lower, upper


def _asian_handicap_settlement(
    matrix: list[tuple[int, int, float]],
    line: float,
    *,
    side: str,
) -> dict[str, float]:
    """Return full/half win, push, and full/half loss probabilities."""
    split = _split_quarter_line(line)
    if split is None:
        win, push, loss = _asian_handicap_probabilities(matrix, line, side=side)
        return {
            "full_win": win,
            "half_win": 0.0,
            "push": push,
            "half_loss": 0.0,
            "full_loss": loss,
        }

    first_line, second_line = split
    totals = {
        "full_win": 0.0,
        "half_win": 0.0,
        "push": 0.0,
        "half_loss": 0.0,
        "full_loss": 0.0,
    }

    for home_goals, away_goals, probability in matrix:
        goal_difference = home_goals - away_goals

        def outcome(component: float) -> int:
            adjusted = (
                goal_difference + component
                if side == "home"
                else -goal_difference + component
            )
            if adjusted > 0:
                return 1
            if adjusted < 0:
                return -1
            return 0

        outcomes = (outcome(first_line), outcome(second_line))
        if outcomes == (1, 1):
            totals["full_win"] += probability
        elif 1 in outcomes and 0 in outcomes:
            totals["half_win"] += probability
        elif outcomes == (0, 0):
            totals["push"] += probability
        elif -1 in outcomes and 0 in outcomes:
            totals["half_loss"] += probability
        elif outcomes == (-1, -1):
            totals["full_loss"] += probability
        else:
            # Defensive fallback for an unusual split result.
            totals["push"] += probability

    return totals


def _total_line_settlement(
    matrix: list[tuple[int, int, float]],
    line: float,
    *,
    side: str,
) -> dict[str, float]:
    """Return settlement probabilities for over/under whole, half, and quarter totals."""
    split = _split_quarter_line(line)
    if split is None:
        win = push = loss = 0.0
        for home_goals, away_goals, probability in matrix:
            total = home_goals + away_goals
            adjusted = total - line if side == "over" else line - total
            if adjusted > 0:
                win += probability
            elif adjusted == 0:
                push += probability
            else:
                loss += probability
        return {
            "full_win": win,
            "half_win": 0.0,
            "push": push,
            "half_loss": 0.0,
            "full_loss": loss,
        }

    first_line, second_line = split
    totals = {
        "full_win": 0.0,
        "half_win": 0.0,
        "push": 0.0,
        "half_loss": 0.0,
        "full_loss": 0.0,
    }
    for home_goals, away_goals, probability in matrix:
        total = home_goals + away_goals

        def outcome(component: float) -> int:
            adjusted = total - component if side == "over" else component - total
            if adjusted > 0:
                return 1
            if adjusted < 0:
                return -1
            return 0

        outcomes = (outcome(first_line), outcome(second_line))
        if outcomes == (1, 1):
            totals["full_win"] += probability
        elif 1 in outcomes and 0 in outcomes:
            totals["half_win"] += probability
        elif outcomes == (0, 0):
            totals["push"] += probability
        elif -1 in outcomes and 0 in outcomes:
            totals["half_loss"] += probability
        elif outcomes == (-1, -1):
            totals["full_loss"] += probability
        else:
            totals["push"] += probability
    return totals


def _format_settlement(label: str, settlement: dict[str, float]) -> str:
    parts = []
    for key, text_label in (
        ("full_win", "win"),
        ("half_win", "half-win"),
        ("push", "push"),
        ("half_loss", "half-loss"),
        ("full_loss", "lose"),
    ):
        value = settlement[key]
        if value > 0.0001:
            parts.append(f"{text_label} {value * 100:.1f}%")
    return f"- {label}: " + ", ".join(parts)


def _betting_market_report(context: dict[str, Any], market_request: str = "all") -> str:
    """Build neutral market-probability estimates from the score distribution."""
    home, away, matrix = _fixture_order_projection(context)
    request = (market_request or "all").casefold()

    home_win = sum(p for h, a, p in matrix if h > a)
    draw = sum(p for h, a, p in matrix if h == a)
    away_win = sum(p for h, a, p in matrix if h < a)
    btts_yes = sum(p for h, a, p in matrix if h > 0 and a > 0)
    btts_no = 1.0 - btts_yes

    def over_probability(line: float) -> float:
        return sum(p for h, a, p in matrix if h + a > line)

    def home_over(line: float) -> float:
        return sum(p for h, _a, p in matrix if h > line)

    def away_over(line: float) -> float:
        return sum(p for _h, a, p in matrix if a > line)

    home_clean = sum(p for _h, a, p in matrix if a == 0)
    away_clean = sum(p for h, _a, p in matrix if h == 0)
    home_win_nil = sum(p for h, a, p in matrix if h > a and a == 0)
    away_win_nil = sum(p for h, a, p in matrix if a > h and h == 0)

    top_scores = sorted(matrix, key=lambda item: item[2], reverse=True)[:3]

    lines = [
        f"Betting-market probability sheet: {home} vs {away}",
        context["venue_note"],
        "",
    ]

    wants_all = request.strip() in {"", "all"} or any(
        phrase in request
        for phrase in ("betting prediction", "bet prediction", "market prediction", "betting markets")
    )

    if wants_all or any(
        term in request
        for term in ("1x2", "match result", "moneyline", "winner")
    ):
        lines.extend(
            [
                "1X2 / Match result:",
                f"- 1 ({home}): {home_win * 100:.1f}%",
                f"- X (Draw): {draw * 100:.1f}%",
                f"- 2 ({away}): {away_win * 100:.1f}%",
                "",
            ]
        )

    if "double chance" in request and not wants_all:
        lines.append("Double chance:")

        if "1x" in request:
            lines.append(
                f"- 1X ({home} or Draw): {(home_win + draw) * 100:.1f}%"
            )
        elif "x2" in request:
            lines.append(
                f"- X2 (Draw or {away}): {(draw + away_win) * 100:.1f}%"
            )
        elif re.search(r"\b12\b", request):
            lines.append(
                f"- 12 ({home} or {away}): {(home_win + away_win) * 100:.1f}%"
            )
        else:
            lines.extend(
                [
                    f"- 1X ({home} or Draw): {(home_win + draw) * 100:.1f}%",
                    f"- X2 (Draw or {away}): {(draw + away_win) * 100:.1f}%",
                    f"- 12 ({home} or {away}): {(home_win + away_win) * 100:.1f}%",
                ]
            )

        lines.append("")

    elif wants_all:
        lines.extend(
            [
                "Double chance:",
                f"- 1X ({home} or Draw): {(home_win + draw) * 100:.1f}%",
                f"- X2 (Draw or {away}): {(draw + away_win) * 100:.1f}%",
                f"- 12 ({home} or {away}): {(home_win + away_win) * 100:.1f}%",
                "",
            ]
        )

    if wants_all or "draw no bet" in request or "dnb" in request:
        lines.extend(
            [
                "Draw no bet:",
                f"- {home} win: {home_win * 100:.1f}%",
                f"- Draw / refund: {draw * 100:.1f}%",
                f"- {away} win: {away_win * 100:.1f}%",
                "",
            ]
        )

    if wants_all or "btts" in request or "both teams to score" in request:
        lines.extend(
            [
                "Both teams to score:",
                f"- Yes: {btts_yes * 100:.1f}%",
                f"- No: {btts_no * 100:.1f}%",
                "",
            ]
        )

    total_match = re.search(r"\b(over|under)\s*([0-9]+(?:\.[0-9]+)?)", request)
    if total_match and not wants_all:
        side = total_match.group(1)
        line = float(total_match.group(2))
        settlement = _total_line_settlement(matrix, line, side=side)
        lines.extend(
            [
                "Total goals:",
                _format_settlement(f"{side.title()} {line:g}", settlement),
                "",
            ]
        )
    elif wants_all or "over" in request or "under" in request or "total" in request:
        lines.append("Total goals:")
        for line in (0.5, 1.5, 2.5, 3.5, 4.5):
            over = over_probability(line)
            lines.append(
                f"- Over {line:.1f}: {over * 100:.1f}% | "
                f"Under {line:.1f}: {(1.0 - over) * 100:.1f}%"
            )
        lines.append("")

    if wants_all or "team total" in request:
        lines.append("Team totals:")
        for line in (0.5, 1.5, 2.5):
            hp = home_over(line)
            ap = away_over(line)
            lines.append(
                f"- {home} Over {line:.1f}: {hp * 100:.1f}% | "
                f"{away} Over {line:.1f}: {ap * 100:.1f}%"
            )
        lines.append("")

    if wants_all or "clean sheet" in request or "win to nil" in request:
        lines.extend(
            [
                "Clean sheet / win to nil:",
                f"- {home} clean sheet: {home_clean * 100:.1f}%",
                f"- {away} clean sheet: {away_clean * 100:.1f}%",
                f"- {home} win to nil: {home_win_nil * 100:.1f}%",
                f"- {away} win to nil: {away_win_nil * 100:.1f}%",
                "",
            ]
        )

    handicap_match = re.search(
        r"([+-]?\d+(?:\.\d+)?)\s*(?:asian\s+)?handicap|"
        r"(?:asian\s+)?handicap\s*([+-]?\d+(?:\.\d+)?)",
        request,
    )
    if handicap_match and not wants_all:
        raw_line = handicap_match.group(1) or handicap_match.group(2)
        line = float(raw_line)
        requested_side = None
        home_key = _normalize_team_name(home)
        away_key = _normalize_team_name(away)
        request_team_key = _normalize_team_name(request)
        if home_key and home_key in request_team_key:
            requested_side = (home, "home")
        elif away_key and away_key in request_team_key:
            requested_side = (away, "away")

        sides = [requested_side] if requested_side else [(home, "home"), (away, "away")]
        lines.append("Asian handicap:")
        for side_name, side_key in sides:
            settlement = _asian_handicap_settlement(matrix, line, side=side_key)
            lines.append(
                _format_settlement(f"{side_name} {line:+g}", settlement)
            )
        lines.append("")
    elif wants_all or "handicap" in request:
        lines.append("Common Asian handicap lines:")
        for side_name, side_key in ((home, "home"), (away, "away")):
            for line in (-1.5, -1.0, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1.0, 1.5):
                settlement = _asian_handicap_settlement(
                    matrix,
                    line,
                    side=side_key,
                )
                lines.append(
                    _format_settlement(f"{side_name} {line:+g}", settlement)
                )
        lines.append("")

    if wants_all or "correct score" in request or "exact score" in request:
        lines.append("Most likely exact scores:")
        for index, (home_goals, away_goals, probability) in enumerate(top_scores, start=1):
            lines.append(
                f"{index}. {home} {home_goals}-{away_goals} {away} "
                f"({probability * 100:.1f}%)"
            )
        lines.append("")

    if wants_all:
        lines.extend(
            [
                "Not modelled from the current data sources:",
                "- Corners, cards, player shots, player cards, player goalscorer props, and other event-level props require dedicated historical data.",
                "",
            ]
        )

    lines.append(
        "These are model probability estimates, not guaranteed outcomes or betting advice."
    )
    return "\n".join(lines)


def _remember_matchup(user_id: str, team1: str, team2: str) -> None:
    if not user_id:
        return
    with _MATCHUP_CONTEXT_LOCK:
        _LAST_MATCHUPS[user_id] = (team1, team2)


def _last_matchup(user_id: str) -> tuple[str, str] | None:
    if not user_id:
        return None
    with _MATCHUP_CONTEXT_LOCK:
        return _LAST_MATCHUPS.get(user_id)

def _build_prediction_context(
    team1: str,
    team2: str,
    *,
    known_fixture: dict[str, str] | None = None,
    recent_only: bool = False,
    prefer_season_only: bool = False,
) -> dict[str, Any]:
    """Gather performance data, fixture venue, and score projection."""
    first_season = second_season = None
    first_recent = second_recent = None

    if not recent_only:
        try:
            first_season = _season_team_summary(team1)
        except Exception:
            pass
        try:
            second_season = _season_team_summary(team2)
        except Exception:
            pass
    if not (prefer_season_only and first_season is not None):
        try:
            first_recent = _form_summary(team1)
        except Exception:
            pass
    if not (prefer_season_only and second_season is not None):
        try:
            second_recent = _form_summary(team2)
        except Exception:
            pass

    first_strength, first_source = _prediction_strength(first_season, first_recent)
    second_strength, second_source = _prediction_strength(second_season, second_recent)

    first_name = str((first_season or first_recent or {"team": team1})["team"])
    second_name = str((second_season or second_recent or {"team": team2})["team"])

    fixture = known_fixture
    if fixture is None:
        try:
            fixture = _find_upcoming_fixture(first_name, second_name)
        except Exception:
            fixture = None

    first_is_home: bool | None = None
    venue_note = "No upcoming head-to-head fixture was found, so no home advantage was applied."
    if fixture is not None:
        first_is_home = (
            _normalize_team_name(fixture["home"]) == _normalize_team_name(first_name)
        )
        if first_is_home:
            first_strength += 0.12
        else:
            second_strength += 0.12
        date_text = f" on {fixture['date']}" if fixture.get("date") else ""
        venue_note = f"Fixture: {fixture['home']} vs {fixture['away']}{date_text}."

    first_rates = _scoring_rates(first_season, first_recent)
    second_rates = _scoring_rates(second_season, second_recent)
    projection = None
    if first_rates is not None and second_rates is not None:
        projection = _score_projection(
            first_rates,
            second_rates,
            first_is_home=first_is_home,
        )
        first_prob = float(projection["first_win"])
        draw_prob = float(projection["draw"])
        second_prob = float(projection["second_win"])
        probability_source = "Poisson scoring model"
    else:
        first_prob, draw_prob, second_prob = _prediction_probabilities(
            first_strength,
            second_strength,
        )
        probability_source = "strength model"

    outcomes = {
        first_name: first_prob,
        "Draw": draw_prob,
        second_name: second_prob,
    }
    likely_outcome = max(outcomes, key=outcomes.get)
    ordered = sorted(outcomes.values(), reverse=True)
    gap = ordered[0] - ordered[1]
    confidence = "high" if gap >= 0.20 else "medium" if gap >= 0.10 else "low"

    return {
        "first_name": first_name,
        "second_name": second_name,
        "first_season": first_season,
        "second_season": second_season,
        "first_recent": first_recent,
        "second_recent": second_recent,
        "first_source": first_source,
        "second_source": second_source,
        "first_probability": first_prob,
        "draw_probability": draw_prob,
        "second_probability": second_prob,
        "likely_outcome": likely_outcome,
        "confidence": confidence,
        "venue_note": venue_note,
        "fixture": fixture,
        "first_is_home": first_is_home,
        "projection": projection,
        "probability_source": probability_source,
    }

def _clean_web_snippet(value: Any, limit: int = 220) -> str:
    """Collapse noisy search-result text into a short plain-text snippet."""
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    text = text.replace("**", "")
    if len(text) > limit:
        text = text[: limit - 3].rstrip() + "..."
    return text


def _relevant_sentences(value: Any, terms: tuple[str, ...], limit: int = 200) -> str:
    """Keep only useful sentences from noisy search-result text."""
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not text:
        return ""

    sentences = re.split(r"(?<=[.!?])\s+", text)
    lowered_terms = tuple(term.casefold() for term in terms)
    relevant = [
        sentence.strip()
        for sentence in sentences
        if any(term in sentence.casefold() for term in lowered_terms)
    ]
    selected = " ".join(relevant[:2]).strip()
    if not selected:
        return ""

    selected = selected.replace("**", "").replace("`", "")
    if len(selected) > limit:
        selected = selected[: limit - 3].rstrip() + "..."
    return selected

def _extract_player_records(payload: Any) -> list[dict[str, Any]]:
    """Handle the different response shapes returned by the player-search API."""
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]

    if not isinstance(payload, dict):
        return []

    if any(key in payload for key in ("name", "playerName", "strPlayer")):
        return [payload]

    for key in ("response", "players", "suggestions", "results", "data", "items"):
        if key not in payload:
            continue
        records = _extract_player_records(payload[key])
        if records:
            return records

    for value in payload.values():
        records = _extract_player_records(value)
        if records:
            return records

    return []


@tool
def get_team_info(team_name: str) -> str:
    """Get basic information about a football team."""
    try:
        data = _safe_get_json(
            f"{SPORTSDB_BASE_URL}/searchteams.php",
            params={"t": team_name},
        )
        teams = data.get("teams") or []
        if not teams:
            return f"Team '{team_name}' not found."
        team = teams[0]
        return (
            f"Team: {team.get('strTeam', 'Unknown')}\n"
            f"League: {team.get('strLeague', 'Unknown')}\n"
            f"Country: {team.get('strCountry', 'Unknown')}\n"
            f"Stadium: {team.get('strStadium', 'Unknown')}"
        )
    except Exception as exc:
        return f"I couldn't retrieve team information right now: {exc}"


@tool
def get_recent_results(team_name: str) -> str:
    """Get the five most recent results for a football team."""
    try:
        canonical_name, events = _recent_events(team_name, 5)
        if not events:
            return f"No recent results found for {canonical_name}."
        lines = [f"Recent results for {canonical_name}:"]
        for event in events:
            lines.append(
                f"- {event.get('strEvent', 'Match')}: "
                f"{event.get('intHomeScore', '?')}-{event.get('intAwayScore', '?')}"
            )
        return "\n".join(lines)
    except Exception as exc:
        return f"I couldn't retrieve recent results right now: {exc}"


@tool
def get_upcoming_fixtures(team_name: str) -> str:
    """Get upcoming fixtures for a football team."""
    try:
        team_id, canonical_name = _team_record(team_name)
        data = _safe_get_json(
            f"{SPORTSDB_BASE_URL}/eventsnext.php",
            params={"id": team_id},
        )
        events = data.get("events") or []
        if not events:
            return f"No upcoming fixtures found for {canonical_name}."
        lines = [f"Upcoming fixtures for {canonical_name}:"]
        for event in events[:5]:
            lines.append(
                f"- {event.get('strEvent', 'Match')} - {event.get('dateEvent', 'Date unavailable')}"
            )
        return "\n".join(lines)
    except Exception as exc:
        return f"I couldn't retrieve upcoming fixtures right now: {exc}"


@tool
def get_team_players(team_name: str) -> str:
    """Get a list of players for a football team."""
    try:
        team_id, canonical_name = _team_record(team_name)
        data = _safe_get_json(
            f"{SPORTSDB_BASE_URL}/lookup_all_players.php",
            params={"id": team_id},
        )
        players = data.get("player") or []
        if not players:
            return f"No players found for {canonical_name}."
        lines = [f"Players for {canonical_name}:"]
        for player in players[:15]:
            lines.append(
                f"- {player.get('strPlayer', 'Unknown')} "
                f"({player.get('strPosition', 'Position unavailable')})"
            )
        return "\n".join(lines)
    except Exception as exc:
        return f"I couldn't retrieve the squad right now: {exc}"


@tool
def search_player(player_name: str) -> str:
    """Search for a football player and return matching player records."""
    try:
        data = _safe_get_json(
            "https://free-api-live-football-data.p.rapidapi.com/football-players-search",
            headers={
                "x-rapidapi-host": "free-api-live-football-data.p.rapidapi.com",
                "x-rapidapi-key": _require_env("RAPIDAPI_KEY"),
            },
            params={"search": player_name},
        )
        players = _extract_player_records(data)
        if not players:
            return f"No player found for '{player_name}'."

        lines = ["Players found:"]
        for player in players[:5]:
            name = (
                player.get("name")
                or player.get("playerName")
                or player.get("strPlayer")
                or "Unknown"
            )
            team_value = player.get("team")
            if isinstance(team_value, dict):
                team = team_value.get("name") or team_value.get("shortName")
            else:
                team = team_value
            team = team or player.get("teamName") or player.get("clubName") or "Team unavailable"
            position = player.get("position") or player.get("strPosition")
            suffix = f" - {team}"
            if position:
                suffix += f" ({position})"
            lines.append(f"- {name}{suffix}")
        return "\n".join(lines)
    except Exception as exc:
        return f"I couldn't search for that player right now: {exc}"


@tool
def get_league_standings(league_name: str) -> str:
    """Get current standings for a supported football league, with a web-search fallback."""
    code = _league_code(league_name)
    if not code:
        return (
            f"League '{league_name}' is not supported. Try Premier League, "
            "La Liga, Bundesliga, Serie A, Ligue 1, or Champions League."
        )

    primary_error = None
    try:
        data = _safe_get_json(
            f"{FOOTBALL_DATA_BASE_URL}/competitions/{code}/standings",
            headers=_football_data_headers(),
        )
        standings = data.get("standings") or []
        table = standings[0].get("table", []) if standings else []
        if table:
            lines = [f"{league_name.title()} standings (football-data.org):"]
            for item in table[:10]:
                lines.append(
                    f"{item.get('position', '?')}. "
                    f"{item.get('team', {}).get('name', 'Unknown')} - "
                    f"{item.get('points', 0)} pts"
                )
            return "\n".join(lines)
        primary_error = "standings table was empty"
    except Exception as exc:
        primary_error = str(exc)

    try:
        tavily = TavilyClient(api_key=_require_env("TAVILY_API_KEY"))
        year = datetime.now(timezone.utc).year
        results = tavily.search(
            query=f"{league_name} current standings table {year}",
            max_results=3,
        )
        items = results.get("results") or []
        if not items:
            raise RuntimeError("no fallback search results")

        lines = [
            f"{league_name.title()} standings are temporarily unavailable from the structured data provider.",
            "I found related current web references, but they are not reliable enough to reconstruct a complete league table.",
            "Do not infer, fill in, or invent positions, matches played, points, goals, or missing teams from these snippets.",
        ]
        for item in items:
            lines.append(f"- Reference: {item.get('title', 'Untitled')}")
        return "\n".join(lines)
    except Exception as fallback_exc:
        return (
            "I couldn't retrieve the standings right now. "
            f"Primary source error: {primary_error}. "
            f"Fallback error: {fallback_exc}"
        )


@tool
def get_live_scores() -> str:
    """Get football matches currently marked as live."""
    try:
        data = _safe_get_json(
            f"{FOOTBALL_DATA_BASE_URL}/matches",
            headers=_football_data_headers(),
            params={"status": "LIVE"},
        )
        matches = data.get("matches") or []
        if not matches:
            return "No live matches are available right now."

        lines = ["Live scores:"]
        for match in matches[:10]:
            score = match.get("score", {})
            full_time = score.get("fullTime") or {}
            home_score = full_time.get("home")
            away_score = full_time.get("away")
            if home_score is None or away_score is None:
                half_time = score.get("halfTime") or {}
                home_score = half_time.get("home", "?")
                away_score = half_time.get("away", "?")
            lines.append(
                f"- {match.get('homeTeam', {}).get('name', 'Home')} "
                f"{home_score} - {away_score} "
                f"{match.get('awayTeam', {}).get('name', 'Away')}"
            )
        return "\n".join(lines)
    except Exception as exc:
        return f"I couldn't retrieve live scores right now: {exc}"


@tool
def get_transfer_news(query: str) -> str:
    """Search the web for recent football transfer news, injuries, and updates."""
    try:
        tavily = TavilyClient(api_key=_require_env("TAVILY_API_KEY"))
        year = datetime.now(timezone.utc).year
        results = tavily.search(
            query=f"football {query} {year}",
            max_results=3,
        )
        items = results.get("results") or []
        if not items:
            return "No recent football news was found."

        lines = [f"Latest football news on '{query}':"]
        for item in items:
            snippet = _clean_web_snippet(item.get("content"), 180)
            url = str(item.get("url") or "").strip()
            entry = f"- {item.get('title', 'Untitled')}"
            if snippet:
                entry += f"\n  {snippet}"
            if url:
                entry += f"\n  Source: {url}"
            lines.append(entry)
        return "\n\n".join(lines)
    except Exception as exc:
        return f"I couldn't search the latest football news right now: {exc}"


@tool
def get_top_scorers(league_name: str) -> str:
    """Get top scorers for a supported football league."""
    code = _league_code(league_name)
    if not code:
        return (
            f"League '{league_name}' is not supported. Try Premier League, "
            "La Liga, Bundesliga, Serie A, Ligue 1, or Champions League."
        )

    try:
        data = _safe_get_json(
            f"{FOOTBALL_DATA_BASE_URL}/competitions/{code}/scorers",
            headers=_football_data_headers(),
            params={"season": str(_current_season_start_year())},
        )
        scorers = data.get("scorers") or []
        if not scorers:
            return "No scorer data is available right now."

        lines = [f"Top scorers - {league_name.title()}:"]
        for index, scorer in enumerate(scorers[:10], start=1):
            goals = int(scorer.get("goals") or 0)
            assists = int(scorer.get("assists") or 0)
            goal_word = "goal" if goals == 1 else "goals"
            assist_word = "assist" if assists == 1 else "assists"
            lines.append(
                f"{index}. {scorer.get('player', {}).get('name', 'Unknown')} "
                f"({scorer.get('team', {}).get('shortName', 'Unknown')}) - "
                f"{goals} {goal_word}, {assists} {assist_word}"
            )
        return "\n".join(lines)
    except Exception as exc:
        return f"I couldn't retrieve top scorers right now: {exc}"


@tool
def get_player_injury(player_name: str) -> str:
    """Search for recent injury information about a football player."""
    try:
        tavily = TavilyClient(api_key=_require_env("TAVILY_API_KEY"))
        year = datetime.now(timezone.utc).year
        results = tavily.search(
            query=f"{player_name} injury update return training status football {year}",
            max_results=4,
        )
        items = results.get("results") or []
        if not items:
            return f"No recent injury news was found for {player_name}."

        surname = player_name.strip().split()[-1] if player_name.strip() else player_name
        terms = (surname, "injury", "return", "training", "fitness", "out", "available")
        lines = [f"Latest injury sources for {player_name}:"]
        added = 0
        for item in items:
            snippet = _relevant_sentences(item.get("content"), terms, 190)
            url = str(item.get("url") or "").strip()
            title = str(item.get("title") or "Untitled").strip()
            if not snippet and not url:
                continue
            entry = f"- {title}"
            if snippet:
                entry += f"\n  {snippet}"
            if url:
                entry += f"\n  Source: {url}"
            lines.append(entry)
            added += 1
            if added >= 3:
                break

        if added == 0:
            return f"I found search results for {player_name}, but none had a clean injury update I could verify."
        return "\n\n".join(lines)
    except Exception as exc:
        return f"I couldn\'t search injury information right now: {exc}"


@tool
def predict_match(team1: str, team2: str) -> str:
    """Predict a match using season performance, recent form, venue, and a Poisson scoring model."""
    try:
        context = _build_prediction_context(team1, team2)
        first_name = context["first_name"]
        second_name = context["second_name"]

        evidence_lines = []
        for name, season, recent, source in (
            (first_name, context["first_season"], context["first_recent"], context["first_source"]),
            (second_name, context["second_season"], context["second_recent"], context["second_source"]),
        ):
            details = []
            if season is not None:
                details.append(f"{season['points']} pts from {season['played']} league matches")
                details.append(f"position {season['position']}")
            if recent is not None:
                details.append(
                    f"recent sample {recent['wins']}W {recent['draws']}D "
                    f"{recent['losses']}L from {recent['played']} "
                    f"{'match' if int(recent['played']) == 1 else 'matches'}"
                )
            evidence_lines.append(f"- {name}: {source}; " + "; ".join(details))

        score_line = ""
        projection = context["projection"]
        fixture = context["fixture"]
        first_is_home = context["first_is_home"]
        if projection is not None:
            best = projection["top_scores"][0]
            if fixture is not None and first_is_home is False:
                score_home = best[1]
                score_away = best[0]
                score_label = (
                    f"{fixture['home']} {score_home}-{score_away} {fixture['away']}"
                )
            elif fixture is not None:
                score_label = (
                    f"{fixture['home']} {best[0]}-{best[1]} {fixture['away']}"
                )
            else:
                score_label = f"{first_name} {best[0]}-{best[1]} {second_name}"

            score_line = (
                f"\nMost likely exact scoreline: {score_label} "
                f"({best[2] * 100:.1f}% as a single exact score)\n"
            )

        return (
            f"Match prediction: {first_name} vs {second_name}\n\n"
            f"{context['venue_note']}\n\n"
            "Data used:\n"
            + "\n".join(evidence_lines)
            + f"\n\nModel: {context['probability_source']}\n"
            + "\nEstimated probabilities:\n"
            f"- {first_name}: {context['first_probability'] * 100:.1f}%\n"
            f"- Draw: {context['draw_probability'] * 100:.1f}%\n"
            f"- {second_name}: {context['second_probability'] * 100:.1f}%\n\n"
            f"Overall outcome edge: {context['likely_outcome']}\n"
            f"Confidence: {context['confidence']}"
            + score_line
            + "\nThis is a statistical estimate, not a guaranteed result or betting advice."
        )
    except Exception as exc:
        return f"I couldn't generate a match prediction right now: {exc}"


@tool
def predict_correct_score(team1: str, team2: str) -> str:
    """Project the most likely exact scorelines using a simple Poisson scoring model."""
    try:
        context = _build_prediction_context(team1, team2)
        projection = context["projection"]
        first_name = context["first_name"]
        second_name = context["second_name"]
        if projection is None:
            return (
                f"I can predict the match outcome for {first_name} vs {second_name}, "
                "but I do not have enough scoring data for an exact-score projection."
            )

        fixture = context["fixture"]
        first_is_home = context["first_is_home"]
        if fixture is not None:
            matchup_title = f"{fixture['home']} vs {fixture['away']}"
        else:
            matchup_title = f"{first_name} vs {second_name}"

        lines = [
            f"Correct-score projection: {matchup_title}",
            context["venue_note"],
            "",
            "Most likely scorelines:",
        ]
        for index, (first_goals, second_goals, probability) in enumerate(
            projection["top_scores"],
            start=1,
        ):
            if fixture is not None and first_is_home is False:
                home_goals = second_goals
                away_goals = first_goals
                score_text = (
                    f"{fixture['home']} {home_goals}-{away_goals} {fixture['away']}"
                )
            elif fixture is not None:
                score_text = (
                    f"{fixture['home']} {first_goals}-{second_goals} {fixture['away']}"
                )
            else:
                score_text = (
                    f"{first_name} {first_goals}-{second_goals} {second_name}"
                )

            lines.append(
                f"{index}. {score_text} ({probability * 100:.1f}%)"
            )
        lines.extend(
            [
                "",
                "These are model estimates from current scoring/conceding rates, not guaranteed scores.",
            ]
        )
        return "\n".join(lines)
    except Exception as exc:
        return f"I couldn't generate a correct-score projection right now: {exc}"


@tool
def predict_betting_markets(team1: str, team2: str, market: str = "all") -> str:
    """Estimate common football betting-market probabilities for a matchup."""
    try:
        context = _build_prediction_context(team1, team2)
        if context.get("projection") is None:
            return (
                f"I can estimate the match outcome for {team1} vs {team2}, "
                "but I do not have enough scoring data to model betting markets."
            )
        return _betting_market_report(context, market)
    except Exception as exc:
        return f"I couldn't generate betting-market probabilities right now: {exc}"


def _concise_platform_prediction_report(
    context: dict[str, Any],
    adapter: Any,
) -> str:
    """Build a short platform-oriented model snapshot suitable for WhatsApp."""
    home, away, matrix = _fixture_order_projection(context)

    home_win = sum(p for h, a, p in matrix if h > a)
    draw = sum(p for h, a, p in matrix if h == a)
    away_win = sum(p for h, a, p in matrix if h < a)
    btts_yes = sum(p for h, a, p in matrix if h > 0 and a > 0)

    over_15 = sum(p for h, a, p in matrix if h + a > 1.5)
    under_35 = sum(p for h, a, p in matrix if h + a < 3.5)
    home_plus_05 = home_win + draw
    away_plus_05 = away_win + draw

    top_score = max(matrix, key=lambda item: item[2])
    score_h, score_a, score_p = top_score

    lines = [
        f"Platform: {adapter.display_name}",
        f"{home} vs {away}",
        context["venue_note"],
        "",
        "Model snapshot:",
        f"- {platform_market_name(adapter, '1x2')}: {home} {home_win * 100:.1f}% | Draw {draw * 100:.1f}% | {away} {away_win * 100:.1f}%",
        f"- {platform_market_name(adapter, 'double_chance')} 1X: {(home_win + draw) * 100:.1f}%",
        f"- {platform_market_name(adapter, 'double_chance')} X2: {(draw + away_win) * 100:.1f}%",
        f"- {platform_market_name(adapter, 'total_goals')} Over 1.5: {over_15 * 100:.1f}%",
        f"- {platform_market_name(adapter, 'total_goals')} Under 3.5: {under_35 * 100:.1f}%",
        f"- {platform_market_name(adapter, 'btts')} Yes: {btts_yes * 100:.1f}% | No: {(1.0 - btts_yes) * 100:.1f}%",
        f"- {platform_market_name(adapter, 'asian_handicap')} {home} +0.5: {home_plus_05 * 100:.1f}%",
        f"- {platform_market_name(adapter, 'asian_handicap')} {away} +0.5: {away_plus_05 * 100:.1f}%",
        "",
        f"Most likely exact score: {home} {score_h}-{score_a} {away} ({score_p * 100:.1f}%)",
        "",
        "Ask 'all betting predictions for them' for the full market sheet.",
        "Model probabilities only; not guaranteed outcomes or betting advice.",
    ]
    return "\n".join(lines)


@tool
def format_prediction_for_platform(
    team1: str,
    team2: str,
    platform: str,
) -> str:
    """Format modelled football markets for a supported betting platform without placing a wager."""
    try:
        adapter = get_platform(platform)
        if adapter is None:
            return (
                "That betting platform is not configured yet. "
                "Supported platforms: SportyBet, Bet9ja, BetKing, MSport, 1xBet, and Betway."
            )

        context = _build_prediction_context(team1, team2)
        if context.get("projection") is None:
            return (
                f"I can identify {adapter.display_name}, but I do not have enough scoring data "
                f"to build market probabilities for {team1} vs {team2}."
            )

        report = _concise_platform_prediction_report(context, adapter)
        return report + "\n\n" + platform_capability_summary(adapter)
    except Exception as exc:
        return f"I couldn't format that platform analysis right now: {exc}"


def _specifier_number(text: str, key: str) -> float | None:
    match = re.search(
        rf"(?:^|[;,&\s]){re.escape(key)}\s*=\s*([+-]?\d+(?:\.\d+)?)",
        text or "",
        flags=re.IGNORECASE,
    )
    if match:
        return float(match.group(1))
    return None


def _settlement_support(settlement: dict[str, float]) -> float:
    return (
        float(settlement.get("full_win", 0.0))
        + 0.5 * float(settlement.get("half_win", 0.0))
    )


def _sportybet_market_can_model(selection: dict[str, Any]) -> bool:
    """Return False before any sports API calls for markets this model cannot support."""
    market = str(selection.get("market_name") or "").casefold().strip()
    market_id = str(selection.get("market_id") or "")

    unsupported_scope_terms = (
        "corner",
        "1st half",
        "first half",
        "2nd half",
        "second half",
        "both halves",
        "either half",
        "player",
        "card",
        "booking",
        "shots",
    )
    if any(term in market for term in unsupported_scope_terms):
        return False

    if market_id in {"1", "10", "11", "16", "18", "23", "24", "29", "45"}:
        return True

    supported_terms = (
        "1x2",
        "match result",
        "double chance",
        "draw no bet",
        "gg/ng",
        "both teams to score",
        "btts",
        "over/under",
        "total goals",
        "asian handicap",
        "correct score",
    )
    if any(term in market for term in supported_terms):
        return True

    if re.search(
        r"(?:home(?: team)?|away(?: team)?)\s+or\s+over\s+[0-9]+(?:\.[0-9]+)?",
        market,
    ):
        return True

    return False


def _sportybet_selection_model_support(
    context: dict[str, Any],
    selection: dict[str, Any],
) -> tuple[float | None, str]:
    """Return model support for a normalized SportyBet selection."""
    home, away, matrix = _fixture_order_projection(context)
    market = str(selection.get("market_name") or "").casefold().strip()
    market_id = str(selection.get("market_id") or "")
    outcome = str(selection.get("outcome_name") or "").casefold().strip()
    outcome_id = str(selection.get("outcome_id") or "")
    specifier = str(selection.get("specifier") or "")

    home_win = sum(p for h, a, p in matrix if h > a)
    draw = sum(p for h, a, p in matrix if h == a)
    away_win = sum(p for h, a, p in matrix if h < a)

    # Do not reinterpret event-level or half-specific markets as full-match goals.
    unsupported_scope_terms = (
        "corner",
        "1st half",
        "first half",
        "2nd half",
        "second half",
        "both halves",
        "either half",
        "player",
        "card",
        "booking",
        "shots",
    )
    if any(term in market for term in unsupported_scope_terms):
        return None, str(selection.get("market_name") or "Unsupported market")

    # Full-match 1X2 only.
    if market_id == "1" or market in {"1x2", "match result"}:
        if outcome_id == "1" or "home" in outcome or outcome == "1":
            return home_win, "1X2 home"
        if outcome_id == "2" or "draw" in outcome or outcome == "x":
            return draw, "1X2 draw"
        if outcome_id == "3" or "away" in outcome or outcome == "2":
            return away_win, "1X2 away"

    # Full-match Double Chance. SportyBet may use either 1/X/2 or words.
    if market_id == "10" or market == "double chance":
        normalized_outcome = re.sub(r"[^a-z0-9]", "", outcome)
        if (
            "homeordraw" in normalized_outcome
            or normalized_outcome in {"1x", "x1"}
        ):
            return home_win + draw, "Double Chance 1X"
        if (
            "draworaway" in normalized_outcome
            or normalized_outcome in {"x2", "2x"}
        ):
            return draw + away_win, "Double Chance X2"
        if (
            "homeoraway" in normalized_outcome
            or normalized_outcome in {"12", "21"}
        ):
            return home_win + away_win, "Double Chance 12"

    # Full-match BTTS only.
    if (
        market_id == "29"
        or market in {"gg/ng", "both teams to score", "btts"}
    ):
        yes = sum(p for h, a, p in matrix if h > 0 and a > 0)
        is_yes = (
            "yes" in outcome
            or outcome == "gg"
            or outcome_id.casefold() in {"yes", "gg"}
        )
        is_no = (
            "no" in outcome
            or outcome == "ng"
            or outcome_id.casefold() in {"no", "ng"}
        )
        if is_yes:
            return yes, "Both Teams to Score - Yes"
        if is_no:
            return 1.0 - yes, "Both Teams to Score - No"

    if market_id == "11" or market == "draw no bet":
        decisive = home_win + away_win
        if decisive <= 0:
            return None, "Draw No Bet"
        if "home" in outcome or outcome == "1":
            return home_win / decisive, "Draw No Bet - Home"
        if "away" in outcome or outcome == "2":
            return away_win / decisive, "Draw No Bet - Away"

    # Team-specific goal totals can be derived from the same score matrix.
    team_total_side: str | None = None
    home_key = _normalize_team_name(home)
    away_key = _normalize_team_name(away)
    market_key = _normalize_team_name(market)
    if "over/under" in market:
        if home_key and home_key in market_key:
            team_total_side = "home"
        elif away_key and away_key in market_key:
            team_total_side = "away"

    if team_total_side is not None:
        line = _specifier_number(specifier, "total")
        if line is None:
            number = re.search(r"([0-9]+(?:\.[0-9]+)?)", outcome)
            if number:
                line = float(number.group(1))

        side = "over" if "over" in outcome else "under" if "under" in outcome else None
        if line is not None and side:
            if team_total_side == "home":
                over_probability = sum(p for h, _a, p in matrix if h > line)
                team_name = home
            else:
                over_probability = sum(p for _h, a, p in matrix if a > line)
                team_name = away

            probability = over_probability if side == "over" else 1.0 - over_probability
            return probability, f"{team_name} {side.title()} {line:g} Goals"

    # Full-match goal total only. This deliberately excludes corner/card totals.
    if market_id == "18" or market in {"over/under", "total goals", "goals over/under"}:
        line = _specifier_number(specifier, "total")
        if line is None:
            joined = f"{market} {outcome}"
            number = re.search(r"([0-9]+(?:\.[0-9]+)?)", joined)
            if number:
                line = float(number.group(1))
        side = None
        if "over" in outcome or outcome_id == "12":
            side = "over"
        elif "under" in outcome or outcome_id == "13":
            side = "under"
        if line is not None and side:
            settlement = _total_line_settlement(matrix, line, side=side)
            return (
                _settlement_support(settlement),
                f"{side.title()} {line:g} Goals",
            )

    # SportyBet combo markets such as "Away or Over 2.5".
    combo = re.search(r"(home(?: team)?|away(?: team)?)\s+or\s+over\s+([0-9]+(?:\.[0-9]+)?)", market)
    if combo:
        selected_side = combo.group(1)
        line = float(combo.group(2))
        probability = sum(
            p
            for h, a, p in matrix
            if (
                (h > a if selected_side.startswith("home") else a > h)
                or (h + a > line)
            )
        )
        if "no" in outcome:
            probability = 1.0 - probability
        elif "yes" not in outcome:
            return None, str(selection.get("market_name") or "Unsupported market")
        label_side = home if selected_side.startswith("home") else away
        return probability, f"{label_side} or Over {line:g} Goals"

    if "asian handicap" in market or market_id == "16":
        home_line = _specifier_number(specifier, "hcp")

        # Some SportyBet payloads include the selected line in the pick text.
        pick_line_match = re.search(r"([+-]\d+(?:\.\d+)?)", outcome)
        pick_line = float(pick_line_match.group(1)) if pick_line_match else None

        if "home" in outcome or outcome == "1":
            line = pick_line if pick_line is not None else home_line
            if line is not None:
                settlement = _asian_handicap_settlement(matrix, line, side="home")
                return _settlement_support(settlement), f"{home} {line:+g}"

        if "away" in outcome or outcome == "2":
            if pick_line is not None:
                line = pick_line
            elif home_line is not None:
                line = -home_line
            else:
                line = None
            if line is not None:
                settlement = _asian_handicap_settlement(matrix, line, side="away")
                return _settlement_support(settlement), f"{away} {line:+g}"

    if market_id == "45" or market == "correct score":
        score = re.search(r"(\d+)\s*[-:]\s*(\d+)", outcome)
        if score:
            wanted_home = int(score.group(1))
            wanted_away = int(score.group(2))
            probability = sum(
                p
                for h, a, p in matrix
                if h == wanted_home and a == wanted_away
            )
            return probability, f"Correct Score {wanted_home}-{wanted_away}"

    return None, str(selection.get("market_name") or "Unsupported market")

def _sportybet_context_has_enough_data(context: dict[str, Any]) -> bool:
    """Avoid confident booking-slip probabilities from tiny recent-form samples."""
    for season_key, recent_key in (
        ("first_season", "first_recent"),
        ("second_season", "second_recent"),
    ):
        if context.get(season_key) is not None:
            continue
        recent = context.get(recent_key)
        if recent is None or int(recent.get("played") or 0) < 3:
            return False
    return True


def _sportybet_platform_probability(
    selection: dict[str, Any],
) -> tuple[float | None, str]:
    """Return SportyBet/provider probability metadata without treating it as our model."""
    source_probability = selection.get("source_probability")
    if isinstance(source_probability, (int, float)) and 0.0 <= source_probability <= 1.0:
        return float(source_probability), "SportyBet feed probability"

    odds = selection.get("odds")
    if isinstance(odds, (int, float)) and odds > 1.0:
        return min(1.0, 1.0 / float(odds)), "Raw odds-implied chance"

    return None, ""


def _model_alignment(probability: float) -> str:
    if probability >= 0.65:
        return "higher model support"
    if probability >= 0.50:
        return "moderate model support"
    return "lower model support"


def _run_sportybet_booking_analysis(booking_code: str) -> dict[str, Any]:
    """Load and analyse a SportyBet booking code once, returning structured results."""
    clean = normalize_booking_code(booking_code)
    payload = fetch_booking(clean)
    booking = extract_booking(payload)
    selections = booking.get("selections") or []
    if not selections:
        raise SportyBetLookupError(
            f"SportyBet code {clean} loaded, but no readable selections were returned. "
            "The code may be expired or the SportyBet response format may have changed."
        )

    def evaluate_selection(selection: dict[str, Any]) -> dict[str, Any]:
        market_name = str(selection.get("market_name") or "Unknown market")
        if not _sportybet_market_can_model(selection):
            return {
                "probability": None,
                "label": market_name,
                "reason": "unsupported_market",
            }

        home = str(selection.get("home_team") or "").strip()
        away = str(selection.get("away_team") or "").strip()
        if not home or not away:
            return {
                "probability": None,
                "label": market_name,
                "reason": "missing_match_data",
            }

        try:
            context = _build_prediction_context(
                home,
                away,
                known_fixture={"home": home, "away": away, "date": ""},
                recent_only=False,
                prefer_season_only=True,
            )
            if (
                context.get("projection") is None
                or not _sportybet_context_has_enough_data(context)
            ):
                return {
                    "probability": None,
                    "label": market_name,
                    "reason": "insufficient_independent_data",
                }

            probability, label = _sportybet_selection_model_support(
                context,
                selection,
            )
            return {
                "probability": probability,
                "label": label,
                "reason": "modelled" if probability is not None else "unsupported_market",
            }
        except Exception:
            return {
                "probability": None,
                "label": market_name,
                "reason": "independent_data_error",
            }

    with ThreadPoolExecutor(max_workers=8) as executor:
        evaluations = list(executor.map(evaluate_selection, selections))

    records: list[dict[str, Any]] = []
    counts = {"higher": 0, "moderate": 0, "lower": 0, "unmodelled": 0}

    for index, (selection, evaluation) in enumerate(
        zip(selections, evaluations),
        start=1,
    ):
        platform_probability, platform_probability_label = (
            _sportybet_platform_probability(selection)
        )
        model_probability = evaluation["probability"]
        model_label = evaluation["label"]
        model_reason = evaluation.get("reason") or (
            "modelled" if model_probability is not None else "independent_data_unavailable"
        )

        alignment = None
        gap = None
        if model_probability is None:
            counts["unmodelled"] += 1
        else:
            alignment = _model_alignment(model_probability)
            if model_probability >= 0.65:
                counts["higher"] += 1
            elif model_probability >= 0.50:
                counts["moderate"] += 1
            else:
                counts["lower"] += 1

            if platform_probability is not None:
                gap = (model_probability - platform_probability) * 100.0

        records.append(
            {
                "index": index,
                "source_index": index,
                "event_id": str(selection.get("event_id") or ""),
                "market_id": str(selection.get("market_id") or ""),
                "specifier": selection.get("specifier"),
                "outcome_id": str(selection.get("outcome_id") or ""),
                "home_team": str(selection.get("home_team") or "Home"),
                "away_team": str(selection.get("away_team") or "Away"),
                "market_name": str(selection.get("market_name") or "Unknown market"),
                "outcome_name": str(selection.get("outcome_name") or "Unknown pick"),
                "odds": selection.get("odds"),
                "platform_probability": platform_probability,
                "platform_probability_label": platform_probability_label,
                "model_probability": model_probability,
                "model_label": model_label,
                "alignment": alignment,
                "gap": gap,
                "model_reason": model_reason,
            }
        )

    return {
        "code": clean,
        "selection_count": len(records),
        "records": records,
        "counts": counts,
    }


def _format_sportybet_analysis_summary(analysis: dict[str, Any]) -> str:
    """Format a compact default WhatsApp summary for a booking-code analysis."""
    records = analysis["records"]
    counts = analysis["counts"]
    modelled = len(records) - int(counts["unmodelled"])

    lines = [
        f"SportyBet Code: {analysis['code']}",
        f"{analysis['selection_count']} selections found",
        "",
        "Independent model coverage:",
        f"- Modelled: {modelled}",
        f"- Not independently modelled: {counts['unmodelled']}",
        f"- Higher support: {counts['higher']}",
        f"- Moderate support: {counts['moderate']}",
        f"- Lower support: {counts['lower']}",
    ]

    disagreements = [
        record
        for record in records
        if record.get("gap") is not None
    ]
    disagreements.sort(key=lambda item: abs(float(item["gap"])), reverse=True)

    if disagreements:
        lines.extend(["", "Biggest model vs platform differences:"])
        for record in disagreements[:4]:
            lines.append(
                f"- {record['home_team']} vs {record['away_team']} | "
                f"{record['outcome_name']} | "
                f"platform {record['platform_probability'] * 100:.1f}% vs "
                f"model {record['model_probability'] * 100:.1f}% "
                f"({record['gap']:+.1f} pp)"
            )

    unsupported_markets: list[str] = []
    seen_markets: set[str] = set()
    insufficient_count = 0
    error_count = 0

    for record in records:
        if record.get("model_probability") is not None:
            continue

        reason = str(record.get("model_reason") or "")
        if reason == "unsupported_market":
            market = str(record.get("market_name") or "Unknown market")
            key = market.casefold()
            if key not in seen_markets:
                seen_markets.add(key)
                unsupported_markets.append(market)
        elif reason in {"insufficient_independent_data", "missing_match_data"}:
            insufficient_count += 1
        elif reason == "independent_data_error":
            error_count += 1

    if unsupported_markets:
        preview = ", ".join(unsupported_markets[:5])
        if len(unsupported_markets) > 5:
            preview += f", +{len(unsupported_markets) - 5} more"
        lines.extend(["", f"Unsupported market types: {preview}"])

    if insufficient_count:
        lines.append(
            f"Supported selections lacking enough independent football data: "
            f"{insufficient_count}"
        )

    if error_count:
        lines.append(
            f"Selections skipped because an independent data source failed: "
            f"{error_count}"
        )

    lines.extend(
        [
            "",
            "Reply:",
            "- SHOW MODELLED",
            "- SHOW UNMODELLED",
            "- SHOW ALL",
            "",
            "The detailed result is cached, so these follow-ups do not rerun the full analysis.",
            "Probabilities are estimates, not guaranteed outcomes or betting advice.",
        ]
    )
    return "\n".join(lines)


def _format_sportybet_record(record: dict[str, Any]) -> str:
    lines = [
        f"{record['index']}. {record['home_team']} vs {record['away_team']}",
        f"- Market: {record['market_name']}",
        f"- Pick: {record['outcome_name']}",
    ]

    odds = record.get("odds")
    if isinstance(odds, (int, float)):
        lines.append(f"- SportyBet odds: {odds:.2f}")

    platform_probability = record.get("platform_probability")
    if platform_probability is not None:
        lines.append(
            f"- {record['platform_probability_label']}: "
            f"{platform_probability * 100:.1f}%"
        )

    model_probability = record.get("model_probability")
    if model_probability is None:
        reason = str(record.get("model_reason") or "")
        if reason == "unsupported_market":
            lines.append("- Independent model comparison: market not currently modelled")
        elif reason in {"insufficient_independent_data", "missing_match_data"}:
            lines.append("- Independent model comparison: not enough independent football data")
        elif reason == "independent_data_error":
            lines.append("- Independent model comparison: independent data source unavailable")
        else:
            lines.append("- Independent model comparison: unavailable")
    else:
        lines.append(
            f"- Independent model support: {model_probability * 100:.1f}% "
            f"({record['alignment']})"
        )
        gap = record.get("gap")
        if gap is not None:
            lines.append(f"- Model vs platform gap: {gap:+.1f} percentage points")

        model_label = str(record.get("model_label") or "")
        market_name = str(record.get("market_name") or "")
        if model_label and model_label.casefold() != market_name.casefold():
            lines.append(f"- Interpreted as: {model_label}")

    return "\n".join(lines)


def _format_sportybet_cached_view(
    analysis: dict[str, Any],
    view: str,
) -> str:
    records = analysis["records"]
    normalized_view = view.casefold().strip()

    if normalized_view == "modelled":
        selected = [
            record
            for record in records
            if record.get("model_probability") is not None
        ]
        title = "Selections with independent model comparison"
    elif normalized_view == "unmodelled":
        selected = [
            record
            for record in records
            if record.get("model_probability") is None
        ]
        title = "Selections without independent model comparison"
    else:
        selected = records
        title = "All selections"

    if not selected:
        return (
            f"SportyBet Code: {analysis['code']}\n"
            f"{title}: none."
        )

    lines = [
        f"SportyBet Code: {analysis['code']}",
        f"{title}: {len(selected)}",
        "",
    ]
    for record in selected:
        lines.append(_format_sportybet_record(record))
        lines.append("")

    lines.append("Loaded from the cached analysis; no full re-analysis was run.")
    return "\n".join(lines).strip()


def _sportybet_state_db_path() -> str:
    return os.getenv("MEMORY_DB_PATH", "memory.db")


def _ensure_sportybet_slip_table() -> None:
    connection = sqlite3.connect(_sportybet_state_db_path())
    try:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS sportybet_slip_state (
                user_id TEXT PRIMARY KEY,
                state_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.commit()
    finally:
        connection.close()


def _persist_sportybet_slip_state(
    user_id: str,
    state: dict[str, Any],
) -> None:
    if not user_id:
        return
    _ensure_sportybet_slip_table()
    payload = json.dumps(state, separators=(",", ":"))
    updated_at = datetime.now(timezone.utc).isoformat()
    connection = sqlite3.connect(_sportybet_state_db_path())
    try:
        connection.execute(
            """
            INSERT INTO sportybet_slip_state (user_id, state_json, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                state_json = excluded.state_json,
                updated_at = excluded.updated_at
            """,
            (user_id, payload, updated_at),
        )
        connection.commit()
    finally:
        connection.close()


def _load_sportybet_slip_state(
    user_id: str,
) -> dict[str, Any] | None:
    if not user_id:
        return None
    _ensure_sportybet_slip_table()
    connection = sqlite3.connect(_sportybet_state_db_path())
    try:
        row = connection.execute(
            """
            SELECT state_json
            FROM sportybet_slip_state
            WHERE user_id = ?
            """,
            (user_id,),
        ).fetchone()
    finally:
        connection.close()

    if not row:
        return None

    try:
        state = json.loads(str(row[0]))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return state if isinstance(state, dict) else None


def _clone_sportybet_records(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [dict(record) for record in records]


def _reindex_sportybet_records(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    cloned = _clone_sportybet_records(records)
    for index, record in enumerate(cloned, start=1):
        record["index"] = index
    return cloned


def _cache_sportybet_analysis(user_id: str, analysis: dict[str, Any]) -> None:
    if not user_id:
        return
    with _SPORTYBET_ANALYSIS_CACHE_LOCK:
        _SPORTYBET_ANALYSIS_CACHE[user_id] = analysis


def _cached_sportybet_analysis(user_id: str) -> dict[str, Any] | None:
    if not user_id:
        return None
    with _SPORTYBET_ANALYSIS_CACHE_LOCK:
        return _SPORTYBET_ANALYSIS_CACHE.get(user_id)


def _initialize_sportybet_working_slip(
    user_id: str,
    analysis: dict[str, Any],
) -> None:
    if not user_id:
        return
    current = _reindex_sportybet_records(
        list(analysis.get("records") or [])
    )
    state = {
        "source_code": str(analysis.get("code") or ""),
        "current": current,
        "undo": [],
        "redo": [],
    }
    with _SPORTYBET_SLIP_STATE_LOCK:
        _SPORTYBET_SLIP_STATES[user_id] = state
    _persist_sportybet_slip_state(user_id, state)


def _sportybet_working_slip(user_id: str) -> dict[str, Any] | None:
    if not user_id:
        return None

    with _SPORTYBET_SLIP_STATE_LOCK:
        state = _SPORTYBET_SLIP_STATES.get(user_id)

    if state is None:
        state = _load_sportybet_slip_state(user_id)
        if state is None:
            return None
        with _SPORTYBET_SLIP_STATE_LOCK:
            _SPORTYBET_SLIP_STATES[user_id] = state

    return {
        "source_code": state.get("source_code", ""),
        "current": _clone_sportybet_records(state.get("current") or []),
        "undo": [
            _clone_sportybet_records(snapshot)
            for snapshot in (state.get("undo") or [])
        ],
        "redo": [
            _clone_sportybet_records(snapshot)
            for snapshot in (state.get("redo") or [])
        ],
    }


def _replace_sportybet_working_slip(
    user_id: str,
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not user_id:
        raise ValueError("No WhatsApp user context is available.")

    updated = _reindex_sportybet_records(records)
    with _SPORTYBET_SLIP_STATE_LOCK:
        state = _SPORTYBET_SLIP_STATES.get(user_id)
        if state is None:
            raise ValueError(
                "Load or analyse a SportyBet code first."
            )
        undo = state.setdefault("undo", [])
        undo.append(
            _clone_sportybet_records(state.get("current") or [])
        )
        if len(undo) > 20:
            del undo[:-20]
        state["current"] = updated
        state["redo"] = []
        persisted = {
            "source_code": state.get("source_code", ""),
            "current": _clone_sportybet_records(state.get("current") or []),
            "undo": [
                _clone_sportybet_records(snapshot)
                for snapshot in (state.get("undo") or [])
            ],
            "redo": [],
        }
    _persist_sportybet_slip_state(user_id, persisted)
    return _clone_sportybet_records(updated)


def _undo_sportybet_slip(user_id: str) -> str:
    with _SPORTYBET_SLIP_STATE_LOCK:
        state = _SPORTYBET_SLIP_STATES.get(user_id)
        if state is None:
            return "Load or analyse a SportyBet code first."
        undo = state.get("undo") or []
        if not undo:
            return "There is nothing to undo on the current SportyBet slip."

        current = _clone_sportybet_records(state.get("current") or [])
        previous = undo.pop()
        state.setdefault("redo", []).append(current)
        state["current"] = _reindex_sportybet_records(previous)
        count = len(state["current"])
        persisted = {
            "source_code": state.get("source_code", ""),
            "current": _clone_sportybet_records(state.get("current") or []),
            "undo": [
                _clone_sportybet_records(snapshot)
                for snapshot in (state.get("undo") or [])
            ],
            "redo": [
                _clone_sportybet_records(snapshot)
                for snapshot in (state.get("redo") or [])
            ],
        }

    _persist_sportybet_slip_state(user_id, persisted)
    return f"Undone. The working SportyBet slip now has {count} selections."


def _redo_sportybet_slip(user_id: str) -> str:
    with _SPORTYBET_SLIP_STATE_LOCK:
        state = _SPORTYBET_SLIP_STATES.get(user_id)
        if state is None:
            return "Load or analyse a SportyBet code first."
        redo = state.get("redo") or []
        if not redo:
            return "There is nothing to redo on the current SportyBet slip."

        current = _clone_sportybet_records(state.get("current") or [])
        next_state = redo.pop()
        state.setdefault("undo", []).append(current)
        state["current"] = _reindex_sportybet_records(next_state)
        count = len(state["current"])
        persisted = {
            "source_code": state.get("source_code", ""),
            "current": _clone_sportybet_records(state.get("current") or []),
            "undo": [
                _clone_sportybet_records(snapshot)
                for snapshot in (state.get("undo") or [])
            ],
            "redo": [
                _clone_sportybet_records(snapshot)
                for snapshot in (state.get("redo") or [])
            ],
        }

    _persist_sportybet_slip_state(user_id, persisted)
    return f"Redone. The working SportyBet slip now has {count} selections."


def _sportybet_record_rank_score(record: dict[str, Any]) -> tuple[int, float]:
    model_probability = record.get("model_probability")
    if isinstance(model_probability, (int, float)):
        return 2, float(model_probability)

    platform_probability = record.get("platform_probability")
    if isinstance(platform_probability, (int, float)):
        return 1, float(platform_probability)

    odds = record.get("odds")
    if isinstance(odds, (int, float)) and float(odds) > 1.0:
        return 0, 1.0 / float(odds)

    return 0, 0.0


def _sportybet_record_confidence(record: dict[str, Any]) -> float:
    tier, probability = _sportybet_record_rank_score(record)
    if tier == 2:
        return probability
    if tier == 1:
        return probability * 0.82
    return probability * 0.70


def _sportybet_combined_odds(
    records: list[dict[str, Any]],
) -> float | None:
    odds = [
        float(record["odds"])
        for record in records
        if isinstance(record.get("odds"), (int, float))
        and float(record["odds"]) > 0
    ]
    if not records or len(odds) != len(records):
        return None
    return math.prod(odds)


def _format_sportybet_working_slip(
    user_id: str,
    *,
    detail_limit: int = 12,
) -> str:
    state = _sportybet_working_slip(user_id)
    if state is None:
        return "Load or analyse a SportyBet code first."

    records = state["current"]
    modelled = sum(
        1 for record in records
        if record.get("model_probability") is not None
    )
    combined = _sportybet_combined_odds(records)

    lines = [
        f"Working SportyBet slip from: {state.get('source_code') or 'new slip'}",
        f"Selections: {len(records)}",
        f"Independently modelled: {modelled}",
    ]
    if combined is not None:
        lines.append(f"Current combined odds snapshot: {combined:.2f}")

    lines.append("")
    for record in records[:detail_limit]:
        support = record.get("model_probability")
        support_text = (
            f" | model {float(support) * 100:.1f}%"
            if isinstance(support, (int, float))
            else ""
        )
        odds = record.get("odds")
        odds_text = (
            f" @ {float(odds):.2f}"
            if isinstance(odds, (int, float))
            else ""
        )
        lines.append(
            f"{record['index']}. {record['home_team']} vs "
            f"{record['away_team']} - {record['outcome_name']}"
            f"{odds_text}{support_text}"
        )

    if len(records) > detail_limit:
        lines.append(f"...and {len(records) - detail_limit} more.")

    lines.extend(
        [
            "",
            "You can REMOVE numbers, KEEP TOP N, REMOVE UNMODELLED, "
            "REMOVE LOWER SUPPORT, TARGET ODDS, UNDO, REDO, or CREATE NEW CODE.",
        ]
    )
    return "\n".join(lines)


def _remove_sportybet_slip_indexes(
    user_id: str,
    indexes: list[int],
) -> str:
    state = _sportybet_working_slip(user_id)
    if state is None:
        return "Load or analyse a SportyBet code first."

    current = state["current"]
    wanted = {int(index) for index in indexes if int(index) > 0}
    valid = {int(record["index"]) for record in current}
    missing = sorted(wanted - valid)
    if missing:
        return (
            "Those selection numbers are not in the current slip: "
            + ", ".join(str(item) for item in missing)
        )

    updated = [
        record
        for record in current
        if int(record["index"]) not in wanted
    ]
    if not updated:
        return "That would remove every selection. Keep at least one selection."

    updated = _replace_sportybet_working_slip(user_id, updated)
    return (
        f"Removed {len(wanted)} selection(s). "
        f"{len(updated)} remain.\n\n"
        + _format_sportybet_working_slip(user_id, detail_limit=8)
    )


def _keep_top_sportybet_slip(
    user_id: str,
    count: int,
) -> str:
    state = _sportybet_working_slip(user_id)
    if state is None:
        return "Load or analyse a SportyBet code first."

    current = state["current"]
    count = int(count)
    if count < 1:
        return "Keep at least one selection."
    if count >= len(current):
        return (
            f"The current slip already has {len(current)} selections."
        )

    ranked = sorted(
        current,
        key=_sportybet_record_rank_score,
        reverse=True,
    )
    selected_ids = {
        int(record["index"])
        for record in ranked[:count]
    }
    updated = [
        record for record in current
        if int(record["index"]) in selected_ids
    ]
    _replace_sportybet_working_slip(user_id, updated)
    return (
        f"Shortened the working slip to the strongest {count} selections "
        "available from the current analysis.\n\n"
        + _format_sportybet_working_slip(user_id, detail_limit=10)
    )


def _remove_unmodelled_sportybet_slip(user_id: str) -> str:
    state = _sportybet_working_slip(user_id)
    if state is None:
        return "Load or analyse a SportyBet code first."

    current = state["current"]
    updated = [
        record for record in current
        if record.get("model_probability") is not None
    ]
    if not updated:
        return (
            "None of the current selections has an independent model "
            "comparison, so I left the slip unchanged."
        )
    if len(updated) == len(current):
        return "Every current selection already has an independent model comparison."

    removed = len(current) - len(updated)
    _replace_sportybet_working_slip(user_id, updated)
    return (
        f"Removed {removed} unmodelled selection(s). "
        f"{len(updated)} remain.\n\n"
        + _format_sportybet_working_slip(user_id, detail_limit=10)
    )


def _remove_lower_support_sportybet_slip(user_id: str) -> str:
    state = _sportybet_working_slip(user_id)
    if state is None:
        return "Load or analyse a SportyBet code first."

    current = state["current"]
    updated = [
        record
        for record in current
        if (
            record.get("model_probability") is None
            or float(record["model_probability"]) >= 0.50
        )
    ]
    if len(updated) == len(current):
        return "There are no lower-support modelled selections to remove."
    if not updated:
        return "That would remove every selection, so I left the slip unchanged."

    removed = len(current) - len(updated)
    _replace_sportybet_working_slip(user_id, updated)
    return (
        f"Removed {removed} lower-support selection(s). "
        f"{len(updated)} remain.\n\n"
        + _format_sportybet_working_slip(user_id, detail_limit=10)
    )


def _remove_lowest_sportybet_slip(
    user_id: str,
    count: int,
) -> str:
    state = _sportybet_working_slip(user_id)
    if state is None:
        return "Load or analyse a SportyBet code first."

    current = state["current"]
    count = int(count)
    if count < 1:
        return "Remove at least one selection."
    if count >= len(current):
        return "That would remove every selection. Reduce the number."

    ranked_low = sorted(
        current,
        key=_sportybet_record_rank_score,
    )
    remove_ids = {
        int(record["index"])
        for record in ranked_low[:count]
    }
    updated = [
        record for record in current
        if int(record["index"]) not in remove_ids
    ]
    _replace_sportybet_working_slip(user_id, updated)
    return (
        f"Removed the lowest-ranked {count} selection(s). "
        f"{len(updated)} remain.\n\n"
        + _format_sportybet_working_slip(user_id, detail_limit=10)
    )


def _select_records_for_target_odds(
    records: list[dict[str, Any]],
    target_odds: float,
    *,
    exact_count: int | None = None,
) -> list[dict[str, Any]]:
    if target_odds <= 1.0:
        raise ValueError("Target combined odds must be above 1.00.")

    usable = [
        record
        for record in records
        if isinstance(record.get("odds"), (int, float))
        and 1.01 <= float(record["odds"]) <= target_odds * 1.5
    ]
    if exact_count is not None:
        exact_count = int(exact_count)
        if exact_count < 1:
            raise ValueError("Number of selections must be at least 1.")
        if exact_count > len(usable):
            raise ValueError(
                f"Only {len(usable)} selections have usable odds."
            )

    # Beam search in log-odds space. This stays fast for large slips while
    # balancing target closeness with independent/platform support.
    target_log = math.log(target_odds)
    states: list[tuple[float, float, tuple[int, ...]]] = [(0.0, 0.0, ())]

    for index, record in enumerate(usable):
        odd = float(record["odds"])
        confidence = _sportybet_record_confidence(record)
        additions: list[tuple[float, float, tuple[int, ...]]] = []
        for log_sum, confidence_sum, chosen in states:
            max_allowed = exact_count if exact_count is not None else 20
            if len(chosen) >= max_allowed:
                continue
            new_log = log_sum + math.log(odd)
            if new_log > math.log(target_odds * 2.0):
                continue
            additions.append(
                (
                    new_log,
                    confidence_sum + confidence,
                    chosen + (index,),
                )
            )

        states.extend(additions)
        states.sort(
            key=lambda state: (
                abs(state[0] - target_log)
                - 0.08
                * (
                    state[1] / max(1, len(state[2]))
                ),
                len(state[2]) == 0,
            )
        )
        states = states[:3000]

    candidates = [
        state for state in states
        if state[2]
        and (
            exact_count is None
            or len(state[2]) == exact_count
        )
        and (
            exact_count is not None
            or len(state[2]) >= min(2, len(usable))
        )
    ]
    if not candidates:
        raise ValueError(
            "I couldn't build a suitable subset for that target odds."
        )

    best = min(
        candidates,
        key=lambda state: (
            abs(state[0] - target_log),
            -state[1] / len(state[2]),
            len(state[2]),
        ),
    )
    return [usable[index] for index in best[2]]


def _target_sportybet_slip_odds(
    user_id: str,
    target_odds: float,
    *,
    exact_count: int | None = None,
) -> str:
    state = _sportybet_working_slip(user_id)
    if state is None:
        return "Load or analyse a SportyBet code first."

    current_combined = _sportybet_combined_odds(state["current"])
    requested_target = float(target_odds)
    if (
        current_combined is not None
        and abs(current_combined - requested_target)
        <= max(0.02, requested_target * 0.01)
    ):
        return (
            f"The current working slip is already about "
            f"{current_combined:.2f} odds, so I left it unchanged.\n\n"
            + _format_sportybet_working_slip(user_id, detail_limit=10)
        )

    if (
        current_combined is not None
        and current_combined < requested_target * 0.95
    ):
        return (
            f"The current working slip is only about {current_combined:.2f} odds, "
            f"so shortening it cannot raise it to {requested_target:.2f}.\n"
            "Use UNDO, reload the original code, or ask for a fresh today's "
            "code around that target."
        )

    try:
        selected = _select_records_for_target_odds(
            state["current"],
            float(target_odds),
            exact_count=exact_count,
        )
    except ValueError as exc:
        return str(exc)

    _replace_sportybet_working_slip(user_id, selected)
    combined = _sportybet_combined_odds(selected)
    target_text = f"{float(target_odds):.2f}"
    actual_text = (
        f"{combined:.2f}" if combined is not None else "unavailable"
    )
    return (
        f"Adjusted the working slip toward {target_text} combined odds. "
        f"Current odds snapshot: {actual_text}.\n"
        "Live odds can still move before a new share code is created.\n\n"
        + _format_sportybet_working_slip(user_id, detail_limit=10)
    )


def _refresh_sportybet_slip_records(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    if not records:
        return [], ["The working slip is empty."]

    team_pairs = [
        (
            str(record.get("home_team") or ""),
            str(record.get("away_team") or ""),
        )
        for record in records
    ]
    market_ids = tuple(
        sorted(
            {
                str(record.get("market_id") or "")
                for record in records
                if str(record.get("market_id") or "")
            }
        )
    )
    fixtures = fetch_upcoming_fixtures(
        team_pairs=team_pairs,
        market_ids=market_ids,
        max_pages=20,
    )
    by_event = {
        str(fixture.get("event_id") or ""): fixture
        for fixture in fixtures
    }

    refreshed: list[dict[str, Any]] = []
    errors: list[str] = []

    for record in records:
        event_id = str(record.get("event_id") or "")
        fixture = by_event.get(event_id)
        if fixture is None:
            try:
                fixture = _resolve_sportybet_fixture(
                    fixtures,
                    str(record.get("home_team") or ""),
                    str(record.get("away_team") or ""),
                )
            except ValueError:
                errors.append(
                    f"{record.get('home_team')} vs {record.get('away_team')}: "
                    "fixture is no longer available."
                )
                continue

        market_id = str(record.get("market_id") or "")
        specifier = record.get("specifier")
        markets = [
            market
            for market in fixture.get("markets") or []
            if str(market.get("market_id") or "") == market_id
            and (
                specifier is None
                or str(market.get("specifier") or "") == str(specifier)
            )
            and int(market.get("status") or 0) == 0
        ]
        if not markets:
            errors.append(
                f"{fixture.get('home_team')} vs {fixture.get('away_team')}: "
                "selected market is no longer available."
            )
            continue

        market = markets[0]
        outcome_id = str(record.get("outcome_id") or "")
        outcome = next(
            (
                item
                for item in market.get("outcomes") or []
                if str(item.get("outcome_id") or "") == outcome_id
                and item.get("is_active", False)
            ),
            None,
        )
        if outcome is None:
            errors.append(
                f"{fixture.get('home_team')} vs {fixture.get('away_team')}: "
                "selected outcome is no longer available."
            )
            continue

        refreshed.append(
            {
                **record,
                "event_id": fixture.get("event_id"),
                "home_team": fixture.get("home_team"),
                "away_team": fixture.get("away_team"),
                "market_id": market.get("market_id"),
                "market_name": market.get("market_name"),
                "specifier": market.get("specifier"),
                "outcome_id": outcome.get("outcome_id"),
                "outcome_name": outcome.get("outcome_name"),
                "odds": outcome.get("odds"),
            }
        )

    return refreshed, errors


def _create_code_from_working_sportybet_slip(user_id: str) -> str:
    state = _sportybet_working_slip(user_id)
    if state is None:
        return "Load or analyse a SportyBet code first."

    current = state["current"]
    if len(current) > 20:
        return (
            f"The working slip has {len(current)} selections. "
            "Shorten it to 20 or fewer before creating a new SportyBet code."
        )

    refreshed, errors = _refresh_sportybet_slip_records(current)
    if errors:
        preview = "\n".join(f"- {item}" for item in errors[:6])
        more = (
            f"\n- ...and {len(errors) - 6} more"
            if len(errors) > 6
            else ""
        )
        return (
            "I didn't create a new code because some working selections "
            "are no longer available on SportyBet:\n"
            f"{preview}{more}\n\n"
            "Remove or replace those selections and try again."
        )

    booking = create_booking(refreshed)
    booked = booking.get("selections") or []
    if booked and len(booked) != len(refreshed):
        return (
            "SportyBet returned a code with fewer selections than the working "
            "slip, so I did not present it as a complete replacement. "
            "Refresh the slip and try again."
        )

    combined = _sportybet_combined_odds(booked or refreshed)
    lines = [
        f"New SportyBet code: {booking.get('share_code')}",
        f"Selections: {len(booked) or len(refreshed)}",
    ]
    if combined is not None:
        lines.append(f"Combined odds: {combined:.2f}")

    lines.append("")
    for index, item in enumerate(booked or refreshed, start=1):
        odds = item.get("odds")
        odds_text = (
            f" @ {float(odds):.2f}"
            if isinstance(odds, (int, float))
            else ""
        )
        lines.append(
            f"{index}. {item.get('home_team')} vs {item.get('away_team')} - "
            f"{item.get('market_name')}: {item.get('outcome_name')}{odds_text}"
        )

    share_url = str(booking.get("share_url") or "").strip()
    if share_url:
        lines.extend(["", f"Share URL: {share_url}"])

    lines.extend(
        [
            "",
            "The original code was not changed. This is a new share code built from the edited working slip.",
            "No wager or stake was submitted.",
        ]
    )
    return "\n".join(lines)


def _replace_sportybet_slip_with_safer_market(
    user_id: str,
    index: int,
) -> str:
    state = _sportybet_working_slip(user_id)
    if state is None:
        return "Load or analyse a SportyBet code first."

    current = state["current"]
    target = next(
        (
            record for record in current
            if int(record.get("index") or 0) == int(index)
        ),
        None,
    )
    if target is None:
        return f"Selection {index} is not in the current slip."

    fixtures = fetch_upcoming_fixtures(
        team_pairs=[
            (
                str(target.get("home_team") or ""),
                str(target.get("away_team") or ""),
            )
        ],
        market_ids=("1", "10", "18", "29"),
        max_pages=20,
    )
    try:
        fixture = _resolve_sportybet_fixture(
            fixtures,
            str(target.get("home_team") or ""),
            str(target.get("away_team") or ""),
        )
    except ValueError as exc:
        return str(exc)

    try:
        context = _build_prediction_context(
            str(fixture.get("home_team") or ""),
            str(fixture.get("away_team") or ""),
            known_fixture={
                "home": str(fixture.get("home_team") or ""),
                "away": str(fixture.get("away_team") or ""),
                "date": "",
            },
            recent_only=False,
            prefer_season_only=True,
        )
    except Exception:
        return (
            "I couldn't get enough independent football data to find a "
            "safer replacement for that selection."
        )

    candidates = _sportybet_auto_candidates_for_fixture(
        fixture,
        context,
    )
    candidates = [
        candidate
        for candidate in candidates
        if not (
            str(candidate.get("market_id") or "") == str(target.get("market_id") or "")
            and str(candidate.get("specifier") or "") == str(target.get("specifier") or "")
            and str(candidate.get("outcome_id") or "") == str(target.get("outcome_id") or "")
        )
    ]
    if not candidates:
        return (
            "I couldn't find a different independently modelled market "
            "that passes the current quality filters for that match."
        )

    replacement = candidates[0]
    old_support = target.get("model_probability")
    if (
        isinstance(old_support, (int, float))
        and float(replacement["model_probability"]) <= float(old_support)
    ):
        return (
            f"I found alternatives for selection {index}, but none has "
            "higher independent model support than the current pick, so I "
            "left it unchanged."
        )

    replacement_record = {
        **target,
        **replacement,
        "platform_probability": (
            1.0 / float(replacement["odds"])
            if isinstance(replacement.get("odds"), (int, float))
            and float(replacement["odds"]) > 1.0
            else None
        ),
        "platform_probability_label": "Raw odds-implied chance",
        "model_probability": replacement["model_probability"],
        "model_label": replacement["model_label"],
        "alignment": _model_alignment(
            float(replacement["model_probability"])
        ),
        "gap": None,
        "model_reason": "modelled",
    }

    updated = []
    for record in current:
        if int(record["index"]) == int(index):
            updated.append(replacement_record)
        else:
            updated.append(record)
    _replace_sportybet_working_slip(user_id, updated)

    return (
        f"Replaced selection {index} with a higher-support available market:\n"
        f"{replacement_record['home_team']} vs {replacement_record['away_team']} - "
        f"{replacement_record['model_label']} @ {float(replacement_record['odds']):.2f}\n"
        f"Independent model support: "
        f"{float(replacement_record['model_probability']) * 100:.1f}%\n\n"
        "Use UNDO if you want the previous selection back."
    )


def _analyse_and_cache_sportybet_booking_code(
    booking_code: str,
    user_id: str,
) -> str:
    analysis = _run_sportybet_booking_analysis(booking_code)
    _cache_sportybet_analysis(user_id, analysis)
    _initialize_sportybet_working_slip(user_id, analysis)
    return (
        _format_sportybet_analysis_summary(analysis)
        + "\n\nWorking slip loaded. You can now shorten, remove, replace, undo, redo, or create a new code."
    )


@tool
def analyse_sportybet_booking_code(booking_code: str) -> str:
    """Read a SportyBet booking code and return a compact model-comparison summary."""
    try:
        analysis = _run_sportybet_booking_analysis(booking_code)
        return _format_sportybet_analysis_summary(analysis)
    except ValueError as exc:
        return str(exc)
    except SportyBetLookupError as exc:
        return str(exc)
    except Exception as exc:
        return f"I couldn't analyse that SportyBet code right now: {exc}"


def _parse_sportybet_booking_request(request: str) -> list[dict[str, str]]:
    """Parse one selection per line/semicolon from a SportyBet code request."""
    body = re.sub(
        r"^\s*(?:create|make|generate|prepare)\s+(?:a\s+)?sportybet\s+"
        r"(?:(?:booking|share)\s+)?code(?:\s+(?:for|from))?\s*:?\s*",
        "",
        request.strip(),
        count=1,
        flags=re.IGNORECASE,
    )

    parts = [
        re.sub(r"^\s*(?:[-•]|\d+[.)])\s*", "", part).strip()
        for part in re.split(r"[;\n]+", body)
        if part.strip()
    ]
    if not parts:
        raise ValueError(
            "Add at least one selection after 'Create SportyBet code:'."
        )
    if len(parts) > 20:
        raise ValueError(
            "SportyBet booking-code creation is limited to 20 selections."
        )

    parsed: list[dict[str, str]] = []
    for part in parts:
        match = re.match(
            r"^(.+?)\s+(?:vs\.?|versus)\s+(.+?)"
            r"(?:\s*\|\s*|\s+-\s+|\s*:\s+)(.+)$",
            part,
            flags=re.IGNORECASE,
        )
        if not match:
            raise ValueError(
                "I couldn't read this selection: "
                f"{part}. Use 'Team A vs Team B | Pick' with one selection "
                "per line or separated by semicolons."
            )

        parsed.append(
            {
                "team1": match.group(1).strip(),
                "team2": match.group(2).strip(),
                "pick": match.group(3).strip(),
            }
        )

    return parsed


def _sportybet_team_key(value: str) -> str:
    tokens = _normalize_team_name(value).split()
    aliases = {
        "utd": "united",
        "st": "saint",
    }
    return " ".join(aliases.get(token, token) for token in tokens)


def _sportybet_team_match_score(requested: str, actual: str) -> int:
    req = _sportybet_team_key(requested)
    act = _sportybet_team_key(actual)
    if not req or not act:
        return 0
    if req == act:
        return 100
    if len(req) >= 4 and req in act:
        return 88
    if len(act) >= 4 and act in req:
        return 84

    req_tokens = set(req.split())
    act_tokens = set(act.split())
    overlap = len(req_tokens & act_tokens)
    if not overlap:
        return 0

    union = len(req_tokens | act_tokens)
    return int(70 * overlap / max(1, union))


def _sportybet_fixture_suggestions(
    fixtures: list[dict[str, Any]],
    team1: str,
    team2: str,
    limit: int = 4,
) -> list[str]:
    """Suggest actual upcoming SportyBet fixtures involving either requested team."""
    suggestions: list[tuple[int, str]] = []
    seen: set[str] = set()

    for fixture in fixtures:
        home = str(fixture.get("home_team") or "")
        away = str(fixture.get("away_team") or "")
        if not home or not away:
            continue

        score = max(
            _sportybet_team_match_score(team1, home),
            _sportybet_team_match_score(team1, away),
            _sportybet_team_match_score(team2, home),
            _sportybet_team_match_score(team2, away),
        )
        if score < 70:
            continue

        label = f"{home} vs {away}"
        if label in seen:
            continue
        seen.add(label)
        suggestions.append((score, label))

    suggestions.sort(key=lambda item: item[0], reverse=True)
    return [label for _score, label in suggestions[:limit]]


def _resolve_sportybet_fixture(
    fixtures: list[dict[str, Any]],
    team1: str,
    team2: str,
) -> dict[str, Any]:
    candidates: list[tuple[int, dict[str, Any]]] = []

    for fixture in fixtures:
        home = str(fixture.get("home_team") or "")
        away = str(fixture.get("away_team") or "")

        direct = (
            _sportybet_team_match_score(team1, home)
            + _sportybet_team_match_score(team2, away)
        )
        reverse = (
            _sportybet_team_match_score(team1, away)
            + _sportybet_team_match_score(team2, home)
        )
        score = max(direct, reverse)
        if score >= 130:
            candidates.append((score, fixture))

    if not candidates:
        suggestions = _sportybet_fixture_suggestions(
            fixtures,
            team1,
            team2,
        )
        message = (
            f"I couldn't find an upcoming SportyBet fixture for "
            f"{team1} vs {team2}."
        )
        if suggestions:
            message += (
                "\n\nUpcoming SportyBet fixtures involving those teams:"
                + "".join(f"\n- {item}" for item in suggestions)
                + "\n\nUse one of the actual fixtures above."
            )
        else:
            message += (
                "\nThe matchup may not be in SportyBet's current pre-match "
                "catalogue yet. Use a fixture currently visible on SportyBet."
            )
        raise ValueError(message)

    candidates.sort(key=lambda item: item[0], reverse=True)
    best_score = candidates[0][0]
    best = [fixture for score, fixture in candidates if score == best_score]

    if len(best) > 1:
        names = ", ".join(
            f"{item.get('home_team')} vs {item.get('away_team')}"
            for item in best[:3]
        )
        raise ValueError(
            f"More than one SportyBet fixture matched {team1} vs {team2}: {names}. "
            "Use the exact SportyBet team names."
        )

    fixture = best[0]
    if str(fixture.get("match_status") or "Not start") != "Not start":
        raise ValueError(
            f"{fixture.get('home_team')} vs {fixture.get('away_team')} "
            "is no longer open as a pre-match fixture."
        )

    start_ms = int(fixture.get("start_ms") or 0)
    if start_ms and start_ms <= int(datetime.now(timezone.utc).timestamp() * 1000):
        raise ValueError(
            f"{fixture.get('home_team')} vs {fixture.get('away_team')} has already started."
        )

    return fixture


def _sportybet_market(
    fixture: dict[str, Any],
    market_id: str,
    *,
    total: float | None = None,
) -> dict[str, Any]:
    matches = [
        market
        for market in fixture.get("markets") or []
        if str(market.get("market_id") or "") == market_id
    ]

    if total is not None:
        filtered: list[dict[str, Any]] = []
        for market in matches:
            specifier = str(market.get("specifier") or "")
            line = _specifier_number(specifier, "total")
            if line is not None and abs(line - total) < 0.001:
                filtered.append(market)
        matches = filtered

    matches = [
        market
        for market in matches
        if int(market.get("status") or 0) == 0
    ]

    if not matches:
        label = f" {total:g}" if total is not None else ""
        raise ValueError(
            f"SportyBet does not currently offer the requested market{label} "
            f"for {fixture.get('home_team')} vs {fixture.get('away_team')}."
        )

    return matches[0]


def _sportybet_outcome(
    market: dict[str, Any],
    accepted_names: set[str],
) -> dict[str, Any]:
    normalized_names = {
        re.sub(r"[^a-z0-9]+", " ", name.casefold()).strip()
        for name in accepted_names
    }

    for outcome in market.get("outcomes") or []:
        if not outcome.get("is_active", False):
            continue
        name = re.sub(
            r"[^a-z0-9]+",
            " ",
            str(outcome.get("outcome_name") or "").casefold(),
        ).strip()
        if name in normalized_names:
            return outcome

    available = ", ".join(
        str(outcome.get("outcome_name") or "")
        for outcome in market.get("outcomes") or []
        if outcome.get("is_active", False)
    )
    raise ValueError(
        f"That selection is not currently available on SportyBet. "
        f"Available outcomes: {available or 'none'}."
    )


def _resolve_sportybet_pick(
    fixture: dict[str, Any],
    pick: str,
) -> dict[str, Any]:
    raw = re.sub(r"\s+", " ", pick.strip())
    lowered = raw.casefold()
    home = str(fixture.get("home_team") or "")
    away = str(fixture.get("away_team") or "")

    total_match = re.fullmatch(
        r"(over|under)\s*([0-9]+(?:\.[0-9]+)?)\s*(?:goals?)?",
        lowered,
    )
    if total_match:
        side = total_match.group(1)
        line = float(total_match.group(2))
        market = _sportybet_market(fixture, "18", total=line)
        outcome = _sportybet_outcome(
            market,
            {side, f"{side} {line:g}"},
        )
        return {
            "event_id": fixture["event_id"],
            "market_id": market["market_id"],
            "specifier": market.get("specifier"),
            "outcome_id": outcome["outcome_id"],
            "home_team": home,
            "away_team": away,
            "market_name": market["market_name"],
            "outcome_name": outcome["outcome_name"],
            "odds": outcome.get("odds"),
        }

    btts_match = re.fullmatch(
        r"(?:btts|both teams to score)\s*(yes|no|gg|ng)",
        lowered,
    )
    if btts_match:
        choice = btts_match.group(1)
        market = _sportybet_market(fixture, "29")
        accepted = {"yes", "gg"} if choice in {"yes", "gg"} else {"no", "ng"}
        outcome = _sportybet_outcome(market, accepted)
        return {
            "event_id": fixture["event_id"],
            "market_id": market["market_id"],
            "specifier": market.get("specifier"),
            "outcome_id": outcome["outcome_id"],
            "home_team": home,
            "away_team": away,
            "market_name": market["market_name"],
            "outcome_name": outcome["outcome_name"],
            "odds": outcome.get("odds"),
        }

    dc_match = re.fullmatch(
        r"(?:double chance\s*)?(1x|x2|12|home or draw|draw or away|home or away)",
        lowered,
    )
    if dc_match:
        choice = dc_match.group(1)
        market = _sportybet_market(fixture, "10")
        aliases = {
            "1x": {"home or draw", "1x"},
            "home or draw": {"home or draw", "1x"},
            "x2": {"draw or away", "x2"},
            "draw or away": {"draw or away", "x2"},
            "12": {"home or away", "12"},
            "home or away": {"home or away", "12"},
        }
        outcome = _sportybet_outcome(market, aliases[choice])
        return {
            "event_id": fixture["event_id"],
            "market_id": market["market_id"],
            "specifier": market.get("specifier"),
            "outcome_id": outcome["outcome_id"],
            "home_team": home,
            "away_team": away,
            "market_name": market["market_name"],
            "outcome_name": outcome["outcome_name"],
            "odds": outcome.get("odds"),
        }

    market = _sportybet_market(fixture, "1")
    home_key = _sportybet_team_key(home)
    away_key = _sportybet_team_key(away)
    pick_key = _sportybet_team_key(
        re.sub(r"\b(?:to win|win|wins)\b", "", lowered).strip()
    )

    if lowered in {"1", "home", "home win"} or pick_key == home_key:
        outcome = _sportybet_outcome(market, {"home", "1"})
    elif lowered in {"x", "draw"}:
        outcome = _sportybet_outcome(market, {"draw", "x"})
    elif lowered in {"2", "away", "away win"} or pick_key == away_key:
        outcome = _sportybet_outcome(market, {"away", "2"})
    else:
        raise ValueError(
            f"I can't create '{pick}' yet. Supported code-creation picks are "
            "1X2/Home/Draw/Away, Double Chance 1X/X2/12, BTTS Yes/No, "
            "and full-match Over/Under goals."
        )

    return {
        "event_id": fixture["event_id"],
        "market_id": market["market_id"],
        "specifier": market.get("specifier"),
        "outcome_id": outcome["outcome_id"],
        "home_team": home,
        "away_team": away,
        "market_name": market["market_name"],
        "outcome_name": outcome["outcome_name"],
        "odds": outcome.get("odds"),
    }


def _create_sportybet_code_from_request(request: str) -> str:
    requested = _parse_sportybet_booking_request(request)
    team_pairs = [
        (item["team1"], item["team2"])
        for item in requested
    ]

    fixtures = fetch_upcoming_fixtures(
        team_pairs=team_pairs,
        market_ids=("1", "10", "18", "29"),
    )

    resolved: list[dict[str, Any]] = []
    for item in requested:
        fixture = _resolve_sportybet_fixture(
            fixtures,
            item["team1"],
            item["team2"],
        )
        resolved.append(
            _resolve_sportybet_pick(
                fixture,
                item["pick"],
            )
        )

    booking = create_booking(resolved)
    booked_selections = booking.get("selections") or []

    live_odds = [
        float(item["odds"])
        for item in booked_selections
        if isinstance(item.get("odds"), (int, float))
        and float(item["odds"]) > 0
    ]
    combined_odds = None
    if booked_selections and len(live_odds) == len(booked_selections):
        combined_odds = math.prod(live_odds)

    lines = [
        f"SportyBet booking code created: {booking.get('share_code')}",
        f"Selections: {len(booked_selections) or len(resolved)}",
    ]
    if combined_odds is not None:
        lines.append(f"Combined odds: {combined_odds:.2f}")

    lines.append("")
    display_records = booked_selections or resolved
    for index, item in enumerate(display_records, start=1):
        odds = item.get("odds")
        odds_text = (
            f" @ {float(odds):.2f}"
            if isinstance(odds, (int, float))
            else ""
        )
        lines.append(
            f"{index}. {item.get('home_team')} vs {item.get('away_team')} - "
            f"{item.get('market_name')}: {item.get('outcome_name')}{odds_text}"
        )

    share_url = str(booking.get("share_url") or "").strip()
    if share_url:
        lines.extend(["", f"Share URL: {share_url}"])

    lines.extend(
        [
            "",
            "This only prepares a SportyBet betslip reservation. No wager was placed and no stake was submitted.",
        ]
    )
    return "\n".join(lines)


@tool
def create_sportybet_booking_code(request: str) -> str:
    """Create a non-staking SportyBet booking/share code from explicit selections."""
    try:
        return _create_sportybet_code_from_request(request)
    except ValueError as exc:
        return str(exc)
    except SportyBetLookupError as exc:
        return str(exc)
    except Exception as exc:
        return f"I couldn't create that SportyBet booking code right now: {exc}"


def _sportybet_auto_candidates_for_fixture(
    fixture: dict[str, Any],
    context: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build model-supported candidate selections from one live SportyBet fixture."""
    if (
        context.get("projection") is None
        or not _sportybet_context_has_enough_data(context)
    ):
        return []

    candidates: list[dict[str, Any]] = []
    for market in fixture.get("markets") or []:
        market_id = str(market.get("market_id") or "")
        market_name = str(market.get("market_name") or "")
        if market_id not in {"1", "10", "18", "29"}:
            continue

        specifier = market.get("specifier")
        if market_id == "18":
            line = _specifier_number(str(specifier or ""), "total")
            if line is None or line < 1.5 or line > 3.5:
                continue

        for outcome in market.get("outcomes") or []:
            if not outcome.get("is_active", False):
                continue

            odds = outcome.get("odds")
            if not isinstance(odds, (int, float)):
                continue
            odds = float(odds)
            if odds < 1.20 or odds > 3.50:
                continue

            selection = {
                "event_id": fixture.get("event_id"),
                "home_team": fixture.get("home_team"),
                "away_team": fixture.get("away_team"),
                "market_id": market_id,
                "market_name": market_name,
                "specifier": specifier,
                "outcome_id": str(outcome.get("outcome_id") or ""),
                "outcome_name": str(outcome.get("outcome_name") or ""),
                "odds": odds,
            }

            probability, label = _sportybet_selection_model_support(
                context,
                selection,
            )
            if probability is None or probability < 0.55:
                continue

            candidates.append(
                {
                    **selection,
                    "model_probability": probability,
                    "model_label": label,
                }
            )

    candidates.sort(
        key=lambda item: (
            float(item["model_probability"]),
            float(item["odds"]),
        ),
        reverse=True,
    )
    return candidates


def _sportybet_today_bounds_ms() -> tuple[int, int, str]:
    """Return today's start/end in WAT (UTC+1) as epoch milliseconds."""
    wat = timezone(timedelta(hours=1))
    now = datetime.now(wat)
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)
    return (
        int(start.timestamp() * 1000),
        int(end.timestamp() * 1000),
        start.date().isoformat(),
    )


def _select_fixture_candidates_for_target_odds(
    fixture_candidates: list[list[dict[str, Any]]],
    target_odds: float,
    *,
    exact_count: int | None = None,
    randomize: bool = False,
) -> list[dict[str, Any]]:
    """Choose at most one market per fixture while approaching target odds."""
    if target_odds <= 1.0:
        raise ValueError("Target combined odds must be above 1.00.")

    groups = [
        [
            candidate
            for candidate in candidates[:8]
            if isinstance(candidate.get("odds"), (int, float))
            and float(candidate["odds"]) > 1.0
        ]
        for candidates in fixture_candidates
        if candidates
    ]
    groups = [group for group in groups if group]
    if not groups:
        raise ValueError("No usable model-supported selections are available.")

    if exact_count is not None:
        exact_count = int(exact_count)
        if exact_count < 1:
            raise ValueError("Number of selections must be at least 1.")
        if exact_count > len(groups):
            raise ValueError(
                f"Only {len(groups)} fixtures have usable model-supported selections."
            )

    target_log = math.log(target_odds)
    max_count = exact_count if exact_count is not None else min(20, len(groups))
    states: list[tuple[float, float, tuple[tuple[int, int], ...]]] = [
        (0.0, 0.0, ())
    ]

    for group_index, group in enumerate(groups):
        additions: list[tuple[float, float, tuple[tuple[int, int], ...]]] = []
        for log_sum, support_sum, chosen in states:
            if len(chosen) >= max_count:
                continue
            for candidate_index, candidate in enumerate(group):
                odds = float(candidate["odds"])
                support = float(candidate.get("model_probability") or 0.0)
                new_log = log_sum + math.log(odds)
                if new_log > math.log(target_odds * 1.75):
                    continue
                additions.append(
                    (
                        new_log,
                        support_sum + support,
                        chosen + ((group_index, candidate_index),),
                    )
                )

        states.extend(additions)
        states.sort(
            key=lambda state: (
                abs(state[0] - target_log),
                -state[1] / max(1, len(state[2])),
            )
        )
        states = states[:5000]

    candidates = [
        state
        for state in states
        if state[2]
        and (
            exact_count is None
            or len(state[2]) == exact_count
        )
    ]
    if not candidates:
        raise ValueError("No combination passed the current quality filters.")

    tolerance = math.log(1.10)
    close = [
        state
        for state in candidates
        if abs(state[0] - target_log) <= tolerance
    ]
    if close:
        close.sort(
            key=lambda state: (
                -(state[1] / len(state[2])),
                abs(state[0] - target_log),
            )
        )
        if randomize:
            best = random.SystemRandom().choice(close[: min(10, len(close))])
        else:
            best = close[0]
    else:
        candidates.sort(
            key=lambda state: (
                abs(state[0] - target_log),
                -state[1] / len(state[2]),
            )
        )
        if randomize:
            best = random.SystemRandom().choice(
                candidates[: min(8, len(candidates))]
            )
        else:
            best = candidates[0]

    return [
        groups[group_index][candidate_index]
        for group_index, candidate_index in best[2]
    ]


def _build_model_ranked_sportybet_code(
    match_count: int = 0,
    target_odds: float = 0.0,
    randomize: bool = False,
) -> str:
    """Build a non-staking SportyBet code from today's model-supported fixtures."""
    requested_count = int(match_count) if int(match_count) > 0 else None
    if requested_count is not None:
        requested_count = min(requested_count, 10)

    requested_target = (
        float(target_odds)
        if float(target_odds) > 1.0
        else None
    )
    if requested_count is None and requested_target is None:
        requested_count = 3

    start_ms, end_ms, date_text = _sportybet_today_bounds_ms()

    fixtures = fetch_upcoming_fixtures(
        market_ids=("1", "10", "18", "29"),
        timeline_hours=48,
        max_pages=20,
    )
    today_all = [
        fixture
        for fixture in fixtures
        if start_ms <= int(fixture.get("start_ms") or 0) < end_ms
        and str(fixture.get("match_status") or "Not start") == "Not start"
    ]

    if not today_all:
        return (
            f"I couldn't find any SportyBet fixtures for {date_text}. "
            "Try another date or create a code from explicit selections."
        )

    # Prefer leagues with structured season data. If the user requests more
    # games than that pool can support, a small fallback set is evaluated
    # with the same independent model using recent-form data. The existing
    # quality guard still requires enough data for both teams.
    today_primary = [
        fixture
        for fixture in today_all
        if _league_code(str(fixture.get("league") or "")) is not None
    ][:20]
    today_fallback = [
        fixture
        for fixture in today_all
        if _league_code(str(fixture.get("league") or "")) is None
    ][:40]

    def evaluate(fixture: dict[str, Any]) -> dict[str, Any]:
        home = str(fixture.get("home_team") or "")
        away = str(fixture.get("away_team") or "")
        base = {
            "fixture": fixture,
            "home_team": home,
            "away_team": away,
            "candidates": [],
            "reason": "",
        }
        if not home or not away:
            return {**base, "reason": "missing_match_data"}

        try:
            context = _build_prediction_context(
                home,
                away,
                known_fixture={"home": home, "away": away, "date": date_text},
                recent_only=False,
                prefer_season_only=True,
            )
        except Exception:
            return {**base, "reason": "independent_data_error"}

        if context.get("projection") is None:
            return {**base, "reason": "no_score_projection"}
        if not _sportybet_context_has_enough_data(context):
            return {**base, "reason": "insufficient_independent_data"}

        candidates = _sportybet_auto_candidates_for_fixture(
            fixture,
            context,
        )
        if not candidates:
            return {**base, "reason": "no_market_passed_filters"}

        return {
            **base,
            "candidates": candidates,
            "reason": "qualified",
        }

    with ThreadPoolExecutor(max_workers=6) as executor:
        primary_results = list(executor.map(evaluate, today_primary))

    evaluated_results = list(primary_results)
    usable_groups = [
        result["candidates"]
        for result in primary_results
        if result["candidates"]
    ]

    if (
        requested_count is not None
        and len(usable_groups) < requested_count
        and today_fallback
    ):
        # Expand the fallback search progressively instead of hammering every
        # remaining fixture at once. Stop as soon as the requested number of
        # independently modelled matches is available.
        fallback_batch_size = 8
        for offset in range(0, len(today_fallback), fallback_batch_size):
            batch = today_fallback[offset : offset + fallback_batch_size]
            with ThreadPoolExecutor(max_workers=6) as executor:
                fallback_results = list(
                    executor.map(evaluate, batch)
                )
            evaluated_results.extend(fallback_results)
            usable_groups.extend(
                result["candidates"]
                for result in fallback_results
                if result["candidates"]
            )
            if len(usable_groups) >= requested_count:
                break
    if not usable_groups:
        return (
            f"I couldn't find independently modelled SportyBet selections "
            f"meeting the current quality filters for {date_text}."
        )

    if requested_target is not None:
        try:
            selected = _select_fixture_candidates_for_target_odds(
                usable_groups,
                requested_target,
                exact_count=requested_count,
                randomize=randomize,
            )
        except ValueError as exc:
            return (
                f"I couldn't build a suitable SportyBet code around "
                f"{requested_target:.2f} odds: {exc}"
            )
    else:
        count = requested_count or 3
        pool = [candidates[0] for candidates in usable_groups]
        pool.sort(
            key=lambda item: (
                float(item["model_probability"]),
                float(item["odds"]),
            ),
            reverse=True,
        )
        if len(pool) < count:
            reason_counts: dict[str, int] = {}
            for result in evaluated_results:
                reason = str(result.get("reason") or "unknown")
                reason_counts[reason] = reason_counts.get(reason, 0) + 1

            lines = [
                f"I found only {len(pool)} independently modelled SportyBet "
                f"selection(s) meeting the current quality filters for {date_text}; "
                f"{count} were requested. No booking code was created.",
                "",
                f"Fixtures checked: {len(evaluated_results)}",
            ]

            diagnostic_labels = (
                ("insufficient_independent_data", "Not enough independent team data"),
                ("independent_data_error", "Independent data lookup failed"),
                ("no_score_projection", "No usable score projection"),
                ("no_market_passed_filters", "No supported market passed the model/odds filters"),
                ("missing_match_data", "Missing fixture/team data"),
            )
            for key, label in diagnostic_labels:
                value = reason_counts.get(key, 0)
                if value:
                    lines.append(f"- {label}: {value}")

            if pool:
                lines.extend(["", "Qualified selections currently available:"])
                for index, item in enumerate(pool[:count], start=1):
                    lines.append(
                        f"{index}. {item['home_team']} vs {item['away_team']} - "
                        f"{item['model_label']} @ {float(item['odds']):.2f} "
                        f"({float(item['model_probability']) * 100:.1f}% model support)"
                    )

            lines.extend(
                [
                    "",
                    "I kept the quality filters unchanged rather than adding an unqualified fifth selection.",
                ]
            )
            return "\n".join(lines)

        if randomize:
            strong_pool = [
                item
                for item in pool
                if float(item.get("model_probability") or 0.0) >= 0.70
            ]
            source_pool = strong_pool if len(strong_pool) >= count else pool
            selected = random.SystemRandom().sample(source_pool, count)
            selected.sort(
                key=lambda item: float(item.get("model_probability") or 0.0),
                reverse=True,
            )
        else:
            selected = pool[:count]

    booking = create_booking(selected)
    booked = booking.get("selections") or []

    live_odds = [
        float(item["odds"])
        for item in booked
        if isinstance(item.get("odds"), (int, float))
        and float(item["odds"]) > 0
    ]
    combined = (
        math.prod(live_odds)
        if booked and len(live_odds) == len(booked)
        else _sportybet_combined_odds(selected)
    )

    lines = [
        f"Model-built SportyBet code: {booking.get('share_code')}",
        f"Date: {date_text}",
        f"Selections: {len(booked) or len(selected)}",
    ]
    if requested_target is not None:
        lines.append(f"Requested target odds: {requested_target:.2f}")
    if combined is not None:
        lines.append(f"Combined odds: {combined:.2f}")

    lines.extend(["", "Selections:"])
    booked_by_event = {
        str(item.get("event_id") or ""): item
        for item in booked
    }

    for index, item in enumerate(selected, start=1):
        live = booked_by_event.get(str(item.get("event_id") or ""), item)
        odds = live.get("odds")
        odds_text = (
            f" @ {float(odds):.2f}"
            if isinstance(odds, (int, float))
            else ""
        )
        lines.append(
            f"{index}. {item['home_team']} vs {item['away_team']} - "
            f"{item['model_label']}{odds_text}"
        )
        lines.append(
            f"   Independent model support: "
            f"{item['model_probability'] * 100:.1f}%"
        )

    share_url = str(booking.get("share_url") or "").strip()
    if share_url:
        lines.extend(["", f"Share URL: {share_url}"])

    if requested_target is not None:
        if combined is not None:
            difference = abs(combined - requested_target)
            if difference <= max(0.15, requested_target * 0.10):
                target_note = (
                    f"This is reasonably close to the requested "
                    f"{requested_target:.2f} target."
                )
            else:
                target_note = (
                    f"The closest quality-filtered combination found was "
                    f"{combined:.2f}, which is {difference:.2f} away from "
                    f"the {requested_target:.2f} target."
                )
        else:
            target_note = "Live odds can move, so the target is approximate."

        lines.extend(
            [
                "",
                target_note,
                "The bot does not force an exact total by adding a weaker selection.",
            ]
        )

    lines.extend(
        [
            "",
            "These selections were ranked by the experimental model from markets currently available on SportyBet. They are not guaranteed outcomes.",
            "The code only prepares a betslip reservation. No wager or stake was submitted.",
        ]
    )
    return "\n".join(lines)


@tool
def build_model_ranked_sportybet_code(
    match_count: int = 0,
    target_odds: float = 0.0,
    randomize: bool = False,
) -> str:
    """Build a non-staking SportyBet code from today's high model-supported selections."""
    try:
        return _build_model_ranked_sportybet_code(
            match_count=match_count,
            target_odds=target_odds,
            randomize=randomize,
        )
    except ValueError as exc:
        return str(exc)
    except SportyBetLookupError as exc:
        return str(exc)
    except Exception as exc:
        return f"I couldn't build that SportyBet code right now: {exc}"


@tool
def search_betting_knowledge(question: str) -> str:
    """Explain football betting terms and market settlement using the local knowledge base."""
    try:
        exact = lookup_betting_term(question)
        if exact:
            return "Betting term explanation:\n\n" + exact

        passages = search_knowledge(
            question,
            k=1,
            category="betting_terms",
        )
        if not passages:
            return "No relevant betting term was found in the local knowledge base."
        return "Betting term explanation:\n\n" + passages[0]
    except Exception as exc:
        return f"I couldn't search the betting terminology knowledge base right now: {exc}"


@tool
def search_football_knowledge(question: str) -> str:
    """Search the local football rules knowledge base using semantic retrieval."""
    try:
        passages = search_knowledge(
            question,
            k=1,
            category="football_rules",
        )
        if not passages:
            return "No relevant information was found in the football knowledge base."
        return "From the football knowledge base:\n\n" + "\n\n".join(passages)
    except Exception as exc:
        return f"I couldn't search the football knowledge base right now: {exc}"


@tool
def get_league_fixtures(league_name: str) -> str:
    """Get upcoming fixtures for a supported football league."""
    code = _league_code(league_name)
    if not code:
        return f"League '{league_name}' is not supported."

    try:
        data = _safe_get_json(
            f"{FOOTBALL_DATA_BASE_URL}/competitions/{code}/matches",
            headers=_football_data_headers(),
            params={"status": "SCHEDULED"},
        )
        matches = data.get("matches") or []
        if not matches:
            return f"No upcoming fixtures found for {league_name}."

        lines = [f"Upcoming {league_name.title()} fixtures:"]
        for match in matches[:10]:
            utc_date = str(match.get("utcDate", ""))[:10] or "Date unavailable"
            lines.append(
                f"- {utc_date}: "
                f"{match.get('homeTeam', {}).get('name', 'Home')} vs "
                f"{match.get('awayTeam', {}).get('name', 'Away')}"
            )
        return "\n".join(lines)
    except Exception as exc:
        return f"I couldn't retrieve league fixtures right now: {exc}"


@tool
def get_head_to_head(team1: str, team2: str) -> str:
    """Search for recent head-to-head information between two football teams."""
    try:
        tavily = TavilyClient(api_key=_require_env("TAVILY_API_KEY"))
        results = tavily.search(
            query=f"{team1} vs {team2} head to head football results",
            max_results=3,
        )
        items = results.get("results") or []
        if not items:
            return "No head-to-head information was found."

        lines = [f"Head to head references: {team1} vs {team2}"]
        for item in items:
            snippet = _clean_web_snippet(item.get("content"), 170)
            url = str(item.get("url") or "").strip()
            entry = f"- {item.get('title', 'Untitled')}"
            if snippet:
                entry += f"\n  {snippet}"
            if url:
                entry += f"\n  Source: {url}"
            lines.append(entry)
        return "\n\n".join(lines)
    except Exception as exc:
        return f"I couldn't retrieve head-to-head information right now: {exc}"


@tool
def get_team_form(team_name: str) -> str:
    """Get a team's last five results in W/D/L form."""
    try:
        summary = _form_summary(team_name)
        played = int(summary["played"])
        match_word = "match" if played == 1 else "matches"
        return (
            f"{summary['team']} recent form: "
            f"{summary['wins']}W {summary['draws']}D {summary['losses']}L "
            f"from the last {played} {match_word}."
        )
    except Exception as exc:
        return f"I couldn't retrieve team form right now: {exc}"


@tool
def get_team_stats(team_name: str) -> str:
    """Get transparent recent-performance stats computed from the team's returned match history."""
    try:
        summary = _form_summary(team_name)
        played = int(summary["played"])
        return (
            f"{summary['team']} recent performance stats "
            f"(based on {played} match{'es' if played != 1 else ''} returned by the data provider):\n"
            f"- Record: {summary['wins']}W {summary['draws']}D {summary['losses']}L\n"
            f"- Goals scored: {summary['goals_for']} ({float(summary['goals_for_per_game']):.2f} per game)\n"
            f"- Goals conceded: {summary['goals_against']} ({float(summary['goals_against_per_game']):.2f} per game)\n"
            f"- Points per game: {float(summary['points_per_game']):.2f}\n"
            f"- Goal difference per game: {float(summary['goal_diff_per_game']):+.2f}"
        )
    except Exception as exc:
        return f"I couldn't retrieve team stats right now: {exc}"


TOOLS = [
    get_team_info,
    get_recent_results,
    get_upcoming_fixtures,
    get_team_players,
    search_player,
    get_league_standings,
    get_live_scores,
    get_transfer_news,
    get_top_scorers,
    get_player_injury,
    predict_match,
    predict_correct_score,
    predict_betting_markets,
    format_prediction_for_platform,
    analyse_sportybet_booking_code,
    create_sportybet_booking_code,
    build_model_ranked_sportybet_code,
    search_betting_knowledge,
    search_football_knowledge,
    get_league_fixtures,
    get_head_to_head,
    get_team_form,
    get_team_stats,
]


@lru_cache(maxsize=1)
def get_agent():
    model = ChatAnthropic(
        model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5"),
        temperature=0,
        api_key=_require_env("ANTHROPIC_API_KEY"),
    )

    connection = sqlite3.connect(
        os.getenv("MEMORY_DB_PATH", "memory.db"),
        check_same_thread=False,
    )
    memory = SqliteSaver(connection)

    prompt = """
You are a friendly football assistant operating on WhatsApp.

Rules:
1. Use an appropriate tool before answering factual or time-sensitive football questions.
2. Never expose function names, tool calls, XML, JSON, internal reasoning, or code to the user.
3. Use the football knowledge-base tool for rules such as offside, VAR, cards, and penalties.
4. Use live-data tools for fixtures, results, standings, scorers, players, and live scores.
5. Use the match prediction tool when the user asks who is likely to win, the correct-score projection tool for exact-score requests, and the betting-market probability tool for markets such as 1X2, double chance, draw no bet, BTTS, totals, team totals, clean sheets, win to nil, Asian handicap, and correct score.
6. Use the betting terminology knowledge-base tool when the user asks what a betting term or market means, including Asian handicap, double chance, draw no bet, BTTS, over/under, accumulators, push/void, and similar terms.
7. Use the betting-platform formatting tool when the user asks for predictions formatted for SportyBet, Bet9ja, BetKing, MSport, 1xBet, or Betway. Use the SportyBet booking-code analysis tool when the user asks to load, check, review, or analyse an existing SportyBet code. Use the explicit SportyBet booking-code creation tool when the user supplies selections. Use the model-ranked SportyBet code tool only when the user explicitly asks the bot to build a code from today's games. Booking-code creation is a non-staking betslip reservation only; never claim a wager was placed and never submit a stake. The SportyBet web integration is undocumented and may change.
8. Betting-market outputs are statistical probability estimates only. Never promise a guaranteed win, call a selection risk-free, or recommend a stake size or bankroll percentage.
9. Keep WhatsApp responses concise, readable, and conversational.
10. Use plain text only. Never use Markdown formatting markers such as asterisks, underscores, hash headers, or backticks. Use emojis and hyphen lists when useful.
11. Never invent, infer, reconstruct, or embellish statistics that were not returned by a tool in the current turn. Do not add xG, shot counts, possession, table positions, points, goals, or other metrics unless a current-turn tool explicitly returned them.
12. Never reuse numeric sports data from conversation memory as if it were current. For standings, results, fixtures, form, injuries, scorers, or stats, current-turn tool output is the only authoritative source.
13. If a standings tool says the structured table is unavailable, do not build a partial table from web snippets and do not fill missing rows with guesses or dashes.
14. When a data provider returns fewer than five recent matches, clearly say how many matches the summary is based on. Do not describe one or two matches as proof of "good form", "bad form", title contention, or another broad conclusion.
15. If a tool reports that data is unavailable, say so rather than inventing an answer.
""".strip()

    return create_react_agent(
        model=model,
        tools=TOOLS,
        prompt=prompt,
        checkpointer=memory,
    )


def _content_to_text(content: Any) -> str:
    """Normalize LangChain/Anthropic message content into plain text."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text")
                if text:
                    parts.append(str(text))
                continue

            text = getattr(block, "text", None)
            if text:
                parts.append(str(text))

        return "\n".join(parts)

    return str(content)


def sanitize_whatsapp_response(response: str) -> str:
    """Strip model/tool markup and Markdown before sending to WhatsApp."""
    response = re.sub(
        r"<function=\w+>.*?</function>",
        "",
        response,
        flags=re.DOTALL | re.IGNORECASE,
    )
    response = re.sub(r"^\s*#{1,6}\s*", "", response, flags=re.MULTILINE)
    response = response.replace("**", "")
    response = response.replace("__", "")
    response = response.replace("*", "")
    response = response.replace("`", "")
    response = re.sub(r"\n{3,}", "\n\n", response)
    response = response.strip()

    if not response:
        return "Sorry, I couldn\'t process that. Please try again!"

    return response

def _current_turn_tool_outputs(messages: list[Any]) -> list[str]:
    """Return unique tool outputs produced after the most recent user message."""
    last_human_index = -1
    for index, message in enumerate(messages):
        if getattr(message, "type", "") == "human":
            last_human_index = index

    outputs: list[str] = []
    seen: set[str] = set()
    for message in messages[last_human_index + 1:]:
        if getattr(message, "type", "") != "tool":
            continue

        text = _content_to_text(getattr(message, "content", "")).strip()
        if text and text not in seen:
            outputs.append(text)
            seen.add(text)

    return outputs


def _parse_auto_sportybet_code_request(
    message: str,
) -> tuple[int, float, bool] | None:
    """Parse casual requests for a model-ranked SportyBet code for today."""
    lowered = re.sub(r"\s+", " ", message.strip().casefold())
    if "today" not in lowered:
        return None

    mentions_request_shape = (
        "code" in lowered
        or re.search(r"\b\d+(?:\.\d+)?\s*(?:odds|odd)\b", lowered)
        or re.search(
            r"\b\d+\s*[- ]?(?:game|games|match|matches|leg|legs)\b",
            lowered,
        )
    )
    if not mentions_request_shape:
        return None

    if not any(
        token in lowered
        for token in (
            "give", "build", "make", "create", "generate", "prepare",
            "random", "sure", "safe", "high support",
        )
    ):
        return None

    count_match = re.search(
        r"\b(\d+)\s*[- ]?(?:game|games|match|matches|leg|legs)\b",
        lowered,
    )
    odds_match = re.search(
        r"\b(\d+(?:\.\d+)?)\s*(?:odds|odd)\b",
        lowered,
    )

    match_count = int(count_match.group(1)) if count_match else 0
    target_odds = float(odds_match.group(1)) if odds_match else 0.0

    if match_count == 0 and target_odds == 0.0:
        match_count = 3

    randomize = "random" in lowered
    return match_count, target_odds, randomize


def _direct_guarded_tool_response(message: str, user_id: str) -> str | None:
    """Route guarded factual intents directly to deterministic tools."""
    raw_commands = [
        part.strip()
        for part in re.split(r"[\r\n;]+", message)
        if part.strip()
    ]
    if len(raw_commands) > 1:
        first_load = re.fullmatch(
            r"(?:analyse|analyze|check|review|load|get)\s+(?:this\s+)?"
            r"(?:sportybet\s+)?(?:booking\s+|share\s+)?code[:\s]+"
            r"([A-Z0-9]{4,12})[?.!]?",
            raw_commands[0],
            flags=re.IGNORECASE,
        )
        if first_load:
            try:
                analysis = _run_sportybet_booking_analysis(
                    first_load.group(1)
                )
                _cache_sportybet_analysis(user_id, analysis)
                _initialize_sportybet_working_slip(user_id, analysis)
            except (ValueError, SportyBetLookupError) as exc:
                return str(exc)

            responses: list[str] = []
            for command in raw_commands[1:6]:
                response = _direct_guarded_tool_response(command, user_id)
                if response is not None:
                    responses.append(response)

            if responses:
                return (
                    f"Loaded SportyBet code {analysis['code']} and applied "
                    f"{len(responses)} follow-up command"
                    f"{'s' if len(responses) != 1 else ''}.\n\n"
                    + "\n\n".join(responses)
                )

            return (
                _format_sportybet_analysis_summary(analysis)
                + "\n\nWorking slip loaded."
            )
    normalized = re.sub(r"\s+", " ", message.strip())
    lowered = normalized.casefold()

    # Existing-code editor commands operate on a per-user working slip.
    direct_code_shorten = re.fullmatch(
        r"(?:shorten|reduce|trim)\s+(?:sportybet\s+)?(?:booking\s+)?code\s+"
        r"([A-Z0-9]{4,12})\s+(?:to|down to)\s+(\d+)\s+"
        r"(?:selections|picks|legs|games)",
        normalized,
        flags=re.IGNORECASE,
    )
    if direct_code_shorten:
        try:
            analysis = _run_sportybet_booking_analysis(
                direct_code_shorten.group(1)
            )
            _cache_sportybet_analysis(user_id, analysis)
            _initialize_sportybet_working_slip(user_id, analysis)
            return _keep_top_sportybet_slip(
                user_id,
                int(direct_code_shorten.group(2)),
            )
        except (ValueError, SportyBetLookupError) as exc:
            return str(exc)

    direct_code_target = re.fullmatch(
        r"(?:shorten|reduce|trim|edit)\s+(?:sportybet\s+)?(?:booking\s+)?code\s+"
        r"([A-Z0-9]{4,12})\s+(?:to|around|about|towards?)\s+"
        r"(\d+(?:\.\d+)?)\s*(?:odds|odd)",
        normalized,
        flags=re.IGNORECASE,
    )
    if direct_code_target:
        try:
            analysis = _run_sportybet_booking_analysis(
                direct_code_target.group(1)
            )
            _cache_sportybet_analysis(user_id, analysis)
            _initialize_sportybet_working_slip(user_id, analysis)
            return _target_sportybet_slip_odds(
                user_id,
                float(direct_code_target.group(2)),
            )
        except (ValueError, SportyBetLookupError) as exc:
            return str(exc)

    if re.fullmatch(
        r"(?:show|display|list)(?:\s+the|\s+my)?\s+(?:current\s+)?"
        r"(?:working\s+)?(?:sportybet\s+)?slip[?.!]?",
        normalized,
        flags=re.IGNORECASE,
    ):
        return _format_sportybet_working_slip(user_id)

    if re.fullmatch(r"undo(?:\s+last)?[?.!]?", normalized, flags=re.IGNORECASE):
        return _undo_sportybet_slip(user_id)

    if re.fullmatch(r"redo(?:\s+last)?[?.!]?", normalized, flags=re.IGNORECASE):
        return _redo_sportybet_slip(user_id)

    if re.fullmatch(
        r"(?:remove|drop|delete)\s+(?:all\s+)?(?:the\s+)?"
        r"(?:unmodelled|unmodeled)(?:\s+(?:selections|picks|legs))?[?.!]?",
        normalized,
        flags=re.IGNORECASE,
    ):
        return _remove_unmodelled_sportybet_slip(user_id)

    if re.fullmatch(
        r"(?:remove|drop|delete)\s+(?:all\s+)?(?:the\s+)?"
        r"(?:lower[- ]support|low[- ]support)(?:\s+(?:selections|picks|legs))?[?.!]?",
        normalized,
        flags=re.IGNORECASE,
    ):
        return _remove_lower_support_sportybet_slip(user_id)

    lowest_match = re.fullmatch(
        r"(?:remove|drop|delete)\s+(?:the\s+)?lowest\s+(\d+)"
        r"(?:\s+(?:selections|picks|legs))?[?.!]?",
        normalized,
        flags=re.IGNORECASE,
    )
    if lowest_match:
        return _remove_lowest_sportybet_slip(
            user_id,
            int(lowest_match.group(1)),
        )

    keep_top_match = re.fullmatch(
        r"(?:(?:keep|retain)\s+(?:only\s+)?(?:the\s+)?"
        r"(?:strongest|best|top)\s+(\d+)|"
        r"(?:shorten|reduce|trim)(?:\s+(?:it|this|the slip))?\s+"
        r"(?:to|down to)\s+(\d+))"
        r"(?:\s+(?:selections|picks|legs|games))?[?.!]?",
        normalized,
        flags=re.IGNORECASE,
    )
    if keep_top_match:
        count_text = keep_top_match.group(1) or keep_top_match.group(2)
        return _keep_top_sportybet_slip(user_id, int(count_text))

    remove_numbers_match = re.fullmatch(
        r"(?:remove|drop|delete)(?:\s+(?:selections?|picks?|legs?))?\s+"
        r"([0-9,\sand]+)[?.!]?",
        normalized,
        flags=re.IGNORECASE,
    )
    if remove_numbers_match:
        indexes = [
            int(item)
            for item in re.findall(r"\d+", remove_numbers_match.group(1))
        ]
        return _remove_sportybet_slip_indexes(user_id, indexes)

    replace_safer_match = re.fullmatch(
        r"(?:replace|change)\s+(?:selection\s+)?(\d+)\s+"
        r"(?:with|to)\s+(?:a\s+)?(?:safer|stronger|higher[- ]support)"
        r"(?:\s+(?:market|pick|selection))?[?.!]?",
        normalized,
        flags=re.IGNORECASE,
    )
    if replace_safer_match:
        return _replace_sportybet_slip_with_safer_market(
            user_id,
            int(replace_safer_match.group(1)),
        )

    target_current_match = re.fullmatch(
        r"(?:(?:shorten|reduce|trim|adjust)(?:\s+(?:it|this|the slip))?\s+"
        r"(?:to|around|about|towards?)\s+|target\s+)"
        r"(\d+(?:\.\d+)?)\s*(?:odds|odd)[?.!]?",
        normalized,
        flags=re.IGNORECASE,
    )
    if target_current_match:
        return _target_sportybet_slip_odds(
            user_id,
            float(target_current_match.group(1)),
        )

    if re.fullmatch(
        r"(?:create|make|generate|prepare)\s+(?:the\s+)?(?:new|edited|current)"
        r"(?:\s+sportybet)?\s+(?:booking\s+|share\s+)?code[?.!]?",
        normalized,
        flags=re.IGNORECASE,
    ):
        return _create_code_from_working_sportybet_slip(user_id)

    auto_request = _parse_auto_sportybet_code_request(message)
    if auto_request is not None:
        match_count, target_odds, randomize = auto_request
        return str(
            build_model_ranked_sportybet_code.invoke(
                {
                    "match_count": match_count,
                    "target_odds": target_odds,
                    "randomize": randomize,
                }
            )
        )

    sportybet_create_match = re.match(
        r"^\s*(?:create|make|generate|prepare)\s+(?:a\s+)?sportybet\s+"
        r"(?:(?:booking|share)\s+)?code\b",
        message,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if sportybet_create_match:
        return str(
            create_sportybet_booking_code.invoke(
                {"request": message}
            )
        )

    cached_analysis = _cached_sportybet_analysis(user_id)
    cached_view_match = re.fullmatch(
        r"(?:show|list)(?:\s+me)?\s+(all|modelled|modeled|unmodelled|unmodeled)"
        r"(?:\s+(?:selections|picks|legs))?[?.!]?",
        normalized,
        flags=re.IGNORECASE,
    )
    if cached_view_match:
        if cached_analysis is None:
            return "Analyse a SportyBet code first, then I can show the cached selections."
        requested_view = cached_view_match.group(1).casefold()
        if requested_view == "modeled":
            requested_view = "modelled"
        elif requested_view == "unmodeled":
            requested_view = "unmodelled"
        return _format_sportybet_cached_view(cached_analysis, requested_view)

    if re.fullmatch(
        r"(?:show|repeat)(?:\s+the)?\s+(?:sportybet\s+)?summary[?.!]?",
        normalized,
        flags=re.IGNORECASE,
    ):
        if cached_analysis is None:
            return "Analyse a SportyBet code first, then I can show the cached summary."
        return _format_sportybet_analysis_summary(cached_analysis)

    sportybet_code_match = re.search(
        r"(?:analyse|analyze|check|review|load|get)\s+(?:this\s+)?"
        r"(?:sportybet\s+)?(?:booking\s+|share\s+)?code[:\s]+([A-Z0-9]{4,12})\b",
        normalized,
        flags=re.IGNORECASE,
    )
    if sportybet_code_match and "sportybet" in lowered:
        try:
            return _analyse_and_cache_sportybet_booking_code(
                sportybet_code_match.group(1),
                user_id,
            )
        except ValueError as exc:
            return str(exc)
        except SportyBetLookupError as exc:
            return str(exc)
        except Exception as exc:
            return f"I couldn't analyse that SportyBet code right now: {exc}"

    sportybet_code_short = re.fullmatch(
        r"(?:sportybet\s+)?(?:booking\s+|share\s+)?code[:\s]+([A-Z0-9]{4,12})[?.!]?",
        normalized,
        flags=re.IGNORECASE,
    )
    if sportybet_code_short and "sportybet" in lowered:
        try:
            return _analyse_and_cache_sportybet_booking_code(
                sportybet_code_short.group(1),
                user_id,
            )
        except ValueError as exc:
            return str(exc)
        except SportyBetLookupError as exc:
            return str(exc)
        except Exception as exc:
            return f"I couldn't analyse that SportyBet code right now: {exc}"

    rule_keywords = ("offside", "var", "yellow card", "red card", "penalty rule", "penalty kick")
    if any(keyword in lowered for keyword in rule_keywords):
        return str(search_football_knowledge.invoke({"question": normalized}))

    platform_pattern = r"(SportyBet|Bet9ja|BetKing|MSport|1xBet|Betway)"
    platform_patterns = (
        rf"^(?:give me\s+)?(.+?)\s+(?:vs\.?|versus)\s+(.+?)\s+"
        rf"(?:betting\s+)?predictions?\s+(?:for|on)\s+{platform_pattern}[?.!]?$",
        rf"^(?:predict|prediction|predictions)\s+(.+?)\s+(?:vs\.?|versus)\s+(.+?)\s+"
        rf"(?:for|on)\s+{platform_pattern}[?.!]?$",
    )
    for pattern in platform_patterns:
        platform_match = re.match(pattern, normalized, flags=re.IGNORECASE)
        if platform_match:
            team1 = platform_match.group(1).strip()
            team2 = platform_match.group(2).strip()
            platform = platform_match.group(3).strip()
            _remember_matchup(user_id, team1, team2)
            return str(
                format_prediction_for_platform.invoke(
                    {
                        "team1": team1,
                        "team2": team2,
                        "platform": platform,
                    }
                )
            )

    platform_first_match = re.match(
        rf"^{platform_pattern}\s+(?:prediction|predictions|betting prediction|betting predictions)\s+"
        r"(.+?)\s+(?:vs\.?|versus)\s+(.+?)[?.!]?$",
        normalized,
        flags=re.IGNORECASE,
    )
    if platform_first_match:
        platform = platform_first_match.group(1).strip()
        team1 = platform_first_match.group(2).strip()
        team2 = platform_first_match.group(3).strip()
        _remember_matchup(user_id, team1, team2)
        return str(
            format_prediction_for_platform.invoke(
                {
                    "team1": team1,
                    "team2": team2,
                    "platform": platform,
                }
            )
        )

    betting_terms = (
        "1x2", "moneyline", "double chance", "draw no bet", "dnb",
        "btts", "both teams to score", "over/under", "over under",
        "asian handicap", "european handicap", "team total",
        "clean sheet", "win to nil", "half-time/full-time",
        "half time full time", "accumulator", "parlay", "bet builder",
        "push", "void bet", "half win", "half loss", "cash out",
        "implied probability", "betting odds", "stake", "return", "profit",
    )

    betting_prediction_match = re.search(
        r"(?:betting prediction|bet prediction|betting markets|market prediction|all betting predictions)"
        r"(?:\s+for)?\s+(.+?)\s+(?:vs\.?|versus)\s+(.+?)(?:[?.!]|$)",
        normalized,
        flags=re.IGNORECASE,
    )
    if betting_prediction_match:
        team1 = betting_prediction_match.group(1).strip()
        team2 = betting_prediction_match.group(2).strip()
        _remember_matchup(user_id, team1, team2)
        return str(
            predict_betting_markets.invoke(
                {"team1": team1, "team2": team2, "market": "all"}
            )
        )

    if (
        ("betting prediction" in lowered or "bet predictions" in lowered or "betting markets" in lowered)
        and ("them" in lowered or "between them" in lowered)
    ):
        matchup = _last_matchup(user_id)
        if matchup is not None:
            team1, team2 = matchup
            return str(
                predict_betting_markets.invoke(
                    {"team1": team1, "team2": team2, "market": "all"}
                )
            )

    total_market_match = re.match(
        r"^(over|under)\s+([0-9]+(?:\.[0-9]+)?)\s+"
        r"(.+?)\s+(?:vs\.?|versus)\s+(.+?)(?:[?.!]|$)",
        normalized,
        flags=re.IGNORECASE,
    )
    if total_market_match:
        side = total_market_match.group(1).title()
        line = total_market_match.group(2)
        team1 = total_market_match.group(3).strip()
        team2 = total_market_match.group(4).strip()

        _remember_matchup(user_id, team1, team2)

        return str(
            predict_betting_markets.invoke(
                {
                    "team1": team1,
                    "team2": team2,
                    "market": f"{side} {line}",
                }
            )
        )

    handicap_selection_match = re.match(
        r"^(.+?)\s+([+-]\d+(?:\.\d+)?)\s+"
        r"(?:asian\s+)?handicap\s+"
        r"(.+?)\s+(?:vs\.?|versus)\s+(.+?)(?:[?.!]|$)",
        normalized,
        flags=re.IGNORECASE,
    )
    if handicap_selection_match:
        selection = handicap_selection_match.group(1).strip()
        line = handicap_selection_match.group(2)
        team1 = handicap_selection_match.group(3).strip()
        team2 = handicap_selection_match.group(4).strip()

        _remember_matchup(user_id, team1, team2)

        return str(
            predict_betting_markets.invoke(
                {
                    "team1": team1,
                    "team2": team2,
                    "market": f"{selection} {line} Asian Handicap",
                }
            )
        )

    general_market_match = re.match(
        r"^(1x2|moneyline|asian handicap|"
        r"double chance(?:\s+(?:1x|x2|12))?|"
        r"(?:btts|both teams to score)(?:\s+(?:yes|no))?|"
        r"draw no bet|dnb|clean sheet|win to nil)\s+"
        r"(.+?)\s+(?:vs\.?|versus)\s+(.+?)(?:[?.!]|$)",
        normalized,
        flags=re.IGNORECASE,
    )
    if general_market_match:
        market = general_market_match.group(1).strip()
        team1 = general_market_match.group(2).strip()
        team2 = general_market_match.group(3).strip()

        _remember_matchup(user_id, team1, team2)

        return str(
            predict_betting_markets.invoke(
                {
                    "team1": team1,
                    "team2": team2,
                    "market": market,
                }
            )
        )
    
    exact_score_match = re.search(
        r"(?:correct|exact)\s+score(?:\s+(?:for|between))?\s+(.+?)\s+(?:vs\.?|versus|and)\s+(.+?)(?:[?.!]|$)",
        normalized,
        flags=re.IGNORECASE,
    )
    if exact_score_match:
        team1 = exact_score_match.group(1).strip()
        team2 = exact_score_match.group(2).strip()
        _remember_matchup(user_id, team1, team2)
        return str(predict_correct_score.invoke({"team1": team1, "team2": team2}))

    if ("correct score" in lowered or "exact score" in lowered) and (
        "them" in lowered or "between them" in lowered
    ):
        matchup = _last_matchup(user_id)
        if matchup is not None:
            team1, team2 = matchup
            return str(predict_correct_score.invoke({"team1": team1, "team2": team2}))

    prediction_match = re.search(
        r"\bpredict\s+(.+?)\s+(?:vs\.?|versus)\s+(.+?)(?:[?.!]|$)",
        normalized,
        flags=re.IGNORECASE,
    )
    if prediction_match:
        team1 = prediction_match.group(1).strip()
        team2 = prediction_match.group(2).strip()
        _remember_matchup(user_id, team1, team2)
        return str(predict_match.invoke({"team1": team1, "team2": team2}))

    # Pure terminology questions can be answered locally without live sports data.
    if " vs " not in lowered and any(term in lowered for term in betting_terms):
        return str(search_betting_knowledge.invoke({"question": normalized}))

    return None

def _requires_current_tool_data(message: str) -> bool:
    """Detect requests that should never be answered from stale conversation memory."""
    lowered = message.casefold()
    keywords = (
        "live", "score", "table", "standing", "fixture", "next match",
        "recent", "form", "top scorer", "scorer", "player", "injury",
        "transfer", "news", "head-to-head", "head to head", "stats",
        "statistics", "predict", "prediction", "who will win",
        "betting prediction", "bet predictions", "betting markets",
        "btts", "both teams to score", "double chance", "draw no bet",
        "asian handicap", "team total", "sportybet code", "booking code",
    )
    return any(keyword in lowered for keyword in keywords)


def ask_agent(message: str, user_id: str) -> str:
    direct_response = _direct_guarded_tool_response(message, user_id)
    if direct_response is not None:
        return sanitize_whatsapp_response(direct_response)

    result = get_agent().invoke(
        {"messages": [{"role": "user", "content": message}]},
        config={"configurable": {"thread_id": f"{MEMORY_NAMESPACE}:{user_id}"}},
    )

    messages = result["messages"]
    tool_outputs = _current_turn_tool_outputs(messages)

    if tool_outputs:
        return sanitize_whatsapp_response("\n\n".join(tool_outputs))

    if _requires_current_tool_data(message):
        return (
            "I couldn't verify that with a current data source right now. "
            "Please try again in a moment."
        )

    raw_content = messages[-1].content
    return sanitize_whatsapp_response(_content_to_text(raw_content))


def _split_whatsapp_message(body: str, max_chars: int = 1500) -> list[str]:
    """Split long WhatsApp replies into Twilio-safe chunks."""
    text = body.strip()
    if not text:
        return [""]
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    current = ""

    def flush() -> None:
        nonlocal current
        if current.strip():
            chunks.append(current.strip())
            current = ""

    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        candidate = paragraph if not current else f"{current}\n\n{paragraph}"
        if len(candidate) <= max_chars:
            current = candidate
            continue

        flush()

        if len(paragraph) <= max_chars:
            current = paragraph
            continue

        for line in paragraph.splitlines():
            line = line.strip()
            if not line:
                continue

            if len(line) <= max_chars:
                candidate = line if not current else f"{current}\n{line}"
                if len(candidate) <= max_chars:
                    current = candidate
                else:
                    flush()
                    current = line
                continue

            words = line.split()
            for word in words:
                if len(word) > max_chars:
                    flush()
                    for start in range(0, len(word), max_chars):
                        chunks.append(word[start:start + max_chars])
                    continue

                candidate = word if not current else f"{current} {word}"
                if len(candidate) <= max_chars:
                    current = candidate
                else:
                    flush()
                    current = word

    flush()
    return chunks


def _send_whatsapp_message(to: str, body: str) -> None:
    client = Client(
        _require_env("TWILIO_ACCOUNT_SID"),
        _require_env("TWILIO_AUTH_TOKEN"),
    )
    chunks = _split_whatsapp_message(body)
    total_parts = len(chunks)
    for index, chunk in enumerate(chunks, start=1):
        if total_parts > 1:
            chunk = f"Part {index}/{total_parts}\n{chunk}"
        client.messages.create(
            from_=TWILIO_WHATSAPP_FROM,
            to=to,
            body=chunk,
        )


def process_and_reply(message: str, sender: str) -> None:
    lowered_message = message.casefold()
    auto_code_request = (
        _parse_auto_sportybet_code_request(message) is not None
    )
    if auto_code_request:
        try:
            _send_whatsapp_message(
                sender,
                "I’m checking today’s SportyBet fixtures and ranking the supported markets with the independent model now.",
            )
        except Exception as exc:
            print(f"Twilio acknowledgement error for {sender}: {exc}")

    if (
        not auto_code_request
        and "sportybet" in lowered_message
        and "code" in lowered_message
        and any(
            lowered_message.lstrip().startswith(word)
            for word in ("create", "make", "generate", "prepare")
        )
    ):
        try:
            _send_whatsapp_message(
                sender,
                "SportyBet selections received. I’m resolving the live markets and preparing the share code now.",
            )
        except Exception as exc:
            print(f"Twilio acknowledgement error for {sender}: {exc}")

    if (
        "sportybet" in lowered_message
        and "code" in lowered_message
        and any(word in lowered_message for word in ("analyse", "analyze", "check", "review", "load"))
    ):
        try:
            _send_whatsapp_message(
                sender,
                "SportyBet code received. I’m checking the slip and comparing the markets now.",
            )
        except Exception as exc:
            print(f"Twilio acknowledgement error for {sender}: {exc}")

    try:
        response = ask_agent(message, sender)
    except Exception as exc:
        print(f"Agent error for {sender}: {exc}")
        response = (
            "Sorry, I couldn't process that request right now. "
            "Please try again in a moment."
        )

    try:
        _send_whatsapp_message(sender, response)
    except Exception as exc:
        print(f"Twilio send error for {sender}: {exc}")


def _mark_message_sid_processed(message_sid: str) -> bool:
    """Return False for a duplicate Twilio webhook message SID."""
    if not message_sid:
        return True

    with _PROCESSED_MESSAGE_LOCK:
        if message_sid in _PROCESSED_MESSAGE_SIDS:
            return False

        _PROCESSED_MESSAGE_SIDS.add(message_sid)
        _PROCESSED_MESSAGE_ORDER.append(message_sid)

        while len(_PROCESSED_MESSAGE_ORDER) > _MAX_PROCESSED_MESSAGE_SIDS:
            oldest = _PROCESSED_MESSAGE_ORDER.popleft()
            _PROCESSED_MESSAGE_SIDS.discard(oldest)

    return True


def _validate_twilio_request(request: Request, form_data: dict[str, str]) -> bool:
    if not VERIFY_TWILIO_SIGNATURE:
        return True

    auth_token = _require_env("TWILIO_AUTH_TOKEN")
    signature = request.headers.get("X-Twilio-Signature", "")
    if not signature:
        return False

    validator = RequestValidator(auth_token)
    return validator.validate(str(request.url), form_data, signature)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/webhook", status_code=204)
async def webhook(
    request: Request,
    background_tasks: BackgroundTasks,
) -> Response:
    raw_form = await request.form()
    form_data = {key: str(value) for key, value in raw_form.items()}

    if not _validate_twilio_request(request, form_data):
        raise HTTPException(status_code=403, detail="Invalid Twilio signature.")

    body = form_data.get("Body", "").strip()
    sender = form_data.get("From", "").strip()
    message_sid = form_data.get("MessageSid", "").strip()

    if not body or not sender:
        raise HTTPException(status_code=400, detail="Missing WhatsApp message data.")

    if not _mark_message_sid_processed(message_sid):
        print(f"Ignoring duplicate Twilio webhook for MessageSid={message_sid}")
        return Response(status_code=204)

    background_tasks.add_task(process_and_reply, body, sender)
    return Response(status_code=204)


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
