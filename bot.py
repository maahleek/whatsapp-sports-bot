import math
import os
import re
import sqlite3
from collections import deque
from datetime import datetime, timezone
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
from sportybet import SportyBetLookupError, extract_booking, fetch_booking, normalize_booking_code

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

LEAGUE_CODES = {
    "premier league": "PL",
    "english premier league": "PL",
    "epl": "PL",
    "la liga": "PD",
    "spanish la liga": "PD",
    "bundesliga": "BL1",
    "german bundesliga": "BL1",
    "serie a": "SA",
    "italian serie a": "SA",
    "ligue 1": "FL1",
    "french ligue 1": "FL1",
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
    data = _safe_get_json(
        f"{SPORTSDB_BASE_URL}/searchteams.php",
        params={"t": team_name},
    )
    teams = data.get("teams") or []
    if not teams:
        raise ValueError(f"Team '{team_name}' not found.")
    team = teams[0]
    return str(team["idTeam"]), str(team["strTeam"])


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


def _season_team_summary(team_name: str) -> dict[str, float | int | str] | None:
    profile = _team_profile(team_name)
    code = _league_code(profile["league"])
    if not code:
        return None

    data = _safe_get_json(
        f"{FOOTBALL_DATA_BASE_URL}/competitions/{code}/standings",
        headers=_football_data_headers(),
    )
    standings = data.get("standings") or []
    table = standings[0].get("table", []) if standings else []
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

def _build_prediction_context(team1: str, team2: str) -> dict[str, Any]:
    """Gather structured season/form data, fixture venue, and score projection."""
    first_season = second_season = None
    first_recent = second_recent = None

    try:
        first_season = _season_team_summary(team1)
    except Exception:
        pass
    try:
        second_season = _season_team_summary(team2)
    except Exception:
        pass
    try:
        first_recent = _form_summary(team1)
    except Exception:
        pass
    try:
        second_recent = _form_summary(team2)
    except Exception:
        pass

    first_strength, first_source = _prediction_strength(first_season, first_recent)
    second_strength, second_source = _prediction_strength(second_season, second_recent)

    first_name = str((first_season or first_recent or {"team": team1})["team"])
    second_name = str((second_season or second_recent or {"team": team2})["team"])

    fixture = None
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
        rf"(?:^|[;,&\\s]){re.escape(key)}\\s*=\\s*([+-]?\\d+(?:\\.\\d+)?)",
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


def _sportybet_selection_model_support(
    context: dict[str, Any],
    selection: dict[str, Any],
) -> tuple[float | None, str]:
    """Return model support for a normalized SportyBet selection."""
    home, away, matrix = _fixture_order_projection(context)
    market = str(selection.get("market_name") or "").casefold()
    market_id = str(selection.get("market_id") or "")
    outcome = str(selection.get("outcome_name") or "").casefold()
    outcome_id = str(selection.get("outcome_id") or "")
    specifier = str(selection.get("specifier") or "")

    home_win = sum(p for h, a, p in matrix if h > a)
    draw = sum(p for h, a, p in matrix if h == a)
    away_win = sum(p for h, a, p in matrix if h < a)

    if market_id == "1" or "1x2" in market or "match result" in market:
        if outcome_id == "1" or "home" in outcome or outcome.strip() == "1":
            return home_win, "1X2 home"
        if outcome_id == "2" or "draw" in outcome or outcome.strip() == "x":
            return draw, "1X2 draw"
        if outcome_id == "3" or "away" in outcome or outcome.strip() == "2":
            return away_win, "1X2 away"

    if "double chance" in market or market_id == "10":
        compact = re.sub(r"[^12x]", "", outcome)
        if compact == "1x":
            return home_win + draw, "Double Chance 1X"
        if compact == "x2":
            return draw + away_win, "Double Chance X2"
        if compact == "12":
            return home_win + away_win, "Double Chance 12"

    if (
        "both teams to score" in market
        or "gg/ng" in market
        or market_id == "29"
    ):
        yes = sum(p for h, a, p in matrix if h > 0 and a > 0)
        is_yes = (
            "yes" in outcome
            or outcome.strip() == "gg"
            or outcome_id.casefold() in {"yes", "gg"}
        )
        is_no = (
            "no" in outcome
            or outcome.strip() == "ng"
            or outcome_id.casefold() in {"no", "ng"}
        )
        if is_yes:
            return yes, "Both Teams to Score - Yes"
        if is_no:
            return 1.0 - yes, "Both Teams to Score - No"

    if "draw no bet" in market or market_id == "11":
        decisive = home_win + away_win
        if decisive <= 0:
            return None, "Draw No Bet"
        if "home" in outcome or outcome.strip() == "1":
            return home_win / decisive, "Draw No Bet - Home"
        if "away" in outcome or outcome.strip() == "2":
            return away_win / decisive, "Draw No Bet - Away"

    if "over/under" in market or "total" in market or market_id == "18":
        line = _specifier_number(specifier, "total")
        if line is None:
            joined = f"{market} {outcome}"
            number = re.search(r"([0-9]+(?:\\.[0-9]+)?)", joined)
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

    if "asian handicap" in market or market_id == "16":
        home_line = _specifier_number(specifier, "hcp")
        if home_line is not None:
            if "home" in outcome or outcome.strip() == "1":
                settlement = _asian_handicap_settlement(
                    matrix,
                    home_line,
                    side="home",
                )
                return _settlement_support(settlement), f"{home} {home_line:+g}"
            if "away" in outcome or outcome.strip() == "2":
                away_line = -home_line
                settlement = _asian_handicap_settlement(
                    matrix,
                    away_line,
                    side="away",
                )
                return _settlement_support(settlement), f"{away} {away_line:+g}"

    if "correct score" in market or market_id == "45":
        score = re.search(r"(\\d+)\\s*[-:]\\s*(\\d+)", outcome)
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


def _model_alignment(probability: float) -> str:
    if probability >= 0.65:
        return "higher model support"
    if probability >= 0.50:
        return "moderate model support"
    return "lower model support"


@tool
def analyse_sportybet_booking_code(booking_code: str) -> str:
    """Read a SportyBet booking code and compare supported selections with the model."""
    try:
        clean = normalize_booking_code(booking_code)
        payload = fetch_booking(clean)
        booking = extract_booking(payload)
        selections = booking.get("selections") or []
        if not selections:
            return (
                f"SportyBet code {clean} loaded, but no readable selections were returned. "
                "The code may be expired or the SportyBet response format may have changed."
            )

        lines = [
            f"SportyBet code: {clean}",
            f"Selections found: {len(selections)}",
            "",
        ]

        counts = {"higher": 0, "moderate": 0, "lower": 0, "unmodelled": 0}

        for index, selection in enumerate(selections, start=1):
            home = str(selection.get("home_team") or "Home")
            away = str(selection.get("away_team") or "Away")
            market_name = str(selection.get("market_name") or "Unknown market")
            outcome_name = str(selection.get("outcome_name") or "Unknown pick")
            odds = selection.get("odds")

            lines.append(f"{index}. {home} vs {away}")
            lines.append(f"- Market: {market_name}")
            lines.append(f"- Pick: {outcome_name}")
            if isinstance(odds, (int, float)):
                lines.append(f"- SportyBet odds: {odds:.2f}")

            try:
                context = _build_prediction_context(home, away)
                if context.get("projection") is None:
                    probability = None
                    label = market_name
                else:
                    probability, label = _sportybet_selection_model_support(
                        context,
                        selection,
                    )
            except Exception:
                probability = None
                label = market_name

            if probability is None:
                counts["unmodelled"] += 1
                lines.append("- Model comparison: not currently modelled")
            else:
                alignment = _model_alignment(probability)
                if probability >= 0.65:
                    counts["higher"] += 1
                elif probability >= 0.50:
                    counts["moderate"] += 1
                else:
                    counts["lower"] += 1
                lines.append(
                    f"- Model support: {probability * 100:.1f}% ({alignment})"
                )
                if label and label.casefold() != market_name.casefold():
                    lines.append(f"- Interpreted as: {label}")
            lines.append("")

        lines.extend(
            [
                "Model comparison summary:",
                f"- Higher support: {counts['higher']}",
                f"- Moderate support: {counts['moderate']}",
                f"- Lower support: {counts['lower']}",
                f"- Not modelled: {counts['unmodelled']}",
                "",
                "This is an experimental read-only lookup of SportyBet's undocumented website endpoint. It does not place or submit a wager.",
                "Model probabilities are estimates, not guaranteed outcomes or betting advice.",
            ]
        )
        return "\n".join(lines)
    except ValueError as exc:
        return str(exc)
    except SportyBetLookupError as exc:
        return str(exc)
    except Exception as exc:
        return f"I couldn't analyse that SportyBet code right now: {exc}"


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
7. Use the betting-platform formatting tool when the user asks for predictions formatted for SportyBet, Bet9ja, BetKing, MSport, 1xBet, or Betway. Use the SportyBet booking-code analysis tool when the user asks to load, check, review, or analyse an existing SportyBet code. The SportyBet lookup is experimental and read-only because it uses an undocumented website endpoint. Do not claim to place a wager or generate a booking code unless a supported integration is explicitly configured.
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


def _direct_guarded_tool_response(message: str, user_id: str) -> str | None:
    """Route guarded factual intents directly to deterministic tools."""
    normalized = re.sub(r"\s+", " ", message.strip())
    lowered = normalized.casefold()

    sportybet_code_match = re.search(
        r"(?:analyse|analyze|check|review|load|get)\\s+(?:this\\s+)?"
        r"(?:sportybet\\s+)?(?:booking\\s+|share\\s+)?code[:\\s]+([A-Z0-9]{4,12})\\b",
        normalized,
        flags=re.IGNORECASE,
    )
    if sportybet_code_match and "sportybet" in lowered:
        return str(
            analyse_sportybet_booking_code.invoke(
                {"booking_code": sportybet_code_match.group(1)}
            )
        )

    sportybet_code_short = re.fullmatch(
        r"(?:sportybet\\s+)?(?:booking\\s+|share\\s+)?code[:\\s]+([A-Z0-9]{4,12})[?.!]?",
        normalized,
        flags=re.IGNORECASE,
    )
    if sportybet_code_short and "sportybet" in lowered:
        return str(
            analyse_sportybet_booking_code.invoke(
                {"booking_code": sportybet_code_short.group(1)}
            )
        )

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
        "asian handicap", "team total",
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
