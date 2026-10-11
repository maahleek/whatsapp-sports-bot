from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class BettingPlatform:
    key: str
    display_name: str
    aliases: tuple[str, ...]
    market_names: dict[str, str]
    supports_booking_codes: bool
    programmatic_booking_code_api: bool = False
    booking_code_lookup: bool = False
    booking_code_lookup_note: str = ""
    booking_code_creation: bool = False
    booking_code_creation_note: str = ""


_COMMON_MARKETS = {
    "1x2": "1X2 / Match Result",
    "double_chance": "Double Chance",
    "draw_no_bet": "Draw No Bet",
    "btts": "Both Teams to Score",
    "total_goals": "Over/Under Goals",
    "team_total": "Team Total Goals",
    "asian_handicap": "Asian Handicap",
    "correct_score": "Correct Score",
    "clean_sheet": "Clean Sheet",
    "win_to_nil": "Win to Nil",
}


PLATFORMS: dict[str, BettingPlatform] = {
    "sportybet": BettingPlatform(
        key="sportybet",
        display_name="SportyBet",
        aliases=("sportybet", "sporty bet"),
        market_names=_COMMON_MARKETS,
        supports_booking_codes=True,
        booking_code_lookup=True,
        booking_code_lookup_note=(
            "experimental lookup uses SportyBet's undocumented website endpoint "
            "and may stop working if the site changes"
        ),
        booking_code_creation=True,
        booking_code_creation_note=(
            "experimental non-staking share-code creation uses SportyBet's "
            "undocumented website endpoint and does not place a wager"
        ),
    ),
    "bet9ja": BettingPlatform(
        key="bet9ja",
        display_name="Bet9ja",
        aliases=("bet9ja", "bet 9ja"),
        market_names=_COMMON_MARKETS,
        supports_booking_codes=True,
    ),
    "betking": BettingPlatform(
        key="betking",
        display_name="BetKing",
        aliases=("betking", "bet king"),
        market_names=_COMMON_MARKETS,
        supports_booking_codes=True,
    ),
    "msport": BettingPlatform(
        key="msport",
        display_name="MSport",
        aliases=("msport", "m sport"),
        market_names=_COMMON_MARKETS,
        supports_booking_codes=True,
    ),
    "1xbet": BettingPlatform(
        key="1xbet",
        display_name="1xBet",
        aliases=("1xbet", "1x bet"),
        market_names=_COMMON_MARKETS,
        supports_booking_codes=True,
    ),
    "betway": BettingPlatform(
        key="betway",
        display_name="Betway",
        aliases=("betway", "bet way"),
        market_names=_COMMON_MARKETS,
        supports_booking_codes=True,
    ),
}


def normalize_platform_name(value: str) -> str | None:
    normalized = re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()
    for key, platform in PLATFORMS.items():
        candidates = (key, *platform.aliases)
        if any(
            re.sub(r"[^a-z0-9]+", " ", candidate.casefold()).strip() == normalized
            for candidate in candidates
        ):
            return key
    return None


def get_platform(value: str) -> BettingPlatform | None:
    key = normalize_platform_name(value)
    if key is None:
        return None
    return PLATFORMS[key]


def platform_market_name(platform: BettingPlatform, market_key: str) -> str:
    return platform.market_names.get(market_key, market_key.replace("_", " ").title())


def platform_capability_summary(platform: BettingPlatform) -> str:
    if platform.supports_booking_codes:
        booking = "supports booking/share codes"
    else:
        booking = "booking/share-code support not configured"

    capability_notes: list[str] = []
    if platform.booking_code_lookup:
        capability_notes.append(platform.booking_code_lookup_note)
    if platform.booking_code_creation:
        capability_notes.append(platform.booking_code_creation_note)

    if capability_notes:
        automation = (
            "; ".join(note for note in capability_notes if note)
            + "; automatic wager placement is not enabled"
        )
    elif platform.programmatic_booking_code_api:
        automation = "programmatic booking-code integration is configured"
    else:
        automation = (
            "no public programmatic booking-code API is configured in this project; "
            "the assistant can format selections but does not place wagers"
        )

    return f"{platform.display_name}: {booking}; {automation}."
