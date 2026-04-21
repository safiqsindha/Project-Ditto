"""
Pokémon Showdown battle log parser.

Parses the Showdown protocol text format into structured event dicts.
Supports both raw .log files and the JSON replay format from HuggingFace.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# Events that carry no information for constraint extraction
_SKIP_PREFIXES = {
    "poke", "rule", "clearpoke", "teampreview", "start", "upkeep",
    "inactive", "inactiveoff", "chat", "c", "c:", "raw", "html",
    "uhtml", "uhtmlchange", "name", "join", "leave", "player",
    "gametype", "gen", "tier", "rated", "seed", "message", "-message",
    "-hint", "request", "callback", "-singleturn", "-singlemove",
    "detailschange", "-formechange", "-transform", "-mega", "-primal",
    "-burst", "-zpower", "-zbroken", "-activate", "-fieldactivate",
    "-end", "-enditem", "-endability", "-ability", "-item", "-fail",
    "-miss", "-notarget", "-immune", "-ohko", "-crit", "-supereffective",
    "-resisted", "-block", "-mustrecharge", "-prepare", "-hitcount",
    "-waiting", "-combine", "-nothing", "expire", "debug", "askreg",
    "qualifiesfor", "tournament", "win", "tie", "error", "bigerror",
    "init", "title", "j", "l", "n",
}

# The 12 events we emit constraints for (§4.3)
_CONSTRAINT_EVENTS = {
    "switch", "faint", "-damage", "-heal", "-status", "-boost",
    "-unboost", "move", "-sidestart", "-weather", "-fieldstart", "turn",
}


@dataclass
class BattleEvent:
    type: str
    player: str | None       # "p1" or "p2", if applicable
    args: list[str]
    raw_line: str
    turn: int = 0


@dataclass
class BattleLog:
    match_id: str
    format: str
    players: dict[str, str]  # {"p1": "username1", "p2": "username2"}
    rating: dict[str, int]   # {"p1": 1800, "p2": 1750}
    winner: str | None
    events: list[BattleEvent] = field(default_factory=list)


def parse_showdown_log(text: str, match_id: str = "unknown") -> BattleLog:
    """Parse Showdown protocol text into a BattleLog."""
    log = BattleLog(
        match_id=match_id,
        format="",
        players={},
        rating={},
        winner=None,
    )
    current_turn = 0

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or not line.startswith("|"):
            continue

        parts = line.split("|")
        # parts[0] is empty (before leading |), parts[1] is event type
        if len(parts) < 2:
            continue

        etype = parts[1]
        args = parts[2:] if len(parts) > 2 else []

        # Metadata extraction
        if etype == "tier":
            log.format = args[0] if args else ""
        elif etype == "player":
            if len(args) >= 2:
                log.players[args[0]] = args[1]
            if len(args) >= 4 and args[3].isdigit():
                log.rating[args[0]] = int(args[3])
        elif etype == "win":
            log.winner = args[0] if args else None
        elif etype == "turn":
            if args and args[0].isdigit():
                current_turn = int(args[0])

        # Skip non-constraint events
        if etype in _SKIP_PREFIXES and etype not in _CONSTRAINT_EVENTS:
            continue

        # Determine player side from first arg (e.g. "p1a: Garchomp")
        player = None
        if args:
            m = re.match(r"^(p[12])[ab]?:", args[0])
            if m:
                player = m.group(1)

        evt = BattleEvent(
            type=etype,
            player=player,
            args=args,
            raw_line=raw_line,
            turn=current_turn,
        )
        log.events.append(evt)

    return log


def parse_huggingface_replay(record: dict[str, Any]) -> BattleLog | None:
    """Parse one record from HolidayOugi/pokemon-showdown-replays."""
    log_text = record.get("log", "")
    if not log_text:
        return None
    match_id = str(record.get("id", "unknown"))
    return parse_showdown_log(log_text, match_id=match_id)


def load_log_file(path: Path) -> BattleLog:
    """Load a raw Showdown .log file from disk."""
    text = path.read_text(encoding="utf-8", errors="replace")
    return parse_showdown_log(text, match_id=path.stem)


def filter_match(log: BattleLog, min_elo: int = 1700, min_turns: int = 15) -> bool:
    """Return True if the match meets quality criteria (§3.2)."""
    fmt_normalized = log.format.lower().replace(" ", "").replace("[", "").replace("]", "")
    if "gen9ou" not in fmt_normalized and "gen9" not in fmt_normalized:
        return False
    if log.winner is None:
        return False
    p1_elo = log.rating.get("p1", 0)
    p2_elo = log.rating.get("p2", 0)
    if p1_elo < min_elo or p2_elo < min_elo:
        return False
    max_turn = max((e.turn for e in log.events), default=0)
    if max_turn < min_turns:
        return False
    return True


def constraint_events(log: BattleLog) -> list[BattleEvent]:
    """Return only the events that emit constraints (§4.3)."""
    return [e for e in log.events if e.type in _CONSTRAINT_EVENTS]


if __name__ == "__main__":
    import sys

    path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if path and path.exists():
        log = load_log_file(path)
        evts = constraint_events(log)
        print(f"Match: {log.match_id}")
        print(f"Format: {log.format}")
        print(f"Players: {log.players}")
        print(f"Rating: {log.rating}")
        print(f"Winner: {log.winner}")
        print(f"Total events: {len(log.events)}")
        print(f"Constraint events: {len(evts)}")
        from collections import Counter
        counts = Counter(e.type for e in evts)
        print("Event type breakdown:", dict(counts))
    else:
        print("Usage: python -m src.parser <path_to_log_file>")
