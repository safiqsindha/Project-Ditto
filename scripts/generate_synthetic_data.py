"""
Generate synthetic Pokémon Showdown battle logs for pipeline development.

These are NOT real battle logs — they are structurally valid Showdown-protocol
text used to develop and smoke-test the pipeline when the real dataset is
unavailable. Replace with real data before running actual evaluations.
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

POKEMON = [
    "Garchomp", "Kingambit", "Gholdengo", "Dragapult", "IronValiant",
    "Landorus-Therian", "Volcarona", "Zamazenta", "Roaring-Moon", "TingLu",
    "Toxapex", "Slowking-Galar", "Amoonguss", "Corviknight", "GreatTusk",
    "Palafin", "SnorlaxStout", "Skeledirge", "Annihilape", "Dondozo",
]

MOVES = {p: [f"move_{i}" for i in range(1, 5)] for p in POKEMON}

STATUSES = ["par", "brn", "slp", "psn", "tox"]
WEATHER = ["rain", "sun", "sand", "snow"]
TERRAIN = ["electricterrain", "grassyterrain", "mistyterrain", "psychicterrain"]
HAZARDS = ["stealthrock", "spikes", "toxicspikes", "stickyweb"]


def _make_battle(rng: random.Random, match_id: str, elo1: int = 1850, elo2: int = 1780) -> str:
    p1_team = rng.sample(POKEMON, 6)
    p2_team = rng.sample(POKEMON, 6)

    p1_hp = {p: rng.randint(280, 380) for p in p1_team}
    p2_hp = {p: rng.randint(280, 380) for p in p2_team}
    p1_cur = {p: p1_hp[p] for p in p1_team}
    p2_cur = {p: p2_hp[p] for p in p2_team}

    p1_fainted = []
    p2_fainted = []

    p1_active = p1_team[0]
    p2_active = p2_team[0]

    lines = [
        f"|player|p1|Player1||{elo1}",
        f"|player|p2|Player2||{elo2}",
        "|gametype|singles",
        "|gen|9",
        "|tier|[Gen 9] OU",
        "|rated|",
        "|start",
        f"|switch|p1a: {p1_active}|{p1_active}, L50, M|{p1_cur[p1_active]}/{p1_hp[p1_active]}",
        f"|switch|p2a: {p2_active}|{p2_active}, L50, F|{p2_cur[p2_active]}/{p2_hp[p2_active]}",
    ]

    turn = 0
    while len(p1_fainted) < 6 and len(p2_fainted) < 6:
        turn += 1
        lines.append(f"|turn|{turn}")

        # Occasionally set weather/terrain early
        if turn == 2 and rng.random() < 0.3:
            w = rng.choice(WEATHER)
            lines.append(f"|-weather|{w}|[from] move: WeatherMove")
        if turn == 3 and rng.random() < 0.2:
            t = rng.choice(TERRAIN)
            lines.append(f"|-fieldstart|move: {t}|[from] move: TerrainMove")
        if turn == 4 and rng.random() < 0.25:
            h = rng.choice(HAZARDS)
            lines.append(f"|-sidestart|p2: Player2|move: {h}")

        # Moves
        m1 = rng.choice(MOVES[p1_active])
        m2 = rng.choice(MOVES[p2_active])
        lines.append(f"|move|p1a: {p1_active}|{m1}|p2a: {p2_active}")
        lines.append(f"|move|p2a: {p2_active}|{m2}|p1a: {p1_active}")

        # Damage to p2
        dmg2 = rng.randint(30, 90)
        p2_cur[p2_active] = max(0, p2_cur[p2_active] - dmg2)
        lines.append(f"|-damage|p2a: {p2_active}|{p2_cur[p2_active]}/{p2_hp[p2_active]}")

        # Damage to p1
        dmg1 = rng.randint(20, 80)
        p1_cur[p1_active] = max(0, p1_cur[p1_active] - dmg1)
        lines.append(f"|-damage|p1a: {p1_active}|{p1_cur[p1_active]}/{p1_hp[p1_active]}")

        # Occasional status
        if rng.random() < 0.08:
            st = rng.choice(STATUSES)
            target = rng.choice(["p1", "p2"])
            poke = p1_active if target == "p1" else p2_active
            lines.append(f"|-status|{target}a: {poke}|{st}")

        # Occasional boost
        if rng.random() < 0.1:
            stat = rng.choice(["atk", "def", "spa", "spd", "spe"])
            lines.append(f"|-boost|p1a: {p1_active}|{stat}|1")

        # Check faints
        if p2_cur[p2_active] <= 0:
            lines.append(f"|faint|p2a: {p2_active}")
            p2_fainted.append(p2_active)
            remaining = [p for p in p2_team if p not in p2_fainted]
            if not remaining:
                break
            p2_active = rng.choice(remaining)
            lines.append(f"|switch|p2a: {p2_active}|{p2_active}, L50, M|{p2_cur[p2_active]}/{p2_hp[p2_active]}")

        if p1_cur[p1_active] <= 0:
            lines.append(f"|faint|p1a: {p1_active}")
            p1_fainted.append(p1_active)
            remaining = [p for p in p1_team if p not in p1_fainted]
            if not remaining:
                break
            p1_active = rng.choice(remaining)
            lines.append(f"|switch|p1a: {p1_active}|{p1_active}, L50, F|{p1_cur[p1_active]}/{p1_hp[p1_active]}")

        # Occasional voluntary switch
        if rng.random() < 0.12:
            available = [p for p in p1_team if p not in p1_fainted and p != p1_active]
            if available:
                p1_active = rng.choice(available)
                lines.append(f"|switch|p1a: {p1_active}|{p1_active}, L50, M|{p1_cur[p1_active]}/{p1_hp[p1_active]}")

        if turn > 60:  # safety cap
            break

    winner = "Player1" if len(p2_fainted) >= len(p1_fainted) else "Player2"
    lines.append(f"|win|{winner}")

    return "\n".join(lines)


def generate(n: int = 2500, out_dir: Path = Path("data/raw"), seed: int = 0):
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    batch: list[dict] = []
    chunk = 1

    for i in range(n):
        match_id = f"synthetic_{i:05d}"
        elo1 = rng.randint(1700, 2100)
        elo2 = rng.randint(1700, 2100)
        log_text = _make_battle(rng, match_id, elo1, elo2)

        batch.append({
            "match_id": match_id,
            "format": "[Gen 9] OU",
            "players": {"p1": "Player1", "p2": "Player2"},
            "rating": {"p1": elo1, "p2": elo2},
            "winner": "Player1",
            "log": log_text,
        })

        if len(batch) == 1000:
            path = out_dir / f"matches_{chunk:04d}.jsonl"
            with open(path, "w") as f:
                for r in batch:
                    f.write(json.dumps(r) + "\n")
            print(f"Wrote {len(batch)} synthetic matches to {path}")
            batch = []
            chunk += 1

    if batch:
        path = out_dir / f"matches_{chunk:04d}.jsonl"
        with open(path, "w") as f:
            for r in batch:
                f.write(json.dumps(r) + "\n")
        print(f"Wrote {len(batch)} synthetic matches to {path}")

    print(f"Generated {n} synthetic matches in {out_dir}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=2500)
    p.add_argument("--out", type=Path, default=Path("data/raw"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    generate(args.n, args.out, args.seed)
