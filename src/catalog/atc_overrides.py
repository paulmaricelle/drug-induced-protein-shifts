# src/catalog/atc_overrides.py
"""Corrections manuelles des codes ATC4, prioritaires sur l'attribution algorithmique.

La table atc4_overrides.csv est versionnée : toute régénération du mapping ATC4 ou du
catalogue applique ces corrections aux ingrédients présents, quel que soit le code voté.
"""
from __future__ import annotations

import csv
from pathlib import Path

OVERRIDES_PATH = Path(__file__).with_name("atc4_overrides.csv")


def load_atc4_overrides(path: Path = OVERRIDES_PATH) -> dict[int, tuple[str, str]]:
    """ingredient_id -> (atc4_code, atc4_name)."""
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return {
            int(row["ingredient_id"]): (row["atc4_code"], row["atc4_name"])
            for row in csv.DictReader(f)
        }


def load_atc4_override_names(path: Path = OVERRIDES_PATH) -> dict[int, str]:
    """ingredient_id -> ingredient_name (pour les ingrédients absents du vocabulaire ATC)."""
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return {int(row["ingredient_id"]): row["ingredient_name"] for row in csv.DictReader(f)}
