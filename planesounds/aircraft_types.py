"""Rough propulsion classification of ICAO type designators.

ADS-B gives an ICAO type code (e.g. ``C172``, ``B738``) but not "prop vs jet".
Following the approach in the reference method, we *exclude known jets*
rather than include known props, so unknown/missing types are kept.
"""
from __future__ import annotations

import re

# Common jet families (prefix match). Not exhaustive but covers most traffic.
JET_PREFIXES = (
    "A19", "A20", "A21", "A22", "A30", "A31", "A32", "A33", "A34", "A35", "A38", "A3ST", "BCS",
    "B37", "B38", "B39", "B70", "B71", "B72", "B73", "B74", "B75", "B76", "B77", "B78", "B79", "B52",
    "MD8", "MD9", "MD1", "DC9", "DC10", "DC1",
    "CRJ", "CL30", "CL35", "CL60", "GL5T", "GL6T", "GL7T", "GLEX", "GLF", "G150", "G280",
    "E13", "E14", "E17", "E19", "E29", "E35", "E45", "E50", "E55", "E75",
    "C25", "C50", "C51", "C52", "C55", "C56", "C65", "C68", "C70", "C75",
    "LJ", "H25", "F2TH", "F900", "FA", "PRM1", "HDJT", "SF50", "EA50", "PC24", "BE40",
    "ASTR", "GALX", "WW24", "J328", "F70", "F100", "RJ", "B46", "BA11", "SU95", "AJ27", "C919",
    "CONC", "IL", "TU", "YK", "AN12", "AN22", "AN72", "AN74", "AN124", "AN225",
    "F15", "F16", "F18", "F22", "F35", "A10", "T38", "C17", "C5", "KC", "E3", "E6", "B1", "B2",
)

# Explicit non-jet types that would otherwise collide with a jet prefix.
PROP_OVERRIDES = {
    "C150", "C152", "C162", "C170", "C172", "C175", "C177", "C180", "C182", "C185", "C195", "C205",
    "C206", "C207", "C208", "C210", "C303", "C310", "C320", "C335", "C337", "C340", "C350", "C400",
    "C402", "C404", "C414", "C421", "C425", "C441", "C72R", "C77R", "C82R", "C82S", "C82T",
    "C130", "C30J", "C160", "C212", "C295", "CN35", "C27J", "C2", "C46", "C47",
    "F27", "F50", "F406", "E110", "E120", "E121", "E314", "E312",
    "AN26", "AN28", "AN30", "AN32", "AN2", "AN38", "AN140", "IL18", "IL114",
    "B190", "BE20", "BE30", "BE9L", "BE9T", "BE10", "BE99", "BE18",
}

PROP_PREFIXES = (
    "PA", "P28", "P32", "P46", "BE", "SR2", "S22", "M20", "DA4", "DA2", "DA6", "DH8", "DHC", "AT4", "AT7",
    "B350", "PC6", "PC7", "PC9", "PC12", "PC21", "TBM", "KODI", "GA8",
    "SW4", "JS3", "JS4", "SF34", "SB20", "D228", "L410", "BN2", "TRIS", "P68", "P180", "AA5",
    "AC", "DR40", "J3", "CUB", "RV", "GLAS", "LNC", "EVOT", "COL",
)

HELI_PREFIXES = ("R22", "R44", "R66", "EC", "AS3", "AS5", "B06", "B407", "B412", "B429", "H60", "S76",
                 "S92", "A109", "A119", "A139", "A169", "AW", "H125", "H130", "H135", "H145", "H160",
                 "MD5", "MD6", "EXPL", "UH1", "CH47", "AS65", "BK17")


def classify(typecode) -> str:
    """Return 'jet', 'prop', 'heli' or 'unknown'."""
    if not isinstance(typecode, str) or not typecode.strip():
        return "unknown"
    tc = typecode.strip().upper()
    if not re.fullmatch(r"[A-Z0-9]{2,4}", tc):
        return "unknown"
    if tc in PROP_OVERRIDES:
        return "prop"
    if tc.startswith(HELI_PREFIXES):
        return "heli"
    if tc.startswith(JET_PREFIXES):
        return "jet"
    if tc.startswith(PROP_PREFIXES):
        return "prop"
    return "unknown"
