"""Scenario model for the DO (Declaração de Óbito).

A "scenario" is a realistic combination of: where/how the death happened,
whether it was natural or violent/accidental, and (for phase 2) whether the
deceased is an adult, an infant (<1 year) or a fetal death. Each scenario
declares a *distribution over DO place-of-occurrence categories* (field 20),
so the generator picks the scenario, then the place, then (from
`cause_chains.eligible_chains_for_scenario`) a chain that is valid for both —
never a chain that implies a different scenario or an impossible setting.

Phase 1 implements the adult scenarios. Infant and fetal scenarios are
defined here as data (so field-applicability bookkeeping is already in one
place) but are NOT yet wired into the generator — see NOT_IMPLEMENTED.
"""
from __future__ import annotations

from dataclasses import dataclass

# Field groups, named after the blocks they gate. Used both by the generator
# (death() consults these directly — see src/semantic/generators.py) and for
# documentation/tests of which of the 103 DO fields apply to a scenario.
FIELD_GROUPS = {
    "core": ("1", "2", "2_hora", "4", "5", "6", "7", "8", "9_anos", "10", "14", "14_codigo_cbo",
             "15", "15_numero", "16", "17", "18", "18_codigo", "19", "25", "25_codigo", "26",
             "38", "39", "41", "42", "43", "46", "53", "54", "55", "56", "57",
             "3", "11", "12", "13", "13_serie", "45"),
    "hospital_occurrence": ("20", "21", "21_codigo_cnes"),
    "home_or_public_occurrence": ("20", "22", "22_numero", "22_complemento", "23", "24"),
    "cause_chain": ("40_a_causa", "40_a_tempo", "40_a_cid", "40_b_causa", "40_b_tempo", "40_b_cid",
                    "40_c_causa", "40_c_tempo", "40_c_cid", "40_d_causa", "40_d_tempo", "40_d_cid",
                    "40_ii_1_causa", "40_ii_1_cid", "40_ii_2_causa", "40_ii_2_cid"),
    "external_cause": ("48", "49", "50", "51", "52"),
    "iml_svo": ("44", "44_uf"),
    "no_physician": ("58", "59_a", "59_b"),  # block IX — phase 2, see NOT_IMPLEMENTED
    "infant_under_1": ("9_meses", "9_dias", "9_horas", "9_minutos"),
    "maternal_history": ("27", "28", "28_serie", "29", "29_codigo_cbo", "30_nascidos_vivos",
                          "30_nascidos_vivos_ignorado", "30_perdas_fetais", "30_perdas_fetais_ignorado",
                          "31", "31_ignorado", "32", "33", "34", "35", "36", "37"),
}


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    label: str
    manner: str  # "Natural" or "Não natural"
    weight: float
    # Distribution over field-20 place categories: "1" hospital, "2" other health
    # establishment, "3" domicílio, "4" via pública. Must sum to 100 (checked by tests).
    place_weights: dict[str, float]
    age_group: str = "adult"  # "adult" | "child" | "infant" | "fetal"


# Adult scenarios implemented in phase 1. Weights are relative frequencies
# approximating Brazilian mortality statistics (SIM/DATASUS): most deaths are
# natural and occur in a hospital; among external causes, traffic accidents,
# falls (mostly elderly), homicides and suicides dominate. Place distributions
# follow the independent verifier's review (M1): traffic deaths are mostly
# hospital (after rescue) or via pública, essentially never at home; falls are
# mostly hospital or home; homicides are public-road or home; suicides are
# mostly home; natural deaths are rarely on a public road.
ADULT_SCENARIOS: tuple[Scenario, ...] = (
    Scenario("natural_hospital", "Morte natural em hospital", "Natural", 65, {"1": 90, "2": 10}),
    Scenario("natural_home", "Morte natural em domicílio", "Natural", 20, {"3": 92, "4": 8}),
    Scenario("external_traffic", "Causa externa — acidente de trânsito", "Não natural", 6,
             {"1": 60, "2": 5, "4": 35}),
    Scenario("external_fall", "Causa externa — queda", "Não natural", 3, {"1": 55, "2": 5, "3": 35, "4": 5}),
    Scenario("external_homicide", "Causa externa — agressão/homicídio", "Não natural", 3,
             {"4": 50, "3": 30, "1": 15, "2": 5}),
    Scenario("external_suicide", "Causa externa — lesão autoprovocada", "Não natural", 3,
             {"3": 75, "1": 15, "4": 10}),
)

# Phase 2: fetal (óbito fetal, field 1 = "1"), neonatal/infant (<1 year, age expressed in
# minutes/hours/days/months) and child/adolescent (1-17) scenarios. Weights are relative to
# ADULT_SCENARIOS' sum of 100: Brazilian SIM data puts fetal deaths (stillbirths) and infant
# deaths each at a few percent of all deaths recorded, and child/adolescent deaths lower still
# (infant mortality itself has fallen under ~15/1000 live births, and under-1 and fetal deaths
# are a small slice of total mortality, which is dominated by the elderly) — so these scenarios
# get small weights added on top of, not carved out of, the adult total.
FETAL_SCENARIOS: tuple[Scenario, ...] = (
    Scenario("fetal_death", "Óbito fetal", "Natural", 2.2, {"1": 90, "2": 5, "3": 4, "4": 1}, age_group="fetal"),
)
INFANT_SCENARIOS: tuple[Scenario, ...] = (
    Scenario("infant_death", "Óbito de menor de 1 ano", "Natural", 2.6, {"1": 85, "2": 9, "3": 5, "4": 1},
             age_group="infant"),
)
# Natural and external deaths of children/adolescents (1-17): rare in absolute terms, external
# causes (drowning, traffic) now outweigh natural disease deaths in this age band in Brazil.
CHILD_SCENARIOS: tuple[Scenario, ...] = (
    Scenario("child_natural", "Morte natural de criança/adolescente", "Natural", 0.4,
             {"1": 80, "2": 10, "3": 10}, age_group="child"),
    Scenario("child_external", "Causa externa de criança/adolescente", "Não natural", 0.5,
             {"1": 45, "4": 35, "3": 20}, age_group="child"),
)

ALL_SCENARIOS: tuple[Scenario, ...] = ADULT_SCENARIOS + FETAL_SCENARIOS + INFANT_SCENARIOS + CHILD_SCENARIOS

# Declared without physician (block IX — a leigo/non-medical declarant, e.g. a remote area
# without a certifying doctor) is still not implemented; everything else phase 1 documented
# here (infant/fetal/child scenarios, their field groups and cause chains) now is.
NOT_IMPLEMENTED: tuple[str, ...] = ("declared_without_physician",)

SCENARIOS_BY_ID = {s.scenario_id: s for s in ALL_SCENARIOS}


def pick_scenario(rng) -> Scenario:
    return rng.choices(ALL_SCENARIOS, weights=[s.weight for s in ALL_SCENARIOS])[0]


def pick_place(rng, scenario: Scenario, allowed_categories: tuple[str, ...] | None = None) -> str:
    """Pick a field-20 place category consistent with the scenario.

    `allowed_categories` narrows the scenario's own distribution (e.g. to the
    categories for which at least one eligible chain exists); if that
    intersection would be empty the scenario's own distribution is used
    unfiltered (every scenario has at least one "any"-setting chain, so this
    never actually triggers, but it keeps this function total).
    """
    categories = list(scenario.place_weights)
    if allowed_categories is not None:
        narrowed = [c for c in categories if c in allowed_categories]
        if narrowed:
            categories = narrowed
    weights = [scenario.place_weights[c] for c in categories]
    return rng.choices(categories, weights)[0]


def applicable_field_groups(scenario: Scenario, hospital: bool) -> tuple[str, ...]:
    # "core" is coarse-grained (it bundles "9_anos" with unrelated fields); fetal/infant drop
    # the main age-in-years field specifically — see FIELD_GROUPS["infant_under_1"] and the
    # generator, which blanks "9_anos" directly for both age groups (R: exactly one age unit).
    groups = ["core", "cause_chain"]
    groups.append("hospital_occurrence" if hospital else "home_or_public_occurrence")
    if scenario.manner == "Não natural":
        groups.append("external_cause")
    if scenario.age_group == "infant":
        groups.append("infant_under_1")
    if scenario.age_group in ("infant", "fetal"):
        groups.append("maternal_history")
    return tuple(groups)
