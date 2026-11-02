"""Curated library of coherent cause-of-death chains for the DO (Declaração de Óbito).

Phase-1-v2 (post-verification fixes): a chain now declares exactly which
scenario(s) it may be used for (`scenario_ids`) and whether it needs a
hospital (`setting`). The generator picks the scenario FIRST, then a chain
whose `scenario_ids` contains that scenario (never crossing manner or
mechanism), then draws age/sex from the chain's own range — so age/sex gates
and scenario/chain/field-48 coherence are true by construction, not by a
fallback.

Each chain models the real Part I / Part II structure of the Brazilian DO:
- Part I has up to four lines (a -> b -> c -> d), read bottom-up: ``d`` is the
  underlying ("causa de base") condition, ``a`` is the terminal ("causa
  imediata") event. Chains may use fewer than four lines.
- Part II (0-2 lines) lists *contributing* conditions, drawn separately (see
  `PART2_POOL`) so they never duplicate a Part I diagnosis.
- Each line has 1-3 text variants (for OCR variety), a real ICD-10 code (for
  V/W/X/Y external-cause codes, the 4th "place of occurrence" character is
  appended dynamically by the generator from the DO's own field 20, so the
  table stores the 3-character base code for those), and an interval range;
  intervals are sampled so they stay non-decreasing from ``a`` (shortest) to
  ``d`` (longest) — lines sharing a `group` id share one sampled draw so
  near-simultaneous events (e.g. a gunshot wound and the assault that caused
  it) never invert.

CID-10 codes were corrected against the independent verifier's review:
I60.7 (ruptured SAH aneurysm, not I67.1 unruptured), I11.0 (hypertensive
heart disease WITH failure, not I11.9), I50.1 (cardiogenic pulmonary edema;
J81 excludes it), W01 (fall on the same level; W19 is "unspecified fall"),
C78.0 (lung metastasis, split from the primary site onto its own line),
A41.9 for intra-abdominal sepsis, and the broncopneumonia_alzheimer order was
reversed (Alzheimer's causes dementia, not the other way around; the
redundant "dementia" line was dropped).
"""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field

UNIT_RANK = {"minutos": 0, "horas": 1, "dias": 2, "semanas": 3, "meses": 4, "anos": 5}
UNIT_SINGULAR = {"minutos": "minuto", "horas": "hora", "dias": "dia", "semanas": "semana", "meses": "mês", "anos": "ano"}


def format_interval(unit: str, n: int) -> str:
    word = UNIT_SINGULAR[unit] if n == 1 else unit
    return f"{n} {word}"


@dataclass(frozen=True)
class CauseLine:
    text_variants: tuple[str, ...]
    cid: str  # for V/W/X/Y codes, a 3-char base code; the generator appends the place digit
    unit: str
    lo: int
    hi: int
    group: int = 0  # lines sharing a group id within a chain share one sampled interval draw


@dataclass(frozen=True)
class CauseChain:
    chain_id: str
    category: str
    manner: str  # "Natural" or "Não natural"
    external_type: str | None  # value of field 48 (Acidente/Suicídio/Homicídio) when manner != Natural
    scenario_ids: tuple[str, ...]  # which death_scenarios.ADULT_SCENARIOS ids may draw this chain
    setting: str  # "hospital" | "any" — "hospital" chains require death.place.category in {1,2}
    min_age: int
    max_age: int
    sex: str | None  # "M", "F" or None (either)
    weight: float
    part1: tuple[CauseLine, ...]  # ordered a, b, c, d (a first / most immediate)
    # "chronic" (long-diagnosed, followed disease: neoplasms, Alzheimer, DPOC, cirrhosis,
    # nephropathy, AVC sequela, ICC) vs "sudden" (IAM, acute AVC, HSA, EAP...) — drives whether a
    # home death plausibly had medical assistance (R2). Defaults to "sudden"; set explicitly below
    # for chronic chains via a trailing keyword argument.
    course: str = "sudden"


def _c(texts: tuple[str, ...] | str, cid: str, unit: str, lo: int, hi: int, group: int = 0) -> CauseLine:
    variants = (texts,) if isinstance(texts, str) else texts
    return CauseLine(variants, cid, unit, lo, hi, group)


CAUSE_CHAINS: tuple[CauseChain, ...] = (
    # ---- circulatory (natural; both hospital and home) ----
    CauseChain("iam_aterosclerose", "circulatory", "Natural", None, ("natural_hospital", "natural_home"), "any",
               40, 95, None, 10,
               (_c(("Infarto agudo do miocárdio", "Infarto agudo do miocárdio extenso"), "I21.9", "horas", 1, 6),
                _c(("Aterosclerose coronariana", "Doença aterosclerótica coronariana"), "I25.1", "anos", 3, 12),
                _c("Hipertensão arterial sistêmica", "I10", "anos", 10, 25))),
    CauseChain("avc_isquemico", "circulatory", "Natural", None, ("natural_hospital", "natural_home"), "any",
               50, 98, None, 9,
               (_c(("Acidente vascular cerebral isquêmico", "AVC isquêmico extenso"), "I63.9", "dias", 1, 5),
                _c("Fibrilação atrial crônica", "I48", "anos", 2, 9),
                _c("Hipertensão arterial sistêmica", "I10", "anos", 8, 20))),
    CauseChain("choque_cardiogenico", "circulatory", "Natural", None, ("natural_hospital",), "hospital",
               45, 95, None, 6,
               (_c("Choque cardiogênico", "R57.0", "horas", 2, 10),
                _c("Infarto agudo do miocárdio", "I21.9", "horas", 6, 30),
                _c("Doença arterial coronariana", "I25.9", "anos", 4, 14))),
    CauseChain("icc_hipertensiva", "circulatory", "Natural", None, ("natural_hospital", "natural_home"), "any",
               55, 99, None, 8,
               (_c("Insuficiência cardíaca congestiva descompensada", "I50.0", "dias", 2, 8),
                _c("Cardiopatia hipertensiva com insuficiência cardíaca", "I11.0", "anos", 5, 15)),
               course="chronic"),
    CauseChain("tep_tvp", "circulatory", "Natural", None, ("natural_hospital",), "hospital",
               35, 90, None, 4,
               (_c(("Tromboembolismo pulmonar agudo", "Tromboembolismo pulmonar maciço"), "I26.9", "dias", 1, 4),
                _c("Trombose venosa profunda de membro inferior", "I80.2", "dias", 5, 20))),
    CauseChain("eap_cardiomiopatia", "circulatory", "Natural", None, ("natural_hospital", "natural_home"), "any",
               40, 90, None, 4,
               (_c("Edema agudo de pulmão cardiogênico", "I50.1", "horas", 1, 6),
                _c("Insuficiência cardíaca congestiva", "I50.0", "anos", 1, 4),
                _c("Cardiomiopatia dilatada", "I42.0", "anos", 3, 9)),
               course="chronic"),
    CauseChain("hsa_aneurisma", "circulatory", "Natural", None, ("natural_hospital", "natural_home"), "any",
               30, 80, None, 3,
               (_c(("Hemorragia subaracnóidea por ruptura de aneurisma cerebral", "Hemorragia subaracnóidea aneurismática"),
                   "I60.7", "horas", 2, 24),
                _c("Hipertensão arterial sistêmica", "I10", "anos", 3, 10))),
    # ---- respiratory (natural; both hospital and home) ----
    CauseChain("pneumonia_dpoc", "respiratory", "Natural", None, ("natural_hospital", "natural_home"), "any",
               55, 97, None, 8,
               (_c(("Pneumonia bacteriana", "Pneumonia bacteriana extensa"), "J15.9", "dias", 3, 10),
                _c("Doença pulmonar obstrutiva crônica", "J44.9", "anos", 6, 18),
                _c("Tabagismo crônico", "F17.2", "anos", 20, 40)),
               course="chronic"),
    CauseChain("pneumonia_aspiracao", "respiratory", "Natural", None, ("natural_hospital", "natural_home"), "any",
               60, 100, None, 6,
               (_c("Insuficiência respiratória aguda", "J96.0", "dias", 1, 4),
                _c("Pneumonia por aspiração", "J69.0", "dias", 3, 10),
                _c("Sequela de acidente vascular cerebral", "I69.3", "anos", 1, 5)),
               course="chronic"),
    CauseChain("fibrose_pulmonar", "respiratory", "Natural", None, ("natural_hospital", "natural_home"), "any",
               55, 90, None, 3,
               (_c("Insuficiência respiratória crônica agudizada", "J96.1", "dias", 2, 8),
                _c("Fibrose pulmonar idiopática", "J84.1", "anos", 2, 7)),
               course="chronic"),
    CauseChain("broncopneumonia_alzheimer", "respiratory", "Natural", None, ("natural_hospital", "natural_home"), "any",
               70, 102, None, 7,
               (_c("Broncopneumonia", "J18.0", "dias", 2, 8),
                _c(("Doença de Alzheimer", "Doença de Alzheimer avançada"), "G30.9", "anos", 4, 10)),
               course="chronic"),
    # ---- infections / sepsis ----
    CauseChain("sepse_pulmonar", "infections", "Natural", None, ("natural_hospital",), "hospital",
               55, 95, None, 5,
               (_c("Choque séptico", "A41.9", "horas", 6, 24),
                _c("Pneumonia bacteriana", "J15.9", "dias", 2, 7),
                _c("Doença pulmonar obstrutiva crônica", "J44.9", "anos", 4, 12))),
    CauseChain("sepse_urinaria", "infections", "Natural", None, ("natural_hospital", "natural_home"), "any",
               65, 97, "M", 4,
               (_c("Sepse de foco urinário", "A41.9", "dias", 1, 3),
                _c("Pielonefrite aguda", "N10", "dias", 3, 10),
                _c("Hiperplasia prostática benigna", "N40", "anos", 2, 8)),
               course="chronic"),
    CauseChain("peritonite_apendicite", "infections", "Natural", None, ("natural_hospital",), "hospital",
               18, 70, None, 2,
               (_c("Sepse de foco abdominal", "A41.9", "horas", 6, 20),
                _c("Peritonite generalizada", "K65.0", "dias", 1, 3),
                _c("Apendicite aguda perfurada", "K35.2", "dias", 2, 5))),
    # ---- neoplasms ----
    CauseChain("neoplasia_pulmao", "neoplasms", "Natural", None, ("natural_hospital", "natural_home"), "any",
               45, 92, None, 6,
               (_c("Caquexia neoplásica", "R64", "semanas", 2, 8),
                _c(("Neoplasia maligna de pulmão", "Neoplasia maligna de pulmão, não especificada"), "C34.9", "meses", 6, 24),
                _c("Tabagismo crônico", "F17.2", "anos", 15, 35)),
               course="chronic"),
    CauseChain("neoplasia_colorretal", "neoplasms", "Natural", None, ("natural_hospital", "natural_home"), "any",
               45, 95, None, 4,
               (_c("Caquexia neoplásica", "R64", "semanas", 3, 10),
                _c("Neoplasia maligna de cólon", "C18.9", "meses", 8, 30)),
               course="chronic"),
    CauseChain("neoplasia_mama", "neoplasms", "Natural", None, ("natural_hospital", "natural_home"), "any",
               40, 90, "F", 4,
               (_c("Insuficiência respiratória aguda", "J96.0", "dias", 2, 7),
                _c(("Metástases pulmonares de neoplasia de mama", "Metástase pulmonar secundária"), "C78.0", "meses", 3, 12),
                _c("Neoplasia maligna de mama", "C50.9", "anos", 1, 5)),
               course="chronic"),
    CauseChain("neoplasia_prostata", "neoplasms", "Natural", None, ("natural_hospital", "natural_home"), "any",
               55, 95, "M", 3,
               (_c("Insuficiência renal aguda", "N17.9", "dias", 2, 7),
                _c("Uropatia obstrutiva", "N13.9", "meses", 2, 9),
                _c("Neoplasia maligna de próstata", "C61", "anos", 1, 6)),
               course="chronic"),
    # ---- digestive / hepatic ----
    CauseChain("cirrose_viral", "digestive", "Natural", None, ("natural_hospital", "natural_home"), "any",
               40, 85, None, 3,
               (_c("Insuficiência hepática", "K72.9", "dias", 2, 6),
                _c("Cirrose hepática", "K74.6", "anos", 3, 9),
                _c("Hepatite C crônica", "B18.2", "anos", 8, 20)),
               course="chronic"),
    CauseChain("cirrose_alcoolica", "digestive", "Natural", None, ("natural_hospital", "natural_home"), "any",
               40, 80, None, 3,
               (_c("Insuficiência hepática", "K72.9", "dias", 1, 5),
                _c("Cirrose hepática alcoólica", "K70.3", "anos", 4, 12),
                _c("Etilismo crônico", "F10.2", "anos", 10, 25)),
               course="chronic"),
    CauseChain("hda_ulcera", "digestive", "Natural", None, ("natural_hospital",), "hospital",
               40, 90, None, 4,
               (_c("Choque hipovolêmico", "R57.1", "horas", 1, 8),
                _c("Hemorragia digestiva alta", "K92.2", "horas", 6, 24),
                _c("Úlcera gástrica crônica", "K25.4", "anos", 1, 6))),
    # ---- diabetes / renal ----
    CauseChain("coma_hiperosmolar", "diabetes", "Natural", None, ("natural_hospital",), "hospital",
               50, 92, None, 4,
               (_c("Coma hiperosmolar", "E11.0", "horas", 6, 24),
                _c("Diabetes mellitus tipo 2", "E11.9", "anos", 5, 15))),
    CauseChain("nefropatia_diabetica", "diabetes", "Natural", None, ("natural_hospital", "natural_home"), "any",
               55, 93, None, 4,
               (_c("Insuficiência renal crônica agudizada", "N18.9", "dias", 3, 12),
                _c("Nefropatia diabética", "E11.2", "anos", 4, 12)),
               course="chronic"),
    # ---- external: traffic (mostly hospital after rescue, or via pública; never home) ----
    CauseChain("trauma_moto", "external", "Não natural", "Acidente", ("external_traffic",), "any",
               18, 60, None, 6,
               (_c("Traumatismo cranioencefálico grave", "S06.9", "horas", 1, 4, group=1),
                _c(("Acidente de trânsito com motocicleta", "Colisão de motocicleta"), "V29.4", "horas", 1, 4, group=1))),
    CauseChain("politrauma_auto", "external", "Não natural", "Acidente", ("external_traffic",), "any",
               18, 70, None, 5,
               (_c("Choque hemorrágico traumático", "T79.4", "horas", 1, 3, group=1),
                _c("Politraumatismo", "T07", "horas", 1, 3, group=1),
                _c(("Acidente de trânsito com automóvel", "Colisão automobilística"), "V49.4", "horas", 1, 3, group=1))),
    # ---- external: fall (mostly hospital, elderly; sometimes home) ----
    CauseChain("tce_queda_idoso", "external", "Não natural", "Acidente", ("external_fall",), "any",
               65, 102, None, 5,
               (_c("Traumatismo cranioencefálico", "S06.9", "dias", 1, 4, group=1),
                _c(("Queda da própria altura", "Queda ao nível do solo"), "W01", "dias", 1, 4, group=1))),
    CauseChain("fratura_femur_queda", "external", "Não natural", "Acidente", ("external_fall",), "hospital",
               70, 100, None, 4,
               (_c("Tromboembolismo pulmonar pós-operatório", "I26.9", "dias", 5, 14, group=1),
                _c("Fratura do colo do fêmur", "S72.0", "dias", 5, 14, group=1),
                _c("Queda da própria altura", "W01", "dias", 5, 14, group=1))),
    # ---- external: homicide (public road or home) ----
    CauseChain("paf_agressao", "external", "Não natural", "Homicídio", ("external_homicide",), "any",
               18, 45, None, 3,
               (_c("Choque hemorrágico traumático", "T79.4", "minutos", 15, 60, group=1),
                _c(("Ferimento por projétil de arma de fogo no tórax", "Ferimento por projétil de arma de fogo no abdome"),
                   "S21.9", "minutos", 15, 60, group=1),
                _c("Agressão por disparo de arma de fogo", "X95", "minutos", 15, 60, group=1))),
    CauseChain("arma_branca_agressao", "external", "Não natural", "Homicídio", ("external_homicide",), "any",
               18, 50, None, 2,
               (_c("Perfuração de órgãos internos", "S36.9", "minutos", 20, 90, group=1),
                _c("Ferimento por arma branca", "S31.8", "minutos", 20, 90, group=1),
                _c("Agressão por objeto cortante ou penetrante", "X99", "minutos", 20, 90, group=1))),
    # ---- external: suicide (mostly home) ----
    CauseChain("intoxicacao_suicidio", "external", "Não natural", "Suicídio", ("external_suicide",), "any",
               18, 60, None, 2,
               (_c("Insuficiência respiratória aguda", "J96.0", "horas", 2, 10, group=1),
                _c("Intoxicação exógena por medicamentos", "T50.9", "horas", 2, 10, group=1),
                _c("Lesão autoprovocada intencionalmente por envenenamento", "X64", "horas", 2, 10, group=1))),
    CauseChain("enforcamento_suicidio", "external", "Não natural", "Suicídio", ("external_suicide",), "any",
               18, 70, None, 2,
               (_c("Asfixia mecânica", "T71", "minutos", 5, 30, group=1),
                _c("Enforcamento", "X70", "minutos", 5, 30, group=1))),
)

# Part II ("contributing conditions"): a shared pool, filtered per record so it
# never repeats a Part I code/text. Picked 0-2 at a time by the generator,
# weighted so most natural-disease records get at least one plausible
# comorbidity and external/traumatic records get none.
PART2_POOL: tuple[CauseLine, ...] = (
    _c("Diabetes mellitus tipo 2", "E11.9", "anos", 1, 1),
    _c("Hipertensão arterial sistêmica", "I10", "anos", 1, 1),
    _c("Doença pulmonar obstrutiva crônica", "J44.9", "anos", 1, 1),
    _c("Obesidade", "E66.9", "anos", 1, 1),
    _c("Tabagismo crônico", "F17.2", "anos", 1, 1),
    _c("Doença renal crônica", "N18.9", "anos", 1, 1),
    _c("Etilismo crônico", "F10.2", "anos", 1, 1),
)


def chains_for_scenario(scenario_id: str) -> tuple[CauseChain, ...]:
    """All chains valid for a scenario, regardless of setting — the caller (death()) picks the
    chain first and then forces a hospital place for "hospital"-setting chains."""
    return tuple(chain for chain in CAUSE_CHAINS if scenario_id in chain.scenario_ids)


def eligible_chains_for_scenario(scenario_id: str, hospital: bool) -> tuple[CauseChain, ...]:
    return tuple(chain for chain in CAUSE_CHAINS
                 if scenario_id in chain.scenario_ids and (chain.setting != "hospital" or hospital))
