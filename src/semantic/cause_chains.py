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

import math
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
    # A condition present from birth (prematurity, congenital heart disease, birth asphyxia —
    # B2): its interval is never sampled, it is forced equal to the deceased's age (in the
    # SAME unit as this line — only set True on lines whose `unit` equals the chain's own
    # `age_unit`, so the conversion is exact).
    since_birth: bool = False


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
    # Unit `min_age`/`max_age` are expressed in: "anos" (default, adult/child chains), "meses"
    # or "dias" (infant chains — exactly one DO age-unit field gets filled from this), or "fetal"
    # (óbito fetal: no age field at all, min/max unused). Phase-2 (see final report).
    age_unit: str = "anos"
    # Optional per-age relative-likelihood weights, one per integer in range(min_age, max_age+1)
    # — overrides the uniform draw `age_and_birth_date_in_range` otherwise uses. Added for the
    # verifier-flagged adult chains whose plausible age distribution is not flat (falls skew
    # elderly but taper off well before 100+; traffic deaths skew towards young adults, with a
    # smaller mid-age bump) — see FALL_AGE_WEIGHTS / TRAFFIC_AGE_WEIGHTS below.
    age_weights: tuple[float, ...] | None = None
    # Place-of-occurrence (DO field 20) weights specific to this chain (M4) — e.g. a pedestrian
    # death is never plausible "at home", and a drowning is rarely on a public road. When set,
    # this REPLACES the scenario's own place_weights for records drawing this chain; when None
    # (the default — every phase-1 chain), the scenario's distribution is used unchanged.
    place_weights: dict[str, float] | None = None
    # m-b: a realistic M/F split for a chain whose `sex` is None (either) but not actually
    # 50/50 in practice (drowning, motorcycle and pedestrian deaths in Brazil are ~75-85%
    # male). None keeps the old flat 52/48 draw; only set where SIM data says otherwise.
    sex_weights: tuple[float, float] | None = None  # (M, F)
    # Fetal-only (B3): ties field 34 ("a morte ocorreu: Antes/Durante o parto") to the chain, so
    # P20.0 (hypoxia first noted BEFORE labour) only pairs with "Antes" and P20.1/other
    # intraparto codes only pair with "Durante" — never both in the same chain, never
    # contradicting field 34. None for every non-fetal chain.
    fetal_timing: str | None = None
    # Infant SDR/prematurity chains only (B4): restricts the gestational-age draw to this
    # chain's own plausible weeks so the prematurity ICD-10 code (P07.2 vs P07.3) always
    # matches the weeks actually drawn, and an "unknown gestation" draw never gets a term
    # weight for a prematurity chain (deceased_birth_block forces weeks_unknown=False here).
    gestation_weeks_range: tuple[int, int] | None = None


def _c(texts: tuple[str, ...] | str, cid: str, unit: str, lo: int, hi: int, group: int = 0,
       since_birth: bool = False) -> CauseLine:
    variants = (texts,) if isinstance(texts, str) else texts
    return CauseLine(variants, cid, unit, lo, hi, group, since_birth)


def _gauss_weights(lo: int, hi: int, *peaks: tuple[float, float, float]) -> tuple[float, ...]:
    """Sum of Gaussian bumps (center, spread, amplitude) over range(lo, hi+1), floored above 0 —
    used to replace a flat age draw with a realistic shape (R6: falls skew elderly but taper
    off well before 100+; traffic skews to young adults with a smaller mid-age bump)."""
    return tuple(sum(amp * math.exp(-((age - center) / spread) ** 2) for center, spread, amp in peaks) + 0.01
                 for age in range(lo, hi + 1))


FALL_AGE_WEIGHTS = _gauss_weights(65, 102, (76, 14, 1.0))  # peaks ~76, < 2% left at 100+
FRATURA_FEMUR_AGE_WEIGHTS = _gauss_weights(70, 100, (80, 12, 1.0))
# m-c: widened so ~15-18% of motorcyclist deaths land at 50+ (SIM), not ~1% — still centered
# on young adults, but with a visibly heavier tail than the original narrow spread=9.
TRAFFIC_MOTO_AGE_WEIGHTS = _gauss_weights(18, 60, (26, 9, 1.0), (50, 10, 0.38))
TRAFFIC_AUTO_AGE_WEIGHTS = _gauss_weights(18, 70, (32, 14, 1.0), (55, 10, 0.4))  # drivers: wider + a mid-age bump
# B5: broncopneumonia_alzheimer was flagged as "not touched" by the per-chain age weights —
# Alzheimer deaths peak in the 80s and taper sharply (should be rare past 100), same shape
# family as the fall weights above.
ALZHEIMER_AGE_WEIGHTS = _gauss_weights(70, 102, (82, 10, 1.0))
# m4: 15-17 is dominated by homicide/motorcycle in Brazil, not pedestrian-only (added below).
ADOLESCENTE_PAF_AGE_WEIGHTS = _gauss_weights(15, 17, (17, 2, 1.0))

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
               course="chronic", age_weights=ALZHEIMER_AGE_WEIGHTS),
    # ---- infections / sepsis ----
    # Re-verification (medical review): "Choque séptico" is coded R57.2 for consistency with the
    # infant/child sepsis chains; A41.9 (the underlying bacterial sepsis) is kept as line d.
    CauseChain("sepse_pulmonar", "infections", "Natural", None, ("natural_hospital",), "hospital",
               55, 95, None, 5,
               (_c("Choque séptico", "R57.2", "horas", 6, 24),
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
                _c(("Acidente de trânsito com motocicleta", "Colisão de motocicleta"), "V29.4", "horas", 1, 4, group=1)),
               age_weights=TRAFFIC_MOTO_AGE_WEIGHTS),
    CauseChain("politrauma_auto", "external", "Não natural", "Acidente", ("external_traffic",), "any",
               18, 70, None, 5,
               (_c("Choque hemorrágico traumático", "T79.4", "horas", 1, 3, group=1),
                _c("Politraumatismo", "T07", "horas", 1, 3, group=1),
                _c(("Acidente de trânsito com automóvel", "Colisão automobilística"), "V49.4", "horas", 1, 3, group=1)),
               age_weights=TRAFFIC_AUTO_AGE_WEIGHTS),
    # ---- external: fall (mostly hospital, elderly; sometimes home) ----
    CauseChain("tce_queda_idoso", "external", "Não natural", "Acidente", ("external_fall",), "any",
               65, 102, None, 5,
               (_c("Traumatismo cranioencefálico", "S06.9", "dias", 1, 4, group=1),
                _c(("Queda da própria altura", "Queda ao nível do solo"), "W01", "dias", 1, 4, group=1)),
               age_weights=FALL_AGE_WEIGHTS),
    CauseChain("fratura_femur_queda", "external", "Não natural", "Acidente", ("external_fall",), "hospital",
               70, 100, None, 4,
               (_c("Tromboembolismo pulmonar pós-operatório", "I26.9", "dias", 5, 14, group=1),
                _c("Fratura do colo do fêmur", "S72.0", "dias", 5, 14, group=1),
                _c("Queda da própria altura", "W01", "dias", 5, 14, group=1)),
               age_weights=FRATURA_FEMUR_AGE_WEIGHTS),
    # ---- external: homicide (public road or home) ----
    # Re-verification: the "no abdome" text variant is dropped — S21.9 is thorax-only (abdomen
    # would be S31.x), and CauseLine ties every text variant of a line to one shared CID.
    CauseChain("paf_agressao", "external", "Não natural", "Homicídio", ("external_homicide",), "any",
               18, 45, None, 3,
               (_c("Choque hemorrágico traumático", "T79.4", "minutos", 15, 60, group=1),
                _c("Ferimento por projétil de arma de fogo no tórax", "S21.9", "minutos", 15, 60, group=1),
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
    # ---- fetal (óbito fetal — field 1 = "1"; no age field at all) ----
    # B3: field 34 ("a morte ocorreu: Antes/Durante o parto") is tied to each chain via
    # `fetal_timing`, and P20.0 (hypoxia first noted BEFORE labour) vs P20.1/other intraparto
    # codes (DURING labour) are never mixed in one chain. deceased_birth_block() draws field 34
    # FROM the chain (fetal_timing), never independently.
    # Re-verification (medical review): "Trabalho de parto prolongado" is P03.8; "Distócia" is
    # better coded P03.1 (malpresentation/disproportion) — a different CID, so each text now has
    # its own chain (same timing/scenario) instead of sharing one line's single CID.
    CauseChain("fetal_hipoxia_intraparto", "perinatal", "Natural", None, ("fetal_death",), "any",
               0, 0, None, 3,
               (_c("Sofrimento fetal intraparto", "P20.1", "horas", 1, 12, group=1),
                _c("Trabalho de parto prolongado", "P03.8", "horas", 1, 12, group=1)),
               age_unit="fetal", fetal_timing="Durante"),
    CauseChain("fetal_hipoxia_distocia", "perinatal", "Natural", None, ("fetal_death",), "any",
               0, 0, None, 1,
               (_c("Hipóxia intrauterina intraparto", "P20.1", "horas", 1, 12, group=1),
                _c("Distócia de trabalho de parto", "P03.1", "horas", 1, 12, group=1)),
               age_unit="fetal", fetal_timing="Durante"),
    CauseChain("fetal_dpp", "perinatal", "Natural", None, ("fetal_death",), "any",
               0, 0, None, 3,
               (_c("Hipóxia intrauterina aguda", "P20.0", "horas", 1, 6, group=1),
                _c("Descolamento prematuro de placenta", "P02.1", "horas", 1, 6, group=1)),
               age_unit="fetal", fetal_timing="Antes"),
    CauseChain("fetal_corioamnionite", "perinatal", "Natural", None, ("fetal_death",), "any",
               0, 0, None, 2,
               (_c(("Infecção intrauterina", "Sepse intrauterina"), "P39.2", "dias", 1, 3, group=1),
                _c("Corioamnionite materna", "P02.7", "dias", 2, 7, group=1)),
               age_unit="fetal", fetal_timing="Antes"),
    CauseChain("fetal_insuficiencia_placentaria", "perinatal", "Natural", None, ("fetal_death",), "any",
               0, 0, None, 2,
               (_c("Hipóxia intrauterina crônica", "P20.0", "dias", 1, 10, group=1),
                _c("Insuficiência placentária", "P02.2", "semanas", 1, 4, group=1)),
               age_unit="fetal", fetal_timing="Antes"),
    # ---- infant / neonatal (<1 year — age_unit "dias" or "meses" drive which field 9 unit is
    # filled; min/max are the age IN THAT UNIT, not years) ----
    # B4: the single old "infant_sdr_prematuridade" chain always said "Prematuridade extrema"
    # (P07.2) regardless of gestation; split by weeks so the code always matches: P07.2 only
    # < 28 weeks, P07.3 for 28-36 weeks. `gestation_weeks_range` forces deceased_birth_block()
    # to draw gestation inside that band (never "unknown" with a term weight — B4).
    CauseChain("infant_sdr_prematuridade_extrema", "perinatal", "Natural", None, ("infant_death",), "hospital",
               1, 10, None, 2,
               (_c(("Síndrome do desconforto respiratório do recém-nascido", "Membrana hialina pulmonar"),
                   "P22.0", "horas", 2, 20, group=1),
                _c("Prematuridade extrema", "P07.2", "dias", 1, 10, group=1, since_birth=True)),
               age_unit="dias", gestation_weeks_range=(24, 27)),
    CauseChain("infant_sdr_prematuridade_moderada", "perinatal", "Natural", None, ("infant_death",), "hospital",
               1, 10, None, 4,
               (_c(("Síndrome do desconforto respiratório do recém-nascido", "Membrana hialina pulmonar"),
                   "P22.0", "horas", 2, 20, group=1),
                _c("Prematuridade", "P07.3", "dias", 1, 10, group=1, since_birth=True)),
               age_unit="dias", gestation_weeks_range=(28, 36)),
    # B2: P21.1 is mild/moderate birth asphyxia; a fatal case needs P21.0 (severe). B3: the
    # infant-side partner of P20.1 (intraparto) is this same chain, never mixed with P20.0.
    CauseChain("infant_asfixia_perinatal", "perinatal", "Natural", None, ("infant_death",), "hospital",
               1, 23, None, 4,
               (_c("Asfixia grave ao nascer", "P21.0", "horas", 1, 23, group=1, since_birth=True),
                _c("Sofrimento fetal intraparto", "P20.1", "horas", 1, 23, group=1, since_birth=True)),
               age_unit="horas"),
    CauseChain("infant_sepse_neonatal", "infections", "Natural", None, ("infant_death",), "hospital",
               1, 27, None, 4,
               (_c("Choque séptico", "R57.2", "horas", 6, 48, group=1),
                _c("Sepse bacteriana do recém-nascido", "P36.9", "dias", 1, 5, group=1)),
               age_unit="dias"),
    CauseChain("infant_cardiopatia_congenita", "congenital", "Natural", None, ("infant_death",), "any",
               1, 11, None, 3,
               (_c("Insuficiência cardíaca congestiva", "I50.0", "dias", 1, 7, group=1),
                _c(("Cardiopatia congênita complexa", "Malformação cardíaca congênita"), "Q24.9", "meses", 1, 11,
                   group=1, since_birth=True)),
               age_unit="meses"),
    CauseChain("infant_pneumonia", "respiratory", "Natural", None, ("infant_death",), "any",
               1, 11, None, 3,
               (_c("Insuficiência respiratória aguda", "J96.0", "dias", 1, 5, group=1),
                _c("Pneumonia", "J18.9", "dias", 3, 10, group=1)),
               age_unit="meses"),
    CauseChain("infant_diarreia_aguda", "infections", "Natural", None, ("infant_death",), "any",
               1, 11, None, 2,
               (_c("Choque hipovolêmico", "R57.1", "horas", 6, 24, group=1),
                _c(("Diarreia e gastroenterite de origem infecciosa presumível", "Gastroenterite aguda"),
                   "A09", "dias", 2, 6, group=1)),
               age_unit="meses"),
    # ---- child / adolescent (1-17, natural and external) ----
    CauseChain("crianca_pneumonia", "respiratory", "Natural", None, ("child_natural",), "any",
               1, 9, None, 3,
               (_c("Insuficiência respiratória aguda", "J96.0", "dias", 1, 4),
                _c("Pneumonia", "J18.9", "dias", 2, 8))),
    # M3: "Choque séptico" is coded R57.2 everywhere it is used (same convention as the infant
    # chain above); A39.4 (meningococcemia, unspecified) is never used as a label for "Choque
    # séptico" — line b carries the actual meningococcemia diagnosis, A39.2.
    CauseChain("crianca_meningococcemia", "infections", "Natural", None, ("child_natural",), "hospital",
               1, 14, None, 2,
               (_c("Choque séptico", "R57.2", "horas", 4, 24, group=1),
                _c(("Meningococcemia aguda", "Sepse meningocócica"), "A39.2", "dias", 1, 3, group=1))),
    # M3/M4: line a is the actual immediate condition (never a mode-of-dying like "parada
    # cardiorrespiratória"); M4: drowning is overwhelmingly at home (pool) or a natural body of
    # water, essentially never on a public road — `place_weights` overrides the scenario's own
    # distribution for this chain. Field 51 gets a drowning-specific description (see
    # CHILD_EXTERNAL_DESCRIPTIONS in generators.py).
    CauseChain("crianca_afogamento", "external", "Não natural", "Acidente", ("child_external",), "any",
               1, 14, None, 4,
               (_c(("Asfixia por submersão", "Insuficiência respiratória aguda por submersão"), "T75.1", "minutos", 5, 30, group=1),
                _c("Afogamento e submersão acidental", "W74", "minutos", 5, 30, group=1)),
               place_weights={"3": 55, "1": 40, "4": 5}, sex_weights=(78, 22)),
    # M4: a pedestrian struck by a vehicle is never "at home" — place_weights excludes
    # domicílio entirely (category "3"), unlike the scenario's own generic distribution.
    CauseChain("crianca_atropelamento", "external", "Não natural", "Acidente", ("child_external",), "any",
               5, 17, None, 4,
               (_c("Traumatismo cranioencefálico grave", "S06.9", "horas", 1, 6, group=1),
                _c(("Atropelamento por veículo a motor", "Acidente de trânsito, pedestre"), "V09.3", "horas", 1, 6, group=1)),
               place_weights={"4": 55, "1": 45}, sex_weights=(68, 32)),
    # m4: 15-17 is dominated in Brazil by homicide (firearm) and motorcycle deaths, not
    # pedestrian strikes alone.
    CauseChain("adolescente_paf", "external", "Não natural", "Homicídio", ("child_external",), "any",
               15, 17, "M", 3,
               (_c("Choque hemorrágico traumático", "T79.4", "minutos", 15, 60, group=1),
                _c("Ferimento por projétil de arma de fogo no tórax", "S21.9", "minutos", 15, 60, group=1),
                _c("Agressão por disparo de arma de fogo", "X95", "minutos", 15, 60, group=1)),
               place_weights={"4": 55, "3": 30, "1": 15}),
    CauseChain("adolescente_moto", "external", "Não natural", "Acidente", ("child_external",), "any",
               15, 17, None, 3,
               (_c("Traumatismo cranioencefálico grave", "S06.9", "horas", 1, 4, group=1),
                _c(("Acidente de trânsito com motocicleta", "Colisão de motocicleta"), "V29.4", "horas", 1, 4, group=1)),
               place_weights={"1": 60, "4": 40}, sex_weights=(85, 15)),
)

# Part II ("contributing conditions"): a shared pool, filtered per record so it
# never repeats a Part I code/text. Picked 0-2 at a time by the generator,
# weighted so most natural-disease records get at least one plausible
# comorbidity and external/traumatic records get none.
PART2_POOL: tuple[CauseLine, ...] = (
    # M6: realistic, plausible-duration ranges (sampled by the generator) instead of a
    # constant lo=hi=1 ("1 ano" on every record) — capped at the deceased's own age downstream.
    _c("Diabetes mellitus tipo 2", "E11.9", "anos", 2, 25),
    _c("Hipertensão arterial sistêmica", "I10", "anos", 2, 25),
    _c("Doença pulmonar obstrutiva crônica", "J44.9", "anos", 3, 20),
    _c("Obesidade", "E66.9", "anos", 2, 20),
    _c("Tabagismo crônico", "F17.2", "anos", 5, 30),
    _c("Doença renal crônica", "N18.9", "anos", 1, 10),
    _c("Etilismo crônico", "F10.2", "anos", 5, 25),
)

# R1: children must never draw from the adult-comorbidity pool above (an "Etilismo crônico"
# or "Hipertensão arterial" contributing condition in a 1-9 year old is medically absurd and
# was producing intervals longer than the child's own age). A small, age-plausible paediatric
# pool instead, per the verifier's own suggestion — each capped at the child's actual age by
# the generator exactly like the adult pool.
CHILD_PART2_POOL: tuple[CauseLine, ...] = (
    _c("Asma brônquica", "J45.9", "anos", 1, 8),
    _c("Paralisia cerebral", "G80.9", "anos", 1, 12),
    _c("Síndrome de Down", "Q90.9", "anos", 1, 14),
    _c("Desnutrição proteico-calórica", "E46", "anos", 1, 5),
    _c("Cardiopatia congênita", "Q24.9", "anos", 1, 14),
)


def chains_for_scenario(scenario_id: str) -> tuple[CauseChain, ...]:
    """All chains valid for a scenario, regardless of setting — the caller (death()) picks the
    chain first and then forces a hospital place for "hospital"-setting chains."""
    return tuple(chain for chain in CAUSE_CHAINS if scenario_id in chain.scenario_ids)


def eligible_chains_for_scenario(scenario_id: str, hospital: bool) -> tuple[CauseChain, ...]:
    return tuple(chain for chain in CAUSE_CHAINS
                 if scenario_id in chain.scenario_ids and (chain.setting != "hospital" or hospital))
