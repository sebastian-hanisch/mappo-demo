"""Defaults, Slider-Grenzen und Presets für die MAPPO-Demo (PPO mit zentralem Kritiker, CTDE).
Szenario- und IQL-Konstanten sind wortgleich aus marl-demo übernommen (dieselbe Umgebung, derselbe
IQL-Referenzlauf); neu ist der PPO-Block weiter unten."""

DEFAULT_N_JOBS = 8
DEFAULT_N_AGENTS = 3
DEFAULT_DURATION_VARIABILITY = 0.3
DEFAULT_TRAVEL_TIME_PER_UNIT = 1.0
DEFAULT_SEED = 5

# Die Zustandstabelle wächst mit n_jobs (S = n_jobs * 12), die Trainingszeit mit
# n_jobs * n_agents - bei 10 Aufträgen/4 Agenten und 100k Episoden bleibt sie unter ~10 s.
N_JOBS_MIN, N_JOBS_MAX = 4, 10
N_AGENTS_MIN, N_AGENTS_MAX = 2, 4
DURATION_VARIABILITY_MIN, DURATION_VARIABILITY_MAX = 0.0, 1.0
TRAVEL_TIME_PER_UNIT_MIN, TRAVEL_TIME_PER_UNIT_MAX = 0.2, 2.0

POSITION_RANGE_MAX = 20.0
DURATION_BASE_RANGE = (5, 15)
SPIKE_PROBABILITY_SCALE = 0.4
SPIKE_MULTIPLIER = 4.0

# OR-Tools-Referenzlauf: harte Zeitgrenze, damit ein Preset niemals hängt.
ORTOOLS_TIME_LIMIT_SECONDS = 10.0
# Für die vielen Held-out-Lösungen: pro Instanz kürzer (typisch 0.05-0.5 s).
ORTOOLS_HELDOUT_TIME_LIMIT_SECONDS = 5.0

# --- Lernverfahren (IQL) -------------------------------------------------------

# Aktionen jedes Agenten pro angekündigtem Auftrag: ein Aufschlag auf sein eigenes
# Contract-Net-Gebot. Aktion 0 = "normal" - ein untrainierter Agent (alle Q gleich)
# wählt per Tie-Break Aktion 0 und verhält sich exakt wie Contract Net.
BIAS_DELTA_MIN = 25.0
BIAS_ACTIONS = (0.0, BIAS_DELTA_MIN, -BIAS_DELTA_MIN)
ACTION_NAMES = ("normal bieten", "hoch bieten (ablehnen)", "niedrig bieten (greifen)")

ALPHA = 0.1
EPSILON_END = 0.05
EPSILON_DECAY_FRACTION = 0.7  # ε fällt linear über die ersten 70 % der Episoden
REWARD_SCALE = 10.0  # Belohnung = -Makespan / REWARD_SCALE

# Beobachtung: Bucket-Kanten für die eigene Freizeit (als Anteile von n_jobs*MEAN_DURATION/n_agents,
# also der "fairen Last") und für die Anfahrt zum angekündigten Auftrag (min).
MEAN_DURATION = 10.0
FREE_TIME_EDGE_FRACTIONS = (0.3, 0.7, 1.1)
TRAVEL_EDGES = (3.0, 8.0)

ENV_RECURRING = "recurring"
ENV_RANDOM = "random"
ENV_LABELS = {
    ENV_RECURRING: "Wiederkehrendes Szenario",
    ENV_RANDOM: "Zufällige Instanzen",
}
DEFAULT_ENV_MODE = ENV_RECURRING

EPISODES_CHOICES = (100, 300, 1000, 3000, 10000, 20000)
DEFAULT_EPISODES = 20000
SIGMA_MIN, SIGMA_MAX = 0.0, 0.6
DEFAULT_SIGMA = 0.3
TRAIN_SEED_MIN, TRAIN_SEED_MAX = 0, 999
DEFAULT_TRAIN_SEED = 0

# --- PPO (IPPO / MAPPO) ----------------------------------------------------------

METHOD_IPPO = "ippo"      # jeder Agent hat einen eigenen (lokalen) Kritiker
METHOD_MAPPO = "mappo"    # ein zentraler Kritiker sieht im Training den gemeinsamen Zustand
METHOD_LABELS = {
    METHOD_IPPO: "IPPO (lokale Kritiker)",
    METHOD_MAPPO: "MAPPO (zentraler Kritiker)",
}
DEFAULT_METHOD = METHOD_MAPPO

ACTOR_NET = "net"         # kleines MLP, stetige Merkmale, geteilte Parameter
ACTOR_TABLE = "table"     # Softmax-Tabelle je Agent über dieselben Buckets wie IQL
ACTOR_LABELS = {
    ACTOR_NET: "Netz (stetige Merkmale)",
    ACTOR_TABLE: "Tabelle (Buckets)",
}
DEFAULT_ACTOR = ACTOR_NET

SCOPE_JOINT = "joint"     # zentraler Kritiker sieht Zustand aller Agenten + aktuellen Auftrag
SCOPE_FULL = "full"       # ... zusätzlich die Liste der künftigen Aufträge (privilegiert, nur Training)
SCOPE_LABELS = {
    SCOPE_JOINT: "alle Agenten",
    SCOPE_FULL: "alle Agenten + Auftragsliste (nur Training)",
}
DEFAULT_SCOPE = SCOPE_JOINT

PPO_CLIP = 0.2
PPO_CLIP_OFF = 1e9                # "ohne Clipping": das Verhältnis wird nie abgeschnitten
CLIP_CHOICES = (0.1, 0.2, 0.5, None)  # None = aus
CLIP_LABELS = {0.1: "0,1", 0.2: "0,2", 0.5: "0,5", None: "aus"}
DEFAULT_CLIP = PPO_CLIP

# Netz-Akteur/-Kritiker (Prototyp-Endkonfiguration): 2 x 16 tanh, geteilte Parameter
NET_HIDDEN = 16
NET_BATCH_EPISODES = 100
NET_EPOCHS = 4
NET_MINIBATCHES = 4
NET_LR = 3e-3
NET_ENTROPY = 0.01
NET_GAE_LAMBDA = 0.9
NET_MAX_GRAD_NORM = 0.5

# Tabellen-Akteur/-Kritiker
TABLE_BATCH_EPISODES = 64
TABLE_EPOCHS = 4
TABLE_LR_POLICY = 0.03
TABLE_LR_VALUE = 0.3
TABLE_ENTROPY = 0.03
TABLE_GAE_LAMBDA = 0.9

DEFAULT_USE_JOB_INDEX = True

# Kritiker-Vermessung (identische Trajektorien, drei Kritiker gefittet)
CRITIC_LAB_TRAIN_EPISODES = 6000
CRITIC_LAB_TEST_EPISODES = 2000
CRITIC_LAB_STEPS = 1200
CRITIC_LAB_HIDDEN = 16

# 2x2-Miniatur: wie viele Trainings-Seeds erreichen das Optimum
MINIATURE_SEEDS = 10
MINIATURE_EPISODES = 2000

# --- Auswertung ---------------------------------------------------------------

N_HELDOUT = 30              # Held-out-Instanzen (Rauschen-Ziehungen bzw. frische Instanzen)
N_FRESH_SCENARIOS = 40      # frische Szenarien für den Generalisierungs-Check
N_RANDOM_BIAS_DRAWS = 30    # Zufalls-Bias-Baseline
RANDOM_POOL_SIZE = 1000     # Trainingspool im Modus "zufällige Instanzen"
N_LOTTERY_SEEDS = 6
LOTTERY_EPISODE_CAP = 10000  # gleiche Trainingsmenge für ALLE Verfahren in der Lotterie
N_LEARNING_CURVE_POINTS_MAX = len(EPISODES_CHOICES)

LOTTERY_WORSE_TOLERANCE_PCT = 1.0  # ein Lauf zählt als 'schlechter als CNP' ab +1 % (nicht schon bei Rauschen)
LOSE_FRACTION_WARNING = 0.2        # Warnung, wenn > 20 % der Held-out-Instanzen schlechter als CNP sind
ABLATION_NEGLIGIBLE_PTS = 1.5      # IPPO-vs-MAPPO-Unterschied unter 1,5 Punkten gilt als Rauschen (Einzellauf)

# Schwellwerte der Verdict-Kaskade (gegen gemessene Spannen kalibriert)
WORSE_THAN_CNP_THRESHOLD_PCT = 2.0      # IQL gilt als schlechter, wenn nominal ODER Held-out > +2 %
CLEARLY_BETTER_THRESHOLD_PCT = 3.0      # Erfolg nur bei klar besser (> 3 %) nominal UND Held-out
LOTTERY_SPREAD_WARNING_PCT = 5.0        # Seed-Std der Held-out-Ergebnisse in % von CNP
CROSSPLAY_PENALTY_WARNING_PCT = 3.0     # Cross-Play-Strafe in % von CNP

# Preset-Werte empirisch kalibriert (Wegwerf-Sweeps, seither gelöscht) - nicht der erste Versuch
# übernommen. Gemessene Werte (n=8, k=3, Streuung 0.3, Anfahrt 1.0; % vs. Contract Net, nominal/Held-out):
#   1 PPO stabilisiert die Lotterie: s4 wiederkehrend, MAPPO -12.7/-8.9; Lotterie (10k Ep., 6 Seeds):
#     IQL 4/6 schlechter, Std 7.7 | IPPO 0/6, Std 0.22 | MAPPO 0/6, Std 0.06
#   2 IPPO = MAPPO: s8 wiederkehrend, Train-Seed 1, beide -21.6/-17.9 (Train-Seed 0: MAPPO 1.7 Pkt. besser)
#   3 Nichts zu lernen: s9 wiederkehrend, IPPO 0.0/0.0 (IQL +0/+3.1, 5/6 Lotterie-Läufe schlechter)
#   4 Zufällige Instanzen: s13 zufällig, MAPPO -23.1/-4.7, IQL -2.3/+4.9 (6/6 schlechter)
#   5 Privilegierter Kritiker: dasselbe Szenario, MAPPO+Auftragsliste -19.3/-6.6
#   6 Wenig Erfahrung: s4 wiederkehrend, 1000 Ep.: PPO -12.7/-6.9..-9.0, IQL +12.3/+13.4
#   7 Tabelle merkt sich den Plan: s4, Tabellen-Akteur: Held-out -9.1, frische Szenarien +6.6 (Netz +0.3)
#   8 Ohne Clipping: s8 wiederkehrend: MAPPO-Lotterie Std 7.1 (mit Clipping 0.07), Cross-Play +15.4 (+0.0)
_BASE = {
    "n_jobs": 8, "n_agents": 3, "duration_variability": 0.3, "travel_time_per_unit": 1.0,
    "sigma": 0.3, "episodes": 20000, "method": METHOD_MAPPO, "actor": ACTOR_NET, "scope": SCOPE_JOINT,
    "clip": PPO_CLIP, "use_job_index": True, "env_mode": ENV_RECURRING, "train_seed": 0,
}
PRESETS = {
    "PPO stabilisiert die Lotterie": {**_BASE, "seed": 4, "auto_lottery": True},
    "IPPO = MAPPO": {**_BASE, "seed": 8, "train_seed": 1},
    "Nichts zu lernen": {**_BASE, "seed": 9, "method": METHOD_IPPO},
    "Zufällige Instanzen": {**_BASE, "seed": 13, "env_mode": ENV_RANDOM},
    "Privilegierter Kritiker": {**_BASE, "seed": 13, "env_mode": ENV_RANDOM, "scope": SCOPE_FULL},
    "Wenig Erfahrung": {**_BASE, "seed": 4, "episodes": 1000, "train_seed": 1},
    "Tabelle merkt sich den Plan": {**_BASE, "seed": 4, "actor": ACTOR_TABLE},
    "Ohne Clipping": {**_BASE, "seed": 8, "train_seed": 1, "clip": None, "auto_lottery": True},
}

PRESET_HELP = {
    "PPO stabilisiert die Lotterie": "Dasselbe Szenario, IQL schwankt je nach Trainings-Seed stark - "
        "PPO landet in jedem Lauf am selben Ergebnis. Startet die Seed-Lotterie.",
    "IPPO = MAPPO": "Der zentrale Kritiker ändert das Ergebnis hier nicht: IPPO mit lokalen Kritikern "
        "erreicht dasselbe wie MAPPO.",
    "Nichts zu lernen": "Hier hat Contract Net kaum Spielraum: PPO bleibt bei Contract Net (kein Schaden), "
        "IQL wird in den meisten Läufen schlechter.",
    "Zufällige Instanzen": "Immer neue Instanzen im Training: das Netz mit stetigen Merkmalen generalisiert "
        "bescheiden, wo IQL scheitert.",
    "Privilegierter Kritiker": "Dasselbe Szenario, aber der Kritiker sieht im Training auch die Auftragsliste "
        "- ein wenig besser, aber Wissen, das die Akteure im Einsatz nie haben.",
    "Wenig Erfahrung": "Nur 1000 Episoden: PPO hat schon gelernt, IQL ist noch deutlich schlechter als "
        "Contract Net.",
    "Tabelle merkt sich den Plan": "Akteur als Tabelle über Buckets: im wiederkehrenden Szenario genauso gut, "
        "auf frischen Szenarien aber schlechter als das Netz.",
    "Ohne Clipping": "PPO-Clipping ausgeschaltet: der einzelne Lauf sieht gut aus, die Seed-Lotterie zeigt "
        "die zurückgekehrte Instabilität und eine große Cross-Play-Strafe.",
}

# Regressions-Bänder für tests/test_mappo_evaluation.py (Prozent gegenüber Contract Net, negativ = besser):
# das ANGEZEIGTE Verfahren (Preset-Feld "method"); nominal = gezeigte Instanz, heldout = Held-out-Mittel.
PRESET_EXPECTED_BANDS = {
    "PPO stabilisiert die Lotterie": {"nominal_pct": (-16.0, -8.0), "heldout_pct": (-12.0, -6.0)},
    "IPPO = MAPPO": {"nominal_pct": (-26.0, -17.0), "heldout_pct": (-22.0, -14.0)},
    "Nichts zu lernen": {"nominal_pct": (-3.0, 3.0), "heldout_pct": (-3.0, 3.0)},
    "Zufällige Instanzen": {"nominal_pct": (-28.0, -16.0), "heldout_pct": (-8.0, -2.0)},
    "Privilegierter Kritiker": {"nominal_pct": (-28.0, -12.0), "heldout_pct": (-9.5, -3.5)},
    "Wenig Erfahrung": {"nominal_pct": (-16.0, -8.0), "heldout_pct": (-12.0, -5.0)},
    "Tabelle merkt sich den Plan": {"nominal_pct": (-16.0, -8.0), "heldout_pct": (-12.0, -6.0)},
    "Ohne Clipping": {"nominal_pct": (-26.0, -17.0), "heldout_pct": (-22.0, -13.0)},
}
