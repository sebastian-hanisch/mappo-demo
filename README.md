# MAPPO / CTDE (zentraler Kritiker) an der Kran-Auftragsvergabe – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-mappo-demo.streamlit.app/)**

Fünftes Stück der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations
Research und Machine Learning", **Multi-Agenten-Koordinations-Linie** - **Fortsetzung von
[marl-demo](../marl-demo)** (unabhängiges Q-Learning), das Contract Net an der Wurzel hat
([contract-net-demo](../contract-net-demo), [task-swap-demo](../task-swap-demo), [dcop-demo](../dcop-demo)).

## Was dieses Stück tut - und was es überraschend zeigt

marl-demos Schwäche war die **Nicht-Stationarität**: jeder Agent lernt, während die anderen sich gleichzeitig
ändern. **CTDE** (*Centralized Training, Decentralized Execution*) setzt dort an: ein **zentraler Kritiker** sieht im
Training den gemeinsamen Zustand aller Agenten, im Einsatz entscheidet jeder **Akteur** nur aus seiner lokalen
Beobachtung. **MAPPO** (Yu et al. 2022) ist PPO mit zentralem Kritiker; **IPPO** (jeder Agent mit eigenem,
lokalem Kritiker) ist die Ablation, die den Effekt des Kritikers isoliert.

Gemessen (Prototyp, 16 Szenarien x 5 Trainings-Seeds, 30 Held-out-Instanzen je Szenario, beide Umgebungen):

1. **PPO - vor allem das Clipping - stabilisiert das Training, nicht der zentrale Kritiker.** Wiederkehrend:
   IQL Held-out −0,5 % (Seed-Std 6,7 Punkte) gegenüber PPO −8,7…−9,1 % (Std 0,5–1,1). Zufällige Instanzen: IQL
   +8,8 % (in 99 % der Läufe schlechter als Contract Net), MLP-PPO −2,7…−4,7 % (0–6 %). **Ohne Clipping** kommt die
   Instabilität zurück (Preset "Ohne Clipping": MAPPO-Seed-Std 0,07 → 7,1; Cross-Play +0,0 % → +15,4 %).
2. **IPPO vs. MAPPO**: wiederkehrend kein Unterschied; auf zufälligen Instanzen ist MAPPO 0,2–1,4 Punkte besser
   (hyperparameterabhängig, ≈ Rauschen). Die Cross-Play-Strafe unterscheidet sich nicht verlässlich - **CTDE behebt
   die Konventionen nicht**.
3. **Der Kritiker senkt die Varianz, nicht die Endgüte.** Erklärte Varianz des Returns auf *identischen*
   Trajektorien: lokal 0,15–0,24, zentral 0,29–0,55, zentral + Auftragsliste (privilegiert) 0,75–0,88. Der
   privilegierte Kritiker hilft am ehesten - überwiegend durch Wissen über die Instanzschwierigkeit, das der
   Akteur im Einsatz nie hat.
4. **Stetige Merkmale + kleines Netz generalisieren über zufällige Instanzen - bescheiden**: MAPPO −4 %, auf
   ~22 % der Held-out-Instanzen weiterhin schlechter als Contract Net (schlechteste +20…+36 %), schließt nur
   ~1/6 der 20–30 % Lücke zu CP-SAT. Unabhängig vom Auftrags-Index.
5. **Tabellen-Akteur** (als Umschalter): wiederkehrend so gut wie das Netz (1 s Training), memoriert aber
   (frische Szenarien +3…+8 % gegenüber Netz −3…+0,3 %); auf zufälligen Instanzen nur ≈ Contract Net.
6. **Untrainiert = Contract Net exakt** (Null-Init der letzten Schicht), Umgebung = marl-demo-Kernel bitweise.

Ehrliche Einordnung: MAPPO/CTDE ist nur *innerhalb kooperativen Multi-Agenten-Lernens* der etablierte Standard;
das zentrale CP-SAT bleibt der praktische Industriestandard. Die Preset-Szenarien stammen aus marl-demo, wo sie
wegen IQL-Dramatik gewählt wurden - auf unselektierten Szenarien ist IQL wiederkehrend −0,5 %.

## Verfahren

- **Umgebung/Aktionen/Belohnung** wie marl-demo: Aufschlag {0, +25, −25} min auf das eigene Gebot, Zuschlag nach
  Contract Nets Regel, Belohnung −Makespan/10 am Episodenende.
- **Akteur "Netz"**: softmax, MLP 2×16 tanh, **geteilte Parameter** (Yu et al.), Agenten-ID als One-Hot; stetige
  lokale Merkmale (eigene Freizeit, Anfahrt, Position, Auftragsdauer/-position, optional Auftrags-Index j/n).
  **Akteur "Tabelle"**: Softmax-Tabelle je Agent über dieselben Buckets wie IQL.
- **Kritiker**: IPPO - lokal je Agent; MAPPO - zentral (Zustand aller Agenten + aktueller Auftrag), optional
  **privilegiert** (+ Liste der künftigen Aufträge, nur Training).
- **PPO**: GAE (λ 0,9, γ 1), geclipptes Ziel (ε 0,2, abschaltbar), Entropie 0,01, Adam 3e-3, Batch 100
  Episoden, 4 Epochen x 4 Minibatches, Vorteile pro Batch normiert; numpy-only (kein torch).
- **Held-out/Seed-Lotterie/Cross-Play** wie marl-demo, jetzt für IQL, IPPO und MAPPO nebeneinander.
- **Kritiker-Vermessung**: drei Kritiker (lokal/zentral/privilegiert) werden auf denselben Trajektorien
  gefittet - erklärte Varianz auf Testdaten.

## Verifikation

- **GAE** an handgerechneten Beispielen (V=[1,2,3], R=10: λ=0,5 → [3,25; 4,5; 7], λ=1 → [9,8,7], λ=0 → [1,1,7]).
- **PPO-Ziel und Gradient**: Handfälle, geschlossene Form gegen finite Differenzen, Null-Gradient im geclippten
  Bereich, Score-Funktions-Identität; **MLP-Backprop** gegen finite Differenzen.
- **Vektorisierte Umgebung = marl-Kernel bitweise** (200 Zufallspolicies), = `run_protocol` (300 Instanzen), Buckets =
  `make_observer`; `replay_dispatch` reproduziert den Makespan über die Vehikel-Funktionen.
- **Untrainiert = Contract Net** (Netz und Tabelle), Determinismus, Seed-Entkopplung, geteilte Parameter (ID
  genullt ⇒ identisches Verhalten).
- **2x2-Miniatur**: IQL, IPPO, MAPPO (Netz) finden das Optimum 15 in ≥ 18/20 Seeds, Tabellen in 20/20 - kein
  Kritiker-Effekt sichtbar (zu klein).
- **Kritiker-Labor**: erklärte Varianz lokal < zentral < privilegiert; **Clipping aus** ⇒ Seed-Streuung und
  Cross-Play-Strafe steigen (Regressionstest); **Preset-Bänder** für alle Presets.
- **AppTest-Rauchtests**: Default, jedes Preset, Umschalter (versteckte Regler behalten ihren Wert), Kleinstinstanz.

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Hauptablauf: Presets, Einstellungen, Lernkurve, Akteur/Kritiker-Ansicht, "Preis des zentralen Kritikers" mit vier Tabs |
| `cn_constants.py` | Defaults, Regler-Grenzen, PPO-Konstanten, `PRESETS`, Bänder |
| `cn_presets.py` | `SettingSpec`/Permalink-Logik (Umgebung, Episoden, σ, Trainings-Seed, Verfahren, Akteur, Kritiker-Sicht, Clip, Auftrags-Index) |
| `mappo_env.py` | Vektorisierte Dispatch-Umgebung, Sampler, Merkmale (stetig/Buckets), `replay_dispatch` |
| `mappo_nets.py` | Batch-MLP mit manuellem Backprop, Adam, Softmax |
| `mappo_ppo.py` | GAE, PPO-Ziel/Gradient, `train_ppo` (IPPO/MAPPO, Netz/Tabelle), gierige Ausführung, `trace_policy` |
| `mappo_evaluation.py` | Vergleich, Lernkurve, Generalisierung, Cross-Play, Seed-Lotterie, Kritiker-Labor, Miniatur |
| `mappo_visualization.py` | Lernkurven-Vergleich, Aktionswahrscheinlichkeiten, Policy-Landkarte, Kritiker-Vorhersage, Erklärte-Varianz, Lotterie |
| `marl_*.py`, `cn_*.py` | IQL-Referenz, Held-out, exakte Referenzen, Vehikel (wortgleich aus marl-demo) |
| `tests/` | Handrechnungen, Gradient-Checks, Kernel-Äquivalenz, Miniatur, Kritiker-Labor, Preset-Bänder, AppTest |

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

Teil des [Operations-Research-Demo-Portfolios](https://sebastianhanisch.net/demos.html) von
[Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning.
Interesse an einer maßgeschneiderten Lösung? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html).
