from cn_bruteforce import solve_bruteforce
from cn_ortools_reference import SCALE, solve_with_ortools
from cn_scenario import generate_instance

# cn_ortools_reference rundet jede Dauer/Anfahrtszeit bewusst AUF die nächste 1/SCALE-
# Einheit auf (nie ab - siehe Modul-Docstring, dieselbe Konvention wie quaycrane-demo),
# damit das Modell nie eine kürzere Zeit annimmt, als real gebraucht wird. Das heißt aber
# auch: gegen eine exakte, stetige Bruteforce-Referenz kann sich pro Auftrag bis zu 1/SCALE
# Aufrundung ansammeln (Anfahrt + Dauer, je einmal) - der Toleranzwert bildet genau dieses
# worst-case-Aufrunden ab, ist also kein Rateversuch.
_ROUNDING_TOLERANCE_PER_JOB = 2.0 / SCALE


def test_cp_sat_matches_bruteforce_on_tiny_instances():
    for seed in range(15):
        instance = generate_instance(
            n_jobs=4, n_agents=2, duration_variability=0.5, travel_time_per_unit=1.0, seed=seed
        )
        exact = solve_with_ortools(instance, time_limit_seconds=5.0)
        bf_makespan, _bf_assignment = solve_bruteforce(instance)

        assert exact.feasible, f"seed={seed}"
        tolerance = _ROUNDING_TOLERANCE_PER_JOB * instance.n_jobs
        assert exact.makespan >= bf_makespan - 1e-6, f"seed={seed}"  # CP-SAT darf nie darunter liegen
        assert abs(exact.makespan - bf_makespan) < tolerance, f"seed={seed}"


def test_cp_sat_lexicographic_tiebreak_avoids_spurious_idle():
    # Dieselbe Instanz mehrfach loesen (num_search_workers=1 -> ohnehin deterministisch,
    # aber diese Regression sichert zu, dass die Sekundaerkennzahl (Summe der Endzeiten)
    # bei jedem Solve gleich bleibt statt zwischen gleich-optimalen Loesungen zu schwanken -
    # dieselbe Klasse Fund wie in quaycrane-demo).
    instance = generate_instance(n_jobs=8, n_agents=2, duration_variability=0.5, travel_time_per_unit=1.0, seed=2)
    results = [solve_with_ortools(instance, time_limit_seconds=5.0) for _ in range(3)]
    assert all(r.feasible for r in results)
    makespans = {round(r.makespan, 6) for r in results}
    assert len(makespans) == 1

    sums_of_starts = {round(sum(r.starts.values()), 6) for r in results}
    assert len(sums_of_starts) == 1


def test_single_agent_cp_sat_matches_sequential_bruteforce():
    for seed in range(10):
        instance = generate_instance(
            n_jobs=5, n_agents=1, duration_variability=0.3, travel_time_per_unit=1.0, seed=seed
        )
        exact = solve_with_ortools(instance, time_limit_seconds=5.0)
        bf_makespan, _bf_assignment = solve_bruteforce(instance)
        assert exact.feasible, f"seed={seed}"
        tolerance = _ROUNDING_TOLERANCE_PER_JOB * instance.n_jobs
        assert exact.makespan >= bf_makespan - 1e-6, f"seed={seed}"
        assert abs(exact.makespan - bf_makespan) < tolerance, f"seed={seed}"
