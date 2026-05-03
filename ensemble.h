#ifndef ENSEMBLE_H
#define ENSEMBLE_H

#include "fitness.h"
#include "baselines.h"
#include "ga.h"
#include "pso.h"
#include <vector>
#include <iostream>
#include <numeric>

using namespace std;

// ============================================================================
// Ensemble pipeline — INSTRUCTOR'S METHOD (parallel + voting)
// ============================================================================
// 1. Run PSO, GA, Greedy independently on the same input.
// 2. For each (area, hour) cell, count how many algorithms voted to cut.
// 3. If >= 2 of 3 voted to cut, the ensemble cuts.
// 4. Apply local search to remove any redundant cuts.
//
// This is fundamentally different from Hybrid (sequential chaining):
//   Hybrid:     Greedy -> GA -> PSO -> LS    (each refines the previous)
//   Ensemble:   PSO || GA || Greedy -> vote  (independent, then merged)
//
// The vote_threshold is configurable. Default = 2 (majority of 3).
// ============================================================================

struct EnsembleResult {
    Schedule final_schedule;
    Schedule pso_schedule;
    Schedule ga_schedule;
    Schedule greedy_schedule;
    double pso_fitness;
    double ga_fitness;
    double greedy_fitness;
    double pre_local_search_fitness;
    double final_fitness;
    int votes_majority;     // cells where >=2 agreed
    int votes_unanimous;    // cells where all 3 agreed
    int total_cells;
};

// Greedy local search shared with PSO/GA — defined here for self-contained ensemble
inline Schedule ensemble_local_search(
    Schedule s,
    const vector<vector<double>>& demand,
    const vector<AreaConfig>& area_configs,
    bool verbose = true)
{
    int n_areas = demand.size();
    int n_hours = demand[0].size();
    double initial_fitness = fitness(s, demand, area_configs);
    int total_removed = 0;
    int passes = 0;
    bool improved = true;

    while (improved) {
        improved = false;
        passes++;
        for (int i = 0; i < n_areas; ++i) {
            if (area_configs[i].priority_flag == 1) continue;
            for (int t = 0; t < n_hours; ++t) {
                if (s[i][t] == 0) continue;

                double fit_with = fitness(s, demand, area_configs);
                s[i][t] = 0;
                double fit_without = fitness(s, demand, area_configs);

                if (fit_without <= fit_with) {
                    total_removed++;
                    improved = true;
                } else {
                    s[i][t] = 1;
                }
            }
        }
    }

    if (verbose) {
        cout << "  Local search: " << total_removed
             << " redundant cuts removed in " << passes << " passes\n";
    }
    return s;
}

// Force priority areas to never be cut (safety net)
inline void force_priority_safety(Schedule& s, const vector<AreaConfig>& area_configs) {
    int n_areas = s.size();
    if (n_areas == 0) return;
    int n_hours = s[0].size();
    for (int i = 0; i < n_areas; ++i) {
        if (area_configs[i].priority_flag == 1) {
            for (int t = 0; t < n_hours; ++t) s[i][t] = 0;
        }
    }
}

// Main ensemble entry point
inline EnsembleResult ensemble_pipeline(
    const vector<vector<double>>& demand,
    const vector<AreaConfig>& area_configs,
    int vote_threshold = 2,
    bool verbose = true)
{
    EnsembleResult result;
    int n_areas = demand.size();
    if (n_areas == 0) {
        if (verbose) cerr << "[Ensemble] ERROR: empty demand matrix\n";
        return result;
    }
    int n_hours = demand[0].size();

    // Validate that area_configs and demand have the same shape
    if ((int)area_configs.size() != n_areas) {
        if (verbose) cerr << "[Ensemble] ERROR: area_configs size mismatch\n";
        return result;
    }

    // ---- VOTER 1: Greedy ----
    if (verbose) cout << "\n[Ensemble Voter 1: Greedy]\n";
    result.greedy_schedule = greedy_scheduler(demand, area_configs);
    force_priority_safety(result.greedy_schedule, area_configs);
    result.greedy_fitness = fitness(result.greedy_schedule, demand, area_configs);
    if (verbose) cout << "  Greedy fitness: " << result.greedy_fitness << "\n";

    // ---- VOTER 2: GA ----
    if (verbose) cout << "\n[Ensemble Voter 2: GA]\n";
    GA ga(/*pop_size=*/50, /*max_gen=*/200, n_areas, n_hours, /*seed=*/123);
    result.ga_schedule = ga.run(demand, area_configs, /*verbose=*/false);
    force_priority_safety(result.ga_schedule, area_configs);
    result.ga_fitness = ga.global_best_fitness;
    if (verbose) cout << "  GA fitness: " << result.ga_fitness << "\n";

    // ---- VOTER 3: PSO ----
    if (verbose) cout << "\n[Ensemble Voter 3: PSO]\n";
    BPSO pso(/*swarm_size=*/50, /*max_iter=*/200, n_areas, n_hours, /*seed=*/456);
    result.pso_schedule = pso.run(demand, area_configs, /*verbose=*/false);
    force_priority_safety(result.pso_schedule, area_configs);
    result.pso_fitness = pso.global_best_fitness;
    if (verbose) cout << "  PSO fitness: " << result.pso_fitness << "\n";

    // ---- VOTING (>= vote_threshold of 3 must agree) ----
    if (verbose) cout << "\n[Ensemble: voting per (area, hour) cell]\n";
    Schedule voted(n_areas, vector<int>(n_hours, 0));
    int majority_count = 0;
    int unanimous_count = 0;
    int total_cells = n_areas * n_hours;

    for (int i = 0; i < n_areas; ++i) {
        for (int t = 0; t < n_hours; ++t) {
            int votes = result.greedy_schedule[i][t]
                      + result.ga_schedule[i][t]
                      + result.pso_schedule[i][t];

            if (votes >= vote_threshold) {
                voted[i][t] = 1;
                majority_count++;
            }
            if (votes == 3) unanimous_count++;
        }
    }
    // Force priority safety net (redundant but defensive)
    force_priority_safety(voted, area_configs);

    result.votes_majority = majority_count;
    result.votes_unanimous = unanimous_count;
    result.total_cells = total_cells;
    result.pre_local_search_fitness = fitness(voted, demand, area_configs);

    if (verbose) {
        cout << "  Cells with majority cut (>=" << vote_threshold << " of 3): "
             << majority_count << " / " << total_cells << "\n";
        cout << "  Cells with unanimous cut (3 of 3): "
             << unanimous_count << " / " << total_cells << "\n";
        cout << "  Pre-cleanup fitness: " << result.pre_local_search_fitness << "\n";
    }

    // ---- LOCAL SEARCH cleanup ----
    if (verbose) cout << "\n[Ensemble: local search cleanup]\n";
    result.final_schedule = ensemble_local_search(voted, demand, area_configs, verbose);
    result.final_fitness = fitness(result.final_schedule, demand, area_configs);

    if (verbose) cout << "  Final ensemble fitness: " << result.final_fitness << "\n";
    return result;
}

#endif // ENSEMBLE_H