#ifndef HYBRID_H
#define HYBRID_H

#include "fitness.h"
#include "baselines.h"
#include "ga.h"
#include "pso.h"
#include <iostream>

using namespace std;

// Hybrid pipeline: Greedy -> GA -> PSO -> Local Search
// Each stage seeds the next with its best result.
inline Schedule hybrid_pipeline(
    const vector<vector<double>>& demand,
    const vector<AreaConfig>& area_configs,
    bool verbose = true)
{
    int n_areas = demand.size();
    int n_hours = demand[0].size();

    // ---- STAGE 1: Greedy warm-start ----
    if (verbose) cout << "\n[Hybrid Stage 1: Greedy warm-start]\n";
    Schedule s1 = greedy_scheduler(demand, area_configs);
    double f1 = fitness(s1, demand, area_configs);
    if (verbose) cout << "  Greedy fitness: " << f1 << "\n";

    // ---- STAGE 2: GA seeded with Greedy ----
    if (verbose) cout << "\n[Hybrid Stage 2: GA refinement (seeded)]\n";
    GA ga(/*pop_size=*/50, /*max_gen=*/200, n_areas, n_hours, /*seed=*/123);
    Schedule s2 = ga.run(demand, area_configs, /*verbose=*/false, /*seed=*/&s1);
    double f2 = ga.global_best_fitness;
    if (verbose) cout << "  GA fitness: " << f2 << "\n";

    // ---- STAGE 3: PSO seeded with GA ----
    if (verbose) cout << "\n[Hybrid Stage 3: PSO refinement (seeded)]\n";
    BPSO pso(/*swarm_size=*/50, /*max_iter=*/200, n_areas, n_hours, /*seed=*/456);
    Schedule s3 = pso.run(demand, area_configs, /*verbose=*/false, /*seed=*/&s2);
    double f3 = pso.global_best_fitness;
    if (verbose) cout << "  PSO fitness: " << f3 << "\n";

    // PSO already runs local search internally, so s3 is the final result
    if (verbose) cout << "\nHybrid pipeline complete.\n";
    return s3;
}

#endif // HYBRID_H