#ifndef PSO_H
#define PSO_H

#include "fitness.h"
#include <vector>
#include <random>
#include <limits>
#include <iostream>
#include <algorithm>

using namespace std;

// One particle = one candidate schedule + its velocities + its personal best
struct Particle {
    Schedule position;
    vector<vector<double>> velocity;
    Schedule best_position;
    double best_fitness;
};

class BPSO {
public:
    int swarm_size;
    int max_iterations;
    int n_areas;
    int n_hours;

    // PSO hyperparameters (Clerc's constriction)
    double w  = 0.729;
    double c1 = 1.49;
    double c2 = 1.49;

    // Stagnation control
    int stagnation_limit = 25;     // restart bad particles after this many no-improvement iters
    double restart_fraction = 0.5; // fraction of swarm to restart

    vector<Particle> swarm;
    Schedule global_best;
    double global_best_fitness;

    mt19937 rng;
    uniform_real_distribution<double> uni{0.0, 1.0};

    const vector<vector<double>>* demand;
    const vector<AreaConfig>* area_configs;

    BPSO(int swarm_sz, int max_iter, int n_a, int n_h, unsigned seed = 42)
        : swarm_size(swarm_sz), max_iterations(max_iter),
          n_areas(n_a), n_hours(n_h),
          global_best_fitness(numeric_limits<double>::max()),
          rng(seed) {}

    double sigmoid(double v) {
        if (v >  6.0) v =  6.0;
        if (v < -6.0) v = -6.0;
        return 1.0 / (1.0 + exp(-v));
    }

    // HARD CONSTRAINT REPAIR: force priority areas to never be cut
    void repair_priority(Schedule& s) {
        for (int i = 0; i < n_areas; ++i) {
            if ((*area_configs)[i].priority_flag == 1) {
                for (int t = 0; t < n_hours; ++t) {
                    s[i][t] = 0;  // never cut priority areas
                }
            }
        }
    }

    // Generate a random particle (with priority repair)
    Particle make_random_particle() {
        Particle p;
        p.position = Schedule(n_areas, vector<int>(n_hours, 0));
        p.velocity = vector<vector<double>>(n_areas, vector<double>(n_hours, 0.0));

        for (int i = 0; i < n_areas; ++i) {
            for (int t = 0; t < n_hours; ++t) {
                p.position[i][t] = (uni(rng) < 0.15) ? 1 : 0;
            }
        }
        repair_priority(p.position);

        p.best_position = p.position;
        p.best_fitness = fitness(p.position, *demand, *area_configs);
        return p;
    }

    void initialize(const Schedule* seed = nullptr) {
    swarm.clear();
    swarm.reserve(swarm_size);

    if (seed) {
        // Seed mode: 1 untouched copy + (swarm_size - 1) perturbed copies
        // Each perturbed copy flips ~5% of bits
        const double perturb_rate = 0.05;

        // First particle: exact seed
        Particle seeded;
        seeded.position = *seed;
        repair_priority(seeded.position);
        seeded.velocity = vector<vector<double>>(n_areas, vector<double>(n_hours, 0.0));
        seeded.best_position = seeded.position;
        seeded.best_fitness = fitness(seeded.position, *demand, *area_configs);
        swarm.push_back(seeded);
        if (seeded.best_fitness < global_best_fitness) {
            global_best_fitness = seeded.best_fitness;
            global_best = seeded.position;
        }

        // Remaining particles: perturbed copies of the seed
        while ((int)swarm.size() < swarm_size) {
            Particle p;
            p.position = *seed;
            // Flip each bit with perturb_rate probability
            for (int i = 0; i < n_areas; ++i) {
                for (int t = 0; t < n_hours; ++t) {
                    if (uni(rng) < perturb_rate) {
                        p.position[i][t] = 1 - p.position[i][t];
                    }
                }
            }
            repair_priority(p.position);
            p.velocity = vector<vector<double>>(n_areas, vector<double>(n_hours, 0.0));
            p.best_position = p.position;
            p.best_fitness = fitness(p.position, *demand, *area_configs);
            swarm.push_back(p);
            if (p.best_fitness < global_best_fitness) {
                global_best_fitness = p.best_fitness;
                global_best = p.position;
            }
        }
    } else {
        // No seed: pure random initialization (original behavior)
        for (int i = 0; i < swarm_size; ++i) {
            Particle particle = make_random_particle();
            swarm.push_back(particle);
            if (particle.best_fitness < global_best_fitness) {
                global_best_fitness = particle.best_fitness;
                global_best = particle.position;
            }
        }
    }
}
    void update_particle(Particle& p) {
        for (int i = 0; i < n_areas; ++i) {
            for (int t = 0; t < n_hours; ++t) {
                double r1 = uni(rng);
                double r2 = uni(rng);

                p.velocity[i][t] =
                    w  * p.velocity[i][t]
                  + c1 * r1 * (p.best_position[i][t] - p.position[i][t])
                  + c2 * r2 * (global_best[i][t]     - p.position[i][t]);

                double prob = sigmoid(p.velocity[i][t]);
                p.position[i][t] = (uni(rng) < prob) ? 1 : 0;
            }
        }
        repair_priority(p.position);  // enforce hard constraint after each move
    }

    // Restart the worst N particles (keep the best ones, replace bottom half)
    void restart_stagnant_particles() {
        sort(swarm.begin(), swarm.end(),
             [](const Particle& a, const Particle& b) {
                 return a.best_fitness < b.best_fitness;
             });
        int n_restart = (int)(swarm_size * restart_fraction);
        for (int p = swarm_size - n_restart; p < swarm_size; ++p) {
            swarm[p] = make_random_particle();
        }
    }
    
    Schedule local_search(Schedule s, bool verbose = true) {
        double initial_fitness = fitness(s, *demand, *area_configs);
        int total_removed = 0;
        int passes = 0;
        bool improved = true;

        while (improved) {
            improved = false;
            passes++;

            for (int i = 0; i < n_areas; ++i) {
                if ((*area_configs)[i].priority_flag == 1) continue; // skip priority

                for (int t = 0; t < n_hours; ++t) {
                    if (s[i][t] == 0) continue; // not a cut, nothing to remove

                    double fit_with_cut = fitness(s, *demand, *area_configs);
                    s[i][t] = 0; // try removing
                    double fit_without_cut = fitness(s, *demand, *area_configs);

                    if (fit_without_cut <= fit_with_cut) {
                        total_removed++;
                        improved = true;
                        // keep it removed
                    } else {
                        s[i][t] = 1; // put it back
                    }
                }
            }
        }

        double final_fitness = fitness(s, *demand, *area_configs);
        if (verbose) {
            cout << "Local search: " << total_removed
                << " unnecessary cuts removed in " << passes << " passes\n";
            cout << "  Fitness " << initial_fitness << " -> " << final_fitness << "\n";
        }
        return s;
    }
    Schedule run(const vector<vector<double>>& dem,
        const vector<AreaConfig>& cfg,
        bool verbose = true,
    const Schedule* seed = nullptr)
    {
        demand = &dem;
        area_configs = &cfg;
        initialize(seed);

        int iters_since_improvement = 0;
        double last_best = global_best_fitness;
        int restarts = 0;

        for (int iter = 0; iter < max_iterations; ++iter) {
            for (auto& p : swarm) {
                update_particle(p);
                double f = fitness(p.position, *demand, *area_configs);

                if (f < p.best_fitness) {
                    p.best_fitness = f;
                    p.best_position = p.position;
                }
                if (f < global_best_fitness) {
                    global_best_fitness = f;
                    global_best = p.position;
                }
            }

            // Stagnation check
            if (global_best_fitness < last_best - 1e-6) {
                last_best = global_best_fitness;
                iters_since_improvement = 0;
            } else {
                iters_since_improvement++;
                if (iters_since_improvement >= stagnation_limit) {
                    restart_stagnant_particles();
                    iters_since_improvement = 0;
                    restarts++;
                    if (verbose) cout << "  [restart at iter " << iter << "]\n";
                }
            }

            if (verbose && (iter % 20 == 0 || iter == max_iterations - 1)) {
                cout << "Iter " << iter
                     << " | best fitness: " << global_best_fitness << "\n";
            }
        }

        if (verbose) cout << "PSO finished. Total restarts: " << restarts << "\n";


        // Local search refinement
        if (verbose) cout << "\n--- Running local search refinement ---\n";
        Schedule refined = local_search(global_best, verbose);
        global_best = refined;
        global_best_fitness = fitness(refined, *demand, *area_configs);

        return global_best;
    }
    // Greedy local search: remove every cut that isn't actually needed
};

#endif // PSO_H