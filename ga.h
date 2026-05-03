#ifndef GA_H
#define GA_H

#include "fitness.h"
#include <vector>
#include <random>
#include <limits>
#include <iostream>
#include <algorithm>

using namespace std;

// One chromosome = one schedule + its fitness
struct Individual {
    Schedule genes;
    double fitness_value;
};

class GA {
public:
    int pop_size;
    int max_generations;
    int n_areas;
    int n_hours;
    int n_bits;

    double crossover_rate = 0.8;
    double mutation_rate;     // set in constructor
    int tournament_size = 3;
    int elite_count = 2;

    vector<Individual> population;
    Schedule global_best;
    double global_best_fitness;

    mt19937 rng;
    uniform_real_distribution<double> uni{0.0, 1.0};
    uniform_int_distribution<int> rand_idx;

    const vector<vector<double>>* demand;
    const vector<AreaConfig>* area_configs;

    GA(int pop_sz, int max_gen, int n_a, int n_h, unsigned seed = 42)
        : pop_size(pop_sz), max_generations(max_gen),
          n_areas(n_a), n_hours(n_h), n_bits(n_a * n_h),
          global_best_fitness(numeric_limits<double>::max()),
          rng(seed), rand_idx(0, pop_sz - 1)
    {
        mutation_rate = 1.0 / n_bits;  // ~1 bit flip per child on average
    }

    // Force priority areas to never be cut
    void repair_priority(Schedule& s) {
        for (int i = 0; i < n_areas; ++i) {
            if ((*area_configs)[i].priority_flag == 1) {
                for (int t = 0; t < n_hours; ++t) s[i][t] = 0;
            }
        }
    }

    // Generate one random chromosome (with priority repair)
    Individual make_random_individual() {
        Individual ind;
        ind.genes = Schedule(n_areas, vector<int>(n_hours, 0));
        for (int i = 0; i < n_areas; ++i) {
            for (int t = 0; t < n_hours; ++t) {
                ind.genes[i][t] = (uni(rng) < 0.15) ? 1 : 0;
            }
        }
        repair_priority(ind.genes);
        ind.fitness_value = fitness(ind.genes, *demand, *area_configs);
        return ind;
    }

   void initialize(const Schedule* seed = nullptr) {
    population.clear();
    population.reserve(pop_size);

    // If a seed schedule is provided, inject it as the first chromosome
    if (seed) {
        Individual seeded;
        seeded.genes = *seed;
        repair_priority(seeded.genes);
        seeded.fitness_value = fitness(seeded.genes, *demand, *area_configs);
        population.push_back(seeded);
        if (seeded.fitness_value < global_best_fitness) {
            global_best_fitness = seeded.fitness_value;
            global_best = seeded.genes;
        }
    }

    while ((int)population.size() < pop_size) {
        Individual ind = make_random_individual();
        population.push_back(ind);
        if (ind.fitness_value < global_best_fitness) {
            global_best_fitness = ind.fitness_value;
            global_best = ind.genes;
        }
    }
}

    // Pick best of `tournament_size` random individuals
    const Individual& tournament_select() {
        int best_idx = rand_idx(rng);
        for (int t = 1; t < tournament_size; ++t) {
            int candidate = rand_idx(rng);
            if (population[candidate].fitness_value < population[best_idx].fitness_value) {
                best_idx = candidate;
            }
        }
        return population[best_idx];
    }

    // Two-point crossover on flattened bit strings
    pair<Schedule, Schedule> crossover(const Schedule& parentA, const Schedule& parentB) {
        Schedule childA = parentA;
        Schedule childB = parentB;

        if (uni(rng) < crossover_rate) {
            // Pick two cut points
            uniform_int_distribution<int> bit_dist(0, n_bits - 1);
            int p1 = bit_dist(rng);
            int p2 = bit_dist(rng);
            if (p1 > p2) swap(p1, p2);

            // Swap bits between cut points
            for (int b = p1; b <= p2; ++b) {
                int i = b / n_hours;
                int t = b % n_hours;
                childA[i][t] = parentB[i][t];
                childB[i][t] = parentA[i][t];
            }
        }
        return {childA, childB};
    }

    // Bit-flip mutation
    void mutate(Schedule& s) {
        for (int i = 0; i < n_areas; ++i) {
            for (int t = 0; t < n_hours; ++t) {
                if (uni(rng) < mutation_rate) {
                    s[i][t] = 1 - s[i][t];
                }
            }
        }
    }

    // Greedy local search (same as PSO's)
    Schedule local_search(Schedule s, bool verbose = true) {
        double initial_fitness = fitness(s, *demand, *area_configs);
        int total_removed = 0;
        int passes = 0;
        bool improved = true;

        while (improved) {
            improved = false;
            passes++;
            for (int i = 0; i < n_areas; ++i) {
                if ((*area_configs)[i].priority_flag == 1) continue;
                for (int t = 0; t < n_hours; ++t) {
                    if (s[i][t] == 0) continue;

                    double fit_with = fitness(s, *demand, *area_configs);
                    s[i][t] = 0;
                    double fit_without = fitness(s, *demand, *area_configs);

                    if (fit_without <= fit_with) {
                        total_removed++;
                        improved = true;
                    } else {
                        s[i][t] = 1;
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

    // Main run loop
    Schedule run(const vector<vector<double>>& dem,
                 const vector<AreaConfig>& cfg,
                 bool verbose = true,
                const Schedule* seed = nullptr)
    {
        demand = &dem;
        area_configs = &cfg;
        initialize(seed);

        for (int gen = 0; gen < max_generations; ++gen) {
            // Sort population by fitness (lowest first)
            sort(population.begin(), population.end(),
                 [](const Individual& a, const Individual& b) {
                     return a.fitness_value < b.fitness_value;
                 });

            // Build next generation
            vector<Individual> next_gen;
            next_gen.reserve(pop_size);

            // Elitism: keep top elite_count unchanged
            for (int e = 0; e < elite_count; ++e) {
                next_gen.push_back(population[e]);
            }

            // Fill rest with children
            while ((int)next_gen.size() < pop_size) {
                const Individual& parentA = tournament_select();
                const Individual& parentB = tournament_select();
                auto [childA_genes, childB_genes] = crossover(parentA.genes, parentB.genes);

                mutate(childA_genes);
                mutate(childB_genes);
                repair_priority(childA_genes);
                repair_priority(childB_genes);

                Individual childA;
                childA.genes = childA_genes;
                childA.fitness_value = fitness(childA_genes, *demand, *area_configs);
                next_gen.push_back(childA);

                if ((int)next_gen.size() < pop_size) {
                    Individual childB;
                    childB.genes = childB_genes;
                    childB.fitness_value = fitness(childB_genes, *demand, *area_configs);
                    next_gen.push_back(childB);
                }
            }

            population = next_gen;

            // Update global best
            for (auto& ind : population) {
                if (ind.fitness_value < global_best_fitness) {
                    global_best_fitness = ind.fitness_value;
                    global_best = ind.genes;
                }
            }

            if (verbose && (gen % 20 == 0 || gen == max_generations - 1)) {
                cout << "Gen " << gen
                     << " | best fitness: " << global_best_fitness << "\n";
            }
        }

        if (verbose) cout << "\n--- Running local search refinement ---\n";
        Schedule refined = local_search(global_best, verbose);
        global_best = refined;
        global_best_fitness = fitness(refined, *demand, *area_configs);

        return global_best;
    }
};

#endif // GA_H