#ifndef FITNESS_H
#define FITNESS_H

#include "data_io.h"
#include <vector>
#include <cmath>
#include <algorithm>

using namespace std;

typedef vector<vector<int>> Schedule;


struct FitnessWeights {
    double capacity_penalty     = 10000.0;
    double priority_penalty     = 100000.0;
    double damage_weight        = 1.0;
    double streak_penalty       = 50000.0;
    int    max_consecutive_cuts = 4;
};

struct ScheduleMetrics {
    double damage = 0.0;
    double capacity_overage = 0.0;
    int priority_violations = 0;
    double fitness_score = 0.0;
};

inline double fitness(const Schedule& schedule,
                      const vector<vector<double>>& demand,
                      const vector<AreaConfig>& area_configs,
                      const FitnessWeights& w = FitnessWeights())
{
    int n_areas = demand.size();
    int n_hours = demand[0].size();
    double damage = 0.0;
    double capacity_violations = 0.0;
    double priority_violations = 0.0;

    for (int t = 0; t < n_hours; ++t) {
        double current_grid_load = 0.0;
        double total_capacity_t = 0.0;

        for (int i = 0; i < n_areas; ++i) {
            total_capacity_t += area_configs[i].capacity_mw;
            if (schedule[i][t] == 1) {
                damage += area_configs[i].economic_weight * demand[i][t];
                if (area_configs[i].priority_flag == 1) priority_violations += 1.0;
            } else {
                current_grid_load += demand[i][t];
            }
        }

        double over = current_grid_load - total_capacity_t;
        if (over > 0) {
            capacity_violations += over * over;
        }
    }

    double streak_violations = 0.0;
    for (int i = 0; i < n_areas; ++i) {
        int run = 0;
        for (int t = 0; t < n_hours; ++t) {
            if (schedule[i][t] == 1) {
                run++;
                if (run > w.max_consecutive_cuts) {
                    streak_violations += 1.0;
                }
            } else {
                run = 0;
            }
        }
    }

    return (damage * w.damage_weight)
         + (w.capacity_penalty * capacity_violations)
         + (w.priority_penalty * priority_violations)
         + (w.streak_penalty   * streak_violations);
}

inline ScheduleMetrics compute_metrics(const Schedule& schedule,
                                        const vector<vector<double>>& demand,
                                        const vector<AreaConfig>& area_configs,
                                        const FitnessWeights& w = FitnessWeights())
{
    ScheduleMetrics m;
    int n_areas = demand.size();
    int n_hours = demand[0].size();

    for (int t = 0; t < n_hours; ++t) {
        double current_grid_load = 0.0;
        double total_capacity_t = 0.0;
        for (int i = 0; i < n_areas; ++i) {
            total_capacity_t += area_configs[i].capacity_mw;
            if (schedule[i][t] == 1) {
                m.damage += area_configs[i].economic_weight * demand[i][t];
                if (area_configs[i].priority_flag == 1) m.priority_violations += 1;
            } else {
                current_grid_load += demand[i][t];
            }
        }
        double over = current_grid_load - total_capacity_t;
        if (over > 0) m.capacity_overage += over;
    }

    m.fitness_score = fitness(schedule, demand, area_configs, w);
    return m;
}

inline void print_schedule(const Schedule& schedule,
                            const vector<AreaConfig>& area_configs)
{
    int n_areas = schedule.size();
    int n_hours = schedule[0].size();
    for (int i = 0; i < n_areas; ++i) {
        cout << area_configs[i].area_id << ": ";
        for (int t = 0; t < n_hours; ++t) {
            cout << (schedule[i][t] == 1 ? "X" : ".");
        }
        cout << "\n";
    }
}

#endif