#ifndef BASELINES_H
#define BASELINES_H

#include "data_io.h"
#include "fitness.h"
#include <vector>
#include <string>
#include <algorithm>
#include <random>

using namespace std;

// 1. Random Scheduler: Randomly cuts non-priority areas only
inline Schedule random_scheduler(const vector<vector<double>>& demand,
                                 const vector<AreaConfig>& area_configs) {
    int n_areas = demand.size();
    int n_hours = demand[0].size();
    Schedule sched(n_areas, vector<int>(n_hours, 0));

    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, 1);

    for (int i = 0; i < n_areas; ++i) {
        // Priority areas: never cut
        if (area_configs[i].priority_flag == 1) continue;
        for (int t = 0; t < n_hours; ++t) {
            sched[i][t] = dis(gen);
        }
    }
    return sched;
}

// 2. Rotation Scheduler: Rolling blackout, but skips priority areas
inline Schedule rotation_scheduler(const vector<vector<double>>& demand,
                                   const vector<AreaConfig>& area_configs) {
    int n_areas = demand.size();
    int n_hours = demand[0].size();
    Schedule sched(n_areas, vector<int>(n_hours, 0));

    // Build list of cuttable (non-priority) area indices
    vector<int> cuttable;
    for (int i = 0; i < n_areas; ++i) {
        if (area_configs[i].priority_flag == 0) cuttable.push_back(i);
    }
    if (cuttable.empty()) return sched;  // no one to cut

    for (int t = 0; t < n_hours; ++t) {
        int area_to_cut = cuttable[t % cuttable.size()];
        sched[area_to_cut][t] = 1;
    }
    return sched;
}

// 3. Greedy Scheduler: Cut the area with the lowest economic weight until capacity is met
inline Schedule greedy_scheduler(const vector<vector<double>>& demand,
                                 const vector<AreaConfig>& area_configs) {
    int n_areas = demand.size();
    int n_hours = demand[0].size();
    Schedule sched(n_areas, vector<int>(n_hours, 0));

    double total_cap = 0;
    for (const auto& conf : area_configs) total_cap += conf.capacity_mw;

    for (int t = 0; t < n_hours; ++t) {
        double current_demand = 0;
        for (int i = 0; i < n_areas; ++i) current_demand += demand[i][t];

        if (current_demand > total_cap) {
            // Sort areas by economic weight (ascending). Priority areas pushed to end.
            vector<pair<double, int>> weights;
            for (int i = 0; i < n_areas; ++i) {
                double effective_weight = area_configs[i].economic_weight;
                if (area_configs[i].priority_flag == 1) effective_weight += 1000000;
                weights.push_back({effective_weight, i});
            }
            sort(weights.begin(), weights.end());

            double reduced_demand = current_demand;
            for (auto& p : weights) {
                int idx = p.second;
                if (area_configs[idx].priority_flag == 1) break; // never cut priority
                sched[idx][t] = 1;
                reduced_demand -= demand[idx][t];
                if (reduced_demand <= total_cap) break;
            }
        }
    }
    return sched;
}

#endif // BASELINES_H