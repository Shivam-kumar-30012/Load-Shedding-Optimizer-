#ifndef MULTIDAY_H
#define MULTIDAY_H

#include "data_io.h"
#include "matrix.h"
#include "gbt.h"
#include "fitness.h"
#include "baselines.h"
#include "pso.h"
#include "ga.h"
#include "hybrid.h"
#include "ensemble.h"
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <map>
#include <cstdio>

using namespace std;

struct DayResult {
    string method;
    string date;
    double fitness;
    double damage;
    double overage;
    int    priority_viol;
    bool   failed;
};

struct MethodSummary {
    string method;
    double avg_fitness = 0.0;
    double avg_damage = 0.0;
    double avg_overage = 0.0;
    int    days_failed = 0;
    int    days_tested = 0;
};

inline vector<vector<double>> align_demand(
    const vector<vector<double>>& raw,
    const vector<string>& raw_order,
    const vector<string>& area_order)
{
    vector<vector<double>> aligned(area_order.size(), vector<double>(24, 0.0));
    for (size_t i = 0; i < area_order.size(); ++i) {
        for (size_t j = 0; j < raw_order.size(); ++j) {
            if (raw_order[j] == area_order[i]) {
                aligned[i] = raw[j];
                break;
            }
        }
    }
    return aligned;
}

inline DayResult make_result(const string& method,
                             const string& date,
                             const Schedule& s,
                             const vector<vector<double>>& demand_actual,
                             const vector<AreaConfig>& area_configs)
{
    auto m = compute_metrics(s, demand_actual, area_configs);
    DayResult r;
    r.method = method;
    r.date = date;
    r.fitness = m.fitness_score;
    r.damage = m.damage;
    r.overage = m.capacity_overage;
    r.priority_viol = m.priority_violations;
    r.failed = (m.capacity_overage >= 0.5 || m.priority_violations > 0);
    return r;
}

// Run all 9 methods on one test day
inline vector<DayResult> run_one_day(
    const string& test_date,
    const vector<Record>& test_records,
    const vector<string>& area_order,
    const vector<AreaConfig>& area_configs,
    const GBT& gbt,
    bool verbose = true)
{
    vector<DayResult> results;

    vector<string> raw_order;
    auto raw = build_demand_matrix(test_records, test_date, raw_order);
    if (raw.empty()) {
        if (verbose) cerr << "  [skip] no records for " << test_date << "\n";
        return results;
    }
    auto demand_actual = align_demand(raw, raw_order, area_order);

    time_t day_t = 0;
    for (const auto& r : test_records) {
        if (date_string(r.timestamp) == test_date) { day_t = r.timestamp; break; }
    }
    auto demand_forecast = gbt.forecast_matrix(area_order, day_t);

    int n_areas = (int)demand_actual.size();
    int n_hours = (int)demand_actual[0].size();

    if (verbose) cout << "\n--- Day: " << test_date << " ---\n";

    auto random_s   = random_scheduler(demand_actual, area_configs);
    auto rotation_s = rotation_scheduler(demand_actual, area_configs);
    auto greedy_s   = greedy_scheduler(demand_actual, area_configs);

    BPSO pso_solo(50, 200, n_areas, n_hours);
    auto pso_s = pso_solo.run(demand_actual, area_configs, false);

    GA ga_solo(50, 200, n_areas, n_hours);
    auto ga_s = ga_solo.run(demand_actual, area_configs, false);

    auto hybrid_s = hybrid_pipeline(demand_actual, area_configs, false);
    auto ensemble_r = ensemble_pipeline(demand_actual, area_configs, 2, false);

    // Forecast-based methods (now using GBT)
    auto hybrid_gbt_s   = hybrid_pipeline(demand_forecast, area_configs, false);
    auto ensemble_gbt_r = ensemble_pipeline(demand_forecast, area_configs, 2, false);

    results.push_back(make_result("Random",          test_date, random_s,                       demand_actual, area_configs));
    results.push_back(make_result("Rotation",        test_date, rotation_s,                     demand_actual, area_configs));
    results.push_back(make_result("Greedy",          test_date, greedy_s,                       demand_actual, area_configs));
    results.push_back(make_result("PSO",             test_date, pso_s,                          demand_actual, area_configs));
    results.push_back(make_result("GA",              test_date, ga_s,                           demand_actual, area_configs));
    results.push_back(make_result("Hybrid",          test_date, hybrid_s,                       demand_actual, area_configs));
    results.push_back(make_result("Ensemble",        test_date, ensemble_r.final_schedule,      demand_actual, area_configs));
    results.push_back(make_result("Hybrid+GBT",      test_date, hybrid_gbt_s,                   demand_actual, area_configs));
    results.push_back(make_result("Ensemble+GBT",    test_date, ensemble_gbt_r.final_schedule,  demand_actual, area_configs));

    if (verbose) {
        for (auto& r : results) {
            printf("  %-15s | dmg=%9.0f | over=%6.0f | %s\n",
                   r.method.c_str(), r.damage, r.overage, r.failed ? "FAIL" : "OK");
        }
    }
    return results;
}

inline vector<MethodSummary> aggregate(const vector<DayResult>& all_results) {
    map<string, MethodSummary> agg;
    for (const auto& r : all_results) {
        auto& s = agg[r.method];
        s.method = r.method;
        s.avg_fitness += r.fitness;
        s.avg_damage += r.damage;
        s.avg_overage += r.overage;
        if (r.failed) s.days_failed++;
        s.days_tested++;
    }
    vector<MethodSummary> out;
    for (auto& [k, s] : agg) {
        if (s.days_tested > 0) {
            s.avg_fitness /= s.days_tested;
            s.avg_damage  /= s.days_tested;
            s.avg_overage /= s.days_tested;
        }
        out.push_back(s);
    }
    return out;
}

inline void save_results_csv(const vector<DayResult>& results, const string& path) {
    ofstream out(path);
    if (!out.is_open()) {
        cerr << "ERROR: cannot write " << path << "\n";
        return;
    }
    out << "date,method,fitness,damage,overage,priority_viol,status\n";
    for (const auto& r : results) {
        out << r.date << "," << r.method << ","
            << r.fitness << "," << r.damage << ","
            << r.overage << "," << r.priority_viol << ","
            << (r.failed ? "FAIL" : "OK") << "\n";
    }
    out.close();
    cout << "[Saved] " << path << "\n";
}

inline void save_summary_csv(const vector<MethodSummary>& summary, const string& path) {
    ofstream out(path);
    out << "method,avg_fitness,avg_damage,avg_overage,days_failed,days_tested\n";
    for (const auto& s : summary) {
        out << s.method << ","
            << s.avg_fitness << ","
            << s.avg_damage << ","
            << s.avg_overage << ","
            << s.days_failed << ","
            << s.days_tested << "\n";
    }
    out.close();
    cout << "[Saved] " << path << "\n";
}

#endif // MULTIDAY_H