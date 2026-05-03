#ifndef PIPELINE_H
#define PIPELINE_H

#include "data_io.h"
#include "matrix.h"
#include "preprocessing.h"
#include "fitness.h"
#include "baselines.h"
#include "pso.h"
#include "ga.h"
#include "hybrid.h"
#include "ensemble.h"
#include "gbt.h"
#include "multiday.h"
#include <vector>
#include <string>
#include <set>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <cstdio>

using namespace std;
using namespace std::chrono;

inline void log_stage(int stage_no, const string& title) {
    cout << "\n";
    cout << "=================================================================\n";
    cout << " STAGE " << stage_no << ": " << title << "\n";
    cout << "=================================================================\n";
}

inline double seconds_since(steady_clock::time_point t) {
    return duration_cast<milliseconds>(steady_clock::now() - t).count() / 1000.0;
}

// STAGE 1: Data Ingestion
inline vector<Record> stage1_load(const vector<string>& files) {
    log_stage(1, "DATA INGESTION");
    auto t0 = steady_clock::now();

    vector<Record> all_records;
    for (const auto& file : files) {
        auto recs = load_csv(file);
        all_records.insert(all_records.end(), recs.begin(), recs.end());
    }
    sort(all_records.begin(), all_records.end(),
         [](const Record& a, const Record& b) { return a.timestamp < b.timestamp; });

    cout << "  [Stage 1 input ] " << files.size() << " CSV files\n";
    cout << "  [Stage 1 output] " << all_records.size() << " sorted records\n";
    cout << "  [Stage 1 time  ] " << seconds_since(t0) << "s\n";
    return all_records;
}

// STAGE 2: Imputation
inline vector<Record> stage2_impute(const vector<Record>& raw) {
    log_stage(2, "MISSING-VALUE IMPUTATION");
    auto t0 = steady_clock::now();

    auto imputed = impute_missing(raw, false);

    int delta = (int)imputed.size() - (int)raw.size();
    cout << "  [Stage 2 input ] " << raw.size() << " records\n";
    cout << "  [Stage 2 output] " << imputed.size() << " records (+" << delta << " imputed)\n";
    cout << "  [Stage 2 time  ] " << seconds_since(t0) << "s\n";
    return imputed;
}

// STAGE 3: Configuration
struct StageConfig {
    vector<string> area_order;
    vector<AreaConfig> area_configs;
};

inline StageConfig stage3_configure(const vector<Record>& records, const string& cfg_path) {
    log_stage(3, "FEATURE / AREA CONFIGURATION");
    auto t0 = steady_clock::now();

    StageConfig cfg;
    set<string> seen;
    for (const auto& r : records) {
        if (seen.insert(r.area_id).second) cfg.area_order.push_back(r.area_id);
    }
    sort(cfg.area_order.begin(), cfg.area_order.end());

    auto configs_map = load_areas_config(cfg_path);
    for (const auto& a : cfg.area_order) {
        if (configs_map.count(a) == 0) {
            cerr << "  ERROR: missing config for " << a << "\n";
            cfg.area_configs.clear();
            return cfg;
        }
        cfg.area_configs.push_back(configs_map[a]);
    }

    cout << "  [Stage 3 input ] " << records.size() << " records, config file " << cfg_path << "\n";
    cout << "  [Stage 3 output] " << cfg.area_order.size() << " areas configured: ";
    for (auto& a : cfg.area_order) cout << a << " ";
    cout << "\n";
    cout << "  [Stage 3 time  ] " << seconds_since(t0) << "s\n";
    return cfg;
}

// STAGE 4: Train/Test Split
inline DataSplit stage4_split(const vector<Record>& records) {
    log_stage(4, "TRAIN / TEST SPLIT (80/20 middle)");
    auto t0 = steady_clock::now();

    auto split = split_middle_20(records, false);

    cout << "  [Stage 4 input ] " << records.size() << " records\n";
    cout << "  [Stage 4 output] train=" << split.train.size()
         << ", test=" << split.test.size() << "\n";
    if (!split.test.empty()) {
        cout << "  [Stage 4 window] "
             << date_string(split.test.front().timestamp) << " to "
             << date_string(split.test.back().timestamp)  << "\n";
    }
    cout << "  [Stage 4 time  ] " << seconds_since(t0) << "s\n";
    return split;
}

// STAGE 5: Train GBT
inline GBT stage5_train(const DataSplit& split,
                        const vector<string>& area_order)
{
    log_stage(5, "MODEL TRAINING (GBT)");
    auto t0 = steady_clock::now();

    GBT gbt(150, 6, 20, 0.05, 0.7, 8);
    gbt.train(split.train, area_order);

    cout << "  [Stage 5 input ] " << split.train.size() << " training records\n";
    cout << "  [Stage 5 output] GBT model: " << gbt.trees.size() << " trees\n";
    cout << "  [Stage 5 time  ] " << seconds_since(t0) << "s\n";
    return gbt;
}

// STAGE 6: Multi-day optimization (uses GBT)
inline vector<DayResult> stage6_optimize(
    const vector<Record>& test_records,
    const vector<string>& area_order,
    const vector<AreaConfig>& area_configs,
    const GBT& gbt,
    const vector<string>& test_dates)
{
    log_stage(6, "MULTI-DAY OPTIMIZATION");
    auto t0 = steady_clock::now();

    vector<DayResult> all_results;
    for (const auto& d : test_dates) {
        auto day_results = run_one_day(d, test_records, area_order,
                                       area_configs, gbt, true);
        all_results.insert(all_results.end(), day_results.begin(), day_results.end());
    }

    cout << "  [Stage 6 input ] " << test_dates.size() << " test days, 9 methods each\n";
    cout << "  [Stage 6 output] " << all_results.size() << " (day, method) results\n";
    cout << "  [Stage 6 time  ] " << seconds_since(t0) << "s\n";
    return all_results;
}

// STAGE 7: Aggregate + Score
struct PipelineFinal {
    vector<MethodSummary> summary;
    double reliability_pct;
    double improvement_pct;
    double deployment_pct;
    double overall_pct;
};

inline PipelineFinal stage7_aggregate(const vector<DayResult>& results) {
    log_stage(7, "AGGREGATION + SCORING");
    auto t0 = steady_clock::now();

    PipelineFinal out;
    out.summary = aggregate(results);

    double random_dmg = 0, hybrid_dmg = 0, hyb_gbt_dmg = 0;
    int hybrid_fails = 0, days_total = 0;
    for (const auto& s : out.summary) {
        if (s.method == "Random")     random_dmg = s.avg_damage;
        if (s.method == "Hybrid") {
            hybrid_dmg = s.avg_damage;
            hybrid_fails = s.days_failed;
            days_total = s.days_tested;
        }
        if (s.method == "Hybrid+GBT") hyb_gbt_dmg = s.avg_damage;
    }

    out.reliability_pct = (days_total > 0)
        ? 100.0 * (1.0 - (double)hybrid_fails / days_total) : 0.0;

    out.improvement_pct = 0.0;
    if (random_dmg > 0) {
        out.improvement_pct = 100.0 * (1.0 - hybrid_dmg / random_dmg);
        if (out.improvement_pct < 0)   out.improvement_pct = 0.0;
        if (out.improvement_pct > 100) out.improvement_pct = 100.0;
    }

    out.deployment_pct = 0.0;
    if (hyb_gbt_dmg > 0) {
        out.deployment_pct = 100.0 * (hybrid_dmg / hyb_gbt_dmg);
        if (out.deployment_pct > 100) out.deployment_pct = 100.0;
    } else if (hybrid_dmg == 0 && hyb_gbt_dmg == 0) {
        out.deployment_pct = 100.0;
    }

    out.overall_pct = (out.reliability_pct + out.improvement_pct + out.deployment_pct) / 3.0;

    cout << "  [Stage 7 input ] " << results.size() << " (day, method) rows\n";
    cout << "  [Stage 7 output] " << out.summary.size() << " method summaries + score\n";
    cout << "  [Stage 7 time  ] " << seconds_since(t0) << "s\n";
    return out;
}

// STAGE 8: Save + display
inline void stage8_output(const vector<DayResult>& results,
                          const PipelineFinal& final_data)
{
    log_stage(8, "EXPORT + FINAL REPORT");
    auto t0 = steady_clock::now();

    save_results_csv(results, "results_per_day.csv");
    save_summary_csv(final_data.summary, "results_summary.csv");

    cout << "\n========== AVERAGED RESULTS ==========\n";
    cout << "Method           | Avg Damage  | Avg Overage | Days Failed\n";
    cout << "-----------------+-------------+-------------+-------------\n";

    vector<string> method_order = {
        "Random","Rotation","Greedy","PSO","GA",
        "Hybrid","Ensemble","Hybrid+GBT","Ensemble+GBT"
    };
    for (const auto& m_name : method_order) {
        for (const auto& s : final_data.summary) {
            if (s.method == m_name) {
                printf("%-16s | %11.0f | %11.0f |   %d / %d\n",
                       s.method.c_str(),
                       s.avg_damage,
                       s.avg_overage,
                       s.days_failed,
                       s.days_tested);
                break;
            }
        }
    }

    cout << "\n========== PROJECT PERFORMANCE SCORE ==========\n";
    printf("  Reliability     (no blackouts)       : %5.1f %%\n", final_data.reliability_pct);
    printf("  Improvement     (vs Random baseline) : %5.1f %%\n", final_data.improvement_pct);
    printf("  Deployment      (real vs oracle)     : %5.1f %%\n", final_data.deployment_pct);
    cout <<   "  ------------------------------------------\n";
    printf("  OVERALL PROJECT SCORE                : %5.1f %%\n", final_data.overall_pct);
    cout << "  [Stage 8 time  ] " << seconds_since(t0) << "s\n";
}

#endif