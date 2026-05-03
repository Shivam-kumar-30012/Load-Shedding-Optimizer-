#include "pipeline.h"
#include "server.h"
#include "gbt.h"
#include "gbt_io.h"
#include "verify_gbt.h"
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <cstring>

using namespace std;
using namespace std::chrono;

void print_usage(const char* progname) {
    cout << "Usage:\n"
         << "  " << progname << "                           Run full pipeline (multi-day GBT test)\n"
         << "  " << progname << " --save-gbt <file>         Train GBT, save\n"
         << "  " << progname << " --load-gbt <file>         Load saved GBT, evaluate on test set\n"
         << "  " << progname << " --serve                   Start web server on localhost:8080\n"
         << "  " << progname << " --port <N>                Use port N (with --serve)\n"
         << "  " << progname << " --test-gbt                Train GBT, evaluate, predict samples\n"
         << "  " << progname << " --verify-gbt              Verify GBT accuracy on stress days\n"
         << "  " << progname << " --diag <YYYY-MM-DD>       Diagnostic: forecast vs actual on a date\n";
}

inline void evaluate_gbt(const GBT& gbt, const vector<Record>& test_records,
                          const vector<string>& area_order)
{
    cout << "\n=================================================================\n";
    cout << " EVALUATE ON TEST SET (HELD-OUT)\n";
    cout << "=================================================================\n";

    double sum_abs_err = 0.0, sum_sq_err = 0.0, sum_pct_err = 0.0;
    double sum_y = 0.0, sum_y_sq = 0.0;
    int n = 0;

    for (const auto& r : test_records) {
        bool found = false;
        for (const auto& a : area_order) if (a == r.area_id) { found = true; break; }
        if (!found) continue;

        // Strip safety buffer for honest accuracy assessment
        double pred = gbt.predict(r.timestamp, r.area_id) / GBT::SAFETY_BUFFER;
        double actual = r.demand_mw;
        if (actual <= 0) continue;

        double err = pred - actual;
        sum_abs_err += fabs(err);
        sum_sq_err  += err * err;
        sum_pct_err += fabs(err) / actual * 100.0;
        sum_y       += actual;
        sum_y_sq    += actual * actual;
        n++;
    }

    if (n == 0) { cout << "[Error] No test samples evaluated.\n"; return; }

    double mae    = sum_abs_err / n;
    double rmse   = sqrt(sum_sq_err / n);
    double mape   = sum_pct_err / n;
    double mean_y = sum_y / n;
    double sstot  = sum_y_sq - n * mean_y * mean_y;
    double r2     = (sstot > 0) ? 1.0 - (sum_sq_err / sstot) : 0.0;

    cout << "  Samples evaluated : " << n << "\n";
    printf("  MAE               : %8.2f MW\n", mae);
    printf("  RMSE              : %8.2f MW\n", rmse);
    printf("  MAPE              : %8.2f %%\n", mape);
    printf("  R-squared         : %8.4f\n", r2);
}

int main(int argc, char* argv[]) {
    auto t_start = steady_clock::now();

    bool serve_mode = false, test_gbt_mode = false;
    bool save_gbt_mode = false, load_gbt_mode = false;
    bool diag_mode = false, verify_gbt_mode = false;
    int  serve_port = 8080;
    string gbt_path, diag_date;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--save-gbt") == 0 && i + 1 < argc) {
            save_gbt_mode = true; gbt_path = argv[i + 1]; ++i;
        } else if (strcmp(argv[i], "--load-gbt") == 0 && i + 1 < argc) {
            load_gbt_mode = true; gbt_path = argv[i + 1]; ++i;
        } else if (strcmp(argv[i], "--serve") == 0) {
            serve_mode = true;
        } else if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            serve_port = atoi(argv[i + 1]); ++i;
        } else if (strcmp(argv[i], "--test-gbt") == 0) {
            test_gbt_mode = true;
        } else if (strcmp(argv[i], "--verify-gbt") == 0) {
            verify_gbt_mode = true;
        } else if (strcmp(argv[i], "--diag") == 0 && i + 1 < argc) {
            diag_mode = true; diag_date = argv[i + 1]; ++i;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]); return 0;
        }
    }

    cout << "\n#################################################################\n";
    cout << "#   LOAD-SHEDDING OPTIMIZER\n";
    if      (save_gbt_mode)    cout << "#   Mode: TRAIN + SAVE GBT MODEL\n";
    else if (load_gbt_mode)    cout << "#   Mode: LOAD GBT MODEL + EVALUATE\n";
    else if (serve_mode)       cout << "#   Mode: WEB SERVER (uses GBT)\n";
    else if (test_gbt_mode)    cout << "#   Mode: GBT TRAIN + TEST + PREDICT\n";
    else if (verify_gbt_mode)  cout << "#   Mode: GBT VERIFICATION\n";
    else if (diag_mode)        cout << "#   Mode: DIAGNOSTIC for " << diag_date << "\n";
    else                       cout << "#   Mode: FULL PIPELINE (multi-day GBT test)\n";
    cout << "#################################################################\n";

    if (serve_mode) {
        start_server(serve_port, "gbt_model.bin");
        return 0;
    }

    vector<string> input_files = {
        "data/AEP_hourly.csv","data/COMED_hourly.csv","data/DAYTON_hourly.csv",
        "data/DOM_hourly.csv","data/DUQ_hourly.csv"
    };

    auto raw_records   = stage1_load(input_files);
    auto clean_records = stage2_impute(raw_records);
    auto config        = stage3_configure(clean_records, "data/areas_config.csv");
    if (config.area_configs.empty()) return 1;

    // ============================================================
    // VERIFY GBT MODE
    // ============================================================
    if (verify_gbt_mode) {
        GBT gbt(150, 6, 20, 0.05, 0.7, 8);
        if (!load_gbt(gbt, "gbt_model.bin")) {
            cerr << "Run --save-gbt first\n";
            return 1;
        }
        vector<string> stress_dates = {
            "2010-08-04","2010-12-15","2011-04-20","2011-07-21",
            "2012-01-18","2012-07-17","2012-11-08"
        };
        verify_gbt_on_dates(gbt, clean_records, config.area_order, stress_dates);
        verify_gbt_calibration(gbt, clean_records, config.area_order);
        cout << "\n[Done]\n";
        return 0;
    }

    // ============================================================
    // DIAGNOSTIC MODE
    // ============================================================
    if (diag_mode) {
        cout << "\n=================================================================\n";
        cout << " DIAGNOSTIC: " << diag_date << "\n";
        cout << "=================================================================\n";

        GBT gbt(150, 6, 20, 0.05, 0.7, 8);
        if (!load_gbt(gbt, "gbt_model.bin")) {
            cerr << "Run --save-gbt first.\n";
            return 1;
        }

        time_t day_t = parse_datetime(diag_date + " 0:00");
        if (day_t == -1) { cerr << "Bad date format\n"; return 1; }

        auto demand_forecast = gbt.forecast_matrix(config.area_order, day_t);

        vector<string> raw_order;
        auto raw = build_demand_matrix(clean_records, diag_date, raw_order);
        vector<vector<double>> demand_actual(config.area_order.size(), vector<double>(24, 0.0));
        bool actual_found = false;
        for (size_t i = 0; i < config.area_order.size(); ++i) {
            for (size_t j = 0; j < raw_order.size(); ++j) {
                if (raw_order[j] == config.area_order[i]) {
                    demand_actual[i] = raw[j];
                    actual_found = true;
                    break;
                }
            }
        }

        if (!actual_found) cout << "\n[No actual data for this date]\n";

        double total_cap = 0;
        for (auto& c : config.area_configs) total_cap += c.capacity_mw;

        cout << "\nHour | Forecast Sum | Actual Sum | Capacity | Overage(actual)\n";
        cout << "-----+--------------+------------+----------+----------------\n";
        for (int h = 0; h < 24; ++h) {
            double sum_fc = 0, sum_ac = 0;
            for (size_t i = 0; i < config.area_order.size(); ++i) {
                sum_fc += demand_forecast[i][h];
                sum_ac += demand_actual[i][h];
            }
            double over = sum_ac - total_cap;
            printf(" %2d  | %12.0f | %10.0f | %8.0f | %s%.0f\n",
                   h, sum_fc, sum_ac, total_cap, over > 0 ? "+" : " ", over);
        }

        auto sched = hybrid_pipeline(demand_forecast, config.area_configs, false);

        if (actual_found) {
            auto m_fc = compute_metrics(sched, demand_forecast, config.area_configs);
            auto m_ac = compute_metrics(sched, demand_actual, config.area_configs);
            cout << "\n--- On FORECAST ---\n";
            printf("  damage=%.0f overage=%.0f\n", m_fc.damage, m_fc.capacity_overage);
            cout << "--- On ACTUAL ---\n";
            printf("  damage=%.0f overage=%.0f\n", m_ac.damage, m_ac.capacity_overage);
        }

        cout << "\nSchedule (X = cut, . = kept):\n";
        print_schedule(sched, config.area_configs);
        return 0;
    }

    // ============================================================
    // SAVE GBT MODE
    // ============================================================
    if (save_gbt_mode) {
        cout << "\n=================================================================\n";
        cout << " STAGE: TRAIN GBT ON FULL 14 YEARS\n";
        cout << "=================================================================\n";
        GBT gbt(150, 6, 20, 0.05, 0.7, 8);
        gbt.train(clean_records, config.area_order);
        if (save_gbt(gbt, gbt_path)) cout << "[Done] GBT saved.\n";
        else                          cerr << "[Error] Save failed.\n";
        return 0;
    }

    // ============================================================
    // LOAD GBT MODE — eval on test set
    // ============================================================
    if (load_gbt_mode) {
        GBT gbt(150, 6, 20, 0.05, 0.7, 8);
        if (!load_gbt(gbt, gbt_path)) return 1;
        auto split = stage4_split(clean_records);
        evaluate_gbt(gbt, split.test, config.area_order);
        return 0;
    }

    // ============================================================
    // GBT TEST MODE — train on 80%, eval on 20%
    // ============================================================
    if (test_gbt_mode) {
        auto split = stage4_split(clean_records);
        GBT gbt(150, 6, 20, 0.05, 0.7, 8);
        gbt.train(split.train, config.area_order);
        evaluate_gbt(gbt, split.test, config.area_order);
        return 0;
    }

    // ============================================================
    // FULL MULTI-DAY PIPELINE (now uses GBT instead of QRF)
    // ============================================================
    auto split = stage4_split(clean_records);
    auto gbt = stage5_train(split, config.area_order);

    vector<string> test_dates = {
        "2010-08-04","2010-12-15","2011-04-20","2011-07-21",
        "2012-01-18","2012-07-17","2012-11-08"
    };
    auto results    = stage6_optimize(split.test, config.area_order,
                                      config.area_configs, gbt, test_dates);
    auto final_data = stage7_aggregate(results);
    stage8_output(results, final_data);

    double total = duration_cast<milliseconds>(steady_clock::now() - t_start).count() / 1000.0;
    cout << "\n[Total runtime] " << total << "s\n";
    return 0;
}