#ifndef VERIFY_GBT_H
#define VERIFY_GBT_H

#include "gbt.h"
#include "data_io.h"
#include "matrix.h"
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <iostream>

using namespace std;

// =============================================================================
// Honest GBT verification — the same depth of analysis we did for QRF.
// =============================================================================

inline void verify_gbt_on_dates(const GBT& gbt,
                                 const vector<Record>& records,
                                 const vector<string>& area_order,
                                 const vector<string>& test_dates)
{
    cout << "\n=================================================================\n";
    cout << " GBT VERIFICATION ON " << test_dates.size() << " STRESS DAYS\n";
    cout << "=================================================================\n";
    cout << "Date       | Hour | Total Forecast | Total Actual | Error % | Worst Area\n";
    cout << "-----------+------+----------------+--------------+---------+-----------\n";

    double overall_abs_err = 0.0, overall_pct_err = 0.0;
    int overall_n = 0;
    double max_err_pct = 0.0;
    string max_err_loc;

    for (const auto& date : test_dates) {
        vector<string> raw_order;
        auto raw = build_demand_matrix(records, date, raw_order);
        if (raw.empty()) continue;

        vector<vector<double>> actual(area_order.size(), vector<double>(24, 0.0));
        for (size_t i = 0; i < area_order.size(); ++i) {
            for (size_t j = 0; j < raw_order.size(); ++j) {
                if (raw_order[j] == area_order[i]) { actual[i] = raw[j]; break; }
            }
        }

        time_t day_t = parse_datetime(date + " 0:00");
        auto forecast = gbt.forecast_matrix(area_order, day_t);

        // Find peak hour for this day
        int peak_h = 0;
        double peak_actual = 0;
        for (int h = 0; h < 24; ++h) {
            double sum_a = 0;
            for (size_t i = 0; i < area_order.size(); ++i) sum_a += actual[i][h];
            if (sum_a > peak_actual) { peak_actual = sum_a; peak_h = h; }
        }

        // Compute error at peak hour
        double sum_fc = 0, sum_ac = 0;
        double worst_pct = 0;
        string worst_area = "";
        for (size_t i = 0; i < area_order.size(); ++i) {
            // GBT.predict already includes safety buffer — strip it for honest comparison
            double fc_raw = forecast[i][peak_h] / GBT::SAFETY_BUFFER;
            double ac = actual[i][peak_h];
            sum_fc += fc_raw;
            sum_ac += ac;
            if (ac > 0) {
                double pct = fabs(fc_raw - ac) / ac * 100.0;
                if (pct > worst_pct) { worst_pct = pct; worst_area = area_order[i]; }
            }
        }
        double err_pct = (sum_ac > 0) ? fabs(sum_fc - sum_ac) / sum_ac * 100.0 : 0;

        printf("%s |  %2d  | %14.0f | %12.0f | %6.1f%% | %s (%.1f%%)\n",
               date.c_str(), peak_h, sum_fc, sum_ac, err_pct,
               worst_area.c_str(), worst_pct);

        overall_abs_err += fabs(sum_fc - sum_ac);
        overall_pct_err += err_pct;
        overall_n++;

        // Track worst single-area error
        for (size_t i = 0; i < area_order.size(); ++i) {
            for (int h = 0; h < 24; ++h) {
                double fc_raw = forecast[i][h] / GBT::SAFETY_BUFFER;
                double ac = actual[i][h];
                if (ac > 0) {
                    double pct = fabs(fc_raw - ac) / ac * 100.0;
                    if (pct > max_err_pct) {
                        max_err_pct = pct;
                        max_err_loc = date + " hr " + to_string(h) + " " + area_order[i];
                    }
                }
            }
        }
    }

    cout << "\n--- Summary ---\n";
    if (overall_n > 0) {
        printf("  Avg peak-hour MAPE across %d days : %.2f %%\n",
               overall_n, overall_pct_err / overall_n);
        printf("  Worst single (date, hour, area)   : %s (%.1f %% error)\n",
               max_err_loc.c_str(), max_err_pct);
    }
}

// Bias check: does GBT systematically over- or under-predict at high demand?
inline void verify_gbt_calibration(const GBT& gbt,
                                    const vector<Record>& records,
                                    const vector<string>& area_order)
{
    cout << "\n=================================================================\n";
    cout << " GBT CALIBRATION CHECK (bias by demand bucket)\n";
    cout << "=================================================================\n";
    cout << "Demand bucket   | Samples | Mean Pred | Mean Actual | Bias %\n";
    cout << "----------------+---------+-----------+-------------+--------\n";

    // 5 buckets by actual demand
    vector<pair<double,double>> buckets = {
        {0, 5000}, {5000, 10000}, {10000, 15000}, {15000, 20000}, {20000, 999999}
    };

    for (auto& bk : buckets) {
        double sum_pred = 0, sum_actual = 0;
        int n = 0;
        for (const auto& r : records) {
            if (r.demand_mw < bk.first || r.demand_mw >= bk.second) continue;
            bool found = false;
            for (auto& a : area_order) if (a == r.area_id) { found = true; break; }
            if (!found) continue;
            // Strip the safety buffer for honest assessment
            double pred = gbt.predict(r.timestamp, r.area_id) / GBT::SAFETY_BUFFER;
            sum_pred += pred;
            sum_actual += r.demand_mw;
            n++;
            if (n >= 5000) break;  // sample, not exhaustive
        }
        if (n == 0) continue;
        double mp = sum_pred / n, ma = sum_actual / n;
        double bias = (ma > 0) ? (mp - ma) / ma * 100.0 : 0;
        printf("%5.0f - %6.0f  | %7d | %9.0f | %11.0f | %+6.1f%%\n",
               bk.first, bk.second, n, mp, ma, bias);
    }
}

#endif
