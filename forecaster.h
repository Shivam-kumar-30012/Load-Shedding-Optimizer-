#ifndef FORECASTER_H
#define FORECASTER_H

#include "data_io.h"
#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <ctime>
#include <cmath>

using namespace std;

// =============================================================================
// Lightweight short-term forecaster for Flow 2.
//
// Method: hour-of-day × day-of-week historical lookup with
//   - winsorization (cap outliers at 95th percentile)
//   - exponential smoothing (recent days weighted more)
//   - small safety buffer (1.05x) for honest deployment
//
// Trains in milliseconds. Forecasts in milliseconds.
// No ML, no QRF. Pure statistical baseline.
// =============================================================================

struct LiteForecaster {
    // demand_history[(area, dow, hour)] = vector of past values, oldest→newest
    map<tuple<string, int, int>, vector<double>> history;

    static constexpr int HISTORY_DAYS = 14;     // look back 14 days
    static constexpr double ALPHA = 0.4;        // exponential smoothing factor
    static constexpr double WINSOR_PCTL = 0.95; // cap top 5% as outliers
    static constexpr double SAFETY_BUFFER = 1.05;

    // Build history from user's records, looking back from target_date
    void build(const vector<Record>& records, time_t target_date) {
        history.clear();

        // Keep only records within the lookback window
        time_t window_start = target_date - (time_t)HISTORY_DAYS * 86400;

        for (const auto& r : records) {
            if (r.timestamp >= window_start && r.timestamp < target_date) {
                tm* tm_info = localtime(&r.timestamp);
                int dow = tm_info->tm_wday;
                int hour = tm_info->tm_hour;
                history[{r.area_id, dow, hour}].push_back(r.demand_mw);
            }
        }
    }

    // Apply winsorization: cap outliers at the 95th percentile
    static void winsorize(vector<double>& values) {
        if (values.size() < 4) return;  // too few to bother
        vector<double> sorted = values;
        sort(sorted.begin(), sorted.end());
        int idx = (int)(WINSOR_PCTL * sorted.size());
        if (idx >= (int)sorted.size()) idx = sorted.size() - 1;
        double cap = sorted[idx];
        for (auto& v : values) if (v > cap) v = cap;
    }

    // Apply exponential smoothing: oldest first, newest last
    static double exp_smooth(const vector<double>& values) {
        if (values.empty()) return 0.0;
        double f = values[0];
        for (size_t i = 1; i < values.size(); ++i) {
            f = ALPHA * values[i] + (1.0 - ALPHA) * f;
        }
        return f;
    }

    // Predict demand for one (area, day-of-week, hour) cell
    double predict(const string& area_id, int dow, int hour) const {
        auto it = history.find({area_id, dow, hour});
        if (it == history.end() || it->second.empty()) {
            // Fallback: use any same-hour data from this area (ignore dow)
            double sum = 0; int n = 0;
            for (int d = 0; d < 7; ++d) {
                auto it2 = history.find({area_id, d, hour});
                if (it2 != history.end()) {
                    for (double v : it2->second) { sum += v; n++; }
                }
            }
            return (n > 0) ? (sum / n) * SAFETY_BUFFER : 0.0;
        }

        vector<double> values = it->second;
        winsorize(values);
        double f = exp_smooth(values);
        return f * SAFETY_BUFFER;
    }

    // Build a 24-hour forecast matrix for all areas on target_date
    vector<vector<double>> forecast_matrix(
        const vector<string>& area_order,
        time_t target_date) const
    {
        tm* tm_info = localtime(&target_date);
        tm midnight = *tm_info;
        midnight.tm_hour = 0; midnight.tm_min = 0; midnight.tm_sec = 0;
        time_t day_start = mktime(&midnight);

        int n_areas = (int)area_order.size();
        vector<vector<double>> matrix(n_areas, vector<double>(24, 0.0));

        for (int i = 0; i < n_areas; ++i) {
            for (int h = 0; h < 24; ++h) {
                time_t hour_t = day_start + h * 3600;
                tm* th = localtime(&hour_t);
                matrix[i][h] = predict(area_order[i], th->tm_wday, h);
            }
        }
        return matrix;
    }

    // Diagnostic: how many history points per (area, dow, hour) on average
    int total_history_points() const {
        int total = 0;
        for (const auto& [k, v] : history) total += (int)v.size();
        return total;
    }
};

#endif // FORECASTER_H