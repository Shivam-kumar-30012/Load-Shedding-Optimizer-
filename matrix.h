#ifndef MATRIX_H
#define MATRIX_H

#include "data_io.h"
#include <algorithm>

// Convert epoch -> "YYYY-MM-DD"
inline string date_string(time_t t) {
    tm* tm_info = localtime(&t);
    char buf[11];
    strftime(buf, sizeof(buf), "%Y-%m-%d", tm_info);
    return string(buf);
}

// Get hour-of-day (0-23)
inline int hour_of_day(time_t t) {
    tm* tm_info = localtime(&t);
    return tm_info->tm_hour;
}

// Find date with highest total demand
inline string find_peak_day(const vector<Record>& records) {
    map<string, double> daily_total;
    for (const auto& r : records) {
        daily_total[date_string(r.timestamp)] += r.demand_mw;
    }
    string peak_date;
    double peak_value = -1.0;
    for (auto& [date, total] : daily_total) {
        if (total > peak_value) {
            peak_value = total;
            peak_date = date;
        }
    }
    cout << "Peak day: " << peak_date
         << " (total demand " << peak_value << " MW)\n";
    return peak_date;
}

// Build demand[area_idx][hour_idx] matrix for one day
inline vector<vector<double>> build_demand_matrix(
    const vector<Record>& records,
    const string& target_date,
    vector<string>& area_order)
{
    map<string, vector<double>> area_to_hours;
    for (const auto& r : records) {
        if (date_string(r.timestamp) != target_date) continue;
        if (area_to_hours.find(r.area_id) == area_to_hours.end()) {
            area_to_hours[r.area_id] = vector<double>(24, 0.0);
        }
        int h = hour_of_day(r.timestamp);
        area_to_hours[r.area_id][h] = r.demand_mw;
    }

    vector<vector<double>> matrix;
    area_order.clear();
    for (auto& [area, hours] : area_to_hours) {
        area_order.push_back(area);
        matrix.push_back(hours);
    }
    return matrix;
}

#endif // MATRIX_H