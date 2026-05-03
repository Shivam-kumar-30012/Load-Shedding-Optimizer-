#ifndef PREPROCESSING_H
#define PREPROCESSING_H

#include "data_io.h"


#include "matrix.h"
#include <vector>
#include <map>
#include <string>
#include <iostream>
#include <algorithm>
#include<set>

using namespace std;

// =============================================================================
// Stage 2: Missing-value imputation
// Fill any missing (area, hour) slots with average of same (area, hour, dow) from history
// =============================================================================

// Get day of week (0=Sun..6=Sat)
inline int day_of_week(time_t t) {
    tm* tm_info = localtime(&t);
    return tm_info->tm_wday;
}

// Returns a NEW vector with imputed records added (originals preserved).
// For every (area, date) pair, ensures all 24 hours exist.
// If any are missing, imputes them using the historical mean of same (area, hour, dow).
inline vector<Record> impute_missing(const vector<Record>& records, bool verbose = true) {
    if (verbose) cout << "\n[Stage 2: Imputation] Scanning for missing hours...\n";

    // Step 1: build the historical mean lookup table
    // Key: (area_id, day_of_week, hour) -> running sum + count
    struct Accum { double sum = 0.0; int count = 0; };
    map<tuple<string, int, int>, Accum> hist;

    for (const auto& r : records) {
        auto key = make_tuple(r.area_id, day_of_week(r.timestamp), hour_of_day(r.timestamp));
        hist[key].sum += r.demand_mw;
        hist[key].count++;
    }

    // Step 2: detect missing slots per (area, date)
    // Group records by (area_id, date_string)
    map<pair<string, string>, vector<bool>> hour_present;  // 24 booleans per (area,date)
    map<pair<string, string>, time_t> sample_timestamp;     // any timestamp from that day

    for (const auto& r : records) {
        auto key = make_pair(r.area_id, date_string(r.timestamp));
        if (hour_present.find(key) == hour_present.end()) {
            hour_present[key] = vector<bool>(24, false);
            sample_timestamp[key] = r.timestamp;
        }
        int h = hour_of_day(r.timestamp);
        hour_present[key][h] = true;
    }

    // Step 3: for each (area, date), fill missing hours with imputed values
    vector<Record> imputed;
    int imputed_count = 0;
    int total_missing = 0;
    int unable_to_impute = 0;

    for (auto& [key, present] : hour_present) {
        const string& area = key.first;
        time_t day_ts = sample_timestamp[key];
        int dow = day_of_week(day_ts);

        // Compute the date's midnight timestamp so we can construct hourly times
        tm* tm_info = localtime(&day_ts);
        tm midnight = *tm_info;
        midnight.tm_hour = 0;
        midnight.tm_min = 0;
        midnight.tm_sec = 0;
        time_t day_start = mktime(&midnight);

        for (int h = 0; h < 24; ++h) {
            if (present[h]) continue;
            total_missing++;

            // Look up historical mean for this (area, dow, hour)
            auto hist_key = make_tuple(area, dow, h);
            auto it = hist.find(hist_key);
            if (it == hist.end() || it->second.count == 0) {
                unable_to_impute++;
                continue;
            }

            Record r;
            r.area_id = area;
            r.timestamp = day_start + h * 3600;
            r.demand_mw = it->second.sum / it->second.count;
            imputed.push_back(r);
            imputed_count++;
        }
    }

    if (verbose) {
        cout << "[Stage 2: Imputation] Missing hours found: " << total_missing
             << ", imputed: " << imputed_count
             << ", unable to impute: " << unable_to_impute << "\n";
    }

    // Step 4: combine original + imputed records
    vector<Record> result = records;
    result.insert(result.end(), imputed.begin(), imputed.end());

    // Re-sort by timestamp for downstream stages
    sort(result.begin(), result.end(),
         [](const Record& a, const Record& b) { return a.timestamp < b.timestamp; });

    return result;
}

// =============================================================================
// Stage 4: Train/Test split — 20% from the middle, 80% (split front/back) train
// =============================================================================

struct DataSplit {
    vector<Record> train;
    vector<Record> test;
};

inline DataSplit split_middle_20(const vector<Record>& records, bool verbose = true) {
    if (records.empty()) return {};

    // Records assumed sorted by timestamp.
    // Test window = middle 20% of total time range.
    time_t t_min = records.front().timestamp;
    time_t t_max = records.back().timestamp;
    long long total_seconds = (long long)(t_max - t_min);
    long long test_window_seconds = total_seconds * 20 / 100;
    long long mid_seconds = total_seconds / 2;

    time_t test_start = t_min + (mid_seconds - test_window_seconds / 2);
    time_t test_end = test_start + test_window_seconds;

    DataSplit split;
    split.train.reserve(records.size() * 80 / 100);
    split.test.reserve(records.size() * 20 / 100);

    for (const auto& r : records) {
        if (r.timestamp >= test_start && r.timestamp < test_end) {
            split.test.push_back(r);
        } else {
            split.train.push_back(r);
        }
    }

    if (verbose) {
        cout << "[Stage 4: Split] Train: " << split.train.size()
             << " records | Test: " << split.test.size() << " records\n";
        cout << "[Stage 4: Split] Test window: "
             << date_string(test_start) << " to " << date_string(test_end) << "\n";
    }
    return split;
}

// Get unique dates from a record set (sorted, ascending)
inline vector<string> unique_dates(const vector<Record>& records) {
    set<string> dates_set;
    for (const auto& r : records) dates_set.insert(date_string(r.timestamp));
    return vector<string>(dates_set.begin(), dates_set.end());
}

#endif // PREPROCESSING_H