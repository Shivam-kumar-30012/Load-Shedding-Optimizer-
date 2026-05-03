#ifndef GBT_FEATURES_H
#define GBT_FEATURES_H

#include "data_io.h"
#include <vector>
#include <string>
#include <cmath>
#include <ctime>
#include <map>

using namespace std;

// =============================================================================
// Time-only feature builder for GBT.
// NO lag features — every input is computable from the date alone.
// This is what makes GBT able to predict ANY date, including the future.
// =============================================================================

constexpr double GBT_PI = 3.14159265358979323846;

struct GBTFeatureBuilder {
    vector<string> area_order;
    map<string, int> area_to_idx;
    int reference_year = 2004;  // Used to normalize the year feature

    void set_areas(const vector<string>& areas) {
        area_order = areas;
        area_to_idx.clear();
        for (size_t i = 0; i < areas.size(); ++i) area_to_idx[areas[i]] = (int)i;
    }

    int n_features() const {
        // 12 time features + N area one-hot
        return 12 + (int)area_order.size();
    }

    static int season_of_month(int month) {
        if (month == 12 || month == 1 || month == 2) return 0; // winter
        if (month >= 3 && month <= 5)  return 1;                // spring
        if (month >= 6 && month <= 8)  return 2;                // summer
        return 3;                                                // fall
    }

    // Build feature vector for any (timestamp, area). No data lookup needed.
    vector<double> build(time_t t, const string& area_id) const {
        tm* tm_info = localtime(&t);
        int hour     = tm_info->tm_hour;            // 0-23
        int dow      = tm_info->tm_wday;            // 0-6
        int month    = tm_info->tm_mon + 1;         // 1-12
        int year     = tm_info->tm_year + 1900;
        int day_of_year = tm_info->tm_yday;         // 0-365
        int is_weekend  = (dow == 0 || dow == 6) ? 1 : 0;
        int season   = season_of_month(month);

        // Cyclical encoding for periodic features (helps tree learn smooth patterns)
        double hour_sin = sin(2 * GBT_PI * hour / 24.0);
        double hour_cos = cos(2 * GBT_PI * hour / 24.0);
        double dow_sin  = sin(2 * GBT_PI * dow  / 7.0);
        double dow_cos  = cos(2 * GBT_PI * dow  / 7.0);
        double doy_sin  = sin(2 * GBT_PI * day_of_year / 365.0);
        double doy_cos  = cos(2 * GBT_PI * day_of_year / 365.0);

        // Year normalized as years-since-reference (lets tree learn yearly trend)
        double year_norm = (double)(year - reference_year);

        vector<double> f;
        f.reserve(n_features());
        f.push_back((double)hour);
        f.push_back((double)dow);
        f.push_back((double)month);
        f.push_back((double)season);
        f.push_back((double)is_weekend);
        f.push_back(year_norm);
        f.push_back(hour_sin);
        f.push_back(hour_cos);
        f.push_back(dow_sin);
        f.push_back(dow_cos);
        f.push_back(doy_sin);
        f.push_back(doy_cos);

        // Area one-hot
        for (const auto& a : area_order) f.push_back(a == area_id ? 1.0 : 0.0);
        return f;
    }
};

#endif // GBT_FEATURES_H