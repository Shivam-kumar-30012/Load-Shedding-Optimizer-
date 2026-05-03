#ifndef DATA_IO_H
#define DATA_IO_H

#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <sstream>
#include <iostream>
#include <ctime>
#include <cstdlib>

using namespace std;

// =============================================================================
// Core data types and CSV loaders.
// =============================================================================

struct Record {
    string  area_id;
    time_t  timestamp;
    double  demand_mw;
};

struct AreaConfig {
    string  area_id;
    double  capacity_mw;
    double  economic_weight;
    int     priority_flag;
    string  area_type;
};

// Parse "MM/DD/YYYY H:MM" or "YYYY-MM-DD HH:MM:SS"
inline time_t parse_datetime(const string& s) {
    if (s.empty()) return -1;
    struct tm tm_info = {};

    if (s.find('/') != string::npos) {
        // MM/DD/YYYY H:MM
        int month, day, year, hour = 0, minute = 0;
        if (sscanf(s.c_str(), "%d/%d/%d %d:%d",
                   &month, &day, &year, &hour, &minute) < 3) return -1;
        tm_info.tm_year = year - 1900;
        tm_info.tm_mon  = month - 1;
        tm_info.tm_mday = day;
        tm_info.tm_hour = hour;
        tm_info.tm_min  = minute;
    } else {
        // YYYY-MM-DD HH:MM:SS  (or with single-digit hour)
        int year, month, day, hour = 0, minute = 0, second = 0;
        if (sscanf(s.c_str(), "%d-%d-%d %d:%d:%d",
                   &year, &month, &day, &hour, &minute, &second) < 3) return -1;
        tm_info.tm_year = year - 1900;
        tm_info.tm_mon  = month - 1;
        tm_info.tm_mday = day;
        tm_info.tm_hour = hour;
        tm_info.tm_min  = minute;
        tm_info.tm_sec  = second;
    }
    tm_info.tm_isdst = -1;
    return mktime(&tm_info);
}

// Extract area_id from filename: "data/AEP_hourly.csv" -> "AEP"
inline string area_from_filename(const string& path) {
    size_t slash = path.find_last_of("/\\");
    string fname = (slash != string::npos) ? path.substr(slash + 1) : path;
    size_t under = fname.find('_');
    if (under != string::npos) return fname.substr(0, under);
    size_t dot = fname.find('.');
    return (dot != string::npos) ? fname.substr(0, dot) : fname;
}

// Load PJM-style CSV (Datetime, AREA_MW)
inline vector<Record> load_csv(const string& path) {
    vector<Record> records;
    ifstream file(path);
    if (!file.is_open()) {
        cerr << "ERROR: cannot open " << path << "\n";
        return records;
    }

    string area_id = area_from_filename(path);
    string line;
    int rows = 0, loaded = 0, skipped = 0;
    bool first = true;

    while (getline(file, line)) {
        if (first) { first = false; continue; }
        if (line.empty()) continue;
        rows++;

        size_t comma = line.find(',');
        if (comma == string::npos) { skipped++; continue; }

        string dt = line.substr(0, comma);
        string mw_str = line.substr(comma + 1);

        // Trim whitespace and \r
        while (!dt.empty() && (dt.back() == ' ' || dt.back() == '\r')) dt.pop_back();
        while (!mw_str.empty() && (mw_str.back() == ' ' || mw_str.back() == '\r')) mw_str.pop_back();

        Record r;
        r.area_id = area_id;
        r.timestamp = parse_datetime(dt);
        if (r.timestamp == -1) { skipped++; continue; }
        try { r.demand_mw = stod(mw_str); }
        catch (...) { skipped++; continue; }

        records.push_back(r);
        loaded++;
    }

    cout << "File: " << path
         << " | rows: " << rows
         << " | loaded: " << loaded
         << " | skipped: " << skipped << "\n";
    return records;
}

// Load areas_config.csv (area_id, capacity_mw, economic_weight, priority_flag, area_type)
inline map<string, AreaConfig> load_areas_config(const string& path) {
    map<string, AreaConfig> result;
    ifstream file(path);
    if (!file.is_open()) {
        cerr << "ERROR: cannot open " << path << "\n";
        return result;
    }
    string line;
    bool first = true;
    while (getline(file, line)) {
        if (first) { first = false; continue; }
        if (line.empty()) continue;

        stringstream ss(line);
        AreaConfig c;
        string cap_s, w_s, p_s;

        if (!getline(ss, c.area_id, ',')) continue;
        if (!getline(ss, cap_s, ','))     continue;
        if (!getline(ss, w_s, ','))       continue;
        if (!getline(ss, p_s, ','))       continue;
        getline(ss, c.area_type, ',');  // optional

        // Trim
        auto trim = [](string& s) {
            while (!s.empty() && (s.back() == ' ' || s.back() == '\r')) s.pop_back();
            while (!s.empty() && (s.front() == ' ')) s.erase(s.begin());
        };
        trim(c.area_id); trim(cap_s); trim(w_s); trim(p_s); trim(c.area_type);

        try {
            c.capacity_mw     = stod(cap_s);
            c.economic_weight = stod(w_s);
            c.priority_flag   = stoi(p_s);
        } catch (...) { continue; }

        result[c.area_id] = c;
    }
    return result;
}

// DataSplit struct used by preprocessing.h (forward declaration here, used elsewhere)
// struct DataSplit {
//     vector<Record> train;
//     vector<Record> test;
// };

#endif // DATA_IO_H