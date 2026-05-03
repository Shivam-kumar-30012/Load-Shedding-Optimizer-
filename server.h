#ifndef SERVER_H
#define SERVER_H

#define byte win_byte_override
#include "httplib.h"
#undef byte

#include "pipeline.h"
#include "forecaster.h"
#include "gbt.h"
#include "gbt_io.h"
#include "fitness.h"
#include "hybrid.h"
#include "data_io.h"
#include "matrix.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <set>
#include <algorithm>

using namespace std;

struct ServerState {
    GBT gbt{150, 6, 20, 0.05, 0.7, 8};
    vector<Record> all_records;          // full dataset for actual-demand lookup
    vector<string> area_order;
    vector<AreaConfig> area_configs;
    bool ready = false;
};

inline string read_text_file(const string& path) {
    ifstream f(path);
    if (!f.is_open()) return "";
    stringstream ss; ss << f.rdbuf();
    return ss.str();
}

inline string extract_json_field(const string& body, const string& field) {
    string key = "\"" + field + "\"";
    size_t k = body.find(key);
    if (k == string::npos) return "";
    size_t colon = body.find(':', k);
    if (colon == string::npos) return "";
    size_t q1 = body.find('"', colon);
    if (q1 == string::npos) return "";
    size_t q2 = body.find('"', q1 + 1);
    while (q2 != string::npos && body[q2 - 1] == '\\') q2 = body.find('"', q2 + 1);
    if (q2 == string::npos) return "";
    return body.substr(q1 + 1, q2 - q1 - 1);
}

inline string extract_json_array(const string& body, const string& field) {
    string key = "\"" + field + "\"";
    size_t k = body.find(key);
    if (k == string::npos) return "";
    size_t colon = body.find(':', k);
    if (colon == string::npos) return "";
    size_t lb = body.find('[', colon);
    if (lb == string::npos) return "";
    int depth = 1; size_t i = lb + 1;
    while (i < body.size() && depth > 0) {
        if (body[i] == '[') depth++;
        else if (body[i] == ']') depth--;
        if (depth == 0) break;
        i++;
    }
    if (depth != 0) return "";
    return body.substr(lb, i - lb + 1);
}

inline string build_result_json(const string& date,
                                const vector<string>& areas,
                                const Schedule& schedule,
                                const ScheduleMetrics& m,
                                bool actual_available = true,
                                int history_days = -1,
                                int history_points = -1)
{
    stringstream js;
    js << "{";
    js << "\"date\":\"" << date << "\",";
    js << "\"fitness\":" << m.fitness_score << ",";
    js << "\"damage\":" << m.damage << ",";
    js << "\"capacity_overage\":" << m.capacity_overage << ",";
    js << "\"priority_violations\":" << m.priority_violations << ",";
    js << "\"actual_available\":" << (actual_available ? "true" : "false") << ",";
    if (history_days >= 0) {
        js << "\"history_days\":" << history_days << ",";
        js << "\"history_points\":" << history_points << ",";
    }
    js << "\"areas\":[";
    for (size_t i = 0; i < areas.size(); ++i) {
        if (i > 0) js << ",";
        js << "\"" << areas[i] << "\"";
    }
    js << "],";
    js << "\"schedule\":[";
    for (size_t i = 0; i < schedule.size(); ++i) {
        if (i > 0) js << ",";
        js << "[";
        for (size_t h = 0; h < schedule[i].size(); ++h) {
            if (h > 0) js << ",";
            js << schedule[i][h];
        }
        js << "]";
    }
    js << "]";
    js << "}";
    return js.str();
}

inline vector<Record> parse_csv_text(const string& text, const string& default_area = "Area1") {
    vector<Record> records;
    stringstream ss(text);
    string line;
    if (!getline(ss, line)) return records;

    char delim = (line.find('\t') != string::npos) ? '\t' : ',';
    int header_cols = 1;
    for (char c : line) if (c == delim) header_cols++;
    bool is_two_col = (header_cols == 2);

    auto trim = [](string& s){
        while (!s.empty() && (s.back() == ' ' || s.back() == '\r' || s.back() == '\n' || s.back() == '\t')) s.pop_back();
        while (!s.empty() && (s.front() == ' ' || s.front() == '\t')) s.erase(s.begin());
    };

    while (getline(ss, line)) {
        if (line.empty()) continue;
        stringstream ls(line);
        string area_id, dt_str, mw_str;
        if (is_two_col) {
            if (!getline(ls, dt_str, delim)) continue;
            if (!getline(ls, mw_str, delim)) continue;
            area_id = default_area;
        } else {
            if (!getline(ls, area_id, delim)) continue;
            if (!getline(ls, dt_str, delim)) continue;
            if (!getline(ls, mw_str, delim)) continue;
        }
        trim(area_id); trim(dt_str); trim(mw_str);

        Record r;
        r.area_id = area_id;
        r.timestamp = parse_datetime(dt_str);
        if (r.timestamp == -1) continue;
        try { r.demand_mw = stod(mw_str); } catch (...) { continue; }
        records.push_back(r);
    }
    return records;
}

inline map<string, AreaConfig> parse_config_array(const string& arr_json) {
    map<string, AreaConfig> result;
    size_t pos = 0;
    while (pos < arr_json.size()) {
        size_t lb = arr_json.find('{', pos);
        if (lb == string::npos) break;
        int depth = 1; size_t i = lb + 1;
        while (i < arr_json.size() && depth > 0) {
            if (arr_json[i] == '{') depth++;
            else if (arr_json[i] == '}') depth--;
            if (depth == 0) break;
            i++;
        }
        if (depth != 0) break;
        string obj = arr_json.substr(lb, i - lb + 1);
        pos = i + 1;

        AreaConfig c;
        c.area_id = extract_json_field(obj, "area_id");
        if (c.area_id.empty()) continue;

        auto find_number = [&](const string& key) -> double {
            string k = "\"" + key + "\"";
            size_t kp = obj.find(k);
            if (kp == string::npos) return 0.0;
            size_t cp = obj.find(':', kp);
            if (cp == string::npos) return 0.0;
            size_t st = cp + 1;
            while (st < obj.size() && (obj[st] == ' ' || obj[st] == '\t')) st++;
            size_t en = st;
            while (en < obj.size() && (obj[en] == '-' || obj[en] == '.' ||
                   (obj[en] >= '0' && obj[en] <= '9'))) en++;
            try { return stod(obj.substr(st, en - st)); } catch (...) { return 0.0; }
        };

        c.capacity_mw = find_number("capacity_mw");
        c.economic_weight = find_number("economic_weight");
        c.priority_flag = (int)find_number("priority_flag");
        c.area_type = "user";
        result[c.area_id] = c;
    }
    return result;
}

inline bool init_server_state(ServerState& state, const string& gbt_path) {
    cout << "\n[Server Init] Loading state...\n";
    vector<string> input_files = {
        "data/AEP_hourly.csv","data/COMED_hourly.csv","data/DAYTON_hourly.csv",
        "data/DOM_hourly.csv","data/DUQ_hourly.csv"
    };

    auto raw = stage1_load(input_files);
    auto clean = stage2_impute(raw);
    auto cfg = stage3_configure(clean, "data/areas_config.csv");
    if (cfg.area_configs.empty()) return false;

    if (!load_gbt(state.gbt, gbt_path)) {
        cerr << "[Server Init] Failed to load GBT model. Run --save-gbt first.\n";
        return false;
    }

    state.all_records = clean;
    state.area_order = cfg.area_order;
    state.area_configs = cfg.area_configs;
    state.ready = true;

    cout << "[Server Init] Ready. " << state.all_records.size()
         << " records, " << state.area_order.size() << " areas, "
         << state.gbt.trees.size() << " GBT trees.\n";
    return true;
}

// Build actual demand matrix for a date IF data exists. Returns empty if not.
inline vector<vector<double>> get_actual_demand(
    const vector<Record>& records,
    const string& date,
    const vector<string>& area_order)
{
    vector<string> raw_order;
    auto raw = build_demand_matrix(records, date, raw_order);
    if (raw.empty()) return {};

    vector<vector<double>> demand(area_order.size(), vector<double>(24, 0.0));
    bool any_found = false;
    for (size_t i = 0; i < area_order.size(); ++i) {
        for (size_t j = 0; j < raw_order.size(); ++j) {
            if (raw_order[j] == area_order[i]) {
                demand[i] = raw[j];
                any_found = true;
                break;
            }
        }
    }
    return any_found ? demand : vector<vector<double>>{};
}

// Validate date is parseable and within reasonable bounds
inline bool valid_date_string(const string& date) {
    if (date.size() != 10) return false;
    if (date[4] != '-' || date[7] != '-') return false;
    int y = atoi(date.substr(0, 4).c_str());
    int m = atoi(date.substr(5, 2).c_str());
    int d = atoi(date.substr(8, 2).c_str());
    if (y < 2010 || y > 2024) return false;
    if (m < 1 || m > 12) return false;
    if (d < 1 || d > 31) return false;
    return true;
}

inline void start_server(int port = 8080, const string& gbt_path = "gbt_model.bin") {
    auto state = make_shared<ServerState>();
    if (!init_server_state(*state, gbt_path)) {
        cerr << "Server init failed.\n";
        return;
    }

    httplib::Server svr;
    svr.set_payload_max_length(100 * 1024 * 1024);

    svr.Get("/", [](const httplib::Request&, httplib::Response& res) {
        string html = read_text_file("dashboard.html");
        if (html.empty()) {
            res.set_content("dashboard.html not found.", "text/plain");
            res.status = 500; return;
        }
        res.set_content(html, "text/html");
    });

    svr.Get("/ping", [](const httplib::Request&, httplib::Response& res) {
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });

    // FLOW 1 — GBT forecast on any date 2010-2035
    svr.Post("/api/run-flow1", [state](const httplib::Request& req, httplib::Response& res) {
        if (!state->ready) {
            res.set_content("{\"error\":\"server not ready\"}", "application/json");
            res.status = 503; return;
        }

        string date = extract_json_field(req.body, "date");
        if (!valid_date_string(date)) {
            res.set_content("{\"error\":\"invalid date — use YYYY-MM-DD between 2010 and 2024\"}", "application/json");
            res.status = 400; return;
        }

        cout << "[/api/run-flow1] date=" << date << "\n";
        
        time_t day_t = parse_datetime(date + " 0:00");
        if (day_t == -1) {
            res.set_content("{\"error\":\"could not parse date\"}", "application/json");
            res.status = 400; return;
        }

        // Get GBT forecast for the requested date
        auto demand_forecast = state->gbt.forecast_matrix(state->area_order, day_t);

        // Try to get actual demand. If date is in our dataset, score against truth.
        // Otherwise, score against the forecast itself (for future dates).
        auto demand_actual = get_actual_demand(state->all_records, date, state->area_order);
        bool actual_available = !demand_actual.empty();
        if (!actual_available) {
            demand_actual = demand_forecast;  // future date, no truth available
            cout << "  (no actual data for this date — scoring against forecast)\n";
        }

        auto sched = hybrid_pipeline(demand_forecast, state->area_configs, false);
        auto metrics = compute_metrics(sched, demand_actual, state->area_configs);

        cout << "  fitness=" << metrics.fitness_score
             << ", damage=" << metrics.damage
             << ", overage=" << metrics.capacity_overage << "\n";

        string json = build_result_json(date, state->area_order, sched, metrics, actual_available);
        res.set_content(json, "application/json");
    });

    // FLOW 2 — user CSV + lite forecaster
    svr.Post("/api/run-flow2", [state](const httplib::Request& req, httplib::Response& res) {
        cout << "[/api/run-flow2] received\n";

        string date = extract_json_field(req.body, "date");
        string csv_text = extract_json_field(req.body, "csv_text");
        string config_arr = extract_json_array(req.body, "config");

        if (date.empty() || csv_text.empty()) {
            res.set_content("{\"error\":\"missing date or csv_text\"}", "application/json");
            res.status = 400; return;
        }

        auto unescape = [](string s) {
            string out; out.reserve(s.size());
            for (size_t i = 0; i < s.size(); ++i) {
                if (s[i] == '\\' && i + 1 < s.size()) {
                    char nx = s[i + 1];
                    if (nx == 'n') { out.push_back('\n'); i++; }
                    else if (nx == 'r') { out.push_back('\r'); i++; }
                    else if (nx == 't') { out.push_back('\t'); i++; }
                    else if (nx == '"') { out.push_back('"'); i++; }
                    else if (nx == '\\') { out.push_back('\\'); i++; }
                    else out.push_back(s[i]);
                } else out.push_back(s[i]);
            }
            return out;
        };
        csv_text = unescape(csv_text);

        string default_area = extract_json_field(req.body, "default_area");
        if (default_area.empty()) {
            auto fc = parse_config_array(config_arr);
            default_area = fc.empty() ? "Area1" : fc.begin()->first;
        }

        auto records = parse_csv_text(csv_text, default_area);
        if (records.empty()) {
            res.set_content("{\"error\":\"could not parse CSV\"}", "application/json");
            res.status = 400; return;
        }

        sort(records.begin(), records.end(),
             [](const Record& a, const Record& b) { return a.timestamp < b.timestamp; });

        set<string> area_set;
        for (const auto& r : records) area_set.insert(r.area_id);
        vector<string> area_order(area_set.begin(), area_set.end());

        time_t target_t = parse_datetime(date + " 0:00");
        if (target_t == -1) {
            res.set_content("{\"error\":\"invalid target date\"}", "application/json");
            res.status = 400; return;
        }

        // Try to find actual demand for the target date in user data
        auto demand_actual = get_actual_demand(records, date, area_order);
        bool actual_available = !demand_actual.empty();

        // Build configs
        auto config_map = parse_config_array(config_arr);
        vector<AreaConfig> area_configs;
        for (const auto& a : area_order) {
            AreaConfig c;
            c.area_id = a;
            if (config_map.count(a)) c = config_map[a];
            else {
                vector<double> demands;
                for (const auto& r : records)
                    if (r.area_id == a) demands.push_back(r.demand_mw);
                double cap = 1.0;
                if (!demands.empty()) {
                    sort(demands.begin(), demands.end());
                    cap = demands[demands.size() / 4]; // P25
                }
                c.capacity_mw = cap;
                c.economic_weight = 0.5;
                c.priority_flag = 0;
                c.area_type = "user";
            }
            if (c.capacity_mw <= 0) c.capacity_mw = 1.0;
            area_configs.push_back(c);
        }

        // Build forecast using lite forecaster
        LiteForecaster lf;
        lf.build(records, target_t);
        int history_points = lf.total_history_points();

        if (history_points == 0) {
            res.set_content("{\"error\":\"no history available before target date — upload more data\"}", "application/json");
            res.status = 400; return;
        }

        auto demand_forecast = lf.forecast_matrix(area_order, target_t);

        if (!actual_available) demand_actual = demand_forecast;

        auto sched = hybrid_pipeline(demand_forecast, area_configs, false);
        auto metrics = compute_metrics(sched, demand_actual, area_configs);

        cout << "  fitness=" << metrics.fitness_score
             << ", damage=" << metrics.damage << "\n";

        string json = build_result_json(date, area_order, sched, metrics,
                                        actual_available,
                                        LiteForecaster::HISTORY_DAYS, history_points);
        res.set_content(json, "application/json");
    });

    cout << "\n=================================================================\n";
    cout << "  SERVER RUNNING AT http://localhost:" << port << "\n";
    cout << "  Open this URL in your browser.\n";
    cout << "  Press Ctrl+C in this terminal to stop.\n";
    cout << "=================================================================\n";
    svr.listen("0.0.0.0", port);
}

#endif // SERVER_H