#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <pigpio.h>

struct HomeConfig {
    double bearing = 270.0;
    bool panClockwise = false;
};

struct ServoInputs {
    int pan = 0;
    int tilt = 0;
    bool valid = false;
    bool backMode = false;
};

const int PAN_PIN = 18;
const int TILT_PIN = 19;
const int TILT_INPUT_MIN = 30;
const int TILT_INPUT_MAX = 215;
const int TILT_HORIZON_FRONT = 30;
const int TILT_MID_FRONT = 95;
const int TILT_STRAIGHT_UP = 155;
const int TILT_FLIPPED_BACK = 215;
const int PULSE_MIN_US = 500;
const int PULSE_MAX_US = 2500;
const int SERVO_INPUT_MAX = 270;
const double FRONT_MIN_TRACKABLE_ELEV_DEG = 0.0;
const double FRONT_MID_ELEV_DEG = 41.0;
const double BACK_MIN_TRACKABLE_ELEV_DEG = 41.0;

static std::string trim(const std::string& s) {
    const size_t a = s.find_first_not_of(" \t\r\n");
    const size_t b = s.find_last_not_of(" \t\r\n");
    return (a == std::string::npos) ? "" : s.substr(a, b - a + 1);
}

bool loadConfig(const std::string& path, HomeConfig& cfg) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "WARNING: Cannot open " << path << ", using defaults.\n";
        return false;
    }

    std::string line;
    while (std::getline(file, line)) {
        const size_t hash = line.find('#');
        if (hash != std::string::npos) line = line.substr(0, hash);
        const size_t colon = line.find(':');
        if (colon == std::string::npos) continue;

        const std::string key = trim(line.substr(0, colon));
        const std::string val = trim(line.substr(colon + 1));
        if (key.empty() || val.empty()) continue;

        try {
            if (key == "home_bearing") cfg.bearing = std::stod(val);
            else if (key == "pan_clockwise") cfg.panClockwise = (val == "true");
        } catch (...) {
            std::cerr << "WARNING: Could not parse key '" << key << "'\n";
        }
    }
    return true;
}

double bearingDiff(double refBearing, double targetBearing, bool clockwise) {
    if (clockwise) return std::fmod(targetBearing - refBearing + 360.0, 360.0);
    return std::fmod(refBearing - targetBearing + 360.0, 360.0);
}

ServoInputs computeServoInputs(double bearing, double elevDeg, double homeBearing, bool panClockwise) {
    ServoInputs r;

    bearing = std::fmod(bearing + 360.0, 360.0);
    if (elevDeg < FRONT_MIN_TRACKABLE_ELEV_DEG) return r;
    if (elevDeg > 90.0) elevDeg = 90.0;

    const double frontDiff = bearingDiff(homeBearing, bearing, panClockwise);
    const double backHomeBearing = std::fmod(homeBearing + 180.0, 360.0);
    const double backDiff = bearingDiff(backHomeBearing, bearing, panClockwise);

    const bool frontReachable = (frontDiff <= 180.0);
    const bool backReachable = (backDiff <= 180.0);

    double panDiff = 0.0;
    int tiltInput = 0;
    if (frontReachable) {
        r.backMode = false;
        panDiff = frontDiff;
        if (elevDeg <= FRONT_MID_ELEV_DEG) {
            const double lowSpan = static_cast<double>(TILT_MID_FRONT - TILT_HORIZON_FRONT);
            tiltInput = static_cast<int>(std::round(
                TILT_HORIZON_FRONT + elevDeg * (lowSpan / FRONT_MID_ELEV_DEG)
            ));
        } else {
            const double highSpan = static_cast<double>(TILT_STRAIGHT_UP - TILT_MID_FRONT);
            tiltInput = static_cast<int>(std::round(
                TILT_MID_FRONT + (elevDeg - FRONT_MID_ELEV_DEG) * (highSpan / (90.0 - FRONT_MID_ELEV_DEG))
            ));
        }
    } else {
        if (!backReachable || elevDeg < BACK_MIN_TRACKABLE_ELEV_DEG) return r;
        r.backMode = true;
        panDiff = backDiff;
        const double backSpan = static_cast<double>(TILT_FLIPPED_BACK - TILT_STRAIGHT_UP);
        tiltInput = static_cast<int>(std::round(
            TILT_FLIPPED_BACK - (elevDeg - BACK_MIN_TRACKABLE_ELEV_DEG) * (backSpan / (90.0 - BACK_MIN_TRACKABLE_ELEV_DEG))
        ));
    }

    int panInput = static_cast<int>(std::round(panDiff * (270.0 / 180.0)));
    panInput = std::clamp(panInput, 0, SERVO_INPUT_MAX);
    tiltInput = std::clamp(tiltInput, TILT_INPUT_MIN, TILT_INPUT_MAX);

    r.pan = panInput;
    r.tilt = tiltInput;
    r.valid = true;
    return r;
}

std::string modeName(const ServoInputs& servo) {
    return servo.backMode ? "back" : "front";
}

std::string compressRanges(const std::vector<int>& values) {
    if (values.empty()) return "none";
    std::ostringstream out;
    int start = values.front();
    int prev = values.front();
    bool first = true;
    for (size_t i = 1; i < values.size(); ++i) {
        const int value = values[i];
        if (value == prev + 1) {
            prev = value;
            continue;
        }
        if (!first) out << ", ";
        if (start == prev) out << std::setw(3) << std::setfill('0') << start;
        else out << std::setw(3) << std::setfill('0') << start << "-" << std::setw(3) << std::setfill('0') << prev;
        first = false;
        start = prev = value;
    }
    if (!first) out << ", ";
    if (start == prev) out << std::setw(3) << std::setfill('0') << start;
    else out << std::setw(3) << std::setfill('0') << start << "-" << std::setw(3) << std::setfill('0') << prev;
    return out.str();
}

void printSamples(double elev, const HomeConfig& cfg) {
    const int sampleBearings[] = {0, 45, 90, 135, 180, 225, 270, 315};
    std::cout << "  samples:";
    for (int bearing : sampleBearings) {
        const ServoInputs servo = computeServoInputs(static_cast<double>(bearing), elev, cfg.bearing, cfg.panClockwise);
        std::cout << " " << std::setw(3) << bearing << "=";
        if (!servo.valid) std::cout << "invalid";
        else std::cout << servo.pan << "/" << servo.tilt << "/" << modeName(servo);
    }
    std::cout << "\n";
}

void printSummary(const HomeConfig& cfg, int elevStep) {
    std::cout << "home_bearing=" << cfg.bearing
              << " pan_clockwise=" << (cfg.panClockwise ? "true" : "false") << "\n";
    for (int elev = 0; elev <= 90; elev += elevStep) {
        std::vector<int> reachable;
        std::vector<int> unreachable;
        for (int bearing = 0; bearing < 360; ++bearing) {
            const ServoInputs servo = computeServoInputs(static_cast<double>(bearing), static_cast<double>(elev), cfg.bearing, cfg.panClockwise);
            if (servo.valid) reachable.push_back(bearing);
            else unreachable.push_back(bearing);
        }

        std::cout << "elev=" << std::setw(2) << elev
                  << " reachable=" << reachable.size() << "/360"
                  << " unreachable=" << unreachable.size() << "\n";
        std::cout << "  reachable ranges:   " << compressRanges(reachable) << "\n";
        std::cout << "  unreachable ranges: " << compressRanges(unreachable) << "\n";
        printSamples(static_cast<double>(elev), cfg);
    }
}

void printFullGrid(const HomeConfig& cfg, int elevStep, int bearingStep) {
    std::cout << "bearing,elevation,valid,pan,tilt,mode\n";
    for (int elev = 0; elev <= 90; elev += elevStep) {
        for (int bearing = 0; bearing < 360; bearing += bearingStep) {
            const ServoInputs servo = computeServoInputs(static_cast<double>(bearing), static_cast<double>(elev), cfg.bearing, cfg.panClockwise);
            std::cout << bearing << "," << elev << "," << (servo.valid ? 1 : 0) << ","
                      << servo.pan << "," << servo.tilt << "," << modeName(servo) << "\n";
        }
    }
}

void setServo(int pin, int servoInput) {
    const int pulseUs = PULSE_MIN_US + static_cast<int>(std::round(
        static_cast<double>(servoInput) / SERVO_INPUT_MAX * (PULSE_MAX_US - PULSE_MIN_US)
    ));
    gpioServo(pin, pulseUs);
}

void stopServos() {
    gpioServo(PAN_PIN, 0);
    gpioServo(TILT_PIN, 0);
}

void driveGrid(const HomeConfig& cfg, int elevStep, int bearingStep, int settleMs, bool validOnly) {
    if (gpioInitialise() < 0) {
        std::cerr << "ERROR: gpioInitialise failed\n";
        return;
    }
    gpioSetMode(PAN_PIN, PI_OUTPUT);
    gpioSetMode(TILT_PIN, PI_OUTPUT);

    std::cout << "Driving servo grid. Press Ctrl+C to stop.\n";
    for (int elev = 0; elev <= 90; elev += elevStep) {
        for (int bearing = 0; bearing < 360; bearing += bearingStep) {
            const ServoInputs servo = computeServoInputs(static_cast<double>(bearing), static_cast<double>(elev), cfg.bearing, cfg.panClockwise);
            if (!servo.valid && validOnly) continue;

            std::cout << "bearing=" << std::setw(3) << bearing
                      << " elev=" << std::setw(2) << elev;
            if (!servo.valid) {
                std::cout << " -> invalid\n";
                std::this_thread::sleep_for(std::chrono::milliseconds(settleMs));
                continue;
            }

            std::cout << " -> pan=" << std::setw(3) << servo.pan
                      << " tilt=" << std::setw(3) << servo.tilt
                      << " mode=" << modeName(servo) << "\n";
            setServo(PAN_PIN, servo.pan);
            setServo(TILT_PIN, servo.tilt);
            std::this_thread::sleep_for(std::chrono::milliseconds(settleMs));
        }
    }

    stopServos();
    gpioTerminate();
}

int main(int argc, char** argv) {
    std::string configPath = "config/home.yaml";
    int elevStep = 5;
    int bearingStep = 5;
    int settleMs = 1200;
    bool fullGrid = false;
    bool drive = false;
    bool validOnly = true;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--home" && i + 1 < argc) configPath = argv[++i];
        else if (arg == "--step" && i + 1 < argc) elevStep = std::max(1, std::stoi(argv[++i]));
        else if (arg == "--bearing-step" && i + 1 < argc) bearingStep = std::max(1, std::stoi(argv[++i]));
        else if (arg == "--settle-ms" && i + 1 < argc) settleMs = std::max(100, std::stoi(argv[++i]));
        else if (arg == "--full-grid") fullGrid = true;
        else if (arg == "--drive") drive = true;
        else if (arg == "--include-invalid") validOnly = false;
        else {
            std::cerr << "Usage: " << argv[0]
                      << " [--home path] [--step n] [--bearing-step n] [--settle-ms n]"
                      << " [--full-grid] [--drive] [--include-invalid]\n";
            return 1;
        }
    }

    HomeConfig cfg;
    loadConfig(configPath, cfg);

    if (drive) {
        driveGrid(cfg, elevStep, bearingStep, settleMs, validOnly);
        return 0;
    }
    if (fullGrid) printFullGrid(cfg, elevStep, bearingStep);
    else printSummary(cfg, elevStep);
    return 0;
}
