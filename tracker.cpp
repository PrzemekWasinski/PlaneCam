#include <atomic>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <netinet/in.h>
#include <pigpio.h>
#include <sstream>
#include <string>
#include <sys/socket.h>
#include <sys/statvfs.h>
#include <sys/sysinfo.h>
#include <thread>
#include <unistd.h>
#include <vector>
#include <array>

namespace fs = std::filesystem;

//SERVO PINS
const int PAN_PIN  = 18;
const int TILT_PIN = 19;

//SERVO CALIBRATION
// Pan still uses the full 0..270 servo range.
// Tilt is calibrated from field testing:
//   front hemisphere: 30 = horizon, 95 = about 41 deg up, 155 = straight up
//   flipped-back hemisphere: 215 = about 41 deg up on the back side
const int TILT_INPUT_MIN     = 30;
const int TILT_INPUT_MAX     = 215;
const int TILT_HORIZON_FRONT = 30;
const int TILT_MID_FRONT     = 95;
const int TILT_STRAIGHT_UP   = 155;
const int TILT_FLIPPED_BACK  = 215;
const int PWM_FREQ           = 50;
const int PULSE_MIN_US       = 500;
const int PULSE_MAX_US       = 2500;
const int SERVO_INPUT_MAX    = 270;
const double FRONT_MIN_TRACKABLE_ELEV_DEG = 0.0;
const double FRONT_MID_ELEV_DEG           = 41.0;
const double BACK_MIN_TRACKABLE_ELEV_DEG  = 41.0;

const char* OUTPUT_DIR = "images";
const int SERVO_SETTLE_MS = 250;

//ENABLE CAMERA
const bool ENABLE_CAMERA_CAPTURE = true;

std::atomic<bool> busy(false);
std::atomic<bool> gpioReady(false);
std::atomic<bool> holdActive(false);
std::atomic<int>  heldPanInput(0);
std::atomic<int>  heldTiltInput(0);
std::mutex        servoMutex;  // <-- NEW: guards all gpioServo calls

struct HomeConfig {
    double lat           = 0.0;
    double lon           = 0.0;
    double elevation     = 0.0;
    double bearing       = 180.0;
    bool pan_clockwise   = false;
};

static std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    size_t b = s.find_last_not_of(" \t\r\n");
    return (a == std::string::npos) ? "" : s.substr(a, b - a + 1);
}

bool loadConfig(const std::string& path, HomeConfig& cfg) {
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "ERROR: Cannot open " << path << "\n";
        return false;
    }
    std::string line;
    while (std::getline(f, line)) {
        size_t hash = line.find('#');
        if (hash != std::string::npos) line = line.substr(0, hash);
        size_t colon = line.find(':');
        if (colon == std::string::npos) continue;
        std::string key = trim(line.substr(0, colon));
        std::string val = trim(line.substr(colon + 1));
        if (key.empty() || val.empty()) continue;
        try {
            if      (key == "home_lat")       cfg.lat = std::stod(val);
            else if (key == "home_lon")       cfg.lon = std::stod(val);
            else if (key == "home_elevation") cfg.elevation = std::stod(val);
            else if (key == "home_bearing")   cfg.bearing = std::stod(val);
            else if (key == "pan_clockwise")  cfg.pan_clockwise = (val == "true");
        } catch (...) {
            std::cerr << "WARNING: Could not parse value for key '" << key << "'\n";
        }
    }
    return true;
}

const double DEG2RAD = M_PI / 180.0;
const double RAD2DEG = 180.0 / M_PI;
const double EARTH_R = 6371000.0;

double haversineBearing(double latA, double lonA, double latB, double lonB) {
    double dLon = (lonB - lonA) * DEG2RAD;
    double la   = latA * DEG2RAD;
    double lb   = latB * DEG2RAD;
    double y = std::sin(dLon) * std::cos(lb);
    double x = std::cos(la) * std::sin(lb) - std::sin(la) * std::cos(lb) * std::cos(dLon);
    return std::fmod(std::atan2(y, x) * RAD2DEG + 360.0, 360.0);
}

double haversineDistance(double latA, double lonA, double latB, double lonB) {
    double dLat = (latB - latA) * DEG2RAD;
    double dLon = (lonB - lonA) * DEG2RAD;
    double a = std::sin(dLat / 2) * std::sin(dLat / 2)
             + std::cos(latA * DEG2RAD) * std::cos(latB * DEG2RAD)
             * std::sin(dLon / 2) * std::sin(dLon / 2);
    return EARTH_R * 2.0 * std::atan2(std::sqrt(a), std::sqrt(1 - a));
}

double elevationAngle(double distanceM, double homeElevM, double targetElevM) {
    double altDiff = targetElevM - homeElevM;
    return std::atan2(altDiff, distanceM) * RAD2DEG;
}

struct ServoInputs {
    int pan;
    int tilt;
    bool valid;
    bool backMode;
};

struct RecognitionResult {
    bool available = false;
    bool containsAircraft = false;
    double confidence = 0.0;
    double rawScore = 0.0;
    double scoreMargin = 0.0;
    std::string label = "UNKNOWN";
    std::string detail = "unavailable";
    std::string predictor = "unknown";
};

ServoInputs computeServoInputs(double bearing, double elevDeg, double homeBearing, bool panClockwise) {
    ServoInputs r = {0, 0, false, false};

    bearing = std::fmod(bearing + 360.0, 360.0);
    if (elevDeg < FRONT_MIN_TRACKABLE_ELEV_DEG) return r;
    if (elevDeg > 90.0) elevDeg = 90.0;

    const auto bearingDiff = [](double refBearing, double targetBearing, bool clockwise) {
        if (clockwise)
            return std::fmod(targetBearing - refBearing + 360.0, 360.0);
        return std::fmod(refBearing - targetBearing + 360.0, 360.0);
    };

    const double frontDiff = bearingDiff(homeBearing, bearing, panClockwise);
    const double backHomeBearing = std::fmod(homeBearing + 180.0, 360.0);
    const double backDiff = bearingDiff(backHomeBearing, bearing, panClockwise);

    const bool frontReachable = (frontDiff <= 180.0);
    const bool backReachable = (backDiff <= 180.0);

    double panDiff;
    int tiltInput;
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

    if (tiltInput < TILT_INPUT_MIN)  tiltInput = TILT_INPUT_MIN;
    if (panInput  < 0)               panInput  = 0;
    if (panInput  > SERVO_INPUT_MAX) panInput  = SERVO_INPUT_MAX;
    if (tiltInput > TILT_INPUT_MAX)  tiltInput = TILT_INPUT_MAX;

    r.pan   = panInput;
    r.tilt  = tiltInput;
    r.valid = true;
    return r;
}

double readCpuUsagePercent() {
    static unsigned long long prevIdle  = 0;
    static unsigned long long prevTotal = 0;

    std::ifstream statFile("/proc/stat");
    std::string cpu;
    unsigned long long user = 0, nice = 0, system = 0, idle = 0,
                       iowait = 0, irq = 0, softirq = 0, steal = 0;
    if (!(statFile >> cpu >> user >> nice >> system >> idle >> iowait >> irq >> softirq >> steal))
        return 0.0;

    unsigned long long idleAll  = idle + iowait;
    unsigned long long total    = user + nice + system + idle + iowait + irq + softirq + steal;
    if (prevTotal == 0 || total <= prevTotal) { prevIdle = idleAll; prevTotal = total; return 0.0; }

    unsigned long long totalDiff = total    - prevTotal;
    unsigned long long idleDiff  = idleAll  - prevIdle;
    prevIdle  = idleAll;
    prevTotal = total;
    if (totalDiff == 0) return 0.0;
    return 100.0 * (1.0 - static_cast<double>(idleDiff) / static_cast<double>(totalDiff));
}

double readTemperatureC() {
    std::ifstream tempFile("/sys/class/thermal/thermal_zone0/temp");
    double milliC = 0.0;
    if (!(tempFile >> milliC)) return 0.0;
    return milliC / 1000.0;
}

double readRamPercent() {
    struct sysinfo info;
    if (sysinfo(&info) != 0 || info.totalram == 0) return 0.0;
    double total     = static_cast<double>(info.totalram) * info.mem_unit;
    double available = static_cast<double>(info.freeram)  * info.mem_unit;
    return 100.0 * (1.0 - (available / total));
}

double readDiskFreeGb() {
    struct statvfs fsInfo;
    if (statvfs("/", &fsInfo) != 0) return 0.0;
    unsigned long long freeBytes =
        static_cast<unsigned long long>(fsInfo.f_bavail) *
        static_cast<unsigned long long>(fsInfo.f_frsize);
    return static_cast<double>(freeBytes) / (1024.0 * 1024.0 * 1024.0);
}

std::string buildStatsResponse() {
    std::ostringstream response;
    response << std::fixed << std::setprecision(1)
             << readTemperatureC()   << ","
             << readRamPercent()     << ","
             << readCpuUsagePercent()<< ","
             << readDiskFreeGb();
    return response.str();
}

// All gpioServo calls must go through this — never call gpioServo directly.
void setServo(int pin, int servoInput) {
    int pulseUs = PULSE_MIN_US + static_cast<int>(std::round(
        static_cast<double>(servoInput) / SERVO_INPUT_MAX * (PULSE_MAX_US - PULSE_MIN_US)
    ));
    std::lock_guard<std::mutex> lock(servoMutex);
    gpioServo(pin, pulseUs);
}

void stopServos() {
    std::lock_guard<std::mutex> lock(servoMutex);
    gpioServo(PAN_PIN,  0);
    gpioServo(TILT_PIN, 0);
}

// Only update the target values — the hold loop drives the hardware.
void applyAndHoldServos(int panInput, int tiltInput) {
    heldPanInput.store(panInput);
    heldTiltInput.store(tiltInput);
    holdActive.store(true);   // hold loop will pick these up on its next tick
}

void servoHoldLoop() {
    while (gpioReady.load()) {
        if (holdActive.load()) {
            // Snapshot both atomics under a single mutex acquisition so pan
            // and tilt are always written as a matched pair.
            int pan  = heldPanInput.load();
            int tilt = heldTiltInput.load();

            int panPulse  = PULSE_MIN_US + static_cast<int>(std::round(
                static_cast<double>(pan)  / SERVO_INPUT_MAX * (PULSE_MAX_US - PULSE_MIN_US)));
            int tiltPulse = PULSE_MIN_US + static_cast<int>(std::round(
                static_cast<double>(tilt) / SERVO_INPUT_MAX * (PULSE_MAX_US - PULSE_MIN_US)));

            {
                std::lock_guard<std::mutex> lock(servoMutex);
                gpioServo(PAN_PIN,  panPulse);
                gpioServo(TILT_PIN, tiltPulse);
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

void shutdownServosAndGpio() {
    if (!gpioReady.exchange(false)) return;
    holdActive.store(false);
    stopServos();
    gpioTerminate();
}

bool sendAll(int socketFd, const char* data, size_t length) {
    size_t sent = 0;
    while (sent < length) {
        ssize_t bytesSent = send(socketFd, data + sent, length - sent, 0);
        if (bytesSent <= 0) return false;
        sent += static_cast<size_t>(bytesSent);
    }
    return true;
}

bool sendLine(int socketFd, const std::string& line) {
    return sendAll(socketFd, line.c_str(), line.size());
}

std::string shellQuote(const std::string& value) {
    std::string quoted = "'";
    for (char ch : value) {
        if (ch == '\'') quoted += "'\\''";
        else            quoted += ch;
    }
    quoted += "'";
    return quoted;
}

std::string sanitizeHexCode(const std::string& hexCode) {
    std::string cleaned;
    cleaned.reserve(hexCode.size());
    for (char ch : hexCode) {
        if (std::isalnum(static_cast<unsigned char>(ch)))
            cleaned += static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
    }
    return cleaned.empty() ? "UNKNOWN" : cleaned;
}

std::string makeTimestamp() {
    const auto now      = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm local{};
    localtime_r(&t, &local);
    std::ostringstream ss;
    ss << std::put_time(&local, "%Y%m%d_%H%M%S");
    return ss.str();
}

fs::path currentExecutablePath() {
    std::array<char, 4096> buffer{};
    const ssize_t length = readlink("/proc/self/exe", buffer.data(), buffer.size() - 1);
    if (length <= 0) return {};
    buffer[static_cast<size_t>(length)] = '\0';
    return fs::path(buffer.data());
}

std::vector<fs::path> candidatePredictorPaths() {
    std::vector<fs::path> paths = {
        fs::path("build/image_predict"),
        fs::path("camera_module/build/image_predict"),
        fs::path("../camera_module/build/image_predict"),
        fs::path("../tests/image_recognition_test/build/predict"),
        fs::path("tests/image_recognition_test/build/predict"),
        fs::path("build/predict")
    };

    const fs::path exePath = currentExecutablePath();
    if (!exePath.empty()) {
        const fs::path exeDir = exePath.parent_path();
        paths.push_back(exeDir / "image_predict");
        paths.push_back(exeDir / "predict");
        paths.push_back(exeDir.parent_path() / "camera_module" / "build" / "image_predict");
        paths.push_back(exeDir.parent_path() / "tests" / "image_recognition_test" / "build" / "predict");
    }

    return paths;
}

std::string extractTokenValue(const std::string& text, const std::string& key) {
    const std::string needle = key + "=";
    const size_t start = text.find(needle);
    if (start == std::string::npos) return "";
    const size_t valueStart = start + needle.size();
    const size_t valueEnd = text.find_first_of(" \r\n", valueStart);
    if (valueEnd == std::string::npos) return text.substr(valueStart);
    return text.substr(valueStart, valueEnd - valueStart);
}

RecognitionResult classifyCapturedImage(const fs::path& imagePath) {
    RecognitionResult result;

    fs::path predictorPath;
    for (const auto& candidate : candidatePredictorPaths()) {
        std::error_code error;
        const fs::path normalized = fs::weakly_canonical(candidate, error);
        const fs::path resolved = error ? candidate : normalized;
        if (fs::exists(resolved)) {
            predictorPath = resolved;
            result.predictor = resolved.filename().string();
            break;
        }
    }

    if (predictorPath.empty()) {
        result.detail = "predictor_missing";
        return result;
    }

    std::ostringstream command;
    command << shellQuote(predictorPath.string()) << " " << shellQuote(imagePath.string()) << " 2>&1";
    FILE* pipe = popen(command.str().c_str(), "r");
    if (pipe == nullptr) {
        result.detail = "predictor_launch_failed";
        return result;
    }

    std::string output;
    std::array<char, 512> buffer{};
    while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
        output += buffer.data();
    }
    const int exitCode = pclose(pipe);
    if (exitCode != 0) {
        result.detail = "predictor_failed";
        return result;
    }

    if (output.find("AIRCRAFT") != std::string::npos) {
        result.available = true;
        result.containsAircraft = true;
        result.label = "AIRCRAFT";
    } else if (output.find("SKY") != std::string::npos) {
        result.available = true;
        result.containsAircraft = false;
        result.label = "SKY";
    } else {
        result.detail = "prediction_unparsed";
        return result;
    }

    try {
        const std::string confidenceText = extractTokenValue(output, "confidence");
        const std::string rawScoreText = extractTokenValue(output, "raw_score");
        if (!confidenceText.empty()) result.confidence = std::stod(confidenceText);
        if (!rawScoreText.empty()) result.rawScore = std::stod(rawScoreText);
        result.scoreMargin = std::abs(result.rawScore);
    } catch (...) {
    }

    result.detail = "ok";
    return result;
}

bool captureImage(const fs::path& outputPath) {
    std::error_code error;
    fs::create_directories(outputPath.parent_path(), error);
    if (error) {
        std::cerr << "Failed to create output directory: " << outputPath.parent_path() << "\n";
        return false;
    }

    const std::vector<std::string> commands = {"rpicam-still", "libcamera-still"};
    for (const auto& command : commands) {
        std::ostringstream shellCommand;
        shellCommand
            << command
            << " -n"
            << " --immediate"
            << " -t 1"
            << " -o " << shellQuote(outputPath.string());

        std::cout << "Trying " << command << "...\n";
        const int exitCode = std::system(shellCommand.str().c_str());
        if (exitCode == 0 && fs::exists(outputPath) && fs::file_size(outputPath, error) > 0)
            return true;
    }
    return false;
}

bool sendImageResponse(
    int clientSocket,
    const fs::path& imagePath,
    const RecognitionResult& recognition,
    const ServoInputs& servo,
    double bearing,
    double elev,
    double distance,
    double alt) {
    std::error_code error;
    const auto imageSize = fs::file_size(imagePath, error);
    if (error || imageSize == 0)
        return sendLine(clientSocket, "ERROR image_missing\n");

    std::ifstream imageFile(imagePath, std::ios::binary);
    if (!imageFile.is_open())
        return sendLine(clientSocket, "ERROR image_open_failed\n");

    std::ostringstream header;
    header << "IMAGE " << imageSize;
    header << std::fixed << std::setprecision(4);
    header << " bearing_deg=" << bearing
           << " elev_deg=" << elev
           << " distance_m=" << distance
           << " alt_m=" << alt
           << " pan=" << servo.pan
           << " tilt=" << servo.tilt
           << " mode=" << (servo.backMode ? "back" : "front");

    if (recognition.available) {
        header << " label=" << recognition.label
               << " aircraft=" << (recognition.containsAircraft ? "yes" : "no")
               << " confidence=" << recognition.confidence
               << " raw_score=" << recognition.rawScore
               << " score_margin=" << recognition.scoreMargin
               << " detail=" << recognition.detail
               << " predictor=" << recognition.predictor;
    } else {
        header << " label=UNKNOWN aircraft=unknown detail=" << recognition.detail
               << " predictor=" << recognition.predictor;
    }
    header << "\n";
    if (!sendLine(clientSocket, header.str())) return false;

    std::vector<char> buffer(8192);
    while (imageFile) {
        imageFile.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const std::streamsize bytesRead = imageFile.gcount();
        if (bytesRead <= 0) break;
        if (!sendAll(clientSocket, buffer.data(), static_cast<size_t>(bytesRead)))
            return false;
    }
    return true;
}

void finishRequest(int clientSocket) {
    close(clientSocket);
    busy.store(false);
}

void trackPlane(const std::string& hexCode, double lat, double lon, double alt,
                HomeConfig& cfg, int clientSocket)
{
    std::cout << "Track request: lat=" << lat << ", lon=" << lon << ", alt_m=" << alt << "\n";

    const double bearing  = haversineBearing(cfg.lat, cfg.lon, lat, lon);
    const double distance = haversineDistance(cfg.lat, cfg.lon, lat, lon);
    const double elev     = elevationAngle(distance, cfg.elevation, alt);
    const ServoInputs servo = computeServoInputs(bearing, elev, cfg.bearing, cfg.pan_clockwise);

    std::cout << "Computed: bearing=" << bearing
              << ", distance_m="      << distance
              << ", elev_deg="        << elev
              << ", pan="             << servo.pan
              << ", tilt="            << servo.tilt
              << ", valid="           << (servo.valid ? "true" : "false") << "\n";

    if (!servo.valid) {
        std::ostringstream error;
        error << std::fixed << std::setprecision(1)
              << "ERROR invalid_target elev_deg=" << elev
              << " distance_m=" << distance
              << " bearing_deg=" << bearing
              << " alt_m=" << alt << "\n";
        sendLine(clientSocket, error.str());
        finishRequest(clientSocket);
        return;
    }

    if (!gpioReady.load()) {
        sendLine(clientSocket, "ERROR gpio_not_ready\n");
        finishRequest(clientSocket);
        return;
    }

    applyAndHoldServos(servo.pan, servo.tilt);
    std::this_thread::sleep_for(std::chrono::milliseconds(SERVO_SETTLE_MS));

    if (!ENABLE_CAMERA_CAPTURE) {
        sendLine(clientSocket, "OK tracking_only\n");
        finishRequest(clientSocket);
        return;
    }

    const fs::path outputPath =
        fs::path(OUTPUT_DIR) / (sanitizeHexCode(hexCode) + "_" + makeTimestamp() + ".jpg");

    if (!captureImage(outputPath)) {
        sendLine(clientSocket, "ERROR capture_failed\n");
        finishRequest(clientSocket);
        return;
    }

    const RecognitionResult recognition = classifyCapturedImage(outputPath);
    std::cout << "Recognition: available=" << (recognition.available ? "true" : "false")
              << ", label=" << recognition.label
              << ", aircraft=" << (recognition.containsAircraft ? "yes" : "no")
              << ", confidence=" << recognition.confidence
              << ", raw_score=" << recognition.rawScore
              << ", detail=" << recognition.detail << "\n";

    if (!sendImageResponse(clientSocket, outputPath, recognition, servo, bearing, elev, distance, alt))
        std::cerr << "Failed to send image response\n";

    finishRequest(clientSocket);
}

int main() {
    HomeConfig cfg;
    if (!loadConfig("config/home.yaml", cfg)) return 1;

    if (gpioInitialise() < 0) {
        std::cerr << "ERROR: gpioInitialise failed\n";
        return 1;
    }
    gpioReady.store(true);
    std::atexit(shutdownServosAndGpio);
    gpioSetMode(PAN_PIN,  PI_OUTPUT);
    gpioSetMode(TILT_PIN, PI_OUTPUT);

    std::thread holdThread(servoHoldLoop);
    holdThread.detach();

    std::cout << "\n--------------------------------------------\n";
    std::cout << "  Camera Tracker Server\n";
    std::cout << "--------------------------------------------\n";
    std::cout << "  Home   : " << cfg.lat << " deg, " << cfg.lon << " deg @ " << cfg.elevation << " m\n";
    std::cout << "  Home bearing (pan=0): " << cfg.bearing << " deg\n";
    std::cout << "  Pan direction: " << (cfg.pan_clockwise ? "clockwise" : "counter-clockwise") << "\n";
    std::cout << "--------------------------------------------\n";
    std::cout << "  Listening on port 12345...\n\n";

    int serverFd;
    struct sockaddr_in address;
    int addrlen = sizeof(address);

    if ((serverFd = socket(AF_INET, SOCK_STREAM, 0)) == 0) { perror("socket failed"); return 1; }

    int opt = 1;
    setsockopt(serverFd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    address.sin_family      = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port        = htons(12345);

    if (bind(serverFd, reinterpret_cast<struct sockaddr*>(&address), sizeof(address)) < 0) {
        perror("bind failed"); return 1;
    }
    if (listen(serverFd, 3) < 0) { perror("listen"); return 1; }

    while (true) {
        int newSocket = accept(serverFd,
            reinterpret_cast<struct sockaddr*>(&address),
            reinterpret_cast<socklen_t*>(&addrlen));
        if (newSocket < 0) { perror("accept"); continue; }

        char buffer[1024] = {0};
        const int bytesRead = read(newSocket, buffer, sizeof(buffer));
        if (bytesRead <= 0) { close(newSocket); continue; }

        const std::string request = trim(std::string(buffer, bytesRead));
        if (request == "stats") {
            sendLine(newSocket, buildStatsResponse());
            close(newSocket);
            continue;
        }

        std::cout << "Incoming request: " << request << "\n";

        char hexBuffer[64] = {0};
        double lat = 0.0, lon = 0.0, alt = 0.0;
        if (std::sscanf(request.c_str(), "%63[^,],%lf,%lf,%lf", hexBuffer, &lat, &lon, &alt) != 4) {
            sendLine(newSocket, "ERROR invalid_request\n");
            close(newSocket);
            continue;
        }
        const std::string hexCode = sanitizeHexCode(trim(hexBuffer));

        bool expected = false;
        if (!busy.compare_exchange_strong(expected, true)) {
            sendLine(newSocket, "BUSY\n");
            close(newSocket);
            continue;
        }

        std::thread trackerThread(trackPlane, hexCode, lat, lon, alt, std::ref(cfg), newSocket);
        trackerThread.detach();
    }

    return 0;
}
