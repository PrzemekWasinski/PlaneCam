#include "../image_recognition/image_recognition.hpp"

#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include <opencv2/imgcodecs.hpp>

namespace fs = std::filesystem;

namespace {

void predictSingleImage(
    const fs::path& imagePath,
    const cv::Ptr<cv::ml::SVM>& svm,
    const cv::HOGDescriptor& hog,
    const cv::Size& targetSize) {
    const cv::Mat image = cv::imread(imagePath.string(), cv::IMREAD_COLOR);
    if (image.empty()) {
        throw std::runtime_error("Unable to read image: " + imagePath.string());
    }

    float rawScore = 0.0f;
    const float predicted = imgrec::predictLabel(svm, image, hog, targetSize, &rawScore);
    const double confidence = 1.0 / (1.0 + std::exp(-std::abs(rawScore)));

    std::cout << imagePath.string() << ": "
              << imgrec::labelToString(predicted)
              << " confidence=" << std::fixed << std::setprecision(4) << confidence
              << " raw_score=" << rawScore << "\n";
}

void predictDirectory(
    const fs::path& dirPath,
    const cv::Ptr<cv::ml::SVM>& svm,
    const cv::HOGDescriptor& hog,
    const cv::Size& targetSize) {
    const auto images = imgrec::listImagesRecursive(dirPath);
    if (images.empty()) {
        throw std::runtime_error("No images found in directory: " + dirPath.string());
    }

    int aircraftCount = 0;
    int skyCount = 0;
    for (const auto& imagePath : images) {
        const cv::Mat image = cv::imread(imagePath.string(), cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cerr << "Skipping unreadable image: " << imagePath << "\n";
            continue;
        }

        float rawScore = 0.0f;
        const float predicted = imgrec::predictLabel(svm, image, hog, targetSize, &rawScore);
        const double confidence = 1.0 / (1.0 + std::exp(-std::abs(rawScore)));
        if (predicted >= 0.5f) {
            ++aircraftCount;
        } else {
            ++skyCount;
        }

        std::cout << imagePath.string() << ": "
                  << imgrec::labelToString(predicted)
                  << " confidence=" << std::fixed << std::setprecision(4) << confidence
                  << " raw_score=" << rawScore << "\n";
    }

    std::cout << "Summary: AIRCRAFT=" << aircraftCount << " SKY=" << skyCount << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const fs::path executablePath = argc > 0 ? fs::absolute(fs::path(argv[0])) : fs::current_path() / "predict";
        const fs::path projectDir = executablePath.parent_path().parent_path();
        const fs::path defaultInputPath = projectDir / "tests" / "image_recognition_test" / "data" / "test";

        const fs::path inputPath = argc >= 2 ? fs::path(argv[1]) : defaultInputPath;

        fs::path modelPath;
        if (argc >= 3) {
            modelPath = fs::path(argv[2]);
        } else {
            const std::vector<fs::path> modelCandidates = {
                projectDir / "camera_module" / "model" / "aircraft_svm.yml",
                projectDir / "model" / "aircraft_svm.yml",
                executablePath.parent_path().parent_path() / "camera_module" / "model" / "aircraft_svm.yml",
                executablePath.parent_path().parent_path() / "tests" / "image_recognition_test" / "model" / "aircraft_svm.yml",
                projectDir / "tests" / "image_recognition_test" / "model" / "aircraft_svm.yml"
            };
            for (const auto& candidate : modelCandidates) {
                if (fs::exists(candidate)) {
                    modelPath = candidate;
                    break;
                }
            }
        }

        if (modelPath.empty()) {
            throw std::runtime_error("Unable to locate aircraft_svm.yml");
        }

        auto svm = cv::Algorithm::load<cv::ml::SVM>(modelPath.string());
        if (svm.empty()) {
            throw std::runtime_error("Unable to load model: " + modelPath.string());
        }

        const cv::HOGDescriptor hog = imgrec::createHogDescriptor();
        const cv::Size targetSize(128, 128);

        if (fs::is_directory(inputPath)) {
            predictDirectory(inputPath, svm, hog, targetSize);
        } else {
            predictSingleImage(inputPath, svm, hog, targetSize);
        }
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Prediction failed: " << ex.what() << "\n";
        return 1;
    }
}
