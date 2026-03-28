#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"

mkdir -p "$BUILD_DIR"

CXX="${CXX:-g++}"
COMMON_FLAGS=( -std=c++17 -O2 -Wall -Wextra )
EXTRA_FS_FLAGS=()
OPENCV_FLAGS=()
HAVE_OPENCV=0

if pkg-config --exists opencv4; then
    OPENCV_FLAGS=( $(pkg-config --cflags --libs opencv4) )
    HAVE_OPENCV=1
elif pkg-config --exists opencv; then
    OPENCV_FLAGS=( $(pkg-config --cflags --libs opencv) )
    HAVE_OPENCV=1
fi

if ! printf '#include <filesystem>
int main(){ std::filesystem::path p{"."}; return 0; }
'     | "$CXX" -std=c++17 -x c++ - -o /dev/null >/dev/null 2>&1; then
    EXTRA_FS_FLAGS=( -lstdc++fs )
fi

compile() {
    local source_file="$1"
    local output_file="$2"
    shift 2
    echo "Compiling ${source_file} to ${output_file}"
    "$CXX" "${COMMON_FLAGS[@]}" "$SCRIPT_DIR/$source_file" -o "$BUILD_DIR/$output_file" "${EXTRA_FS_FLAGS[@]}" "$@"
}

compile "tools/camera_test.cpp" "camera_test" -pthread
compile "tools/servo_test.cpp" "servo_test" -lpigpio -lrt
compile_opencv() {
    local source_file="$1"
    local output_file="$2"
    shift 2
    echo "Compiling ${source_file} to ${output_file}"
    "$CXX" "${COMMON_FLAGS[@]}" "$SCRIPT_DIR/$source_file" "$SCRIPT_DIR/image_recognition/image_recognition.cpp" -o "$BUILD_DIR/$output_file" "${EXTRA_FS_FLAGS[@]}" "${OPENCV_FLAGS[@]}" "$@"
}

compile "tools/tracker_mapping_test.cpp" "tracker_mapping_test" -lpigpio -lrt -pthread
if [ "$HAVE_OPENCV" -eq 1 ]; then
    compile_opencv "tools/image_predict.cpp" "image_predict"
    compile_opencv "tools/image_train.cpp" "image_train"
else
    echo "Skipping image_predict and image_train: pkg-config could not find opencv4 or opencv"
fi
compile "tracker.cpp" "tracker" -lpigpio -lpthread -lrt

echo "Build complete. Binaries are in: $BUILD_DIR"
