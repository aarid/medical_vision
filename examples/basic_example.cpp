/**
 * @file basic_example.cpp
 * @brief Generic viewer for testing different image processing methods
 */

#include "../include/medical_vision/image_preprocessor.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <filesystem>
#include <vector>
#include <algorithm>

// Processing method to test
void processImage(medical_vision::ImagePreprocessor& processor) {
    // Example: Chain of processing methods
    processor.normalize();
    processor.clahe(2.0);
    processor.sharpen(1.2);
    
    // Alternative examples (commented out):
    // processor.denoise(medical_vision::ImagePreprocessor::NoiseReductionMethod::BILATERAL);
    // processor.histogramProcessing(medical_vision::ImagePreprocessor::HistogramMethod::EQUALIZATION);
    // processor.unsharpMask(1.0, 1.5);
}

/**
 * @brief Create side-by-side comparison view
 * @param original Original image
 * @param processed Processed image
 * @param title1 Title for original image
 * @param title2 Title for processed image
 * @return Combined view
 */
cv::Mat createComparisonView(const cv::Mat& original, const cv::Mat& processed,
                           const std::string& title1, const std::string& title2) {
    // Fixed window size
    cv::Size screenSize(1280, 1024);

    // Calculate target size for each image
    int targetWidth = (screenSize.width / 2) - 20;  // -20 for padding
    int targetHeight = screenSize.height - 100;     // -100 for titles and padding

    // Resize images
    cv::Mat resized1, resized2;
    double scale = std::min(targetWidth / (double)original.cols, 
                           targetHeight / (double)original.rows);
    
    cv::resize(original, resized1, cv::Size(), scale, scale);
    cv::resize(processed, resized2, cv::Size(), scale, scale);

    // Convert to BGR if grayscale
    if (resized1.channels() == 1) {
        cv::cvtColor(resized1, resized1, cv::COLOR_GRAY2BGR);
    }
    if (resized2.channels() == 1) {
        cv::cvtColor(resized2, resized2, cv::COLOR_GRAY2BGR);
    }

    // Create output image
    cv::Mat output = cv::Mat::zeros(targetHeight + 50, targetWidth * 2, CV_8UC3);

    // Copy images
    resized1.copyTo(output(cv::Rect(0, 50, resized1.cols, resized1.rows)));
    resized2.copyTo(output(cv::Rect(targetWidth, 50, resized2.cols, resized2.rows)));

    // Add titles
    cv::putText(output, title1, cv::Point(10, 30), 
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255,255,255), 2);
    cv::putText(output, title2, cv::Point(targetWidth + 10, 30), 
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255,255,255), 2);

    return output;
}

int main() {
    // Create window with fixed size
    cv::namedWindow("Display", cv::WINDOW_NORMAL);
    cv::resizeWindow("Display", 1280, 1024);

    // Load first 100 images from directory
    std::string folderPath = "D:/enhanced_projects/medical_vision/data/test_images/";
    std::vector<std::string> imageFiles;
    int count = 0;
    for (const auto& entry : std::filesystem::directory_iterator(folderPath)) {
        if (entry.path().extension() == ".png" || entry.path().extension() == ".jpg") {
            imageFiles.push_back(entry.path().string());
            count++;
            if (count >= 100) break;
        }
    }
    std::sort(imageFiles.begin(), imageFiles.end());


    if (imageFiles.empty()) {
        std::cout << "No images found in directory!" << std::endl;
        return -1;
    }

    medical_vision::ImagePreprocessor processor;
    size_t currentImageIndex = 0;
    
    // Main processing loop
    while (true) {
        // Load and process current image
        if (!processor.loadImage(imageFiles[currentImageIndex])) {
            std::cout << "Failed to load image: " << imageFiles[currentImageIndex] << std::endl;
            continue;
        }

        // Process image
        processImage(processor);

        // Create comparison view
        cv::Mat display = createComparisonView(
            processor.getOriginalImage(),
            processor.getImage(),
            "Original",
            "Processed"
        );

        // Show results
        std::string windowTitle = "Image " + std::to_string(currentImageIndex + 1) + "/" + 
                                std::to_string(imageFiles.size());
        cv::setWindowTitle("Display", windowTitle);
        cv::imshow("Display", display);

        // Handle keyboard input
        int key = cv::waitKey(0);
        
        if (key == 27) { // ESC - Exit
            break;
        }
        else if (key == 'b') { // Left arrow - Previous image
            if (currentImageIndex > 0) currentImageIndex--;
        }
        else if (key == 'n') { // Right arrow - Next image
            if (currentImageIndex < imageFiles.size() - 1) currentImageIndex++;
        }
    }

    cv::destroyAllWindows();
    return 0;
}