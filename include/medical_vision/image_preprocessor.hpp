/**
 * @file image_preprocessor.hpp
 * @brief Header file for image preprocessing operations
 */

#pragma once

#include <opencv2/core.hpp>
#include <string>

namespace medical_vision {

/**
 * @class ImagePreprocessor
 * @brief Class handling various image preprocessing operations
 */
class ImagePreprocessor {
public:
    /**
     * @brief Enum for different types of noise reduction methods
     */
    enum class NoiseReductionMethod {
        GAUSSIAN,
        MEDIAN,
        BILATERAL,
        NLM  // Non-Local Means
    };

    /**
     * @brief Enum for different types of histogram processing
     */
    enum class HistogramMethod {
        EQUALIZATION,
        CLAHE,  // Contrast Limited Adaptive Histogram Equalization
        STRETCHING
    };

public:
    ImagePreprocessor() = default;
    ~ImagePreprocessor() = default;

    // Disable copy
    ImagePreprocessor(const ImagePreprocessor&) = delete;
    ImagePreprocessor& operator=(const ImagePreprocessor&) = delete;

    // Basic operations
    bool loadImage(const std::string& filepath);
    bool saveImage(const std::string& filepath) const;

    // Image information
    cv::Size getImageSize() const;
    int getChannels() const;
    std::string getImageType() const;
    
    // Noise reduction
    bool denoise(NoiseReductionMethod method = NoiseReductionMethod::GAUSSIAN);
    bool gaussianBlur(int kernelSize = 3, double sigma = 1.0);
    bool medianBlur(int kernelSize = 3);
    bool bilateralFilter(int diameter = 9, double sigmaColor = 75, double sigmaSpace = 75);
    bool nonLocalMeans(float h = 3, int templateWindowSize = 7, int searchWindowSize = 21);

    // Contrast and brightness
    bool normalize(double minValue = 0, double maxValue = 255);
    bool adjustContrast(double alpha = 1.0, double beta = 0);
    bool histogramProcessing(HistogramMethod method);
    bool clahe(double clipLimit = 2.0, cv::Size tileGridSize = cv::Size(8, 8));

    // Edge enhancement
    bool sharpen(double strength = 1.0);
    bool unsharpMask(double sigma = 1.0, double strength = 1.5);

    // Utility functions
    bool isLoaded() const { return !image_.empty(); }
    void reset() { image_ = originalImage_.clone(); }
    
    // Getters
    const cv::Mat& getImage() const { return image_; }
    const cv::Mat& getOriginalImage() const { return originalImage_; }
    cv::Mat getHistogram() const;

private:
    cv::Mat image_;          // Current working image
    cv::Mat originalImage_;  // Original image backup
    
    // Utility functions
    bool checkImageLoaded() const;
    void updateOriginalImage();
    bool validateKernelSize(int kernelSize) const;
};

} // namespace medical_vision