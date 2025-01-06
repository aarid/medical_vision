/**
 * @file segmentation.hpp
 * @brief Header file for image segmentation operations
 */

#pragma once

#include <opencv2/core.hpp>
#include <vector>
#include <string>

namespace medical_vision {

class Segmentation {
public:
    enum class Method {
        THRESHOLD,          // Basic threshold
        OTSU,              // Otsu's method
        ADAPTIVE_MEAN,     // Adaptive threshold with mean
        ADAPTIVE_GAUSSIAN, // Adaptive threshold with gaussian
        REGION_GROWING,    // Region growing from seed points
        WATERSHED,         // Marker-based watershed
        GRAPH_CUT         // Graph cut segmentation
    };

    struct ThresholdParams {
        double threshold{128};
        double maxValue{255};
        bool invertColors{false};
    };

    struct AdaptiveParams {
        int blockSize{11};
        double C{2.0};
        double maxValue{255};
        bool invertColors{false};
    };

    struct RegionGrowingParams {
        std::vector<cv::Point> seeds;
        double threshold{20.0};    // Intensity difference threshold
        int connectivity{8};       // 4 or 8 connectivity
    };

    struct WatershedParams {
    bool useDistanceTransform{true};
    std::vector<cv::Point> foregroundSeeds;
    std::vector<cv::Point> backgroundSeeds;
    };

    struct GraphCutParams {
        cv::Rect foregroundRect;   // Rectangle containing foreground
        cv::Rect backgroundRect;   // Rectangle containing background
        double lambda{50.0};       // Weight parameter
    };

public:
    Segmentation() = default;
    ~Segmentation() = default;

    // Disable copy
    Segmentation(const Segmentation&) = delete;
    Segmentation& operator=(const Segmentation&) = delete;

    /**
     * @brief Main segmentation function
     * @param input Input image
     * @param method Segmentation method to use
     * @param params Parameters for the selected method (as void*)
     * @return Binary mask of segmentation result
     */
    cv::Mat segment(const cv::Mat& input, Method method, const void* params = nullptr);

    /**
     * @brief Apply threshold segmentation
     * @param input Input image
     * @param params Threshold parameters
     * @return Binary mask
     */
    cv::Mat threshold(const cv::Mat& input, const ThresholdParams& params);

    /**
     * @brief Apply Otsu's thresholding
     * @param input Input image
     * @return Binary mask
     */
    cv::Mat otsuThreshold(const cv::Mat& input);

    /**
     * @brief Apply adaptive thresholding
     * @param input Input image
     * @param params Adaptive threshold parameters
     * @return Binary mask
     */
    cv::Mat adaptiveThreshold(const cv::Mat& input, const AdaptiveParams& params);

    /**
     * @brief Apply region growing segmentation
     * @param input Input image
     * @param params Region growing parameters
     * @return Binary mask
     */
    cv::Mat regionGrowing(const cv::Mat& input, const RegionGrowingParams& params);

    /**
     * @brief Apply watershed segmentation
     * @param input Input image
     * @param params Watershed parameters
     * @return Labeled image
     */
    cv::Mat watershed(const cv::Mat& input, const WatershedParams& params);

    /**
     * @brief Apply graph cut segmentation
     * @param input Input image
     * @param params Graph cut parameters
     * @return Binary mask
     */
    cv::Mat graphCut(const cv::Mat& input, const GraphCutParams& params);

    /**
     * @brief Get contours from binary mask
     * @param mask Binary segmentation mask
     * @return Vector of contours
     */
    std::vector<std::vector<cv::Point>> getContours(const cv::Mat& mask);

    /**
     * @brief Draw segmentation result on image
     * @param input Original image
     * @param mask Segmentation mask
     * @param alpha Transparency (0-1)
     * @return Visualization of segmentation
     */
    cv::Mat drawSegmentation(const cv::Mat& input, const cv::Mat& mask, double alpha = 0.5);

private:
    /**
     * @brief Validate input image
     * @param input Input image to validate
     * @return true if valid
     */
    bool validateInput(const cv::Mat& input);

    /**
     * @brief Prepare image for segmentation
     * @param input Input image
     * @return Preprocessed image
     */
    cv::Mat prepareImage(const cv::Mat& input);

    /**
     * @brief Post-process segmentation result
     * @param mask Segmentation mask
     * @return Cleaned mask
     */
    cv::Mat postProcessMask(const cv::Mat& mask);
};

} // namespace medical_vision