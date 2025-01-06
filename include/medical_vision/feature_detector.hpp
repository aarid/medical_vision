/**
 * @file feature_detector.hpp
 * @brief Header file for feature detection operations
 */

#pragma once

#include <opencv2/core.hpp>
#include <vector>
#include <string>

namespace medical_vision {

class FeatureDetector {
public:
    /**
     * @brief Edge detection methods
     */
    enum class EdgeDetector {
        CANNY,
        SOBEL,
        LAPLACIAN
    };

    /**
     * @brief Keypoint detection methods
     */
    enum class KeypointDetector {
        SIFT,
        ORB,
        FAST
    };

    /**
     * @brief Parameters for edge detection
     */
    struct EdgeParams {
        double threshold1{100};    // First threshold for Canny
        double threshold2{200};    // Second threshold for Canny
        int apertureSize{3};      // Aperture size for Sobel/Laplacian
        bool L2gradient{false};    // L2 gradient for Canny
    };

    /**
     * @brief Parameters for keypoint detection
     */
    struct KeypointParams {
        int maxKeypoints{1000};    // Maximum number of keypoints to detect
        float scaleFactor{1.2f};   // Scale factor between levels
        int nlevels{8};           // Number of pyramid levels
        int edgeThreshold{31};    // Edge threshold for ORB
        int fastThreshold{20};    // Threshold for FAST
    };

public:
    FeatureDetector() = default;
    ~FeatureDetector() = default;

    // Disable copy
    FeatureDetector(const FeatureDetector&) = delete;
    FeatureDetector& operator=(const FeatureDetector&) = delete;

    /**
     * @brief Detect edges in an image
     * @param input Input image
     * @param method Edge detection method to use
     * @param params Parameters for edge detection
     * @return Edge map
     */
    cv::Mat detectEdges(const cv::Mat& input, 
                       EdgeDetector method,
                       const EdgeParams& params = EdgeParams());

    /**
     * @brief Detect keypoints in an image
     * @param input Input image
     * @param method Keypoint detection method to use
     * @param params Parameters for keypoint detection
     * @return Vector of detected keypoints
     */
    std::vector<cv::KeyPoint> detectKeypoints(const cv::Mat& input,
                                            KeypointDetector method,
                                            const KeypointParams& params = KeypointParams());

    /**
     * @brief Draw detected keypoints on an image
     * @param input Input image
     * @param keypoints Detected keypoints
     * @return Image with drawn keypoints
     */
    cv::Mat drawKeypoints(const cv::Mat& input,
                         const std::vector<cv::KeyPoint>& keypoints);

    /**
     * @brief Compute texture features using GLCM
     * @param input Input image
     * @return Matrix of texture features
     */
    cv::Mat computeGLCM(const cv::Mat& input);

    /**
     * @brief Extract texture features from GLCM
     * @param input Input GLCM matrix
     * @return Vector of texture features (contrast, correlation, energy, homogeneity)
     */
    std::vector<double> extractTextureFeatures(const cv::Mat& input);

private:
    // Helper functions for edge detection
    cv::Mat applyCanny(const cv::Mat& input, const EdgeParams& params);
    cv::Mat applySobel(const cv::Mat& input, const EdgeParams& params);
    cv::Mat applyLaplacian(const cv::Mat& input, const EdgeParams& params);

    // Helper functions for keypoint detection
    std::vector<cv::KeyPoint> applySIFT(const cv::Mat& input, const KeypointParams& params);
    std::vector<cv::KeyPoint> applyORB(const cv::Mat& input, const KeypointParams& params);
    std::vector<cv::KeyPoint> applyFAST(const cv::Mat& input, const KeypointParams& params);

    // Utility functions
    cv::Mat prepareImage(const cv::Mat& input);
    bool validateInput(const cv::Mat& input);
};

} // namespace medical_vision