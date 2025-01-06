/**
 * @file feature_detector.cpp
 * @brief Implementation of feature detection operations
 */

#include "../include/medical_vision/feature_detector.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <stdexcept>

namespace medical_vision {

// Utility functions
bool FeatureDetector::validateInput(const cv::Mat& input) {
    if (input.empty()) {
        throw std::runtime_error("Input image is empty");
    }
    return true;
}

cv::Mat FeatureDetector::prepareImage(const cv::Mat& input) {
    cv::Mat processed;
    if (input.channels() > 1) {
        cv::cvtColor(input, processed, cv::COLOR_BGR2GRAY);
    } else {
        processed = input.clone();
    }
    return processed;
}

// Edge detection implementations
cv::Mat FeatureDetector::detectEdges(const cv::Mat& input, 
                                   EdgeDetector method,
                                   const EdgeParams& params) {
    try {
        validateInput(input);
        cv::Mat processed = prepareImage(input);

        switch (method) {
            case EdgeDetector::CANNY:
                return applyCanny(processed, params);
            case EdgeDetector::SOBEL:
                return applySobel(processed, params);
            case EdgeDetector::LAPLACIAN:
                return applyLaplacian(processed, params);
            default:
                throw std::runtime_error("Unknown edge detection method");
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error(std::string("Edge detection failed: ") + e.what());
    }
}

cv::Mat FeatureDetector::applyCanny(const cv::Mat& input, const EdgeParams& params) {
    cv::Mat edges;
    cv::Canny(input, edges, 
              params.threshold1, 
              params.threshold2,
              params.apertureSize,
              params.L2gradient);
    return edges;
}

cv::Mat FeatureDetector::applySobel(const cv::Mat& input, const EdgeParams& params) {
    cv::Mat gradX, gradY, grad;
    
    // Compute gradients in x and y directions
    cv::Sobel(input, gradX, CV_16S, 1, 0, params.apertureSize);
    cv::Sobel(input, gradY, CV_16S, 0, 1, params.apertureSize);
    
    // Convert gradients to absolute values
    cv::convertScaleAbs(gradX, gradX);
    cv::convertScaleAbs(gradY, gradY);
    
    // Combine gradients
    cv::addWeighted(gradX, 0.5, gradY, 0.5, 0, grad);
    
    return grad;
}

cv::Mat FeatureDetector::applyLaplacian(const cv::Mat& input, const EdgeParams& params) {
    cv::Mat laplacian, result;
    
    // Apply Laplacian
    cv::Laplacian(input, laplacian, CV_16S, params.apertureSize);
    
    // Convert to absolute values and scale to 8-bit
    cv::convertScaleAbs(laplacian, result);
    
    return result;
}

// Keypoint detection implementations
std::vector<cv::KeyPoint> FeatureDetector::detectKeypoints(const cv::Mat& input,
                                                         KeypointDetector method,
                                                         const KeypointParams& params) {
    try {
        validateInput(input);
        cv::Mat processed = prepareImage(input);

        switch (method) {
            case KeypointDetector::SIFT:
                return applySIFT(processed, params);
            case KeypointDetector::ORB:
                return applyORB(processed, params);
            case KeypointDetector::FAST:
                return applyFAST(processed, params);
            default:
                throw std::runtime_error("Unknown keypoint detection method");
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error(std::string("Keypoint detection failed: ") + e.what());
    }
}

std::vector<cv::KeyPoint> FeatureDetector::applySIFT(const cv::Mat& input, 
                                                    const KeypointParams& params) {
    auto detector = cv::SIFT::create(params.maxKeypoints);
    std::vector<cv::KeyPoint> keypoints;
    detector->detect(input, keypoints);
    return keypoints;
}

std::vector<cv::KeyPoint> FeatureDetector::applyORB(const cv::Mat& input, 
                                                   const KeypointParams& params) {
    auto detector = cv::ORB::create(
        params.maxKeypoints,
        params.scaleFactor,
        params.nlevels,
        params.edgeThreshold
    );
    std::vector<cv::KeyPoint> keypoints;
    detector->detect(input, keypoints);
    return keypoints;
}

std::vector<cv::KeyPoint> FeatureDetector::applyFAST(const cv::Mat& input, 
                                                    const KeypointParams& params) {
    std::vector<cv::KeyPoint> keypoints;
    cv::FAST(input, keypoints, params.fastThreshold);
    
    // Limit number of keypoints if necessary
    if (keypoints.size() > params.maxKeypoints) {
        std::sort(keypoints.begin(), keypoints.end(),
                 [](const cv::KeyPoint& a, const cv::KeyPoint& b) {
                     return a.response > b.response;
                 });
        keypoints.resize(params.maxKeypoints);
    }
    
    return keypoints;
}

cv::Mat FeatureDetector::drawKeypoints(const cv::Mat& input,
                                     const std::vector<cv::KeyPoint>& keypoints) {
    cv::Mat output;
    cv::drawKeypoints(input, keypoints, output, cv::Scalar::all(-1),
                     cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    return output;
}

// Texture analysis implementations
cv::Mat FeatureDetector::computeGLCM(const cv::Mat& input) {
    // Todo : calcul de la matrice de co-occurrence
    throw std::runtime_error("GLCM computation not implemented yet");
}

std::vector<double> FeatureDetector::extractTextureFeatures(const cv::Mat& input) {
    // Todo : extraction des caractéristiques de texture
    throw std::runtime_error("Texture feature extraction not implemented yet");
}

} // namespace medical_vision