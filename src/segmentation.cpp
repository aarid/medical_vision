/**
 * @file segmentation.cpp
 * @brief Implementation of image segmentation operations
 */

#include "../include/medical_vision/segmentation.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>
#include <stdexcept>

namespace medical_vision {

bool Segmentation::validateInput(const cv::Mat& input) {
    if (input.empty()) {
        throw std::runtime_error("Input image is empty");
    }
    return true;
}

cv::Mat Segmentation::prepareImage(const cv::Mat& input) {
    cv::Mat processed;
    if (input.channels() > 1) {
        cv::cvtColor(input, processed, cv::COLOR_BGR2GRAY);
    } else {
        processed = input.clone();
    }
    
    // Ensure 8-bit image
    if (processed.type() != CV_8UC1) {
        processed.convertTo(processed, CV_8UC1);
    }
    
    return processed;
}

cv::Mat Segmentation::postProcessMask(const cv::Mat& mask) {
    cv::Mat processed = mask.clone();
    
    // Ensure binary mask
    if (processed.type() != CV_8UC1) {
        processed.convertTo(processed, CV_8UC1);
    }
    
    // Optional: Remove small objects and fill holes
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(processed, processed, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(processed, processed, cv::MORPH_CLOSE, kernel);
    
    return processed;
}

cv::Mat Segmentation::segment(const cv::Mat& input, Method method, const void* params) {
    try {
        validateInput(input);
        cv::Mat result;

        switch (method) {
            case Method::THRESHOLD:
                result = threshold(input, params ? *static_cast<const ThresholdParams*>(params) 
                                               : ThresholdParams());
                break;
            case Method::OTSU:
                result = otsuThreshold(input);
                break;
            case Method::ADAPTIVE_MEAN:
            case Method::ADAPTIVE_GAUSSIAN:
                result = adaptiveThreshold(input, params ? *static_cast<const AdaptiveParams*>(params) 
                                                       : AdaptiveParams());
                break;
            case Method::REGION_GROWING:
                result = regionGrowing(input, params ? *static_cast<const RegionGrowingParams*>(params) 
                                                   : RegionGrowingParams());
                break;
            case Method::WATERSHED:
                // result = watershed(input, params ? *static_cast<const WatershedParams*>(params) : WatershedParams());
                break;
            case Method::GRAPH_CUT:
                // result = graphCut(input, params ? *static_cast<const GraphCutParams*>(params) : GraphCutParams());
                break;
            default:
                throw std::runtime_error("Unknown segmentation method");
        }

        return postProcessMask(result);
    }
    catch (const std::exception& e) {
        throw std::runtime_error(std::string("Segmentation failed: ") + e.what());
    }
}

cv::Mat Segmentation::threshold(const cv::Mat& input, const ThresholdParams& params) {
    cv::Mat processed = prepareImage(input);
    cv::Mat result;
    
    cv::threshold(processed, result, params.threshold, params.maxValue,
                 params.invertColors ? cv::THRESH_BINARY_INV : cv::THRESH_BINARY);
    
    return result;
}

cv::Mat Segmentation::otsuThreshold(const cv::Mat& input) {
    cv::Mat processed = prepareImage(input);
    cv::Mat result;
    
    cv::threshold(processed, result, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    
    return result;
}

cv::Mat Segmentation::adaptiveThreshold(const cv::Mat& input, const AdaptiveParams& params) {
    cv::Mat processed = prepareImage(input);
    cv::Mat result;
    
    cv::adaptiveThreshold(processed, result,
                         params.maxValue,
                         cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                         params.invertColors ? cv::THRESH_BINARY_INV : cv::THRESH_BINARY,
                         params.blockSize,
                         params.C);
    
    return result;
}

cv::Mat Segmentation::regionGrowing(const cv::Mat& input, const RegionGrowingParams& params) {
    cv::Mat processed = prepareImage(input);
    cv::Mat mask = cv::Mat::zeros(processed.size(), CV_8UC1);
    
    if (params.seeds.empty()) {
        throw std::runtime_error("No seeds provided for region growing");
    }

    // Implementation of region growing algorithm
    std::vector<cv::Point> seeds = params.seeds;
    for (const auto& seed : seeds) {
        if (mask.at<uchar>(seed) == 255) continue;
        
        std::vector<cv::Point> queue;
        queue.push_back(seed);
        
        while (!queue.empty()) {
            cv::Point current = queue.back();
            queue.pop_back();
            
            if (mask.at<uchar>(current) == 255) continue;
            
            mask.at<uchar>(current) = 255;
            uchar currentIntensity = processed.at<uchar>(current);
            
            // Check 8-connected neighbors
            for (int i = -1; i <= 1; i++) {
                for (int j = -1; j <= 1; j++) {
                    if (i == 0 && j == 0) continue;
                    
                    cv::Point neighbor(current.x + j, current.y + i);
                    
                    if (neighbor.x < 0 || neighbor.x >= processed.cols ||
                        neighbor.y < 0 || neighbor.y >= processed.rows)
                        continue;
                    
                    if (mask.at<uchar>(neighbor) == 255) continue;
                    
                    uchar neighborIntensity = processed.at<uchar>(neighbor);
                    if (std::abs(currentIntensity - neighborIntensity) <= params.threshold) {
                        queue.push_back(neighbor);
                    }
                }
            }
        }
    }
    
    return mask;
}

std::vector<std::vector<cv::Point>> Segmentation::getContours(const cv::Mat& mask) {
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    
    cv::findContours(mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    return contours;
}

cv::Mat Segmentation::drawSegmentation(const cv::Mat& input, const cv::Mat& mask, double alpha) {
    cv::Mat result;
    if (input.channels() == 1) {
        cv::cvtColor(input, result, cv::COLOR_GRAY2BGR);
    } else {
        result = input.clone();
    }
    
    cv::Mat overlay = result.clone();
    overlay.setTo(cv::Scalar(0, 0, 255), mask);  // Red overlay for segmentation
    
    cv::addWeighted(overlay, alpha, result, 1.0 - alpha, 0, result);
    
    return result;
}

} // namespace medical_vision