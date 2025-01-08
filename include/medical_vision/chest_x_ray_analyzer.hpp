#pragma once

#include <opencv2/dnn.hpp>
#include <opencv2/core.hpp>
#include <string>
#include <vector>
#include <memory>

namespace medical_vision {

class ChestXRayAnalyzer {
public:
    
    // Structures for detections and configurations
    struct Detection {
        std::string pathology;
        float confidence;
        cv::Rect region;        
        cv::Mat heatmap;        
    };

    struct ModelConfig {
        std::string modelPath;
        std::string configPath;
        cv::Size inputSize{224, 224};
        float confidenceThreshold{0.5f};
        bool useGPU{false};
        bool generateHeatmaps{false};
    };

    struct AnalysisResult {
        std::vector<Detection> detections;
        cv::Mat processedImage;
        double processingTime{0.0};
        bool success{false};
        std::string errorMessage;
    };

    ChestXRayAnalyzer() = default;
    ChestXRayAnalyzer(cv::dnn::Net net, ModelConfig config)
        : net_(std::move(net)), config_(std::move(config)) {}
    ~ChestXRayAnalyzer() = default;

    // Main interface
    bool loadModel(const ModelConfig& config);
    AnalysisResult analyze(const cv::Mat& image);
    
    // Utility functions
    bool isModelLoaded() const;
    std::vector<std::string> getAvailablePathologies() const;
    void setConfidenceThreshold(float threshold);
    
    // Batch processing
    std::vector<AnalysisResult> analyzeBatch(
        const std::vector<cv::Mat>& images, 
        size_t batchSize = 1);

private:
    // Internal processing functions
    cv::Mat preprocessImage(const cv::Mat& image) const;
    std::vector<Detection> postprocessOutputs(const cv::Mat& outputs) const;
    cv::Mat generateHeatmap(const cv::Mat& image, const cv::Mat& features) const;
    bool validateInput(const cv::Mat& image) const;

    // Model and configuration
    cv::dnn::Net net_;
    ModelConfig config_;
    bool isModelLoaded_{false};

    // Constants
    static const std::vector<std::string> pathologyNames_;
    static constexpr int CHANNEL_COUNT = 1;  // Grayscale images
    static constexpr float PIXEL_SCALE = 1.0f/255.0f;
    
    // Preprocessing parameters
    static constexpr int TARGET_SIZE = 224;  // DenseNet input size
    static constexpr float MEAN_VAL = 0.485f;  // ImageNet mean
    static constexpr float STD_VAL = 0.229f;   // ImageNet std

    struct ImageStats {
        double min;
        double max;
        double mean;
        double std;
    };

    // Helper function to compute image statistics
    ImageStats computeImageStats(const cv::Mat& image) const;
};

} // namespace medical_vision