#include "../include/medical_vision/chest_x_ray_analyzer.hpp"
#include <opencv2/imgproc.hpp>
#include <chrono>

namespace medical_vision {

// Initialize static members
const std::vector<std::string> ChestXRayAnalyzer::pathologyNames_ = {
    "Atelectasis", "Consolidation", "Infiltration",
    "Pneumothorax", "Edema", "Emphysema",
    "Fibrosis", "Effusion", "Pneumonia",
    "Pleural_Thickening", "Cardiomegaly",
    "Nodule", "Mass", "Hernia"
};


bool ChestXRayAnalyzer::loadModel(const ModelConfig& config) {
    try {
        // Load network
        net_ = cv::dnn::readNet(config.modelPath, config.configPath);
        
        // Configure backend
        if (config.useGPU) {
            net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        } else {
            net_.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
            net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        }

        config_ = config;
        isModelLoaded_ = true;
        return true;
    }
    catch (const cv::Exception& e) {
        isModelLoaded_ = false;
        throw std::runtime_error("Failed to load model: " + std::string(e.what()));
    }
}

ChestXRayAnalyzer::AnalysisResult ChestXRayAnalyzer::analyze(const cv::Mat& image) {
    AnalysisResult result;
    
    try {
        if (!isModelLoaded()) {
            result.errorMessage = "Model not loaded";
            return result;
        }

        if (!validateInput(image)) {
            result.errorMessage = "Invalid input image";
            return result;
        }

        auto start = std::chrono::high_resolution_clock::now();

        // Preprocessing
        cv::Mat blob = preprocessImage(image);
        
        // Forward pass
        net_.setInput(blob);
        cv::Mat outputs = net_.forward();

        // Postprocessing
        result.detections = postprocessOutputs(outputs);
        
        // Generate heatmaps if requested
        if (config_.generateHeatmaps) {
            cv::Mat features = net_.getLayer(net_.getLayerNames().back())->blobs[0];
            for (auto& detection : result.detections) {
                detection.heatmap = generateHeatmap(image, features);
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        result.processingTime = std::chrono::duration<double>(end - start).count();
        result.success = true;
        result.processedImage = image.clone();

    } catch (const std::exception& e) {
        result.success = false;
        result.errorMessage = e.what();
    }

    return result;
}

cv::Mat ChestXRayAnalyzer::preprocessImage(const cv::Mat& image) const {
    cv::Mat processed;
    
    // Convert to float and normalize
    image.convertTo(processed, CV_32F, PIXEL_SCALE);

    // Resize to network input size
    cv::resize(processed, processed, config_.inputSize);

    // Normalize using ImageNet mean and std
    processed = (processed - MEAN_VAL) / STD_VAL;

    // Create blob
    return cv::dnn::blobFromImage(processed);
}

std::vector<ChestXRayAnalyzer::Detection> ChestXRayAnalyzer::postprocessOutputs(
    const cv::Mat& outputs) const {
    
    std::vector<Detection> detections;
    float* data = (float*)outputs.data;
    
    for (size_t i = 0; i < pathologyNames_.size(); ++i) {
        float confidence = data[i];
        if (confidence >= config_.confidenceThreshold) {
            Detection det;
            det.pathology = pathologyNames_[i];
            det.confidence = confidence;
            detections.push_back(det);
        }
    }

    return detections;
}

cv::Mat ChestXRayAnalyzer::generateHeatmap(
    const cv::Mat& image, const cv::Mat& features) const {
    
    cv::Mat heatmap;
    
    // Convert feature map to heatmap
    cv::normalize(features, heatmap, 0, 255, cv::NORM_MINMAX);
    heatmap.convertTo(heatmap, CV_8U);
    
    // Apply colormap
    cv::applyColorMap(heatmap, heatmap, cv::COLORMAP_JET);
    
    // Resize to original image size
    cv::resize(heatmap, heatmap, image.size());
    
    return heatmap;
}

bool ChestXRayAnalyzer::validateInput(const cv::Mat& image) const {
    if (image.empty()) return false;
    if (image.channels() != CHANNEL_COUNT) return false;
    if (image.type() != CV_8UC1) return false;
    return true;
}

std::vector<ChestXRayAnalyzer::AnalysisResult> ChestXRayAnalyzer::analyzeBatch(
    const std::vector<cv::Mat>& images, size_t batchSize) {
    
    std::vector<AnalysisResult> results;
    results.reserve(images.size());

    for (size_t i = 0; i < images.size(); i += batchSize) {
        size_t currentBatchSize = std::min(batchSize, images.size() - i);
        std::vector<cv::Mat> batch(images.begin() + i, 
                                 images.begin() + i + currentBatchSize);
        
        // Process batch
        for (const auto& image : batch) {
            results.push_back(analyze(image));
        }
    }

    return results;
}

bool ChestXRayAnalyzer::isModelLoaded() const {
    return isModelLoaded_;
}

std::vector<std::string> ChestXRayAnalyzer::getAvailablePathologies() const {
    return pathologyNames_;
}

void ChestXRayAnalyzer::setConfidenceThreshold(float threshold) {
    if (threshold >= 0.0f && threshold <= 1.0f) {
        config_.confidenceThreshold = threshold;
    }
}

} // namespace medical_vision