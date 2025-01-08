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
    try {
        cv::Mat processed;
        
        // 1. Vérifier que c'est bien une image en niveaux de gris
        if (image.channels() != 1) {
            cv::cvtColor(image, processed, cv::COLOR_BGR2GRAY);
        } else {
            processed = image.clone();
        }

        // 2. Redimensionner en gardant le ratio
        cv::Mat resized;
        double scale = std::min(
            config_.inputSize.width / static_cast<double>(processed.cols),
            config_.inputSize.height / static_cast<double>(processed.rows)
        );
        cv::Size newSize(
            static_cast<int>(processed.cols * scale),
            static_cast<int>(processed.rows * scale)
        );
        cv::resize(processed, resized, newSize);

        // 3. Padding pour atteindre la taille cible
        cv::Mat padded = cv::Mat::zeros(config_.inputSize, CV_8UC1);
        int top = (config_.inputSize.height - resized.rows) / 2;
        int left = (config_.inputSize.width - resized.cols) / 2;
        resized.copyTo(padded(cv::Rect(left, top, resized.cols, resized.rows)));

        // 4. Normalisation
        padded.convertTo(processed, CV_32F, PIXEL_SCALE);
        processed = (processed - MEAN_VAL) / STD_VAL;

        // 5. Répéter le canal en niveaux de gris pour simuler 3 canaux
        std::vector<cv::Mat> channels = {processed, processed, processed};
        cv::merge(channels, processed);

        // 6. Créer le blob pour le réseau
        return cv::dnn::blobFromImage(processed);
    }
    catch (const cv::Exception& e) {
        throw std::runtime_error("Preprocessing failed: " + std::string(e.what()));
    }
}

std::vector<ChestXRayAnalyzer::Detection> ChestXRayAnalyzer::postprocessOutputs(
    const cv::Mat& outputs) const {
    
    std::vector<Detection> detections;
    cv::Mat probabilities;
    
    // Appliquer sigmoid pour convertir en probabilités
    cv::exp(-outputs, probabilities);
    probabilities = 1.0 / (1.0 + probabilities);

    // Extraire les détections au-dessus du seuil
    float* data = (float*)probabilities.data;
    for (size_t i = 0; i < pathologyNames_.size(); ++i) {
        float confidence = data[i];
        if (confidence >= config_.confidenceThreshold) {
            Detection det;
            det.pathology = pathologyNames_[i];
            det.confidence = confidence;
            detections.push_back(det);
        }
    }

    // Trier par confiance décroissante
    std::sort(detections.begin(), detections.end(),
              [](const Detection& a, const Detection& b) {
                  return a.confidence > b.confidence;
              });

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
    if (image.empty()) {
        throw std::runtime_error("Empty image");
    }

    // Vérifier la taille minimale
    if (image.rows < 200 || image.cols < 200) {
        throw std::runtime_error("Image too small");
    }

    // Vérifier le type d'image
    if (image.type() != CV_8UC1 && image.type() != CV_8UC3) {
        throw std::runtime_error("Unsupported image type");
    }

    // Vérifier le contraste et la luminosité
    ImageStats stats = computeImageStats(image);
    if (stats.max - stats.min < 50) {
        throw std::runtime_error("Image contrast too low");
    }

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


ChestXRayAnalyzer::ImageStats ChestXRayAnalyzer::computeImageStats(const cv::Mat& image) const {
    ImageStats stats;
    cv::Mat gray;
    
    if (image.channels() == 1) {
        gray = image;
    } else {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    }

    cv::minMaxLoc(gray, &stats.min, &stats.max);
    cv::Scalar mean, stddev;
    cv::meanStdDev(gray, mean, stddev);
    stats.mean = mean[0];
    stats.std = stddev[0];

    return stats;
}

} // namespace medical_vision