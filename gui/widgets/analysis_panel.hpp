#pragma once

#include <QtWidgets/QGroupBox>
#include <QtWidgets/QLabel>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QDoubleSpinBox>
#include <QtWidgets/QProgressBar>
#include <QtWidgets/QTableWidget>
#include "../../include/medical_vision/chest_x_ray_analyzer.hpp"

class AnalysisPanel : public QGroupBox {
    Q_OBJECT

public:
    explicit AnalysisPanel(QWidget* parent = nullptr);
    ~AnalysisPanel() = default;

    // Main interface
    void analyzeImage(const cv::Mat& image);
    bool loadModel(const QString& modelPath, const QString& configPath);
    bool GetIsModelLoaded(){ return isModelLoaded;}

signals:
    void analysisCompleted(const medical_vision::ChestXRayAnalyzer::AnalysisResult& result);
    void heatmapRequested(const QString& pathology);
    void confidenceThresholdChanged(float value);

private slots:
    void updateResults(const medical_vision::ChestXRayAnalyzer::AnalysisResult& result);
    void toggleHeatmap(bool checked);
    void updateConfidenceThreshold(double value);

private:
    void setupUI();
    void setupConnections();
    void updateControlsState(bool enabled);
    void displayError(const QString& message);
    void clearResults();

    // UI Components
    QPushButton* analyzeButton;
    QCheckBox* showHeatmapCheck;
    QDoubleSpinBox* confidenceThresholdSpin;
    QProgressBar* progressBar;
    QTableWidget* resultsTable;
    QLabel* processingTimeLabel;
    QLabel* statusLabel;

    // Analysis
    medical_vision::ChestXRayAnalyzer chest_analyzer;
    bool isModelLoaded{false};

    // Constants
    static constexpr int TABLE_COLUMNS = 2;  // Pathology, Confidence
    static constexpr double DEFAULT_CONFIDENCE_THRESHOLD = 0.5;
};