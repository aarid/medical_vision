#pragma once

#include <QtWidgets/QMainWindow>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QLabel>

#include "widgets/image_viewer.hpp"
#include "widgets/histogram_viewer.hpp"
#include "widgets/processing_panel.hpp"
#include "widgets/feature_panel.hpp"
#include "widgets/segmentation_panel.hpp"

#include "../include/medical_vision/image_preprocessor.hpp"
#include "../include/medical_vision/feature_detector.hpp"
#include "../include/medical_vision/segmentation.hpp"

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(QWidget* parent = nullptr);
    ~MainWindow() = default;

private slots:
    void openFolder();
    void nextImage();
    void previousImage();
    void processImage();
    void handleSeedPlacement(cv::Point pos, Qt::MouseButton button);

private:
    void setupUI();
    void setupMenus();
    void setupConnections();
    void updateNavigationState();
    void loadCurrentImage();

    // UI Components
    ImageViewer* originalViewer{nullptr};
    ImageViewer* processedViewer{nullptr};
    HistogramViewer* histogramViewer{nullptr};
    ProcessingPanel* processingPanel{nullptr};
    FeaturePanel* featurePanel{nullptr};
    SegmentationPanel* segmentationPanel{nullptr};
    
    // Navigation controls
    QPushButton* prevButton{nullptr};
    QPushButton* nextButton{nullptr};
    QLabel* imageCountLabel{nullptr};

    // Processing core
    medical_vision::ImagePreprocessor processor;
    medical_vision::FeatureDetector featureDetector;
    medical_vision::Segmentation segmentation;

    // Image data
    QStringList imageFiles;
    size_t currentImageIndex{0};
};