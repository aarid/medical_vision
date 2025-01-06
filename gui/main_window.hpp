#pragma once

#include <QtWidgets/QMainWindow>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QLabel>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QDoubleSpinBox>
#include <QtWidgets/QListWidget>
#include <QtWidgets/QGroupBox>
#include "../include/medical_vision/image_preprocessor.hpp"
#include "../include/medical_vision/feature_detector.hpp"
#include "../include/medical_vision/segmentation.hpp"

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow() = default;

private slots:
    void selectFolder();
    void nextImage();
    void previousImage();
    void updateImage();
    void processImage();
    void processFeatures();
    void processSegmentation();
    void updateDisplay();

private:
    // UI Setup functions
    void setupUI();
    void createMenus();
    void updateNavigationControls();
    QGroupBox* createProcessingGroup();
    QGroupBox* createFeatureDetectionGroup();
    QGroupBox* createSegmentationGroup();
    QImage matToQImage(const cv::Mat& mat);

    // Core components
    QWidget* centralWidget;
    medical_vision::ImagePreprocessor processor;
    medical_vision::FeatureDetector featureDetector;
    QStringList imageFiles;
    size_t currentImageIndex{0};

    // Navigation components
    QPushButton* prevButton;
    QPushButton* nextButton;
    QLabel* imageCountLabel;

    // Display components
    QLabel* imageViewerOriginal;
    QLabel* imageViewerProcessed;
    QLabel* histogramView;

    // Image processing controls
    QCheckBox* denoiseCheck;
    QCheckBox* claheCheck;
    QCheckBox* sharpenCheck;
    QDoubleSpinBox* strengthSpinner;
    QListWidget* pipelineList;

    // Feature detection controls
    QComboBox* edgeDetectorCombo;
    QComboBox* keypointDetectorCombo;
    QCheckBox* showEdgesCheck;
    QCheckBox* showKeypointsCheck;
    QSpinBox* threshold1Spin;
    QSpinBox* threshold2Spin;
    QSpinBox* apertureSizeSpin;
    QSpinBox* maxKeypointsSpin;

    // Feature detection results
    cv::Mat edgeResult;
    std::vector<cv::KeyPoint> keypointResult;

    // Segmentation controls

    QComboBox* segmentationMethodCombo;
    QSpinBox* thresholdSpin;
    QSpinBox* maxValueSpin;
    QSpinBox* blockSizeSpin;
    QDoubleSpinBox* paramCSpin;
    QCheckBox* invertColorsCheck;
    QCheckBox* showSegmentationCheck;

     // Segmentation
    medical_vision::Segmentation segmentation;
    cv::Mat segmentationResult;
    std::vector<cv::Point> seedPoints;  // for region growing
};