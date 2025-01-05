#pragma once

#include <QtWidgets/QMainWindow>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QLabel>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QDoubleSpinBox>
#include <QtWidgets/QListWidget>
#include <QtWidgets/QGroupBox>
#include "../include/medical_vision/image_preprocessor.hpp"

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

private:
    void setupUI();
    void createMenus();
    void updateNavigationControls();
    QImage matToQImage(const cv::Mat& mat);

    // UI Components
    QWidget* centralWidget;
    
    // Navigation
    QPushButton* prevButton;
    QPushButton* nextButton;
    QLabel* imageCountLabel;
    
    // Image display
    QLabel* imageViewerOriginal;
    QLabel* imageViewerProcessed;
    QLabel* histogramView;
    
    // Processing controls
    QCheckBox* denoiseCheck;
    QCheckBox* claheCheck;
    QCheckBox* sharpenCheck;
    QDoubleSpinBox* strengthSpinner;
    
    // Pipeline
    QListWidget* pipelineList;

    // Data
    medical_vision::ImagePreprocessor processor;
    QStringList imageFiles;
    size_t currentImageIndex{0};
};