#include "main_window.hpp"
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QMessageBox>
#include <QtCore/QDir>
#include <QtWidgets/QFileDialog>
#include <QtWidgets/QScrollArea>
#include <QtGui/QPalette>
#include <QtGui/QColor>
#include <QtCore/QString>
#include <QtCore/QStringList>
#include <QtCore/QFileInfo>
#include <opencv2/imgproc.hpp>

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent) {
    setupUI();
    createMenus();
    setMinimumSize(1280, 1024);
    statusBar()->showMessage("Ready");
}

void MainWindow::setupUI() {
    centralWidget = new QWidget(this);
    setCentralWidget(centralWidget);
    auto mainLayout = new QHBoxLayout(centralWidget);

    // Left panel (Image viewer)
    auto leftPanel = new QWidget;
    auto leftLayout = new QVBoxLayout(leftPanel);
    
    // Navigation controls
    auto navLayout = new QHBoxLayout;
    prevButton = new QPushButton("Previous", this);
    nextButton = new QPushButton("Next", this);
    imageCountLabel = new QLabel("No images loaded", this);
    navLayout->addWidget(prevButton);
    navLayout->addWidget(imageCountLabel);
    navLayout->addWidget(nextButton);
    connect(prevButton, &QPushButton::clicked, this, &MainWindow::previousImage);
    connect(nextButton, &QPushButton::clicked, this, &MainWindow::nextImage);

    // Image viewers
    auto viewersLayout = new QHBoxLayout;
    auto originalGroup = new QGroupBox("Original", this);
    auto processedGroup = new QGroupBox("Processed", this);
    auto originalLayout = new QVBoxLayout(originalGroup);
    auto processedLayout = new QVBoxLayout(processedGroup);

    imageViewerOriginal = new QLabel(this);
    imageViewerProcessed = new QLabel(this);
    imageViewerOriginal->setMinimumSize(400, 400);
    imageViewerProcessed->setMinimumSize(400, 400);
    imageViewerOriginal->setAlignment(Qt::AlignCenter);
    imageViewerProcessed->setAlignment(Qt::AlignCenter);
    originalLayout->addWidget(imageViewerOriginal);
    processedLayout->addWidget(imageViewerProcessed);
    viewersLayout->addWidget(originalGroup);
    viewersLayout->addWidget(processedGroup);

    // Histogram
    auto histogramGroup = new QGroupBox("Histogram", this);
    auto histogramLayout = new QVBoxLayout(histogramGroup);
    histogramView = new QLabel(this);
    histogramView->setMinimumHeight(200);
    histogramView->setAlignment(Qt::AlignCenter);
    histogramLayout->addWidget(histogramView);

    leftLayout->addLayout(navLayout);
    leftLayout->addLayout(viewersLayout);
    leftLayout->addWidget(histogramGroup);

    // Right panel (Controls)
    auto rightPanel = new QWidget;
    auto rightLayout = new QVBoxLayout(rightPanel);
    auto scrollArea = new QScrollArea;
    auto scrollWidget = new QWidget;
    auto scrollLayout = new QVBoxLayout(scrollWidget);

    scrollLayout->addWidget(createProcessingGroup());
    scrollLayout->addWidget(createFeatureDetectionGroup());
    scrollLayout->addWidget(createSegmentationGroup());
    scrollLayout->addStretch();

    scrollWidget->setLayout(scrollLayout);
    scrollArea->setWidget(scrollWidget);
    scrollArea->setWidgetResizable(true);
    scrollArea->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    scrollArea->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    rightLayout->addWidget(scrollArea);

    // Set size policies
    leftPanel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
    rightPanel->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Preferred);
    rightPanel->setFixedWidth(300);

    mainLayout->addWidget(leftPanel);
    mainLayout->addWidget(rightPanel);
    setWindowTitle("Medical Vision");
}

QGroupBox* MainWindow::createProcessingGroup() {
    auto group = new QGroupBox("Image Processing", this);
    auto layout = new QVBoxLayout(group);

    denoiseCheck = new QCheckBox("Denoise", this);
    claheCheck = new QCheckBox("CLAHE", this);
    sharpenCheck = new QCheckBox("Sharpen", this);

    auto strengthLayout = new QHBoxLayout;
    strengthLayout->addWidget(new QLabel("Strength:"));
    strengthSpinner = new QDoubleSpinBox(this);
    strengthSpinner->setRange(0.1, 5.0);
    strengthSpinner->setValue(1.0);
    strengthSpinner->setSingleStep(0.1);
    strengthLayout->addWidget(strengthSpinner);

    auto processButton = new QPushButton("Process", this);

    layout->addWidget(denoiseCheck);
    layout->addWidget(claheCheck);
    layout->addWidget(sharpenCheck);
    layout->addLayout(strengthLayout);
    layout->addWidget(processButton);

    connect(processButton, &QPushButton::clicked, this, &MainWindow::processImage);
    connect(denoiseCheck, &QCheckBox::stateChanged, this, &MainWindow::processImage);
    connect(claheCheck, &QCheckBox::stateChanged, this, &MainWindow::processImage);
    connect(sharpenCheck, &QCheckBox::stateChanged, this, &MainWindow::processImage);
    connect(strengthSpinner, QOverload<double>::of(&QDoubleSpinBox::valueChanged), 
            this, &MainWindow::processImage);

    return group;
}

QGroupBox* MainWindow::createFeatureDetectionGroup() {
    auto group = new QGroupBox("Feature Detection", this);
    auto layout = new QVBoxLayout(group);

    // Edge detection
    auto edgeGroup = new QGroupBox("Edge Detection", this);
    auto edgeLayout = new QGridLayout(edgeGroup);

    edgeDetectorCombo = new QComboBox(this);
    edgeDetectorCombo->addItem("Canny", static_cast<int>(medical_vision::FeatureDetector::EdgeDetector::CANNY));
    edgeDetectorCombo->addItem("Sobel", static_cast<int>(medical_vision::FeatureDetector::EdgeDetector::SOBEL));
    edgeDetectorCombo->addItem("Laplacian", static_cast<int>(medical_vision::FeatureDetector::EdgeDetector::LAPLACIAN));

    showEdgesCheck = new QCheckBox("Show Edges", this);
    threshold1Spin = new QSpinBox(this);
    threshold2Spin = new QSpinBox(this);
    apertureSizeSpin = new QSpinBox(this);
    
    threshold1Spin->setRange(0, 255);
    threshold2Spin->setRange(0, 255);
    apertureSizeSpin->setRange(3, 7);
    apertureSizeSpin->setSingleStep(2);
    threshold1Spin->setValue(100);
    threshold2Spin->setValue(200);
    apertureSizeSpin->setValue(3);

    edgeLayout->addWidget(new QLabel("Method:"), 0, 0);
    edgeLayout->addWidget(edgeDetectorCombo, 0, 1);
    edgeLayout->addWidget(new QLabel("Threshold 1:"), 1, 0);
    edgeLayout->addWidget(threshold1Spin, 1, 1);
    edgeLayout->addWidget(new QLabel("Threshold 2:"), 2, 0);
    edgeLayout->addWidget(threshold2Spin, 2, 1);
    edgeLayout->addWidget(new QLabel("Aperture:"), 3, 0);
    edgeLayout->addWidget(apertureSizeSpin, 3, 1);
    edgeLayout->addWidget(showEdgesCheck, 4, 0, 1, 2);

    // Keypoint detection
    auto keypointGroup = new QGroupBox("Keypoint Detection", this);
    auto keypointLayout = new QGridLayout(keypointGroup);

    keypointDetectorCombo = new QComboBox(this);
    keypointDetectorCombo->addItem("SIFT", static_cast<int>(medical_vision::FeatureDetector::KeypointDetector::SIFT));
    keypointDetectorCombo->addItem("ORB", static_cast<int>(medical_vision::FeatureDetector::KeypointDetector::ORB));
    keypointDetectorCombo->addItem("FAST", static_cast<int>(medical_vision::FeatureDetector::KeypointDetector::FAST));

    showKeypointsCheck = new QCheckBox("Show Keypoints", this);
    maxKeypointsSpin = new QSpinBox(this);
    maxKeypointsSpin->setRange(10, 5000);
    maxKeypointsSpin->setValue(1000);

    keypointLayout->addWidget(new QLabel("Method:"), 0, 0);
    keypointLayout->addWidget(keypointDetectorCombo, 0, 1);
    keypointLayout->addWidget(new QLabel("Max Points:"), 1, 0);
    keypointLayout->addWidget(maxKeypointsSpin, 1, 1);
    keypointLayout->addWidget(showKeypointsCheck, 2, 0, 1, 2);

    layout->addWidget(edgeGroup);
    layout->addWidget(keypointGroup);

    connect(edgeDetectorCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), 
            this, &MainWindow::processFeatures);
    connect(keypointDetectorCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), 
            this, &MainWindow::processFeatures);
    connect(threshold1Spin, QOverload<int>::of(&QSpinBox::valueChanged), 
            this, &MainWindow::processFeatures);
    connect(threshold2Spin, QOverload<int>::of(&QSpinBox::valueChanged), 
            this, &MainWindow::processFeatures);
    connect(apertureSizeSpin, QOverload<int>::of(&QSpinBox::valueChanged), 
            this, &MainWindow::processFeatures);
    connect(maxKeypointsSpin, QOverload<int>::of(&QSpinBox::valueChanged), 
            this, &MainWindow::processFeatures);
    connect(showEdgesCheck, &QCheckBox::stateChanged, 
            this, &MainWindow::updateDisplay);
    connect(showKeypointsCheck, &QCheckBox::stateChanged, 
            this, &MainWindow::updateDisplay);

    return group;
}

QGroupBox* MainWindow::createSegmentationGroup() {
    auto group = new QGroupBox("Segmentation", this);
    auto layout = new QVBoxLayout(group);

    // Method selection
    auto methodLayout = new QHBoxLayout;
    methodLayout->addWidget(new QLabel("Method:"));
    segmentationMethodCombo = new QComboBox(this);
    segmentationMethodCombo->addItem("Threshold", static_cast<int>(medical_vision::Segmentation::Method::THRESHOLD));
    segmentationMethodCombo->addItem("Otsu", static_cast<int>(medical_vision::Segmentation::Method::OTSU));
    segmentationMethodCombo->addItem("Adaptive", static_cast<int>(medical_vision::Segmentation::Method::ADAPTIVE_GAUSSIAN));
    segmentationMethodCombo->addItem("Region Growing", static_cast<int>(medical_vision::Segmentation::Method::REGION_GROWING));
    methodLayout->addWidget(segmentationMethodCombo);

    // Parameters
    auto paramsGroup = new QGroupBox("Parameters");
    auto paramsLayout = new QGridLayout(paramsGroup);

    // Threshold parameters
    thresholdSpin = new QSpinBox(this);
    thresholdSpin->setRange(0, 255);
    thresholdSpin->setValue(128);
    paramsLayout->addWidget(new QLabel("Threshold:"), 0, 0);
    paramsLayout->addWidget(thresholdSpin, 0, 1);

    maxValueSpin = new QSpinBox(this);
    maxValueSpin->setRange(0, 255);
    maxValueSpin->setValue(255);
    paramsLayout->addWidget(new QLabel("Max Value:"), 1, 0);
    paramsLayout->addWidget(maxValueSpin, 1, 1);

    // Adaptive parameters
    blockSizeSpin = new QSpinBox(this);
    blockSizeSpin->setRange(3, 99);
    blockSizeSpin->setValue(11);
    blockSizeSpin->setSingleStep(2);
    paramsLayout->addWidget(new QLabel("Block Size:"), 2, 0);
    paramsLayout->addWidget(blockSizeSpin, 2, 1);

    paramCSpin = new QDoubleSpinBox(this);
    paramCSpin->setRange(-100, 100);
    paramCSpin->setValue(2);
    paramCSpin->setSingleStep(0.5);
    paramsLayout->addWidget(new QLabel("Parameter C:"), 3, 0);
    paramsLayout->addWidget(paramCSpin, 3, 1);

    invertColorsCheck = new QCheckBox("Invert Colors");
    paramsLayout->addWidget(invertColorsCheck, 4, 0, 1, 2);

    showSegmentationCheck = new QCheckBox("Show Segmentation");
    showSegmentationCheck->setChecked(true);

    // Add to main layout
    layout->addLayout(methodLayout);
    layout->addWidget(paramsGroup);
    layout->addWidget(showSegmentationCheck);

    // Connect signals
    connect(segmentationMethodCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &MainWindow::processSegmentation);
    connect(thresholdSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &MainWindow::processSegmentation);
    connect(maxValueSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &MainWindow::processSegmentation);
    connect(blockSizeSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &MainWindow::processSegmentation);
    connect(paramCSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &MainWindow::processSegmentation);
    connect(invertColorsCheck, &QCheckBox::stateChanged,
            this, &MainWindow::processSegmentation);
    connect(showSegmentationCheck, &QCheckBox::stateChanged,
            this, &MainWindow::updateDisplay);

    return group;
}

void MainWindow::createMenus() {
    auto fileMenu = menuBar()->addMenu(tr("&File"));
    auto openAction = new QAction(tr("&Open Folder"), this);
    openAction->setShortcut(QKeySequence::Open);
    connect(openAction, &QAction::triggered, this, &MainWindow::selectFolder);
    fileMenu->addAction(openAction);
}

void MainWindow::selectFolder() {
    QString dir = QFileDialog::getExistingDirectory(this, "Select Image Folder",
                                                  QString(),
                                                  QFileDialog::ShowDirsOnly);
    if (dir.isEmpty()) return;

    imageFiles.clear();
    QDir selectedDir(dir);
    QStringList filters;
    filters << "*.jpg" << "*.jpeg" << "*.png";
    
    QFileInfoList fileInfoList = selectedDir.entryInfoList(filters, QDir::Files);
    for (const QFileInfo& fileInfo : fileInfoList) {
        imageFiles.push_back(fileInfo.absoluteFilePath());
    }

    if (imageFiles.empty()) {
        QMessageBox::warning(this, "Error", "No valid images found in selected folder");
        return;
    }

    currentImageIndex = 0;
    updateImage();
    updateNavigationControls();
}

void MainWindow::nextImage() {
    if (currentImageIndex < imageFiles.size() - 1) {
        currentImageIndex++;
        updateImage();
        updateNavigationControls();
    }
}

void MainWindow::previousImage() {
    if (currentImageIndex > 0) {
        currentImageIndex--;
        updateImage();
        updateNavigationControls();
    }
}

void MainWindow::updateNavigationControls() {
    prevButton->setEnabled(currentImageIndex > 0);
    nextButton->setEnabled(currentImageIndex < imageFiles.size() - 1);
    imageCountLabel->setText(QString("Image %1/%2")
        .arg(currentImageIndex + 1)
        .arg(imageFiles.size()));
}

void MainWindow::updateImage() {
    if (imageFiles.empty()) return;

    if (!processor.loadImage(imageFiles[currentImageIndex].toStdString())) {
        QMessageBox::warning(this, "Error", "Failed to load image");
        return;
    }

    try {
        // Display original image
        cv::Mat originalMat = processor.getOriginalImage();
        QImage originalQImage = matToQImage(originalMat);
        if (originalQImage.isNull()) {
            QMessageBox::warning(this, "Error", "Failed to convert original image");
            return;
        }

        QPixmap originalPixmap = QPixmap::fromImage(originalQImage);
        if (originalPixmap.isNull()) {
            QMessageBox::warning(this, "Error", "Failed to create pixmap from original image");
            return;
        }

        imageViewerOriginal->setPixmap(originalPixmap.scaled(
            imageViewerOriginal->size(), 
            Qt::KeepAspectRatio, 
            Qt::SmoothTransformation));

        // Process and display processed image
        processImage();
        processFeatures();

        // Update histogram
        cv::Mat histMat = processor.getHistogram();
        QImage histQImage = matToQImage(histMat);
        if (!histQImage.isNull()) {
            histogramView->setPixmap(QPixmap::fromImage(histQImage).scaled(
                histogramView->size(), 
                Qt::KeepAspectRatio, 
                Qt::SmoothTransformation));
        }
    }
    catch (const std::exception& e) {
        QMessageBox::warning(this, "Error", 
            QString("Error processing image: %1").arg(e.what()));
    }
}

void MainWindow::processImage() {
    if (!processor.isLoaded()) return;

    // Reset to original image
    processor.reset();

    // Apply selected processing operations
    if (denoiseCheck->isChecked()) {
        processor.denoise(medical_vision::ImagePreprocessor::NoiseReductionMethod::BILATERAL);
    }
    
    if (claheCheck->isChecked()) {
        processor.histogramProcessing(medical_vision::ImagePreprocessor::HistogramMethod::CLAHE);
    }
    
    if (sharpenCheck->isChecked()) {
        processor.sharpen(strengthSpinner->value());
    }

    updateDisplay();
}

void MainWindow::processFeatures() {
    if (!processor.isLoaded()) return;

    try {
        // Get current image
        cv::Mat currentImage = processor.getImage();

        // Edge detection
        medical_vision::FeatureDetector::EdgeParams edgeParams;
        edgeParams.threshold1 = threshold1Spin->value();
        edgeParams.threshold2 = threshold2Spin->value();
        edgeParams.apertureSize = apertureSizeSpin->value();

        auto edgeMethod = static_cast<medical_vision::FeatureDetector::EdgeDetector>(
            edgeDetectorCombo->currentData().toInt());
        edgeResult = featureDetector.detectEdges(currentImage, edgeMethod, edgeParams);

        // Keypoint detection
        medical_vision::FeatureDetector::KeypointParams keypointParams;
        keypointParams.maxKeypoints = maxKeypointsSpin->value();

        auto keypointMethod = static_cast<medical_vision::FeatureDetector::KeypointDetector>(
            keypointDetectorCombo->currentData().toInt());
        keypointResult = featureDetector.detectKeypoints(currentImage, keypointMethod, keypointParams);

        updateDisplay();
    }
    catch (const std::exception& e) {
        QMessageBox::warning(this, "Error", 
            QString("Feature detection failed: %1").arg(e.what()));
    }
}

void MainWindow::processSegmentation() {
    if (!processor.isLoaded()) return;

    try {
        cv::Mat input = processor.getImage();
        auto method = static_cast<medical_vision::Segmentation::Method>(
            segmentationMethodCombo->currentData().toInt());

        switch (method) {
            case medical_vision::Segmentation::Method::THRESHOLD: {
                medical_vision::Segmentation::ThresholdParams params;
                params.threshold = thresholdSpin->value();
                params.maxValue = maxValueSpin->value();
                params.invertColors = invertColorsCheck->isChecked();
                segmentationResult = segmentation.threshold(input, params);
                break;
            }
            case medical_vision::Segmentation::Method::OTSU: {
                segmentationResult = segmentation.otsuThreshold(input);
                break;
            }
            case medical_vision::Segmentation::Method::ADAPTIVE_GAUSSIAN: {
                medical_vision::Segmentation::AdaptiveParams params;
                params.blockSize = blockSizeSpin->value();
                params.C = paramCSpin->value();
                params.maxValue = maxValueSpin->value();
                params.invertColors = invertColorsCheck->isChecked();
                segmentationResult = segmentation.adaptiveThreshold(input, params);
                break;
            }
            case medical_vision::Segmentation::Method::REGION_GROWING: {
                if (!seedPoints.empty()) {
                    medical_vision::Segmentation::RegionGrowingParams params;
                    params.seeds = seedPoints;
                    params.threshold = thresholdSpin->value();
                    segmentationResult = segmentation.regionGrowing(input, params);
                }
                break;
            }
        }

        updateDisplay();
    }
    catch (const std::exception& e) {
        QMessageBox::warning(this, "Error", 
            QString("Segmentation failed: %1").arg(e.what()));
    }
}


void MainWindow::updateDisplay() {
    if (!processor.isLoaded()) return;

    cv::Mat displayImage = processor.getImage().clone();
    
    // Convert grayscale to BGR if needed
    if (displayImage.channels() == 1) {
        cv::cvtColor(displayImage, displayImage, cv::COLOR_GRAY2BGR);
    }

    if (showEdgesCheck->isChecked() && !edgeResult.empty()) {
        // Convert edge result to BGR for overlay
        cv::Mat edges;
        cv::cvtColor(edgeResult, edges, cv::COLOR_GRAY2BGR);
        
        // Create colored edge overlay (red)
        cv::Mat coloredEdges = cv::Mat::zeros(edges.size(), CV_8UC3);
        coloredEdges.setTo(cv::Scalar(0, 0, 255), edgeResult);
        
        // Blend images
        cv::addWeighted(displayImage, 0.7, coloredEdges, 0.3, 0, displayImage);
    }

    if (showKeypointsCheck->isChecked() && !keypointResult.empty()) {
        displayImage = featureDetector.drawKeypoints(displayImage, keypointResult);
    }

    if (showSegmentationCheck->isChecked() && !segmentationResult.empty()) {
        displayImage = segmentation.drawSegmentation(displayImage, segmentationResult, 0.3);
    }

    QImage qimg = matToQImage(displayImage);
    imageViewerProcessed->setPixmap(QPixmap::fromImage(qimg).scaled(
        imageViewerProcessed->size(), 
        Qt::KeepAspectRatio, 
        Qt::SmoothTransformation));
}

QImage MainWindow::matToQImage(const cv::Mat& mat) {
    if (mat.empty()) {
        return QImage();
    }

    if (mat.type() == CV_8UC1) {
        return QImage(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_Grayscale8).copy();
    }

    if (mat.type() == CV_8UC3) {
        cv::Mat rgb;
        cv::cvtColor(mat, rgb, cv::COLOR_BGR2RGB);
        return QImage(rgb.data, rgb.cols, rgb.rows, rgb.step, QImage::Format_RGB888).copy();
    }

    if (mat.type() == CV_8UC4) {
        return QImage(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_RGBA8888).copy();
    }

    return QImage();
}