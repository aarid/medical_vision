#include "main_window.hpp"
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QMessageBox>
#include <QtWidgets/QFileDialog>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QScrollArea>
#include <QtWidgets/QLabel>

MainWindow::MainWindow(QWidget* parent) 
    : QMainWindow(parent) {
    setupUI();
    setupMenus();
    setupConnections();
    setMinimumSize(1280, 800);
}

void MainWindow::setupUI() {
    auto centralWidget = new QWidget(this);
    setCentralWidget(centralWidget);

    auto mainLayout = new QHBoxLayout(centralWidget);

    // Left panel (Images and histogram)
    auto leftPanel = new QWidget;
    auto leftLayout = new QVBoxLayout(leftPanel);

    // Navigation controls
    auto navLayout = new QHBoxLayout;
    prevButton = new QPushButton(tr("Previous"), this);
    nextButton = new QPushButton(tr("Next"), this);
    imageCountLabel = new QLabel(tr("No images"), this);
    
    navLayout->addWidget(prevButton);
    navLayout->addWidget(imageCountLabel);
    navLayout->addWidget(nextButton);
    leftLayout->addLayout(navLayout);

    // Image viewers
    auto viewersLayout = new QHBoxLayout;
    originalViewer = new ImageViewer(tr("Original"), this);
    processedViewer = new ImageViewer(tr("Processed"), this);
    viewersLayout->addWidget(originalViewer);
    viewersLayout->addWidget(processedViewer);
    leftLayout->addLayout(viewersLayout);

    // Histogram
    histogramViewer = new HistogramViewer(this);
    leftLayout->addWidget(histogramViewer);

    // Right panel (Controls)
    auto rightPanel = new QWidget;
    auto rightLayout = new QVBoxLayout(rightPanel);

    // Create control panels
    processingPanel = new ProcessingPanel(this);
    featurePanel = new FeaturePanel(this);
    segmentationPanel = new SegmentationPanel(this);

    // Add panels to right layout with scroll area
    auto scrollArea = new QScrollArea(this);
    auto scrollWidget = new QWidget;
    auto scrollLayout = new QVBoxLayout(scrollWidget);
    
    scrollLayout->addWidget(processingPanel);
    scrollLayout->addWidget(featurePanel);
    scrollLayout->addWidget(segmentationPanel);
    scrollLayout->addStretch();

    scrollWidget->setLayout(scrollLayout);
    scrollArea->setWidget(scrollWidget);
    scrollArea->setWidgetResizable(true);
    rightLayout->addWidget(scrollArea);

    // Set panel sizes
    rightPanel->setFixedWidth(350);

    // Add panels to main layout
    mainLayout->addWidget(leftPanel, 1);
    mainLayout->addWidget(rightPanel, 0);
}

void MainWindow::setupMenus() {
    auto fileMenu = menuBar()->addMenu(tr("&File"));
    
    auto openAction = new QAction(tr("&Open Folder..."), this);
    openAction->setShortcut(QKeySequence::Open);
    connect(openAction, &QAction::triggered, this, &MainWindow::openFolder);
    fileMenu->addAction(openAction);

    fileMenu->addSeparator();

    auto exitAction = new QAction(tr("&Exit"), this);
    exitAction->setShortcut(QKeySequence::Quit);
    connect(exitAction, &QAction::triggered, this, &QWidget::close);
    fileMenu->addAction(exitAction);
}

void MainWindow::setupConnections() {
    // Navigation
    connect(prevButton, &QPushButton::clicked, this, &MainWindow::previousImage);
    connect(nextButton, &QPushButton::clicked, this, &MainWindow::nextImage);

    // Processing
    connect(processingPanel, &ProcessingPanel::settingsChanged, this, &MainWindow::processImage);
    connect(featurePanel, &FeaturePanel::settingsChanged, this, &MainWindow::processImage);
    connect(segmentationPanel, &SegmentationPanel::settingsChanged, this, &MainWindow::processImage);

    // Seed placement for watershed
    connect(processedViewer, &ImageViewer::mousePressed, this, &MainWindow::handleSeedPlacement);
}

void MainWindow::openFolder() {
    QString dir = QFileDialog::getExistingDirectory(this, tr("Select Image Folder"));
    if (dir.isEmpty()) return;

    imageFiles.clear();
    QDir selectedDir(dir);
    QStringList filters{"*.jpg", "*.jpeg", "*.png"};
    imageFiles = selectedDir.entryList(filters, QDir::Files);

    // Prepend path to filenames
    for (int i = 0; i < imageFiles.size(); ++i) {
        imageFiles[i] = selectedDir.filePath(imageFiles[i]);
    }

    if (imageFiles.empty()) {
        QMessageBox::warning(this, tr("Error"), tr("No valid images found in folder"));
        return;
    }

    currentImageIndex = 0;
    loadCurrentImage();
    updateNavigationState();
}

void MainWindow::loadCurrentImage() {
    if (imageFiles.empty()) return;
     try {
        if (!processor.loadImage(imageFiles[currentImageIndex].toStdString())) {
            QMessageBox::warning(this, tr("Error"), tr("Failed to load image"));
            return;
        }

        // Display original image
        originalViewer->setImage(processor.getOriginalImage());
        
        // Update histogram
        histogramViewer->setHistogram(processor.getHistogram());
        
        // Process image with current settings
        processImage();
     } catch (const std::exception& e) {
        QMessageBox::warning(this, tr("Error"), tr("Failed to load image: %1").arg(e.what()));
    }
}

void MainWindow::processImage() {
    if (!processor.isLoaded()) {
        statusBar()->showMessage(tr("No image loaded"), 3000);
        return;
    }

    try {
        // Reset to original image
        processor.reset();

        // Update histogram
        cv::Mat hist = processor.getHistogram();
        if (!hist.empty()) {
            histogramViewer->setHistogram(hist);
        } else {
            statusBar()->showMessage(tr("Failed to compute histogram"), 3000);
        }

        // Apply image processing
        auto procSettings = processingPanel->getCurrentSettings();
        if (procSettings.denoiseEnabled) {
            processor.denoise(procSettings.denoiseMethod);
        }
        if (procSettings.claheEnabled) {
            processor.histogramProcessing(medical_vision::ImagePreprocessor::HistogramMethod::CLAHE);
        }
        if (procSettings.sharpenEnabled) {
            processor.sharpen(procSettings.sharpenStrength);
        }

        cv::Mat displayImage = processor.getImage().clone();

        // Apply feature detection
        auto featureSettings = featurePanel->getCurrentSettings();
        if (featureSettings.edgesEnabled) {
            cv::Mat edges = featureDetector.detectEdges(
                displayImage, featureSettings.edgeMethod, featureSettings.edgeParams);
            processedViewer->setOverlay(edges, 0.3);
        }

        if (featureSettings.keypointsEnabled) {
            auto keypoints = featureDetector.detectKeypoints(
                displayImage, featureSettings.keypointMethod, featureSettings.keypointParams);
            displayImage = featureDetector.drawKeypoints(displayImage, keypoints);
        }

        // Apply segmentation
        auto segSettings = segmentationPanel->getCurrentSettings();
        if (segSettings.enabled) {
            cv::Mat segmentation_image = segmentation.segment(displayImage, segSettings.method);
            processedViewer->setOverlay(segmentation_image, 0.3);
        }

        // Update display
        processedViewer->setImage(displayImage);

    }  catch (const std::exception& e) {
        QMessageBox::warning(this, tr("Processing Error"),
            tr("An error occurred during image processing:\n%1").arg(e.what()));
        statusBar()->showMessage(tr("Processing failed"), 3000);
    }
}

void MainWindow::handleSeedPlacement(cv::Point pos, Qt::MouseButton button) {
    auto segSettings = segmentationPanel->getCurrentSettings();
    if (segSettings.enabled && 
        segSettings.method == medical_vision::Segmentation::Method::WATERSHED) {
        segmentationPanel->addSeed(pos, button == Qt::LeftButton);
    }
}

void MainWindow::nextImage() {
    if (currentImageIndex < imageFiles.size() - 1) {
        ++currentImageIndex;
        loadCurrentImage();
        updateNavigationState();
    }
}

void MainWindow::previousImage() {
    if (currentImageIndex > 0) {
        --currentImageIndex;
        loadCurrentImage();
        updateNavigationState();
    }
}

void MainWindow::updateNavigationState() {
    prevButton->setEnabled(currentImageIndex > 0);
    nextButton->setEnabled(currentImageIndex < imageFiles.size() - 1);
    imageCountLabel->setText(tr("Image %1/%2")
        .arg(currentImageIndex + 1)
        .arg(imageFiles.size()));
}