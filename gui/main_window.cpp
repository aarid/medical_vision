#include "main_window.hpp"

#include <QtWidgets/QMenuBar>

#include <QtWidgets/QStatusBar>
#include <QtWidgets/QMessageBox>
#include <QtCore/QDir>
#include <QtWidgets/QFileDialog>
#include <QtWidgets/QScrollArea>
#include <QtWidgets/QApplication>
#include <QtGui/QPalette>
#include <QtGui/QColor>
#include <QtCore/QString>
#include <QtCore/QStringList>
#include <QtCore/QFileInfo>
#include <opencv2/imgproc.hpp>

MainWindow::MainWindow(QWidget *parent) 
    : QMainWindow(parent)
{
    setupUI();
    createMenus();
    setMinimumSize(1280, 1024);
    statusBar()->showMessage("Ready");
}

void MainWindow::setupUI() {
    centralWidget = new QWidget(this);
    setCentralWidget(centralWidget);

    // Main layout
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
    imageViewerOriginal = new QLabel(this);
    imageViewerProcessed = new QLabel(this);
    
    imageViewerOriginal->setMinimumSize(400, 400);
    imageViewerProcessed->setMinimumSize(400, 400);
    
    imageViewerOriginal->setAlignment(Qt::AlignCenter);
    imageViewerProcessed->setAlignment(Qt::AlignCenter);

    viewersLayout->addWidget(imageViewerOriginal);
    viewersLayout->addWidget(imageViewerProcessed);

    // Histogram view
    histogramView = new QLabel(this);
    histogramView->setMinimumHeight(200);
    histogramView->setAlignment(Qt::AlignCenter);

    leftLayout->addLayout(navLayout);
    leftLayout->addLayout(viewersLayout);
    leftLayout->addWidget(histogramView);

    // Right panel (Controls)
    auto rightPanel = new QWidget;
    auto rightLayout = new QVBoxLayout(rightPanel);

    // Processing controls group
    auto processingGroup = new QGroupBox("Processing Options", this);
    auto processingLayout = new QVBoxLayout(processingGroup);

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

    processingLayout->addWidget(denoiseCheck);
    processingLayout->addWidget(claheCheck);
    processingLayout->addWidget(sharpenCheck);
    processingLayout->addLayout(strengthLayout);

    // Pipeline group
    auto pipelineGroup = new QGroupBox("Processing Pipeline", this);
    auto pipelineLayout = new QVBoxLayout(pipelineGroup);

    pipelineList = new QListWidget(this);
    auto pipelineButtonLayout = new QHBoxLayout;
    
    auto addButton = new QPushButton("Add", this);
    auto removeButton = new QPushButton("Remove", this);
    auto processButton = new QPushButton("Process", this);
    
    pipelineButtonLayout->addWidget(addButton);
    pipelineButtonLayout->addWidget(removeButton);
    
    pipelineLayout->addWidget(pipelineList);
    pipelineLayout->addLayout(pipelineButtonLayout);
    pipelineLayout->addWidget(processButton);

    // Connect processing signals
    connect(processButton, &QPushButton::clicked, this, &MainWindow::processImage);
    connect(denoiseCheck, &QCheckBox::stateChanged, this, &MainWindow::processImage);
    connect(claheCheck, &QCheckBox::stateChanged, this, &MainWindow::processImage);
    connect(sharpenCheck, &QCheckBox::stateChanged, this, &MainWindow::processImage);
    connect(strengthSpinner, QOverload<double>::of(&QDoubleSpinBox::valueChanged), 
            this, &MainWindow::processImage);

    rightLayout->addWidget(processingGroup);
    rightLayout->addWidget(pipelineGroup);
    rightLayout->addStretch();

    // Add panels to main layout
    mainLayout->addWidget(leftPanel, 2);
    mainLayout->addWidget(rightPanel, 1);
}

void MainWindow::createMenus() {
    auto fileMenu = menuBar()->addMenu(tr("&File"));
    
    auto openAction = new QAction(tr("&Open Folder"), this);
    openAction->setShortcut(QKeySequence::Open);
    connect(openAction, &QAction::triggered, this, &MainWindow::selectFolder);
    fileMenu->addAction(openAction);

    fileMenu->addSeparator();
    
    auto exitAction = new QAction(tr("&Exit"), this);
    exitAction->setShortcut(QKeySequence::Quit);
    connect(exitAction, &QAction::triggered, this, &QWidget::close);
    fileMenu->addAction(exitAction);
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
        QMessageBox::warning(this, "Error", QString("Error processing image: %1").arg(e.what()));
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

    // Display processed image
    cv::Mat processedMat = processor.getImage();
    QImage processedQImage = matToQImage(processedMat);
    imageViewerProcessed->setPixmap(QPixmap::fromImage(processedQImage).scaled(
        imageViewerProcessed->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}

QImage MainWindow::matToQImage(const cv::Mat& mat) {
    if (mat.empty()) {
        return QImage();
    }

    // 8-bit, 1 channel
    if (mat.type() == CV_8UC1) {
        return QImage(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_Grayscale8).copy();
    }


    // 8-bit, 3 channels
    if (mat.type() == CV_8UC3) {
        cv::Mat rgb;
        cv::cvtColor(mat, rgb, cv::COLOR_BGR2RGB);
        return QImage(rgb.data, rgb.cols, rgb.rows, rgb.step, QImage::Format_RGB888).copy();
    }


    // 8-bit, 4 channels
    if (mat.type() == CV_8UC4) {
        return QImage(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_RGBA8888).copy();
    }

    return QImage();
}