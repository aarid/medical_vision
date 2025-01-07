#include "analysis_panel.hpp"
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QMessageBox>

AnalysisPanel::AnalysisPanel(QWidget* parent)
    : QGroupBox(tr("Deep Learning Analysis"), parent) {
    setupUI();
    setupConnections();
}

void AnalysisPanel::setupUI() {
    auto mainLayout = new QVBoxLayout(this);

    // Controls
    auto controlsLayout = new QHBoxLayout;
    
    analyzeButton = new QPushButton(tr("Analyze"), this);
    analyzeButton->setEnabled(false);
    
    showHeatmapCheck = new QCheckBox(tr("Show Heatmap"), this);
    showHeatmapCheck->setEnabled(false);

    auto thresholdLayout = new QHBoxLayout;
    thresholdLayout->addWidget(new QLabel(tr("Confidence:")));
    confidenceThresholdSpin = new QDoubleSpinBox(this);
    confidenceThresholdSpin->setRange(0.0, 1.0);
    confidenceThresholdSpin->setValue(DEFAULT_CONFIDENCE_THRESHOLD);
    confidenceThresholdSpin->setSingleStep(0.05);
    thresholdLayout->addWidget(confidenceThresholdSpin);

    controlsLayout->addWidget(analyzeButton);
    controlsLayout->addWidget(showHeatmapCheck);
    controlsLayout->addLayout(thresholdLayout);
    controlsLayout->addStretch();

    mainLayout->addLayout(controlsLayout);

    // Progress
    progressBar = new QProgressBar(this);
    progressBar->setVisible(false);
    mainLayout->addWidget(progressBar);

    // Results table
    resultsTable = new QTableWidget(0, TABLE_COLUMNS, this);
    resultsTable->setHorizontalHeaderLabels({tr("Pathology"), tr("Confidence")});
    resultsTable->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    resultsTable->setSelectionBehavior(QAbstractItemView::SelectRows);
    resultsTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
    mainLayout->addWidget(resultsTable);

    // Status
    auto statusLayout = new QHBoxLayout;
    statusLabel = new QLabel(this);
    processingTimeLabel = new QLabel(this);
    statusLayout->addWidget(statusLabel);
    statusLayout->addStretch();
    statusLayout->addWidget(processingTimeLabel);
    mainLayout->addLayout(statusLayout);
}

void AnalysisPanel::setupConnections() {
    connect(analyzeButton, &QPushButton::clicked, [this]() {
        analyzeButton->setEnabled(false);
        progressBar->setVisible(true);
        progressBar->setRange(0, 0);  // Indeterminate progress
    });

    connect(showHeatmapCheck, &QCheckBox::toggled, this, &AnalysisPanel::toggleHeatmap);
    
    connect(confidenceThresholdSpin, 
           QOverload<double>::of(&QDoubleSpinBox::valueChanged),
           this, &AnalysisPanel::updateConfidenceThreshold);

    connect(resultsTable, &QTableWidget::itemSelectionChanged, [this]() {
        if (showHeatmapCheck->isChecked() && resultsTable->currentRow() >= 0) {
            QString pathology = resultsTable->item(resultsTable->currentRow(), 0)->text();
            emit heatmapRequested(pathology);
        }
    });
}

bool AnalysisPanel::loadModel(const QString& modelPath, const QString& configPath) {
    try {
        medical_vision::ChestXRayAnalyzer::ModelConfig config;
        config.modelPath = modelPath.toStdString();
        config.configPath = configPath.toStdString();
        config.confidenceThreshold = DEFAULT_CONFIDENCE_THRESHOLD;
        config.generateHeatmaps = true;

        if (chest_analyzer.loadModel(config)) {
            isModelLoaded = true;
            analyzeButton->setEnabled(true);
            showHeatmapCheck->setEnabled(true);
            statusLabel->setText(tr("Model loaded successfully"));
            return true;
        }
    }
    catch (const std::exception& e) {
        displayError(tr("Failed to load model: %1").arg(e.what()));
    }
    
    isModelLoaded = false;
    return false;
}

void AnalysisPanel::analyzeImage(const cv::Mat& image) {
    if (!isModelLoaded) {
        displayError(tr("Model not loaded"));
        return;
    }

    try {
        progressBar->setVisible(true);
        auto result = chest_analyzer.analyze(image);
        updateResults(result);
    }
    catch (const std::exception& e) {
        displayError(tr("Analysis failed: %1").arg(e.what()));
    }

    progressBar->setVisible(false);
    analyzeButton->setEnabled(true);
}

void AnalysisPanel::updateResults(
    const medical_vision::ChestXRayAnalyzer::AnalysisResult& result) {
    
    clearResults();

    if (!result.success) {
        statusLabel->setText(tr("Analysis failed: %1").arg(
            QString::fromStdString(result.errorMessage)));
        return;
    }

    // Update table
    resultsTable->setRowCount(result.detections.size());
    for (size_t i = 0; i < result.detections.size(); ++i) {
        const auto& detection = result.detections[i];
        
        auto pathologyItem = new QTableWidgetItem(
            QString::fromStdString(detection.pathology));
        auto confidenceItem = new QTableWidgetItem(
            QString::number(detection.confidence * 100, 'f', 1) + "%");

        resultsTable->setItem(i, 0, pathologyItem);
        resultsTable->setItem(i, 1, confidenceItem);
    }

    // Update status
    statusLabel->setText(tr("Analysis completed"));
    processingTimeLabel->setText(tr("Processing time: %1 ms")
        .arg(result.processingTime * 1000.0, 0, 'f', 1));

    emit analysisCompleted(result);
}

void AnalysisPanel::toggleHeatmap(bool checked) {
    if (checked && resultsTable->currentRow() >= 0) {
        QString pathology = resultsTable->item(resultsTable->currentRow(), 0)->text();
        emit heatmapRequested(pathology);
    }
}

void AnalysisPanel::updateConfidenceThreshold(double value) {
    chest_analyzer.setConfidenceThreshold(static_cast<float>(value));
    emit confidenceThresholdChanged(value);
}

void AnalysisPanel::clearResults() {
    resultsTable->setRowCount(0);
    processingTimeLabel->clear();
}

void AnalysisPanel::displayError(const QString& message) {
    statusLabel->setText(message);
    QMessageBox::warning(this, tr("Error"), message);
}