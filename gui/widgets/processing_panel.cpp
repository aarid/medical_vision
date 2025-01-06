#include "processing_panel.hpp"
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QComboBox>

ProcessingPanel::ProcessingPanel(QWidget* parent)
    : QGroupBox(tr("Image Processing"), parent) {
    setupUI();
    createConnections();
}

void ProcessingPanel::setupUI() {
    auto mainLayout = new QVBoxLayout(this);

    // Add denoise controls
    mainLayout->addWidget(createDenoiseControls());

    // Add CLAHE control
    claheCheck = new QCheckBox(tr("CLAHE"), this);
    mainLayout->addWidget(claheCheck);

    // Add sharpen controls
    mainLayout->addWidget(createSharpenControls());

    // Add stretch to bottom
    mainLayout->addStretch();
}

QWidget* ProcessingPanel::createDenoiseControls() {
    auto container = new QWidget(this);
    auto layout = new QVBoxLayout(container);
    layout->setContentsMargins(0, 0, 0, 0);

    // Denoise checkbox and method selection
    denoiseCheck = new QCheckBox(tr("Denoise"), this);
    
    auto methodLayout = new QHBoxLayout;
    methodLayout->addWidget(new QLabel(tr("Method:")));
    
    denoiseMethodCombo = new QComboBox(this);
    denoiseMethodCombo->addItem(tr("Bilateral"), static_cast<int>(
        medical_vision::ImagePreprocessor::NoiseReductionMethod::BILATERAL));
    denoiseMethodCombo->addItem(tr("Gaussian"), static_cast<int>(
        medical_vision::ImagePreprocessor::NoiseReductionMethod::GAUSSIAN));
    denoiseMethodCombo->addItem(tr("Median"), static_cast<int>(
        medical_vision::ImagePreprocessor::NoiseReductionMethod::MEDIAN));
    denoiseMethodCombo->addItem(tr("NLM"), static_cast<int>(
        medical_vision::ImagePreprocessor::NoiseReductionMethod::NLM));
    
    methodLayout->addWidget(denoiseMethodCombo);
    
    layout->addWidget(denoiseCheck);
    layout->addLayout(methodLayout);

    // Enable/disable method combo based on checkbox
    denoiseMethodCombo->setEnabled(false);
    connect(denoiseCheck, &QCheckBox::toggled,
            denoiseMethodCombo, &QComboBox::setEnabled);

    return container;
}

QWidget* ProcessingPanel::createSharpenControls() {
    auto container = new QWidget(this);
    auto layout = new QVBoxLayout(container);
    layout->setContentsMargins(0, 0, 0, 0);

    // Sharpen checkbox
    sharpenCheck = new QCheckBox(tr("Sharpen"), this);
    
    // Strength control
    auto strengthLayout = new QHBoxLayout;
    strengthLayout->addWidget(new QLabel(tr("Strength:")));
    
    strengthSpinner = new QDoubleSpinBox(this);
    strengthSpinner->setRange(0.1, 5.0);
    strengthSpinner->setValue(1.0);
    strengthSpinner->setSingleStep(0.1);
    strengthSpinner->setEnabled(false);
    
    strengthLayout->addWidget(strengthSpinner);
    
    layout->addWidget(sharpenCheck);
    layout->addLayout(strengthLayout);

    // Enable/disable spinner based on checkbox
    connect(sharpenCheck, &QCheckBox::toggled,
            strengthSpinner, &QDoubleSpinBox::setEnabled);

    return container;
}

void ProcessingPanel::createConnections() {
    // Connect all controls to emit settingsChanged
    connect(denoiseCheck, &QCheckBox::toggled,
            this, &ProcessingPanel::settingsChanged);
    connect(denoiseMethodCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &ProcessingPanel::settingsChanged);
    connect(claheCheck, &QCheckBox::toggled,
            this, &ProcessingPanel::settingsChanged);
    connect(sharpenCheck, &QCheckBox::toggled,
            this, &ProcessingPanel::settingsChanged);
    connect(strengthSpinner, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &ProcessingPanel::settingsChanged);
}

ProcessingPanel::ProcessingSettings ProcessingPanel::getCurrentSettings() const {
    ProcessingSettings settings;
    settings.denoiseEnabled = denoiseCheck->isChecked();
    settings.claheEnabled = claheCheck->isChecked();
    settings.sharpenEnabled = sharpenCheck->isChecked();
    settings.sharpenStrength = strengthSpinner->value();
    settings.denoiseMethod = static_cast<medical_vision::ImagePreprocessor::NoiseReductionMethod>(
        denoiseMethodCombo->currentData().toInt());
    return settings;
}

void ProcessingPanel::resetSettings() {
    denoiseCheck->setChecked(false);
    claheCheck->setChecked(false);
    sharpenCheck->setChecked(false);
    strengthSpinner->setValue(1.0);
    denoiseMethodCombo->setCurrentIndex(0);
}