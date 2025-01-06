#include "segmentation_panel.hpp"
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QStackedWidget>
#include <QtWidgets/QPushButton>

SegmentationPanel::SegmentationPanel(QWidget* parent)
    : QGroupBox(tr("Segmentation"), parent) {
    setupUI();
    createConnections();
}

void SegmentationPanel::setupUI() {
    auto mainLayout = new QVBoxLayout(this);

    // Enable checkbox
    enableCheck = new QCheckBox(tr("Enable Segmentation"), this);
    mainLayout->addWidget(enableCheck);

    // Method selection
    auto methodLayout = new QHBoxLayout;
    methodLayout->addWidget(new QLabel(tr("Method:")));
    
    methodCombo = new QComboBox(this);
    methodCombo->addItem(tr("Threshold"), static_cast<int>(
        medical_vision::Segmentation::Method::THRESHOLD));
    methodCombo->addItem(tr("Otsu"), static_cast<int>(
        medical_vision::Segmentation::Method::OTSU));
    methodCombo->addItem(tr("Adaptive"), static_cast<int>(
        medical_vision::Segmentation::Method::ADAPTIVE_GAUSSIAN));
    methodCombo->addItem(tr("Watershed"), static_cast<int>(
        medical_vision::Segmentation::Method::WATERSHED));
    
    methodLayout->addWidget(methodCombo);
    mainLayout->addLayout(methodLayout);

    // Parameter stack
    paramStack = new QStackedWidget(this);
    paramStack->addWidget(createThresholdControls());
    paramStack->addWidget(createAdaptiveControls());
    paramStack->addWidget(createWatershedControls());
    mainLayout->addWidget(paramStack);

    // Add stretch
    mainLayout->addStretch();

    // Initial state
    updateControlsVisibility();
}

QWidget* SegmentationPanel::createThresholdControls() {
    auto container = new QWidget(this);
    auto layout = new QGridLayout(container);

    thresholdSpin = new QSpinBox(this);
    thresholdSpin->setRange(0, 255);
    thresholdSpin->setValue(128);
    layout->addWidget(new QLabel(tr("Threshold:")), 0, 0);
    layout->addWidget(thresholdSpin, 0, 1);

    maxValueSpin = new QSpinBox(this);
    maxValueSpin->setRange(0, 255);
    maxValueSpin->setValue(255);
    layout->addWidget(new QLabel(tr("Max Value:")), 1, 0);
    layout->addWidget(maxValueSpin, 1, 1);

    invertColorsCheck = new QCheckBox(tr("Invert Colors"), this);
    layout->addWidget(invertColorsCheck, 2, 0, 1, 2);

    return container;
}

QWidget* SegmentationPanel::createAdaptiveControls() {
    auto container = new QWidget(this);
    auto layout = new QGridLayout(container);

    blockSizeSpin = new QSpinBox(this);
    blockSizeSpin->setRange(3, 99);
    blockSizeSpin->setSingleStep(2);
    blockSizeSpin->setValue(11);
    layout->addWidget(new QLabel(tr("Block Size:")), 0, 0);
    layout->addWidget(blockSizeSpin, 0, 1);

    paramCSpin = new QDoubleSpinBox(this);
    paramCSpin->setRange(-100, 100);
    paramCSpin->setValue(2);
    paramCSpin->setSingleStep(0.5);
    layout->addWidget(new QLabel(tr("Parameter C:")), 1, 0);
    layout->addWidget(paramCSpin, 1, 1);

    return container;
}

QWidget* SegmentationPanel::createWatershedControls() {
    auto container = new QWidget(this);
    auto layout = new QVBoxLayout(container);

    // Method selection
    distanceTransformRadio = new QRadioButton(tr("Distance Transform"), this);
    manualSeedingRadio = new QRadioButton(tr("Manual Seeding"), this);
    distanceTransformRadio->setChecked(true);

    layout->addWidget(distanceTransformRadio);
    layout->addWidget(manualSeedingRadio);

    // Seeding controls
    auto seedingControls = new QWidget(this);
    auto seedingLayout = new QVBoxLayout(seedingControls);

    seedInstructionsLabel = new QLabel(
        tr("Left click: Add foreground seed\n"
           "Right click: Add background seed"), this);
    seedInstructionsLabel->setStyleSheet("QLabel { color: blue; }");

    clearSeedsButton = new QPushButton(tr("Clear Seeds"), this);

    seedingLayout->addWidget(seedInstructionsLabel);
    seedingLayout->addWidget(clearSeedsButton);

    layout->addWidget(seedingControls);

    return container;
}

void SegmentationPanel::createConnections() {
    // Main controls
    connect(enableCheck, &QCheckBox::toggled, this, &SegmentationPanel::updateControlsVisibility);
    connect(enableCheck, &QCheckBox::toggled, this, &SegmentationPanel::settingsChanged);
    
    connect(methodCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &SegmentationPanel::updateControlsVisibility);
    connect(methodCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &SegmentationPanel::settingsChanged);

    // Threshold controls
    connect(thresholdSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &SegmentationPanel::settingsChanged);
    connect(maxValueSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &SegmentationPanel::settingsChanged);
    connect(invertColorsCheck, &QCheckBox::toggled,
            this, &SegmentationPanel::settingsChanged);

    // Adaptive controls
    connect(blockSizeSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &SegmentationPanel::settingsChanged);
    connect(paramCSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &SegmentationPanel::settingsChanged);

    // Watershed controls
    connect(distanceTransformRadio, &QRadioButton::toggled,
            this, &SegmentationPanel::settingsChanged);
    connect(manualSeedingRadio, &QRadioButton::toggled,
            [this](bool checked) {
                isSeedingMode = checked;
                emit seedingModeChanged(checked);
            });
    connect(clearSeedsButton, &QPushButton::clicked,
            this, &SegmentationPanel::clearSeeds);
}

void SegmentationPanel::updateControlsVisibility() {
    bool enabled = enableCheck->isChecked();
    methodCombo->setEnabled(enabled);
    paramStack->setEnabled(enabled);

    // Show appropriate parameter widget
    auto method = static_cast<medical_vision::Segmentation::Method>(
        methodCombo->currentData().toInt());
    
    switch (method) {
        case medical_vision::Segmentation::Method::THRESHOLD:
            paramStack->setCurrentIndex(0);
            break;
        case medical_vision::Segmentation::Method::ADAPTIVE_GAUSSIAN:
            paramStack->setCurrentIndex(1);
            break;
        case medical_vision::Segmentation::Method::WATERSHED:
            paramStack->setCurrentIndex(2);
            break;
        default:
            paramStack->setCurrentIndex(0);
            paramStack->setEnabled(false);
            break;
    }
}

SegmentationPanel::SegmentationSettings SegmentationPanel::getCurrentSettings() const {
    SegmentationSettings settings;
    settings.enabled = enableCheck->isChecked();
    settings.method = static_cast<medical_vision::Segmentation::Method>(
        methodCombo->currentData().toInt());

    // Threshold parameters
    settings.thresholdParams.threshold = thresholdSpin->value();
    settings.thresholdParams.maxValue = maxValueSpin->value();
    settings.thresholdParams.invertColors = invertColorsCheck->isChecked();

    // Adaptive parameters
    settings.adaptiveParams.blockSize = blockSizeSpin->value();
    settings.adaptiveParams.C = paramCSpin->value();
    settings.adaptiveParams.maxValue = maxValueSpin->value();

    // Watershed parameters
    settings.useDistanceTransform = distanceTransformRadio->isChecked();
    settings.foregroundSeeds = foregroundSeeds;
    settings.backgroundSeeds = backgroundSeeds;

    return settings;
}

void SegmentationPanel::resetSettings() {
    enableCheck->setChecked(false);
    methodCombo->setCurrentIndex(0);
    
    // Reset threshold parameters
    thresholdSpin->setValue(128);
    maxValueSpin->setValue(255);
    invertColorsCheck->setChecked(false);

    // Reset adaptive parameters
    blockSizeSpin->setValue(11);
    paramCSpin->setValue(2);

    // Reset watershed parameters
    distanceTransformRadio->setChecked(true);
    clearSeeds();
}

void SegmentationPanel::clearSeeds() {
    foregroundSeeds.clear();
    backgroundSeeds.clear();
    emit settingsChanged();
}

void SegmentationPanel::addSeed(const cv::Point& point, bool isForeground) {
    if (isForeground) {
        foregroundSeeds.push_back(point);
    } else {
        backgroundSeeds.push_back(point);
    }
    emit settingsChanged();
}