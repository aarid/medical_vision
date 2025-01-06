#include "feature_panel.hpp"
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QGridLayout>

FeaturePanel::FeaturePanel(QWidget* parent)
    : QGroupBox(tr("Feature Detection"), parent) {
    setupUI();
    createConnections();
}

void FeaturePanel::setupUI() {
    auto mainLayout = new QVBoxLayout(this);

    // Add edge detection controls
    mainLayout->addWidget(createEdgeControls());

    // Add separator line
    auto line = new QFrame(this);
    line->setFrameShape(QFrame::HLine);
    line->setFrameShadow(QFrame::Sunken);
    mainLayout->addWidget(line);

    // Add keypoint detection controls
    mainLayout->addWidget(createKeypointControls());

    // Add stretch to bottom
    mainLayout->addStretch();
}

QWidget* FeaturePanel::createEdgeControls() {
    auto container = new QWidget(this);
    auto layout = new QVBoxLayout(container);
    layout->setContentsMargins(0, 0, 0, 0);

    // Edge detection enable checkbox
    edgesCheck = new QCheckBox(tr("Enable Edge Detection"), this);

    // Edge method selection
    auto methodLayout = new QHBoxLayout;
    methodLayout->addWidget(new QLabel(tr("Method:")));
    
    edgeMethodCombo = new QComboBox(this);
    edgeMethodCombo->addItem(tr("Canny"), static_cast<int>(
        medical_vision::FeatureDetector::EdgeDetector::CANNY));
    edgeMethodCombo->addItem(tr("Sobel"), static_cast<int>(
        medical_vision::FeatureDetector::EdgeDetector::SOBEL));
    edgeMethodCombo->addItem(tr("Laplacian"), static_cast<int>(
        medical_vision::FeatureDetector::EdgeDetector::LAPLACIAN));
    
    methodLayout->addWidget(edgeMethodCombo);

    // Edge parameters
    auto paramsLayout = new QGridLayout;
    
    threshold1Spin = new QSpinBox(this);
    threshold1Spin->setRange(0, 255);
    threshold1Spin->setValue(100);
    paramsLayout->addWidget(new QLabel(tr("Threshold 1:")), 0, 0);
    paramsLayout->addWidget(threshold1Spin, 0, 1);

    threshold2Spin = new QSpinBox(this);
    threshold2Spin->setRange(0, 255);
    threshold2Spin->setValue(200);
    paramsLayout->addWidget(new QLabel(tr("Threshold 2:")), 1, 0);
    paramsLayout->addWidget(threshold2Spin, 1, 1);

    apertureSizeSpin = new QSpinBox(this);
    apertureSizeSpin->setRange(3, 7);
    apertureSizeSpin->setSingleStep(2);
    apertureSizeSpin->setValue(3);
    paramsLayout->addWidget(new QLabel(tr("Aperture:")), 2, 0);
    paramsLayout->addWidget(apertureSizeSpin, 2, 1);

    layout->addWidget(edgesCheck);
    layout->addLayout(methodLayout);
    layout->addLayout(paramsLayout);

    return container;
}

QWidget* FeaturePanel::createKeypointControls() {
    auto container = new QWidget(this);
    auto layout = new QVBoxLayout(container);
    layout->setContentsMargins(0, 0, 0, 0);

    // Keypoint detection enable checkbox
    keypointsCheck = new QCheckBox(tr("Enable Keypoint Detection"), this);

    // Keypoint method selection
    auto methodLayout = new QHBoxLayout;
    methodLayout->addWidget(new QLabel(tr("Method:")));
    
    keypointMethodCombo = new QComboBox(this);
    keypointMethodCombo->addItem(tr("SIFT"), static_cast<int>(
        medical_vision::FeatureDetector::KeypointDetector::SIFT));
    keypointMethodCombo->addItem(tr("ORB"), static_cast<int>(
        medical_vision::FeatureDetector::KeypointDetector::ORB));
    keypointMethodCombo->addItem(tr("FAST"), static_cast<int>(
        medical_vision::FeatureDetector::KeypointDetector::FAST));
    
    methodLayout->addWidget(keypointMethodCombo);

    // Keypoint parameters
    auto paramsLayout = new QGridLayout;
    
    maxKeypointsSpin = new QSpinBox(this);
    maxKeypointsSpin->setRange(10, 5000);
    maxKeypointsSpin->setValue(1000);
    paramsLayout->addWidget(new QLabel(tr("Max Points:")), 0, 0);
    paramsLayout->addWidget(maxKeypointsSpin, 0, 1);

    scaleFactorSpin = new QDoubleSpinBox(this);
    scaleFactorSpin->setRange(1.1, 2.0);
    scaleFactorSpin->setValue(1.2);
    scaleFactorSpin->setSingleStep(0.1);
    paramsLayout->addWidget(new QLabel(tr("Scale Factor:")), 1, 0);
    paramsLayout->addWidget(scaleFactorSpin, 1, 1);

    nLevelsSpin = new QSpinBox(this);
    nLevelsSpin->setRange(1, 16);
    nLevelsSpin->setValue(8);
    paramsLayout->addWidget(new QLabel(tr("Levels:")), 2, 0);
    paramsLayout->addWidget(nLevelsSpin, 2, 1);

    layout->addWidget(keypointsCheck);
    layout->addLayout(methodLayout);
    layout->addLayout(paramsLayout);

    return container;
}

void FeaturePanel::createConnections() {
    // Edge detection connections
    connect(edgesCheck, &QCheckBox::toggled, this, &FeaturePanel::updateControlsState);
    connect(edgesCheck, &QCheckBox::toggled, this, &FeaturePanel::settingsChanged);
    connect(edgeMethodCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &FeaturePanel::settingsChanged);
    connect(threshold1Spin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &FeaturePanel::settingsChanged);
    connect(threshold2Spin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &FeaturePanel::settingsChanged);
    connect(apertureSizeSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &FeaturePanel::settingsChanged);

    // Keypoint detection connections
    connect(keypointsCheck, &QCheckBox::toggled, this, &FeaturePanel::updateControlsState);
    connect(keypointsCheck, &QCheckBox::toggled, this, &FeaturePanel::settingsChanged);
    connect(keypointMethodCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &FeaturePanel::settingsChanged);
    connect(maxKeypointsSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &FeaturePanel::settingsChanged);
    connect(scaleFactorSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &FeaturePanel::settingsChanged);
    connect(nLevelsSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &FeaturePanel::settingsChanged);
}

void FeaturePanel::updateControlsState() {
    // Enable/disable edge detection controls
    bool edgesEnabled = edgesCheck->isChecked();
    edgeMethodCombo->setEnabled(edgesEnabled);
    threshold1Spin->setEnabled(edgesEnabled);
    threshold2Spin->setEnabled(edgesEnabled);
    apertureSizeSpin->setEnabled(edgesEnabled);

    // Enable/disable keypoint detection controls
    bool keypointsEnabled = keypointsCheck->isChecked();
    keypointMethodCombo->setEnabled(keypointsEnabled);
    maxKeypointsSpin->setEnabled(keypointsEnabled);
    scaleFactorSpin->setEnabled(keypointsEnabled);
    nLevelsSpin->setEnabled(keypointsEnabled);
}

FeaturePanel::FeatureSettings FeaturePanel::getCurrentSettings() const {
    FeatureSettings settings;
    
    // Edge detection settings
    settings.edgesEnabled = edgesCheck->isChecked();
    settings.edgeMethod = static_cast<medical_vision::FeatureDetector::EdgeDetector>(
        edgeMethodCombo->currentData().toInt());
    settings.edgeParams.threshold1 = threshold1Spin->value();
    settings.edgeParams.threshold2 = threshold2Spin->value();
    settings.edgeParams.apertureSize = apertureSizeSpin->value();

    // Keypoint detection settings
    settings.keypointsEnabled = keypointsCheck->isChecked();
    settings.keypointMethod = static_cast<medical_vision::FeatureDetector::KeypointDetector>(
        keypointMethodCombo->currentData().toInt());
    settings.keypointParams.maxKeypoints = maxKeypointsSpin->value();
    settings.keypointParams.scaleFactor = scaleFactorSpin->value();
    settings.keypointParams.nlevels = nLevelsSpin->value();

    return settings;
}

void FeaturePanel::resetSettings() {
    edgesCheck->setChecked(false);
    keypointsCheck->setChecked(false);
    edgeMethodCombo->setCurrentIndex(0);
    keypointMethodCombo->setCurrentIndex(0);
    
    threshold1Spin->setValue(100);
    threshold2Spin->setValue(200);
    apertureSizeSpin->setValue(3);
    
    maxKeypointsSpin->setValue(1000);
    scaleFactorSpin->setValue(1.2);
    nLevelsSpin->setValue(8);
    
    updateControlsState();
}