#pragma once

#include <QtWidgets/QGroupBox>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QComboBox>
#include "../include/medical_vision/feature_detector.hpp"

class FeaturePanel : public QGroupBox {
    Q_OBJECT

public:
    explicit FeaturePanel(QWidget* parent = nullptr);
    ~FeaturePanel() = default;

    struct FeatureSettings {
        // Edge detection settings
        bool edgesEnabled{false};
        medical_vision::FeatureDetector::EdgeDetector edgeMethod{
            medical_vision::FeatureDetector::EdgeDetector::CANNY};
        medical_vision::FeatureDetector::EdgeParams edgeParams;

        // Keypoint detection settings
        bool keypointsEnabled{false};
        medical_vision::FeatureDetector::KeypointDetector keypointMethod{
            medical_vision::FeatureDetector::KeypointDetector::SIFT};
        medical_vision::FeatureDetector::KeypointParams keypointParams;
    };

    FeatureSettings getCurrentSettings() const;
    void resetSettings();

signals:
    void settingsChanged();

private:
    void setupUI();
    void createConnections();

    // UI Components
    QWidget* createEdgeControls();
    QWidget* createKeypointControls();

    // Edge detection controls
    QCheckBox* edgesCheck{nullptr};
    QComboBox* edgeMethodCombo{nullptr};
    QSpinBox* threshold1Spin{nullptr};
    QSpinBox* threshold2Spin{nullptr};
    QSpinBox* apertureSizeSpin{nullptr};

    // Keypoint detection controls
    QCheckBox* keypointsCheck{nullptr};
    QComboBox* keypointMethodCombo{nullptr};
    QSpinBox* maxKeypointsSpin{nullptr};
    QDoubleSpinBox* scaleFactorSpin{nullptr};
    QSpinBox* nLevelsSpin{nullptr};

    void updateControlsState();
};