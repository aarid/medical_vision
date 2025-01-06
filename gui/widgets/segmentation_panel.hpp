#pragma once

#include <QtWidgets/QGroupBox>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QStackedWidget>
#include <QtWidgets/QRadioButton>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QLabel>
#include "../../include/medical_vision/segmentation.hpp"

class SegmentationPanel : public QGroupBox {
    Q_OBJECT

public:
    explicit SegmentationPanel(QWidget* parent = nullptr);
    ~SegmentationPanel() = default;

    struct SegmentationSettings {
        bool enabled{false};
        medical_vision::Segmentation::Method method{
            medical_vision::Segmentation::Method::THRESHOLD};
        
        // Threshold parameters
        medical_vision::Segmentation::ThresholdParams thresholdParams;
        medical_vision::Segmentation::AdaptiveParams adaptiveParams;
        
        // Watershed parameters
        bool useDistanceTransform{true};
        std::vector<cv::Point> foregroundSeeds;
        std::vector<cv::Point> backgroundSeeds;
    };

    SegmentationSettings getCurrentSettings() const;
    void resetSettings();
    void clearSeeds();
    void addSeed(const cv::Point& point, bool isForeground);

signals:
    void settingsChanged();
    void seedingModeChanged(bool active);

private:
    void setupUI();
    void createConnections();
    void updateControlsVisibility();

    // UI Components
    QWidget* createThresholdControls();
    QWidget* createAdaptiveControls();
    QWidget* createWatershedControls();

    // Main controls
    QCheckBox* enableCheck{nullptr};
    QComboBox* methodCombo{nullptr};
    QStackedWidget* paramStack{nullptr};

    // Threshold controls
    QSpinBox* thresholdSpin{nullptr};
    QSpinBox* maxValueSpin{nullptr};
    QCheckBox* invertColorsCheck{nullptr};

    // Adaptive controls
    QSpinBox* blockSizeSpin{nullptr};
    QDoubleSpinBox* paramCSpin{nullptr};

    // Watershed controls
    QRadioButton* distanceTransformRadio{nullptr};
    QRadioButton* manualSeedingRadio{nullptr};
    QPushButton* clearSeedsButton{nullptr};
    QLabel* seedInstructionsLabel{nullptr};

    // Internal state
    bool isSeedingMode{false};
    std::vector<cv::Point> foregroundSeeds;
    std::vector<cv::Point> backgroundSeeds;
};