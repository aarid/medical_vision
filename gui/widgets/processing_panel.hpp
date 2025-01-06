#pragma once

#include <QtWidgets/QGroupBox>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QDoubleSpinBox>
#include "../include/medical_vision/image_preprocessor.hpp"

class ProcessingPanel : public QGroupBox {
    Q_OBJECT

public:
    explicit ProcessingPanel(QWidget* parent = nullptr);
    ~ProcessingPanel() = default;

    // Get current settings
    struct ProcessingSettings {
        bool denoiseEnabled{false};
        bool claheEnabled{false};
        bool sharpenEnabled{false};
        double sharpenStrength{1.0};
        medical_vision::ImagePreprocessor::NoiseReductionMethod denoiseMethod{
            medical_vision::ImagePreprocessor::NoiseReductionMethod::BILATERAL};
    };

    ProcessingSettings getCurrentSettings() const;
    void resetSettings();

signals:
    void settingsChanged();

private:
    void setupUI();
    void createConnections();

    // Processing controls
    QCheckBox* denoiseCheck{nullptr};
    QComboBox* denoiseMethodCombo{nullptr};
    QCheckBox* claheCheck{nullptr};
    QCheckBox* sharpenCheck{nullptr};
    QDoubleSpinBox* strengthSpinner{nullptr};

    // Layout helpers
    QWidget* createDenoiseControls();
    QWidget* createSharpenControls();
};