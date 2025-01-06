#pragma once

#include <QtWidgets/QWidget>
#include <opencv2/core.hpp>

class HistogramViewer : public QWidget {
    Q_OBJECT

public:
    explicit HistogramViewer(QWidget* parent = nullptr);
    ~HistogramViewer() = default;

    void setHistogram(const cv::Mat& histogram);
    void setTitle(const QString& title);

protected:
    void paintEvent(QPaintEvent* event) override;

private:
    cv::Mat histogramData;
    QString titleText;
    static constexpr int MARGIN = 20;
    static constexpr int AXIS_LABEL_MARGIN = 15;
    static constexpr int STEPS = 4;
};