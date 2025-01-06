#pragma once

#include <QtWidgets/QWidget>
#include <opencv2/core.hpp>

class HistogramViewer : public QWidget {
    Q_OBJECT

public:
    explicit HistogramViewer(QWidget* parent = nullptr);
    ~HistogramViewer() = default;

    void setHistogram(const cv::Mat& histogram);

protected:
    void paintEvent(QPaintEvent* event) override;

private:
    QImage histogramImage;
    static constexpr int MIN_WIDTH = 512;
    static constexpr int MIN_HEIGHT = 200;
};