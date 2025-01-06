#include "histogram_viewer.hpp"
#include <QtGui/QPainter>
#include <opencv2/imgproc.hpp>

HistogramViewer::HistogramViewer(QWidget* parent)
    : QWidget(parent) {
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    setMinimumSize(MIN_WIDTH, MIN_HEIGHT);
    setBackgroundRole(QPalette::Base);
    setAutoFillBackground(true);
}

void HistogramViewer::setHistogram(const cv::Mat& histogram) {
    if (histogram.empty()) {
        histogramImage = QImage();
    } else {
        // Convert OpenCV Mat to QImage
        if (histogram.type() == CV_8UC3) {
            cv::Mat rgb;
            cv::cvtColor(histogram, rgb, cv::COLOR_BGR2RGB);
            histogramImage = QImage(rgb.data, rgb.cols, rgb.rows, 
                                  rgb.step, QImage::Format_RGB888).copy();
        }
    }
    update();
}

void HistogramViewer::paintEvent(QPaintEvent*) {
    QPainter painter(this);
    
    // Fill background
    painter.fillRect(rect(), Qt::black);

    if (histogramImage.isNull()) {
        // Draw placeholder text
        painter.setPen(Qt::white);
        painter.drawText(rect(), Qt::AlignCenter, tr("No Histogram"));
        return;
    }

    // Draw histogram image scaled to widget size
    painter.drawImage(rect(), histogramImage);
}