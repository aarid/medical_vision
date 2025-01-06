#include "histogram_viewer.hpp"
#include <QtGui/QPainter>
#include <QtGui/QPainterPath>

HistogramViewer::HistogramViewer(QWidget* parent)
    : QWidget(parent) {
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    setMinimumSize(300, 200);
    setBackgroundRole(QPalette::Base);
    setAutoFillBackground(true);
}

void HistogramViewer::setHistogram(const cv::Mat& histogram) {
    if (histogram.empty()) {
        histogramData.release();
    } else {
        histogram.copyTo(histogramData);
    }
    update();
}

void HistogramViewer::setTitle(const QString& title) {
    titleText = title;
    update();
}

void HistogramViewer::paintEvent(QPaintEvent*) {
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);

    // Draw background
    painter.fillRect(rect(), Qt::black);

    // Draw title if present
    if (!titleText.isEmpty()) {
        painter.setPen(Qt::white);
        painter.drawText(rect().adjusted(MARGIN, 5, -MARGIN, 0),
                        Qt::AlignTop | Qt::AlignHCenter, titleText);
    }

    if (histogramData.empty()) {
        painter.setPen(Qt::white);
        painter.drawText(rect(), Qt::AlignCenter, tr("No Data"));
        return;
    }

    // Define drawing area
    QRect drawingRect = rect().adjusted(MARGIN, MARGIN + (titleText.isEmpty() ? 0 : 20),
                                      -MARGIN, -MARGIN);

    try {
        const int bins = histogramData.cols;
        const double binWidth = drawingRect.width() / static_cast<double>(bins);

        // Colors for BGR channels
        const QColor colors[] = {Qt::blue, Qt::green, Qt::red};
        const int channels = histogramData.rows;

        // Draw histogram for each channel
        for (int channel = 0; channel < channels; ++channel) {
            QPainterPath path;
            path.moveTo(drawingRect.left(), drawingRect.bottom());

            // Create path for this channel
            for (int bin = 0; bin < bins; ++bin) {
                float value = histogramData.at<float>(channel, bin);
                int x = drawingRect.left() + bin * binWidth;
                int y = drawingRect.bottom() - (value * drawingRect.height() / 256.0);
                
                if (bin == 0) {
                    path.moveTo(x, y);
                } else {
                    path.lineTo(x, y);
                }
            }

            // Draw line for this channel
            painter.setPen(QPen(colors[channel], 2));
            painter.drawPath(path);
        }

        // Draw axes
        painter.setPen(Qt::white);
        painter.drawLine(drawingRect.bottomLeft(), drawingRect.bottomRight());
        painter.drawLine(drawingRect.bottomLeft(), drawingRect.topLeft());

        // Draw X axis labels
        for (int i = 0; i <= STEPS; ++i) {
            int x = drawingRect.left() + (i * drawingRect.width() / STEPS);
            int value = (i * 255 / STEPS);
            painter.drawText(x - 15, drawingRect.bottom() + AXIS_LABEL_MARGIN, 
                           QString::number(value));
        }

        // Draw channel legend
        if (channels > 1) {
            int legendY = drawingRect.top() + 15;
            for (int i = 0; i < channels; ++i) {
                painter.setPen(colors[i]);
                QString channelName = (i == 0) ? "B" : (i == 1) ? "G" : "R";
                painter.drawText(drawingRect.right() - 50, legendY + i * 15, channelName);
            }
        }
    }
    catch (const cv::Exception& e) {
        painter.setPen(Qt::red);
        painter.drawText(rect(), Qt::AlignCenter, 
                        tr("Error: %1").arg(e.what()));
    }
}