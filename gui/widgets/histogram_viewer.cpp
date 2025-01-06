#include "histogram_viewer.hpp"
#include <QtGui/QPainter>
#include <QtGui/QPainterPath>
#include <QtWidgets/QStyleOption>

HistogramViewer::HistogramViewer(QWidget* parent)
    : QWidget(parent)
    , channelColors{Qt::blue, Qt::green, Qt::red} {  // Default BGR colors
    
    // Set size policy
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    setMinimumSize(MIN_WIDTH, DEFAULT_HEIGHT);

    // Set background
    setBackgroundRole(QPalette::Base);
    setAutoFillBackground(true);
}

void HistogramViewer::setHistogram(const cv::Mat& histogram) {
    if (histogram.empty()) return;

    histogramData = histogram.clone();
    needsUpdate = true;
    update();
}

void HistogramViewer::setChannelColors(const QVector<QColor>& colors) {
    channelColors = colors;
    needsUpdate = true;
    update();
}

void HistogramViewer::showGrid(bool show) {
    showGridLines = show;
    needsUpdate = true;
    update();
}

void HistogramViewer::setTitle(const QString& title) {
    titleText = title;
    needsUpdate = true;
    update();
}

void HistogramViewer::paintEvent(QPaintEvent* event) {
    Q_UNUSED(event);

    // Create painter
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);

    // Draw background
    QStyleOption opt;
    opt.initFrom(this);
    style()->drawPrimitive(QStyle::PE_Widget, &opt, &painter, this);

    // Calculate drawing rect
    QRect drawRect = rect().adjusted(margin, margin, -margin, -margin);
    if (!titleText.isEmpty()) {
        // Adjust for title
        QFontMetrics fm(font());
        int titleHeight = fm.height() + 5;
        drawRect.adjust(0, titleHeight, 0, 0);
        
        // Draw title
        painter.drawText(rect().adjusted(margin, 5, -margin, 0),
                        Qt::AlignTop | Qt::AlignHCenter,
                        titleText);
    }

    // Draw grid if enabled
    if (showGridLines) {
        drawGrid(painter, drawRect);
    }

    // Draw histogram
    drawHistogram(painter, drawRect);

    // Draw labels
    drawLabels(painter, drawRect);
}

void HistogramViewer::drawHistogram(QPainter& painter, const QRect& rect) {
    if (histogramData.empty()) {
        // Draw placeholder text
        painter.setPen(Qt::gray);
        painter.drawText(rect, Qt::AlignCenter, tr("No histogram data"));
        return;
    }

    // Calculate scaling factors
    double binWidth = rect.width() / static_cast<double>(histogramData.cols);
    double maxVal = 0;
    cv::minMaxLoc(histogramData, nullptr, &maxVal);
    double scale = rect.height() / maxVal;

    // Draw each channel
    for (int channel = 0; channel < histogramData.channels(); ++channel) {
        QPainterPath path;
        path.moveTo(rect.left(), rect.bottom());

        // Extract channel data
        std::vector<float> channelData;
        for (int i = 0; i < histogramData.cols; ++i) {
            channelData.push_back(histogramData.at<float>(channel, i));
        }

        // Create path
        for (int bin = 0; bin < histogramData.cols; ++bin) {
            double x = rect.left() + bin * binWidth;
            double y = rect.bottom() - (channelData[bin] * scale);
            path.lineTo(x, y);
        }
        path.lineTo(rect.right(), rect.bottom());
        path.closeSubpath();

        // Draw filled path with transparency
        QColor color = channelColors[channel];
        color.setAlpha(128);
        painter.setBrush(color);
        painter.setPen(Qt::NoPen);
        painter.drawPath(path);

        // Draw line on top
        painter.setPen(channelColors[channel]);
        path = QPainterPath();
        for (int bin = 0; bin < histogramData.cols; ++bin) {
            double x = rect.left() + bin * binWidth;
            double y = rect.bottom() - (channelData[bin] * scale);
            if (bin == 0) path.moveTo(x, y);
            else path.lineTo(x, y);
        }
        painter.drawPath(path);
    }
}

void HistogramViewer::drawGrid(QPainter& painter, const QRect& rect) {
    painter.save();
    
    // Set up pen for grid
    QPen gridPen(Qt::gray);
    gridPen.setStyle(Qt::DotLine);
    painter.setPen(gridPen);

    // Draw vertical lines
    for (int x = 0; x <= 255; x += 51) {  // Draw at 0, 51, 102, 153, 204, 255
        int xPos = rect.left() + (x * rect.width() / 255);
        painter.drawLine(xPos, rect.top(), xPos, rect.bottom());
    }

    // Draw horizontal lines
    int numHLines = 4;
    for (int i = 0; i <= numHLines; ++i) {
        int y = rect.top() + (i * rect.height() / numHLines);
        painter.drawLine(rect.left(), y, rect.right(), y);
    }

    painter.restore();
}

void HistogramViewer::drawLabels(QPainter& painter, const QRect& rect) {
    painter.save();
    
    // Draw x-axis labels
    for (int x = 0; x <= 255; x += 51) {
        int xPos = rect.left() + (x * rect.width() / 255);
        painter.drawText(QRect(xPos - 20, rect.bottom() + 5, 40, margin),
                        Qt::AlignHCenter | Qt::AlignTop,
                        QString::number(x));
    }

    painter.restore();
}

void HistogramViewer::resizeEvent(QResizeEvent* event) {
    Q_UNUSED(event);
    needsUpdate = true;
}