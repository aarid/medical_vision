#pragma once

#include <QtWidgets/QWidget>
#include <opencv2/core.hpp>

class HistogramViewer : public QWidget {
    Q_OBJECT

public:
    explicit HistogramViewer(QWidget* parent = nullptr);
    ~HistogramViewer() = default;

    void setHistogram(const cv::Mat& histogram);
    void setChannelColors(const QVector<QColor>& colors);
    void showGrid(bool show);
    void setTitle(const QString& title);

protected:
    void paintEvent(QPaintEvent* event) override;
    void resizeEvent(QResizeEvent* event) override;

private:
    void updateCache();
    void drawHistogram(QPainter& painter, const QRect& rect);
    void drawGrid(QPainter& painter, const QRect& rect);
    void drawLabels(QPainter& painter, const QRect& rect);

    // Data
    cv::Mat histogramData;
    QVector<QColor> channelColors;
    
    // Display settings
    bool showGridLines{true};
    QString titleText;
    int margin{20};
    int labelSpacing{50};
    
    // Cache
    QPixmap cachedBackground;
    bool needsUpdate{true};

    // Constants
    static constexpr int DEFAULT_HEIGHT = 200;
    static constexpr int MIN_WIDTH = 300;
};