#pragma once

#include <QtWidgets/QWidget>
#include <opencv2/core.hpp>

class ImageViewer : public QWidget {
    Q_OBJECT

public:
    explicit ImageViewer(const QString& title = "", QWidget* parent = nullptr);
    ~ImageViewer() = default;

    // Image handling
    void setImage(const cv::Mat& image);
    void setOverlay(const cv::Mat& overlay, double alpha = 0.3);
    void clearOverlay();
    
    // Coordinate conversion
    cv::Point getImageCoordinates(const QPoint& widgetPos) const;
    
    // Settings
    void setAspectRatioMode(Qt::AspectRatioMode mode);
    void setTitle(const QString& title);

signals:
    void mousePressed(cv::Point imagePos, Qt::MouseButton button);
    void mouseMoved(cv::Point imagePos);

protected:
    void paintEvent(QPaintEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void resizeEvent(QResizeEvent* event) override;

private:
    // Helper functions
    QRect getImageRect() const;
    QImage matToQImage(const cv::Mat& mat) const;
    void updateCache();

    // Image data
    QImage currentImage;
    QImage overlayImage;
    double overlayAlpha{0.3};
    
    // Display settings
    QString title;
    Qt::AspectRatioMode aspectRatioMode{Qt::KeepAspectRatio};
    
    // Cache for better performance
    QPixmap cachedPixmap;
    bool needsUpdate{false};
};