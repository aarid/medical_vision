#include "image_viewer.hpp"
#include <QtGui/QPainter>
#include <QtGui/QMouseEvent>
#include <QtGui/QResizeEvent>
#include <opencv2/imgproc.hpp>

ImageViewer::ImageViewer(const QString& title, QWidget* parent)
    : QWidget(parent)
    , title(title) {
    // Set size policy
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    setMinimumSize(200, 200);
    
    // Enable mouse tracking for mouseMoved signal
    setMouseTracking(true);
}

void ImageViewer::setImage(const cv::Mat& image) {
    if (image.empty()) return;
    
    currentImage = matToQImage(image);
    needsUpdate = true;
    update();
}

void ImageViewer::setOverlay(const cv::Mat& overlay, double alpha) {
    if (overlay.empty()) return;
    
    overlayImage = matToQImage(overlay);
    overlayAlpha = alpha;
    needsUpdate = true;
    update();
}

void ImageViewer::clearOverlay() {
    overlayImage = QImage();
    needsUpdate = true;
    update();
}

void ImageViewer::setAspectRatioMode(Qt::AspectRatioMode mode) {
    aspectRatioMode = mode;
    needsUpdate = true;
    update();
}

void ImageViewer::setTitle(const QString& newTitle) {
    title = newTitle;
    update();
}

QRect ImageViewer::getImageRect() const {
    if (currentImage.isNull()) return QRect();

    QSize viewSize = size();
    QSize imgSize = currentImage.size();

    // Calculate scaled size maintaining aspect ratio
    QSize scaledSize = imgSize;
    scaledSize.scale(viewSize, aspectRatioMode);

    // Center the image
    QPoint topLeft(
        (viewSize.width() - scaledSize.width()) / 2,
        (viewSize.height() - scaledSize.height()) / 2
    );

    return QRect(topLeft, scaledSize);
}

cv::Point ImageViewer::getImageCoordinates(const QPoint& widgetPos) const {
    QRect imageRect = getImageRect();
    if (!imageRect.isValid()) return cv::Point(-1, -1);

    // Check if point is inside image area
    if (!imageRect.contains(widgetPos)) return cv::Point(-1, -1);

    // Calculate relative position in image
    double xRatio = (widgetPos.x() - imageRect.x()) / (double)imageRect.width();
    double yRatio = (widgetPos.y() - imageRect.y()) / (double)imageRect.height();

    // Convert to image coordinates
    return cv::Point(
        static_cast<int>(xRatio * currentImage.width()),
        static_cast<int>(yRatio * currentImage.height())
    );
}

void ImageViewer::paintEvent(QPaintEvent* event) {
    Q_UNUSED(event);
    QPainter painter(this);
    
    // Draw background
    painter.fillRect(rect(), Qt::black);

    if (currentImage.isNull()) {
        // Draw placeholder text
        painter.setPen(Qt::white);
        painter.drawText(rect(), Qt::AlignCenter, "No Image");
        return;
    }

    // Update cached pixmap if needed
    if (needsUpdate) {
        updateCache();
    }

    // Draw cached image
    QRect imageRect = getImageRect();
    painter.drawPixmap(imageRect, cachedPixmap);

    // Draw title if present
    if (!title.isEmpty()) {
        painter.setPen(Qt::white);
        painter.drawText(rect().adjusted(5, 5, -5, -5), Qt::AlignTop | Qt::AlignLeft, title);
    }
}

void ImageViewer::mousePressEvent(QMouseEvent* event) {
    cv::Point imagePos = getImageCoordinates(event->pos());
    if (imagePos.x >= 0 && imagePos.y >= 0) {
        emit mousePressed(imagePos, event->button());
    }
}

void ImageViewer::mouseMoveEvent(QMouseEvent* event) {
    cv::Point imagePos = getImageCoordinates(event->pos());
    if (imagePos.x >= 0 && imagePos.y >= 0) {
        emit mouseMoved(imagePos);
    }
}

void ImageViewer::resizeEvent(QResizeEvent* event) {
    Q_UNUSED(event);
    needsUpdate = true;
}

void ImageViewer::updateCache() {
    if (currentImage.isNull()) return;

    QRect imageRect = getImageRect();
    QImage scaledImage = currentImage.scaled(
        imageRect.size(), 
        aspectRatioMode, 
        Qt::SmoothTransformation
    );

    // Apply overlay if present
    if (!overlayImage.isNull()) {
        QPainter painter(&scaledImage);
        painter.setOpacity(overlayAlpha);
        painter.drawImage(0, 0, overlayImage.scaled(
            imageRect.size(),
            aspectRatioMode,
            Qt::SmoothTransformation
        ));
    }

    cachedPixmap = QPixmap::fromImage(scaledImage);
    needsUpdate = false;
}

QImage ImageViewer::matToQImage(const cv::Mat& mat) const {
    if (mat.empty()) return QImage();

    if (mat.type() == CV_8UC1) {
        return QImage(mat.data, mat.cols, mat.rows, 
                     mat.step, QImage::Format_Grayscale8).copy();
    }

    if (mat.type() == CV_8UC3) {
        cv::Mat rgb;
        cv::cvtColor(mat, rgb, cv::COLOR_BGR2RGB);
        return QImage(rgb.data, rgb.cols, rgb.rows, 
                     rgb.step, QImage::Format_RGB888).copy();
    }

    if (mat.type() == CV_8UC4) {
        return QImage(mat.data, mat.cols, mat.rows, 
                     mat.step, QImage::Format_RGBA8888).copy();
    }

    return QImage();
}