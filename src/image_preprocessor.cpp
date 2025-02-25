/**
 * @file image_preprocessor.cpp
 * @brief Implementation of image preprocessing operations
 */

#include "../include/medical_vision/image_preprocessor.hpp"
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>
#include <stdexcept>

namespace medical_vision {

// ---------- Basic Operations ----------

bool ImagePreprocessor::loadImage(const std::string& filepath) {
    image_ = cv::imread(filepath, cv::IMREAD_UNCHANGED);
    if (image_.empty()) {
        return false;
    }
    updateOriginalImage();
    return true;
}

bool ImagePreprocessor::saveImage(const std::string& filepath) const {
    if (!checkImageLoaded()) return false;
    return cv::imwrite(filepath, image_);
}

// ---------- Image Information ----------

cv::Size ImagePreprocessor::getImageSize() const {
    return image_.size();
}

int ImagePreprocessor::getChannels() const {
    return image_.channels();
}

std::string ImagePreprocessor::getImageType() const {
    std::string r;
    uint8_t depth = image_.type() & CV_MAT_DEPTH_MASK;
    uint8_t chans = 1 + (image_.type() >> CV_CN_SHIFT);
    
    switch (depth) {
        case CV_8U:  r = "8U"; break;
        case CV_8S:  r = "8S"; break;
        case CV_16U: r = "16U"; break;
        case CV_16S: r = "16S"; break;
        case CV_32S: r = "32S"; break;
        case CV_32F: r = "32F"; break;
        case CV_64F: r = "64F"; break;
        default:     r = "User"; break;
    }

    r += "C";
    r += (chans + '0');
    return r;
}

// ---------- Noise Reduction ----------


bool ImagePreprocessor::denoise(NoiseReductionMethod method) {
    if (!checkImageLoaded()) return false;

    try {
        switch (method) {
            case NoiseReductionMethod::GAUSSIAN:
                return gaussianBlur();
            
            case NoiseReductionMethod::MEDIAN:
                return medianBlur();
            
            case NoiseReductionMethod::BILATERAL: {
                cv::Mat temp;
                if (image_.type() != CV_8UC1 && image_.type() != CV_8UC3) {
                    image_.convertTo(temp, CV_8U);
                } else {
                    temp = image_.clone();
                }

                if (temp.channels() == 1) {
                    cv::bilateralFilter(temp, image_, 9, 75, 75);
                } else {
                    cv::Mat result;
                    cv::bilateralFilter(temp, result, 9, 75, 75);
                    result.copyTo(image_);
                }
                return true;
            }
            
            case NoiseReductionMethod::NLM: {
                cv::Mat temp;
                if (image_.type() != CV_8UC1 && image_.type() != CV_8UC3) {
                    image_.convertTo(temp, CV_8U);
                } else {
                    temp = image_.clone();
                }

                if (temp.channels() == 1) {
                    cv::fastNlMeansDenoising(temp, image_);
                } else {
                    cv::fastNlMeansDenoisingColored(temp, image_);
                }
                return true;
            }
            
            default:
                return false;
        }
    }
    catch (const cv::Exception& e) {
        std::cerr << "OpenCV error: " << e.what() << std::endl;
        return false;
    }
}

bool ImagePreprocessor::gaussianBlur(int kernelSize, double sigma) {
    if (!checkImageLoaded() || !validateKernelSize(kernelSize)) return false;
    
    cv::GaussianBlur(image_, image_, cv::Size(kernelSize, kernelSize), sigma);
    return true;
}

bool ImagePreprocessor::medianBlur(int kernelSize) {
    if (!checkImageLoaded() || !validateKernelSize(kernelSize)) return false;
    
    cv::medianBlur(image_, image_, kernelSize);
    return true;
}

bool ImagePreprocessor::bilateralFilter(int diameter, double sigmaColor, double sigmaSpace) {
    if (!checkImageLoaded()) return false;
    
    cv::bilateralFilter(image_, image_, diameter, sigmaColor, sigmaSpace);
    return true;
}

bool ImagePreprocessor::nonLocalMeans(float h, int templateWindowSize, int searchWindowSize) {
    if (!checkImageLoaded()) return false;
    
    cv::fastNlMeansDenoising(image_, image_, h, templateWindowSize, searchWindowSize);
    return true;
}

// ---------- Contrast and Brightness ----------

bool ImagePreprocessor::normalize(double minValue, double maxValue) {
    if (!checkImageLoaded()) return false;
    
    cv::normalize(image_, image_, minValue, maxValue, cv::NORM_MINMAX);
    return true;
}

bool ImagePreprocessor::adjustContrast(double alpha, double beta) {
    if (!checkImageLoaded()) return false;
    
    image_.convertTo(image_, -1, alpha, beta);
    return true;
}


bool ImagePreprocessor::histogramProcessing(HistogramMethod method) {
    if (!checkImageLoaded()) return false;

    switch (method) {
        case HistogramMethod::EQUALIZATION: {
            if (image_.channels() == 1) {
                cv::equalizeHist(image_, image_);
            } else {
                cv::Mat ycrcb;
                cv::cvtColor(image_, ycrcb, cv::COLOR_BGR2YCrCb);
                std::vector<cv::Mat> channels;
                cv::split(ycrcb, channels);
                cv::equalizeHist(channels[0], channels[0]);
                cv::merge(channels, ycrcb);
                cv::cvtColor(ycrcb, image_, cv::COLOR_YCrCb2BGR);
            }
            return true;
        }
        case HistogramMethod::CLAHE:
            return clahe();
        case HistogramMethod::STRETCHING: {
            if (image_.channels() == 1) {
                double minVal, maxVal;
                cv::minMaxLoc(image_, &minVal, &maxVal);
                image_.convertTo(image_, -1, 255.0/(maxVal - minVal), -minVal * 255.0/(maxVal - minVal));
            } else {
                std::vector<cv::Mat> channels;
                cv::split(image_, channels);
                for (auto& channel : channels) {
                    double minVal, maxVal;
                    cv::minMaxLoc(channel, &minVal, &maxVal);
                    channel.convertTo(channel, -1, 255.0/(maxVal - minVal), -minVal * 255.0/(maxVal - minVal));
                }
                cv::merge(channels, image_);
            }
            return true;
        }
        default:
            return false;
    }
}

bool ImagePreprocessor::clahe(double clipLimit, cv::Size tileGridSize) {
    if (!checkImageLoaded()) return false;

    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(clipLimit, tileGridSize);
    
    if (image_.channels() == 1) {
        clahe->apply(image_, image_);
    } else {
        cv::Mat lab;
        cv::cvtColor(image_, lab, cv::COLOR_BGR2Lab);
        std::vector<cv::Mat> channels;
        cv::split(lab, channels);
        clahe->apply(channels[0], channels[0]);
        cv::merge(channels, lab);
        cv::cvtColor(lab, image_, cv::COLOR_Lab2BGR);
    }
    return true;
}

// ---------- Edge Enhancement ----------

bool ImagePreprocessor::sharpen(double strength) {
    if (!checkImageLoaded()) return false;
    
    cv::Mat kernel = (cv::Mat_<float>(3,3) <<
        -1, -1, -1,
        -1,  9, -1,
        -1, -1, -1);
    kernel *= strength;

    if (image_.channels() == 1) {
        cv::filter2D(image_, image_, -1, kernel);
    } else {
        std::vector<cv::Mat> channels;
        cv::split(image_, channels);
        for (auto& channel : channels) {
            cv::filter2D(channel, channel, -1, kernel);
        }
        cv::merge(channels, image_);
    }
    return true;
}

bool ImagePreprocessor::unsharpMask(double sigma, double strength) {
    if (!checkImageLoaded()) return false;
    
    if (image_.channels() == 1) {
        cv::Mat blurred;
        cv::GaussianBlur(image_, blurred, cv::Size(), sigma);
        cv::addWeighted(image_, 1.0 + strength, blurred, -strength, 0, image_);
    } else {
        std::vector<cv::Mat> channels;
        cv::split(image_, channels);
        for (auto& channel : channels) {
            cv::Mat blurred;
            cv::GaussianBlur(channel, blurred, cv::Size(), sigma);
            cv::addWeighted(channel, 1.0 + strength, blurred, -strength, 0, channel);
        }
        cv::merge(channels, image_);
    }
    return true;
}

// ---------- Utility Functions ----------

cv::Mat ImagePreprocessor::getHistogram() const {
    if (!checkImageLoaded()) return cv::Mat();

    // Setup histogram parameters
    int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange = { range };
    bool uniform = true, accumulate = false;
    
    // Create output image
    int hist_w = 512, hist_h = 400;
    int bin_w = cvRound((double)hist_w/histSize);
    cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0,0,0));

    if (image_.channels() == 1) {
        // Grayscale image
        cv::Mat hist;
        cv::calcHist(&image_, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);
        cv::normalize(hist, hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());

        // Draw histogram in white
        for(int i = 1; i < histSize; i++) {
            cv::line(histImage, 
                     cv::Point(bin_w*(i-1), hist_h - cvRound(hist.at<float>(i-1))),
                     cv::Point(bin_w*(i), hist_h - cvRound(hist.at<float>(i))),
                     cv::Scalar(255, 255, 255), 2, 8, 0);
        }
    } else {
        // Color image
        std::vector<cv::Mat> bgr_planes;
        cv::split(image_, bgr_planes);

        // Compute histograms for each channel
        cv::Mat b_hist, g_hist, r_hist;
        cv::calcHist(&bgr_planes[0], 1, 0, cv::Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
        cv::calcHist(&bgr_planes[1], 1, 0, cv::Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
        cv::calcHist(&bgr_planes[2], 1, 0, cv::Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);

        // Normalize histograms
        cv::normalize(b_hist, b_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
        cv::normalize(g_hist, g_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
        cv::normalize(r_hist, r_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());

        // Draw for each channel
        for(int i = 1; i < histSize; i++) {
            cv::line(histImage, 
                     cv::Point(bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1))),
                     cv::Point(bin_w*(i), hist_h - cvRound(b_hist.at<float>(i))),
                     cv::Scalar(255, 0, 0), 2, 8, 0);
            cv::line(histImage, 
                     cv::Point(bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1))),
                     cv::Point(bin_w*(i), hist_h - cvRound(g_hist.at<float>(i))),
                     cv::Scalar(0, 255, 0), 2, 8, 0);
            cv::line(histImage, 
                     cv::Point(bin_w*(i-1), hist_h - cvRound(r_hist.at<float>(i-1))),
                     cv::Point(bin_w*(i), hist_h - cvRound(r_hist.at<float>(i))),
                     cv::Scalar(0, 0, 255), 2, 8, 0);
        }
    }

    return histImage;
}

// ---------- Private Methods ----------

bool ImagePreprocessor::checkImageLoaded() const {
    return !image_.empty();
}

void ImagePreprocessor::updateOriginalImage() {
    originalImage_ = image_.clone();
}

bool ImagePreprocessor::validateKernelSize(int kernelSize) const {
    return kernelSize > 0 && kernelSize % 2 == 1;
}

} // namespace medical_vision