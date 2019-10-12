#include<sys/time.h>
#include "anchor_generator.h"

/**
 * @brief Get current system time
 */
uint64_t current_timestamp();
/**
 * @brief The class of face detector
 */
class Detector {
 public:
  /**
   * @brief Face detector constructor
   *
   * @param model_file caffe pb file path
   * @param weights_file caffemodel file path
   * @param confidence bbox filter confidence threshold
   * @param nms bbox nms filter threshold
   * @param gpu_mode caffe runtime mode
   */
  Detector(const std::string& model_file,
           const std::string& weights_file,
           const float confidence,
           const float nms,
           const std::string& gpu_mode);
  /**
   * @brief detect input frame
   *
   * @param img input image
   */
  std::vector<Anchor> Detect(cv::Mat& img);
 private:
  std::shared_ptr<caffe::Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  float confidence_threshold;
  float nms_threshold;
  float ratio_w=0.0;
  float ratio_h=0.0;
  /**
   * @brief input preprocess
   */
  void preprocess(const  cv::Mat& img, std::vector<cv::Mat>* input_channels);
  /**
   * @brief fill caffe network input with processed image
   */
  void wrapInputLayer(std::shared_ptr<caffe::Net<float> > net_, std::vector<cv::Mat>* input_channels);
};