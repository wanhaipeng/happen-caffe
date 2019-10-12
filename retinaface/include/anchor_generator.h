#ifndef ANCHOR_GENERTOR
#define ANCHOR_GENERTOR

#include <iostream>
#include <caffe/caffe.hpp>
#include <opencv2/opencv.hpp>
#include "config.h"

/**
 * @brief The class of face bbox which save two points coordinates
 *
 * point coordinates (x1, y1, x2, y2)
 */

class CRect2f {
 public:
  /**
   * @brief bbox Constructor
   *
   * @param x1, y1, x2, y2
   */
  CRect2f(float x1, float y1, float x2, float y2) {
    val_[0] = x1;
    val_[1] = y1;
    val_[2] = x2;
    val_[3] = y2;
  }
  ~CRect2f() {}
  /**
   * @brief get coordfinates refrence
   *
   * @param coordinates index
   */
  float& operator[](int i) {
    return val_[i];
  }
  /**
   * @brief get coordfinates value
   *
   * @param coorddinates index
   */
  float operator[](int i) const {
    return val_[i];
  }
  /**
   * @brief print bbox coordfinates info
   */
  void print() {
    std::cout << "rect: " << val_[0] << " "
                          << val_[1] << " "
                          << val_[2] << " "
                          << val_[3] << std::endl;
  }
 private:
  /// coordfinates array
  float val_[4];
};

/**
 * @brief The class of detection anchor
 */
class Anchor {
 public:
  Anchor() {}
  ~Anchor() {}
  /**
   * @brief < overload
   */
  bool operator<(const Anchor &t) const {
    return score < t.score;
  }
  /**
   * @brief > overload
   */
  bool operator>(const Anchor &t) const {
      return score > t.score;
  }
  /**
   * @brief get anchor info reference
   *
   * @param info index
   */
  float& operator[](int i) {
    assert(0 <= i && i <= 4);
    if (i == 0) return finalbox.x;
    if (i == 1) return finalbox.y;
    if (i == 2) return finalbox.width;
    if (i == 3) return finalbox.height;
  }
  /**
   * @brief get anchor info reference
   *
   * @param info index
   */
  float operator[](int i) const {
    assert(0 <= i && i <= 4);
    if (i == 0) return finalbox.x;
    if (i == 1) return finalbox.y;
    if (i == 2) return finalbox.width;
    if (i == 3) return finalbox.height;
  }
  /// anchor x1,y1,x2,y2
  cv::Rect2f anchor;
  /// offset regression
  float reg[4];
  /// anchor feat center
  cv::Point center;
  // anchor cls score
  float score;
  /// prediction landmarks points
  std::vector<cv::Point2f> pts;
  /// final detection bbox
  cv::Rect2f finalbox;
  /**
   * @brief print anchor info
   */
  void print() {
    std::cout << "finalbox: " << finalbox.x << " "
                              << finalbox.y << " "
                              << finalbox.width << " "
                              << finalbox.height << " "
              << "score: " << score << std::endl;
    std::cout << "landmarks: ";
    for (int i = 0; i < pts.size(); ++i) {
      std::cout << pts[i].x << " " << pts[i].y << ",";
    }
    std::cout << std::endl;
  }
};

/**
 * @brief The class of anchor generator
 */
class AnchorGenerator {
 public:
  AnchorGenerator();
  ~AnchorGenerator();
  /**
   * @brief Init different anchor
   * 
   * @param stride Fm stirde
   * @param cfg Anchor config
   * @param dense_anchor Is dense anchor
   */
  int Init(int stride, const AnchorCfg& cfg, bool dense_anchor);

  // anchor plane
  int Generate(int fwidth, int fheight, int stride, float step, std::vector<int>& size, \
      std::vector<float>& ratio, bool dense_anchor);

  // filter anchors and return valid anchors
  // int FilterAnchor(const caffe::Blob<float>* cls, const caffe::Blob<float>* reg, \
  // const caffe::Blob<float>* pts, std::vector<Anchor>& result);
  int FilterAnchor(const caffe::Blob<float> *cls, const caffe::Blob<float> *reg, \
      const caffe::Blob<float> *pts, std::vector<Anchor>& result, float confidence_threshold);

  int FilterAnchor(const caffe::Blob<float> *cls, const caffe::Blob<float> *reg, \
      const caffe::Blob<float> *pts, std::vector<Anchor>& result, float ratio_w, \
      float ratio_h, float confidence_threshold);

 private:
  void _ratio_enum(const CRect2f& anchor, const std::vector<float>& ratios, \
      std::vector<CRect2f>& ratio_anchors);

  void _scale_enum(const std::vector<CRect2f>& ratio_anchor, const std::vector<float>& scales, \
      std::vector<CRect2f>& scale_anchors);

  void bbox_pred(const CRect2f& anchor, const CRect2f& delta, cv::Rect2f& box);

  void landmark_pred(const CRect2f anchor, const std::vector<cv::Point2f>& delta, \
      std::vector<cv::Point2f>& pts);
  /// corrspont to channels
  std::vector<std::vector<Anchor>> anchor_planes;
  /// anchor size array
  std::vector<int> anchor_size;
  /// anchor ratio array
  std::vector<float> anchor_ratio;
  /// scale step
  float anchor_step;
  /// anchor tile stride
  int anchor_stride;
  /// feature map width
  int feature_w;
  /// feature map height
  int feature_h;
  /// the preset anchor
  std::vector<CRect2f> preset_anchors;
  /// anchor type number
  int anchor_num;
  float ratiow=0.0;
  float ratioh=0.0;
};

#endif // ANCHOR_GENERTOR
