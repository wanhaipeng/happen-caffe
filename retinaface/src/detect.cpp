#include "anchor_generator.h"
#include "detect.h"
#include "tools.h"

uint64_t current_timestamp() {
  struct timeval te; 
  gettimeofday(&te, NULL); // get current time
  return te.tv_sec * 1000LL + te.tv_usec / 1000; // caculate milliseconds
}
/*
void printMat(const cv::Mat &image) {
  uint8_t  *myData = image.data;
  int width = image.cols;
  int height = image.rows;
  int _stride = image.step;//in case cols != strides
  for(int i = 0; i < height; i++) {
    for(int j = 0; j < width; j++) {
      uint8_t  val = myData[ i * _stride + j];
      cout << val;
      // do whatever you want with your value
    }
  }
  cout << endl;
}
*/

Detector::Detector(const std::string& model_file,
                   const std::string& weights_file,
                   const float confidence,
                   const float nms,
                   const std::string& gpu_mode) {
  if(gpu_mode == "gpu") {
    #ifdef CPU_ONLY
    caffe::Caffe::set_mode(caffe::Caffe::CPU);
    #else
    caffe::Caffe::set_mode(caffe::Caffe::GPU);
    caffe::Caffe::SetDevice(1);
    #endif // CPU_ONLY  
  } else {
    caffe::Caffe::set_mode(caffe::Caffe::CPU);
  }
  // build the caffe network
  net_.reset(new caffe::Net<float>(model_file, caffe::TEST));
  // load parameters
  net_->CopyTrainedLayersFrom(weights_file);
  CHECK_EQ(net_->num_inputs(), 1)
    << "Network should have exactly one input.";
  caffe::Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  confidence_threshold = confidence;
  nms_threshold = nms;
  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
  // network default input: width 1980 height 1024 default (VGA resolution 640*480)
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
}

void Detector::preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels) {
  cv::Mat sample, sample_resized, sample_float;
  if (img.channels() == 3 && num_channels_ == 1) cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1) cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3) cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3) cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else sample = img;
  // std::cout << "img1" << img.rows << std::endl;
  // cv::cvtColor(img, sample, cv::COLOR_BGR2RGB);
  if(sample.size()!= input_geometry_) {
    // cout << "need to be resized" << endl;
    ratio_w = float(img.cols)/float(input_geometry_.width);
    ratio_h = float(img.rows)/float(input_geometry_.height);
    // std::cout << "ratio_wratio_w" << ratio_w << "ratio_hratio_h " << ratio_h << std::endl;
    cv::resize(sample, sample_resized, input_geometry_);
    // std::cout << "img2" << img.rows << std::endl;
  } else
    sample_resized=sample;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1); 
  // std::cout << "img3" << img.rows << std::endl;
  cv::split(sample_float, *input_channels);
}

void Detector::wrapInputLayer(std::shared_ptr<caffe::Net<float> > net_,
    std::vector<cv::Mat>* input_channels) {
  caffe::Blob<float>* input_layer = net_->input_blobs()[0];
  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

std::vector<Anchor> Detector:: Detect(cv::Mat& img) {
  // std::cout << "input.rows :"<<input.rows << "   "<< "fd_h_ :"<<fd_h_<<endl;
  // std::cout<<"input.cols :"<<input.cols <<"   "<<"fd_w_ :" << fd_w_<<endl;
  // std::cout<< "input.channels():"<<input.channels()<<"  "<<"fd_c_: "<<fd_c_<<endl;
  // assert(input.rows == fd_h_ && input.cols == fd_w_ && input.channels() == fd_c_);
  std::vector<cv::Mat> input_channels;  
  wrapInputLayer(net_, &input_channels);
  preprocess(img, &input_channels);
  net_->Forward();
  // extern std::vector<int> _feat_stride_fpn;
  // extern std::map<int, AnchorCfg> anchor_cfg;
  // _feat_stride_fpn.size()=3 three levels fpn
  std::vector<AnchorGenerator> ac(_feat_stride_fpn.size());
  for (int i = 0; i < _feat_stride_fpn.size(); ++i) {
    int stride = _feat_stride_fpn[i];
    ac[i].Init(stride, anchor_cfg[stride], false);
  }
  std::vector<Anchor> proposals;
  proposals.clear();
  for (int i = 0; i < _feat_stride_fpn.size(); ++i) {
    char clsname[100];
    sprintf(clsname, "face_rpn_cls_prob_reshape_stride%d", _feat_stride_fpn[i]);
    char regname[100];
    sprintf(regname, "face_rpn_bbox_pred_stride%d", _feat_stride_fpn[i]);
    char ptsname[100];
    sprintf(ptsname, "face_rpn_landmark_pred_stride%d", _feat_stride_fpn[i]);
    const caffe::Blob<float>* clsBlob = net_->blob_by_name(clsname).get();
    const caffe::Blob<float>* regBlob = net_->blob_by_name(regname).get();
    const caffe::Blob<float>* ptsBlob = net_->blob_by_name(ptsname).get();
    ac[i].FilterAnchor(clsBlob, regBlob, ptsBlob, proposals, ratio_w, ratio_h, confidence_threshold);
    std::cout << "fp stride: " << _feat_stride_fpn[i]
              << " proposals size: " << proposals.size() << std::endl;
    // for (int r = 0; r < proposals.size(); ++r) {
    //   proposals[r].print();
    // }
  }
  // nms
  std::vector<Anchor> result;
  nms_cpu(proposals, nms_threshold, result);
  // printf("final result %d\n", result.size());
  std::cout << "final detect result size: " << result.size() << std::endl;
  // for (int r = 0; r < proposals.size(); ++r) {
  //   result[r].print();
  // }
  return result;
}


