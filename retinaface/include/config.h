#ifndef FD_CONFIG
#define FD_CONFIG

#include <vector>
#include <map>

/// define in other files, global vairable
extern float pixel_mean[3];
extern float pixel_std[3];
extern float pixel_scale;

/**
 * @brief The class of Anchor config
 */
class AnchorCfg {
 public:
  AnchorCfg() {}
  ~AnchorCfg() {}
  AnchorCfg(const std::vector<float> s, const std::vector<float> r, int size) {
    SCALES = s;
    RATIOS = r;
    BASE_SIZE = size;
  }
  std::vector<float> SCALES;
  std::vector<float> RATIOS;
  int BASE_SIZE;
};

#define fmc 3
extern std::vector<int> _feat_stride_fpn;
extern std::map<int, AnchorCfg> anchor_cfg;
extern bool dense_anchor;
// extern float cls_threshold;
// extern float nms_threshold;

#endif // FD_CONFIG
