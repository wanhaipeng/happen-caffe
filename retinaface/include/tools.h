#ifndef FD_TOOLS
#define FD_TOOLS

#include "anchor_generator.h"

/**
 * @brief Detected bbox nms(cpu version)
 * 
 * @param boxes input bbox before nms
 * @param threshold nms filter threshold
 * @param filterOutBoxes output bbox after nms
 */
void nms_cpu(std::vector<Anchor>& boxes, float threshold, std::vector<Anchor>& filterOutBoxes);

#endif // FD_TOOLS
