#ifndef ROTATION_AVX2_H
#define ROTATION_AVX2_H

#include <stdint.h>

// Rotate image by PI/2 with vectorization
void RotateBy1Pi2FactorAVX2(const uint8_t* srcData, uint8_t* destData, uint32_t width);

// Rotate image by 3PI/2 with vectorization
void RotateBy3Pi2FactorAVX2(const uint8_t* srcData, uint8_t* destData, uint32_t width);

// Solution that should be the most optimized, but way more complicated and error-prone, and doesn't seem to do that well compared to RotateBmpAVX2v2 
void RotateBmpAVX2(const uint8_t* srcData, uint8_t* destData, uint32_t width, float rad);

// Solution that uses same algorithm as RotateBmpAVX2 but doesn't try to speed up by applying different treatments depending on the region of the image
void RotateBmpAVX2v2(const uint8_t* srcData, uint8_t* destData, uint32_t width, float angle);

#endif // ROTATION_AVX2_H