#ifndef ROTATION_H
#define ROTATION_H

#include <stdint.h>

/*
    This file contains utils function that are used in AVX2 version and non vectorized counterpart of the functions to make the code run without AVX2.
    The functions are pretty much duplicates of their AVX2 counterparts without vectorization.
*/
inline
bool IsInbound(uint32_t X, uint32_t Y, uint32_t width)
{
    return X >= 0 && X < width && Y >= 0 && Y < width;
}

// Rotate image by PI
void RotateByPi(const uint8_t* srcData, uint8_t* destData, uint32_t width);

// Rotate image by PI/2
void RotateBy1Pi2Factor(const uint8_t* srcData, uint8_t* destData, uint32_t width);

// Rotate image by 3PI/2
void RotateBy3Pi2Factor(const uint8_t* srcData, uint8_t* destData, uint32_t width);

// Rotates part of a row while checking pixels around at each new pixel
void RotateUnsafePartialRow(uint32_t y, const uint8_t* srcData, uint8_t* destData, uint32_t start, uint32_t end, uint32_t width, float srcX, float srcY, const float dxRow, const float dyRow);

// Rotates part of a row but does not check if pixel are in destination image or not, it is assumed
void RotateSafePartialRow(uint32_t y, const uint8_t* srcData, uint8_t* destData, uint32_t start, uint32_t end, uint32_t width, float srcX, float srcY, const float dxRow, const float dyRow);

// Rotates the bitmap, using line equations instead of rotating each pixel
void RotateBmp(const uint8_t* srcData, uint8_t* destData, uint32_t width, float angle);

#endif // ROTATION_H