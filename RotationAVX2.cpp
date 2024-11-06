#include "RotationAVX2.h"
#include "Rotation.h"
#include <math.h>
#include <stdio.h>
#include <immintrin.h>

/*
* In all functions using vectorization to rotate the image, the approach is to divide the height by the number of values the register can hold, and compute these lines in parallel using the registers
* I feel like there must be a better way to use vectorization though and I will probably keep searching now that I am a bit more confortable with SIMD
*/

void RotateBy1Pi2FactorAVX2(const uint8_t* srcData, uint8_t* destData, uint32_t width)
{
    __m256i widths = _mm256_set1_epi32(width);
    __m256i ones = _mm256_set1_epi32(1);
    __m256i temp;
    __m256i dstX;
    uint32_t x, y;
    for (y = 0; y < width; y += 8)
    {
        dstX = _mm256_set1_epi32(0);
        for (x = 0; x < width; x++)
        {
            temp = _mm256_mullo_epi32(dstX, widths);
            for (uint32_t i = 0; i < 8; ++i)
            {
                destData[(y + i) * width + x] = srcData[(y + i) + temp.m256i_i32[i]];
            }
            dstX = _mm256_add_epi32(dstX, ones);
        }
    }
}

void RotateBy3Pi2FactorAVX2(const uint8_t* srcData, uint8_t* destData, uint32_t width)
{
    __m256i widths = _mm256_set1_epi32(width);
    __m256i ones = _mm256_set1_epi32(1);
    __m256i widths1 = _mm256_set1_epi32(width - 1);
    __m256i temp;
    __m256i dstX;
    uint32_t x, y;
    for (y = 0; y < width; y += 8)
    {
        dstX = _mm256_set1_epi32(0);
        for (x = 0; x < width; x++)
        {
            temp = _mm256_mullo_epi32(_mm256_sub_epi32(widths1, dstX), widths);
            for (uint32_t i = 0; i < 8; ++i)
            {
                destData[(y + i) * width + x] = srcData[(y + i) + temp.m256i_i32[i]];
            }
            dstX = _mm256_add_epi32(dstX, ones);
        }
    }
}

inline
__m256 IsInboundsAVX2(const __m256 floorXf, const __m256 ceilXf, const __m256 floorYf, const __m256 ceilYf, const __m256 widthsf, const __m256 zeros)
{
    __m256 isInboundsFloorX = _mm256_and_ps(_mm256_cmp_ps(floorXf, zeros, _CMP_GE_OS), _mm256_cmp_ps(floorXf, widthsf, _CMP_NGE_US));
    __m256 isInboundsCeilX = _mm256_and_ps(_mm256_cmp_ps(ceilXf, zeros, _CMP_GE_OS), _mm256_cmp_ps(ceilXf, widthsf, _CMP_NGE_US));
    __m256 isInboundsFloorY = _mm256_and_ps(_mm256_cmp_ps(floorYf, zeros, _CMP_GE_OS), _mm256_cmp_ps(floorYf, widthsf, _CMP_NGE_US));
    __m256 isInboundsCeilY = _mm256_and_ps(_mm256_cmp_ps(ceilYf, zeros, _CMP_GE_OS), _mm256_cmp_ps(ceilYf, widthsf, _CMP_NGE_US));

    return _mm256_and_ps(_mm256_and_ps(_mm256_and_ps(isInboundsFloorX, isInboundsFloorY), _mm256_and_ps(isInboundsFloorX, isInboundsCeilY)),
        _mm256_and_ps(_mm256_and_ps(isInboundsCeilX, isInboundsFloorY), _mm256_and_ps(isInboundsCeilX, isInboundsCeilY)));
}

inline
void InitRotateAVX2(__m256& srcXsf, __m256& srcYsf, float rowX, float rowY, float dxCol, float dyCol)
{
    float srcXsf2[8];
    float srcYsf2[8];
    for (uint32_t i = 0; i < 8; i++)
    {
        srcXsf2[i] = rowX + i * dxCol;
        srcYsf2[i] = rowY + i * dyCol;
    }
    srcXsf = _mm256_load_ps(srcXsf2);
    srcYsf = _mm256_load_ps(srcYsf2);
}

/*
* Rotates a block of rows while checking for valid pixels
*/
void RotateTopBottomAVX2(const uint8_t* srcData, uint8_t* destData, uint32_t& y, uint32_t end, uint32_t width, __m256 widthsf, __m256 zeros, __m256i widths, __m256 dxRowS, __m256 dyRowS, float rowX, float rowY, float dxCol, float dyCol)
{
    __m256 srcXsf, srcYsf;
    __m256 floorXf, ceilXf, floorYf, ceilYf, ratioX, ratioY, isInbounds;
    __m256i floorX, ceilX, floorY, ceilY, ulIndices, urIndices, blIndices, brIndices;
    uint8_t ul, ur, br, bl, botInterpolation, upInterpolation, biInterpolation;
    for (y = y; y < end; y += 8, rowX += dxCol * 8, rowY += dyCol * 8)
    {
        InitRotateAVX2(srcXsf, srcYsf, rowX, rowY, dxCol, dyCol);
        for (uint32_t x = 0; x < width; x++)
        {
            // Precomputation of bilinear interpolation. Note that we compute these information even though we might not need them in the end. 
            // If only one value of our register is valid, the computation for the three others will be useless for instance
            floorXf = _mm256_floor_ps(srcXsf);
            ceilXf = _mm256_ceil_ps(srcXsf);
            floorYf = _mm256_floor_ps(srcYsf);
            ceilYf = _mm256_ceil_ps(srcYsf);
            floorX = _mm256_cvtps_epi32(floorXf);
            ceilX = _mm256_cvtps_epi32(ceilXf);
            floorY = _mm256_cvtps_epi32(floorYf);
            ceilY = _mm256_cvtps_epi32(ceilYf);

            // Boolean to verify if index is valid
            isInbounds = IsInboundsAVX2(floorXf, ceilXf, floorYf, ceilYf, widthsf, zeros);

            // Indicess of each surrounding pixels (that might be invalids)
            ulIndices = _mm256_add_epi32(_mm256_mullo_epi32(ceilY, widths), floorX);
            urIndices = _mm256_add_epi32(_mm256_mullo_epi32(ceilY, widths), ceilX);
            blIndices = _mm256_add_epi32(_mm256_mullo_epi32(floorY, widths), floorX);
            brIndices = _mm256_add_epi32(_mm256_mullo_epi32(floorY, widths), ceilX);

            // Bilinear interpolation, if surrounding pixels are valid
            ratioX = _mm256_sub_ps(srcXsf, floorXf);
            ratioY = _mm256_sub_ps(srcYsf, floorYf);
            for (uint32_t i = 0; i < 8; ++i)
            {
                if (isInbounds.m256_f32[i])
                {
                    ul = srcData[ulIndices.m256i_u32[i]];
                    ur = srcData[urIndices.m256i_u32[i]];
                    bl = srcData[blIndices.m256i_u32[i]];
                    br = srcData[blIndices.m256i_u32[i]];

                    botInterpolation = (1 - ratioX.m256_f32[i]) * bl + ratioX.m256_f32[i] * br;
                    upInterpolation = (1 - ratioX.m256_f32[i]) * ul + ratioX.m256_f32[i] * ur;
                    biInterpolation = (1 - ratioY.m256_f32[i]) * botInterpolation + ratioY.m256_f32[i] * upInterpolation;
                    destData[(y + i) * width + x] = biInterpolation;
                }
                else
                {
                    destData[(y + i) * width + x] = 0;
                }
            }

            // Compute the next index of the current rotated row
            srcXsf = _mm256_add_ps(srcXsf, dxRowS);
            srcYsf = _mm256_add_ps(srcYsf, dyRowS);
        }
    }
}

/*
* This function rotates the "center" of the image. This consists of the square of pixels that will lie in the destination image no matter the rotation, as well as the sides of this sub image
*/
void RotateCenterAVX2(const uint8_t* srcData, uint8_t* destData, uint32_t width, uint32_t subImageWidth, __m256 widthsf, __m256 zeros, __m256i widths, __m256 dxRowS, __m256 dyRowS,
    uint32_t offset, float rowX, float rowY, float dxCol, float dyCol)
{
    __m256 floorXf, ceilXf, floorYf, ceilYf, ratioX, ratioY, isInbounds;
    __m256i floorX, ceilX, floorY, ceilY, ulIndices, urIndices, blIndices, brIndices;
    uint8_t ul, ur, br, bl, botInterpolation, upInterpolation, biInterpolation;

    __m256 srcXsf;
    __m256 srcYsf;
    /*
    * For each y, we will draw the line in the center that doesn't need to be checked, the remaining (the right side) that needs to be checked, and the left side of the next line that needs to be checked
    */
    uint32_t end = subImageWidth + offset;
    uint32_t y = 0;
    for (y = offset; y < end - 8; y += 8, rowX += dxCol * 8, rowY += dyCol * 8)
    {
        // Center line
        float currentX = rowX;
        float currentY = rowY;
        currentX += dxRowS.m256_f32[0] * offset;
        currentY += dyRowS.m256_f32[0] * offset;
        InitRotateAVX2(srcXsf, srcYsf, currentX, currentY, dxCol, dyCol);
        for (uint32_t x = offset; x < offset + subImageWidth; x++)
        {
            // Precomputation of bilinear interpolation
            floorXf = _mm256_floor_ps(srcXsf);
            ceilXf = _mm256_ceil_ps(srcXsf);
            floorYf = _mm256_floor_ps(srcYsf);
            ceilYf = _mm256_ceil_ps(srcYsf);
            floorX = _mm256_cvtps_epi32(floorXf);
            ceilX = _mm256_cvtps_epi32(ceilXf);
            floorY = _mm256_cvtps_epi32(floorYf);
            ceilY = _mm256_cvtps_epi32(ceilYf);

            // Indicess of each surrounding pixels (that might be invalids)
            ulIndices = _mm256_add_epi32(_mm256_mullo_epi32(ceilY, widths), floorX);
            urIndices = _mm256_add_epi32(_mm256_mullo_epi32(ceilY, widths), ceilX);
            blIndices = _mm256_add_epi32(_mm256_mullo_epi32(floorY, widths), floorX);
            brIndices = _mm256_add_epi32(_mm256_mullo_epi32(floorY, widths), ceilX);

            ratioX = _mm256_sub_ps(srcXsf, floorXf);
            ratioY = _mm256_sub_ps(srcYsf, floorYf);

            // Bilinear interpolation, no need to check that surrounding pixels are valid
            for (uint32_t i = 0; i < 8; i++)
            {
                ul = srcData[ulIndices.m256i_u32[i]];
                ur = srcData[urIndices.m256i_u32[i]];
                bl = srcData[blIndices.m256i_u32[i]];
                br = srcData[blIndices.m256i_u32[i]];
                botInterpolation = (1 - ratioX.m256_f32[i]) * bl + ratioX.m256_f32[i] * br;
                upInterpolation = (1 - ratioX.m256_f32[i]) * ul + ratioX.m256_f32[i] * ur;
                biInterpolation = (1 - ratioY.m256_f32[i]) * botInterpolation + ratioY.m256_f32[i] * upInterpolation;
                destData[(y + i) * width + x] = biInterpolation;
            }

            // Compute the next index of the current rotated row
            srcXsf = _mm256_add_ps(srcXsf, dxRowS);
            srcYsf = _mm256_add_ps(srcYsf, dyRowS);
        }

        // Line that follows the last pixel of the center that could be rotated without branching
        uint32_t start = width - offset;
        currentX = rowX + dxRowS.m256_f32[0] * start;
        currentY = rowY + dyRowS.m256_f32[0] * start;
        InitRotateAVX2(srcXsf, srcYsf, currentX, currentY, dxCol, dyCol);
        for (uint32_t x = start; x < width; x++)
        {
            // Precomputation of bilinear interpolation
            floorXf = _mm256_floor_ps(srcXsf);
            ceilXf = _mm256_ceil_ps(srcXsf);
            floorYf = _mm256_floor_ps(srcYsf);
            ceilYf = _mm256_ceil_ps(srcYsf);
            floorX = _mm256_cvtps_epi32(floorXf);
            ceilX = _mm256_cvtps_epi32(ceilXf);
            floorY = _mm256_cvtps_epi32(floorYf);
            ceilY = _mm256_cvtps_epi32(ceilYf);

            // Boolean to verify if index is valid
            __m256 isInbounds = IsInboundsAVX2(floorXf, ceilXf, floorYf, ceilYf, widthsf, zeros);


            // Indicess of each surrounding pixels (that might be invalids)
            ulIndices = _mm256_add_epi32(_mm256_mullo_epi32(ceilY, widths), floorX);
            urIndices = _mm256_add_epi32(_mm256_mullo_epi32(ceilY, widths), ceilX);
            blIndices = _mm256_add_epi32(_mm256_mullo_epi32(floorY, widths), floorX);
            brIndices = _mm256_add_epi32(_mm256_mullo_epi32(floorY, widths), ceilX);

            ratioX = _mm256_sub_ps(srcXsf, floorXf);
            ratioY = _mm256_sub_ps(srcYsf, floorYf);

            // Bilinear interpolation, if surrounding pixels are valid
            for (uint32_t i = 0; i < 8; ++i)
            {
                if (isInbounds.m256_f32[i])
                {
                    ul = srcData[ulIndices.m256i_u32[i]];
                    ur = srcData[urIndices.m256i_u32[i]];
                    bl = srcData[blIndices.m256i_u32[i]];
                    br = srcData[blIndices.m256i_u32[i]];

                    botInterpolation = (1 - ratioX.m256_f32[i]) * bl + ratioX.m256_f32[i] * br;
                    upInterpolation = (1 - ratioX.m256_f32[i]) * ul + ratioX.m256_f32[i] * ur;
                    biInterpolation = (1 - ratioY.m256_f32[i]) * botInterpolation + ratioY.m256_f32[i] * upInterpolation;
                    destData[(y + i) * width + x] = biInterpolation;
                }
                else
                {
                    destData[(y + i) * width + x] = 0;
                }
            }

            // Compute the next index of the current rotated row
            srcXsf = _mm256_add_ps(srcXsf, dxRowS);
            srcYsf = _mm256_add_ps(srcYsf, dyRowS);
        }

        // New line, we fill the beginning of the new line until we reach the first pixel that doesn't need branching
        uint32_t ycpy = y + 1;
        currentX = rowX + dxCol;
        currentY = rowY + dyCol;
        InitRotateAVX2(srcXsf, srcYsf, currentX, currentY, dxCol, dyCol);
        for (uint32_t x = 0; x < offset; x++)
        {
            // Precomputation of bilinear interpolation
            floorXf = _mm256_floor_ps(srcXsf);
            ceilXf = _mm256_ceil_ps(srcXsf);
            floorYf = _mm256_floor_ps(srcYsf);
            ceilYf = _mm256_ceil_ps(srcYsf);
            floorX = _mm256_cvtps_epi32(floorXf);
            ceilX = _mm256_cvtps_epi32(ceilXf);
            floorY = _mm256_cvtps_epi32(floorYf);
            ceilY = _mm256_cvtps_epi32(ceilYf);

            // Boolean to verify if index is valid
            __m256 isInbounds = IsInboundsAVX2(floorXf, ceilXf, floorYf, ceilYf, widthsf, zeros);

            // Indicess of each surrounding pixels (that might be invalids)
            ulIndices = _mm256_add_epi32(_mm256_mullo_epi32(ceilY, widths), floorX);
            urIndices = _mm256_add_epi32(_mm256_mullo_epi32(ceilY, widths), ceilX);
            blIndices = _mm256_add_epi32(_mm256_mullo_epi32(floorY, widths), floorX);
            brIndices = _mm256_add_epi32(_mm256_mullo_epi32(floorY, widths), ceilX);

            ratioX = _mm256_sub_ps(srcXsf, floorXf);
            ratioY = _mm256_sub_ps(srcYsf, floorYf);

            // Bilinear interpolation, if surrounding pixels are valid
            for (uint32_t i = 0; i < 8; ++i)
            {
                if (isInbounds.m256_f32[i])
                {
                    ul = srcData[ulIndices.m256i_u32[i]];
                    ur = srcData[urIndices.m256i_u32[i]];
                    bl = srcData[blIndices.m256i_u32[i]];
                    br = srcData[blIndices.m256i_u32[i]];

                    botInterpolation = (1 - ratioX.m256_f32[i]) * bl + ratioX.m256_f32[i] * br;
                    upInterpolation = (1 - ratioX.m256_f32[i]) * ul + ratioX.m256_f32[i] * ur;
                    biInterpolation = (1 - ratioY.m256_f32[i]) * botInterpolation + ratioY.m256_f32[i] * upInterpolation;
                    destData[(ycpy + i) * width + x] = biInterpolation;
                }
                else
                {
                    destData[(ycpy + i) * width + x] = 0;
                }
            }

            // Compute the next index of the current rotated row
            srcXsf = _mm256_add_ps(srcXsf, dxRowS);
            srcYsf = _mm256_add_ps(srcYsf, dyRowS);
        }
    }

    // There might be a remainder if the number of values in register is not a divisor of the subimage size
    if (y == end)
    {
        return;
    }

    // Revert the addition that stopped the loop
    rowX = rowX - (8 * dxCol);
    rowY = rowY - (8 * dyCol);
    for (uint32_t newy = y - 8; newy < end; newy++, rowX += dxCol, rowY += dyCol)
    {
        RotateSafePartialRow(newy, srcData, destData, offset, width - offset, width, rowX, rowY, dxRowS.m256_f32[0], dyRowS.m256_f32[0]);
        RotateUnsafePartialRow(newy, srcData, destData, width - offset, width, width, rowX, rowY, dxRowS.m256_f32[0], dyRowS.m256_f32[0]);
        RotateUnsafePartialRow(newy + 1, srcData, destData, 0, offset, width, rowX + dxCol, rowY + dyCol, dxRowS.m256_f32[0], dyRowS.m256_f32[0]);
    }
}

/*
 The function is quite straightforward in itself but is overly complicated due to the absence of refactoring. There is a lot of duplicate code that I struggle to factorise further as
 it tends to create non inline functions, which are costly specially if called inside loops.
 So, for now, I am leaving it in that state but I will keep trying to find the right balance between factoring and function calls costs.
 
 Regarding the algorithm, instead of rotating each point, the algorithm rotates the top left points, and deduces the other points with the line equation.
 On top of that, the algorithm handles the center of the image differently than the rest since it contains a subimage for which a lot of computation can be avoided.
*/
void RotateBmpAVX2(const uint8_t* srcData, uint8_t* destData, uint32_t width, float angle)
{
    // two pixels of a rotated row will be seperated by sin(theta) in X, and cos(theta) in Y
    float dxCol = (float)sin(angle);
    float dyCol = (float)cos(angle);

    // Similarly, two pixels of a rotated column will be seperated by cos(theta) in X, and -sin(theta) in Y
    float dxRow = dyCol;
    float dyRow = -dxCol;
    float center = (float)width / 2;

    // Rotation of the top left pixel of the image. The rest will be computed by incrementing the x/y steps for each row and columns of the image
    float topLeftX = center - (center * dyCol + center * dxCol);
    float topLeftY = center - (center * dyRow + center * dxRow);

    // Index of the first pixel of the current row rotated in the source image (as a remainder, we are "reverse" rotating the destination image to the source image in order to apply filtering,
    // therefore, rowX and rowY are indices of the source image
    float rowX = topLeftX;
    float rowY = topLeftY;

    // Store necessary data in AVX2 registers for further computations
    __m256 dxRowS = _mm256_set1_ps(dxRow);
    __m256 dyRowS = _mm256_set1_ps(dyRow);
    __m256i widths = _mm256_set1_epi32(width);
    __m256 widthsf = _mm256_set1_ps(width);
    __m256 zeros = _mm256_set1_ps(0);

    // Computes the sub image for which every pixel will lie in the destination image no matter the rotation
    uint32_t subImageWidth = (sqrt(2) * width) / 2;
    uint32_t offset = (width - subImageWidth) / 2;
    subImageWidth--;
    offset++;
    uint32_t y = 0;
    // Rotate the top part of the image with 256 bits registers (8 lines at a time)
    RotateTopBottomAVX2(srcData, destData, y, offset, width, widthsf, zeros, widths, dxRowS, dyRowS, rowX, rowY, dxCol, dyCol);

    // Rotate the single line that connects the left border of the image to the top left pixel ofthe subimage that will lie in the destination image no matter the rotation
    // It is important to note that this is not vectorized, therefore the number of lines to be computed will be 2^n - 1 instead of 2^n, which we won't be able to fully vectorize
    y = offset;
    rowX += dxCol * offset;
    rowY += dyCol * offset;
    RotateUnsafePartialRow(offset, srcData, destData, 0, offset, width, rowX, rowY, dxRow, dyRow);

    // Rotate the center of the image, which is divided in 3 parts discussed in the function
    RotateCenterAVX2(srcData, destData, width, subImageWidth, widthsf, zeros, widths, dxRowS, dyRowS, offset, rowX, rowY, dxCol, dyCol);
    rowX += dxCol * subImageWidth;
    rowY += dyCol * subImageWidth;
    y = offset + subImageWidth;
    // RotateCenterAVX2's last iteration didn't finish to draw a line, so we finish this line.
    RotateUnsafePartialRow(y, srcData, destData, offset, width, width, rowX, rowY, dxRow, dyRow);

    // Now we only need to rotate the bottom part of the image while accounting the fact that we cannot be sure to compute it until the end with vectorization
    rowX += dxCol;
    rowY += dyCol;
    y++;
    RotateTopBottomAVX2(srcData, destData, y, width - 8, width, widthsf, zeros, widths, dxRowS, dyRowS, rowX, rowY, dxCol, dyCol);

    // Handle remainder
    if (y == width)
    {
        return;
    }
    uint32_t newY = y - 8;
    rowX = topLeftX + dxCol * newY;
    rowY = topLeftY + dyCol * newY;

    //Compute the remainding lines without vectorization
    for (newY = newY; newY < width; newY++, rowX += dxCol, rowY += dyCol)
    {
        RotateUnsafePartialRow(newY, srcData, destData, 0, width, width, rowX, rowY, dxRow, dyRow);
    }
}

void RotateBmpAVX2v2(const uint8_t* srcData, uint8_t* destData, uint32_t width, float angle)
{
    // two pixels of a rotated row will be seperated by sin(theta) in X, and cos(theta) in Y
    float dxCol = (float)sin(angle);
    float dyCol = (float)cos(angle);

    // Similarly, two pixels of a rotated column will be seperated by cos(theta) in X, and -sin(theta) in Y
    float dxRow = dyCol;
    float dyRow = -dxCol;
    float center = (float)width / 2;

    // Rotation of the top left pixel of the image. The rest will be computed by incrementing the x/y steps for each row and columns of the image
    float topLeftX = center - (center * dyCol + center * dxCol);
    float topLeftY = center - (center * dyRow + center * dxRow);

    // Store necessary data in AVX2 registers for further computations
    __m256 dxRowS = _mm256_set1_ps(dxRow);
    __m256 dyRowS = _mm256_set1_ps(dyRow);
    __m256i widths = _mm256_set1_epi32(width);
    __m256 widthsf = _mm256_set1_ps(width);
    __m256 zeros = _mm256_set1_ps(0);

    uint32_t y = 0;
    // Rotate the entire image in one go
    RotateTopBottomAVX2(srcData, destData, y, width, width, widthsf, zeros, widths, dxRowS, dyRowS, topLeftX, topLeftY, dxCol, dyCol);
}