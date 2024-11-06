#include "Rotation.h"
#include <math.h>
#include <stdio.h>

/*
* Rotate specified range of pixel on given destination column while checking for pixel validity
*/
void RotateUnsafePartialRow(uint32_t y, const uint8_t* srcData, uint8_t* destData, uint32_t start, uint32_t end, uint32_t width, float srcX, float srcY, const float dxRow, const float dyRow)
{
    // srcX is the beginning of the row, applying dxRow * start gives us X at of the pixel of the row we start at, same for y
    srcX += dxRow * start;
    srcY += dyRow * start;
    for (uint32_t x = start; x < end; ++x)
    {
        // Bilinear interpolation precomputation
        uint32_t floorX = floor(srcX);
        uint32_t ceilX = ceil(srcX);
        uint32_t floorY = floor(srcY);
        uint32_t ceilY = ceil(srcY);

        // Check surrounding pixels
        if (!IsInbound(floorX, floorY, width)
            || !IsInbound(floorX, ceilY, width)
            || !IsInbound(ceilX, floorY, width)
            || !IsInbound(ceilX, ceilY, width))
        {
            destData[x + width * y] = 0;

            // Next index
            srcX += dxRow;
            srcY += dyRow;
            continue;
        }

        // Compute bilinear interpolation
        uint8_t ul = srcData[ceilY * width + floorX];
        uint8_t ur = srcData[ceilY * width + ceilX];
        uint8_t bl = srcData[floorY * width + floorX];
        uint8_t br = srcData[floorY * width + ceilX];

        float ratioX = srcX - floorX;
        float ratioY = srcY - floorY;

        uint8_t botInterpolation = (1 - ratioX) * bl + ratioX * br;
        uint8_t upInterpolation = (1 - ratioX) * ul + ratioX * ur;
        uint8_t biInterpolation = (1 - ratioY) * botInterpolation + ratioY * upInterpolation;

        destData[x + width * y] = biInterpolation;

        // Next index
        srcX += dxRow;
        srcY += dyRow;
    }
}

void RotateSafePartialRow(uint32_t y, const uint8_t* srcData, uint8_t* destData, uint32_t start, uint32_t end, uint32_t width, float srcX, float srcY, const float dxRow, const float dyRow)
{
    // srcX is the beginning of the row, applying dxRow * start gives us X at of the pixel of the row we start at, same for y
    srcX += dxRow * start;
    srcY += dyRow * start;
    //printf("width: %d, end: %d\n");
    for (uint32_t x = start; x < end; ++x)
    {
        // Bilinear interpolation precomputation
        uint32_t floorX = floor(srcX);
        uint32_t ceilX = ceil(srcX);
        uint32_t floorY = floor(srcY);
        uint32_t ceilY = ceil(srcY);

        // Compute bilinear interpolation
        uint8_t ul = srcData[ceilY * width + floorX];
        uint8_t ur = srcData[ceilY * width + ceilX];
        uint8_t bl = srcData[floorY * width + floorX];
        uint8_t br = srcData[floorY * width + ceilX];

        float ratioX = srcX - floorX;
        float ratioY = srcY - floorY;

        uint8_t botInterpolation = (1 - ratioX) * bl + ratioX * br;
        uint8_t upInterpolation = (1 - ratioX) * ul + ratioX * ur;
        uint8_t biInterpolation = (1 - ratioY) * botInterpolation + ratioY * upInterpolation;

        destData[x + width * y] = biInterpolation;

        // Next index
        srcX += dxRow;
        srcY += dyRow;
    }
}

void RotateByPi(const uint8_t* srcData, uint8_t* destData, uint32_t width)
{
    uint32_t nbPixels = width * width;
    for (uint32_t i = 0; i < nbPixels; ++i)
    {
        destData[i] = srcData[nbPixels - 1 - i];
    }
}

void RotateBy1Pi2Factor(const uint8_t* srcData, uint8_t* destData, uint32_t width)
{
    uint32_t x;
    uint32_t y;
    for (uint32_t i = 0; i < width * width; ++i)
    {
        x = i % width;
        y = i / width;
        destData[(width - x - 1) * width + y] = srcData[y * width + x];
    }
}

void RotateBy3Pi2Factor(const uint8_t* srcData, uint8_t* destData, uint32_t width)
{
    uint32_t x;
    uint32_t y;
    for (uint32_t i = 0; i < width * width; ++i)
    {
        x = i % width;
        y = i / width;
        destData[x * width + width - y - 1] = srcData[y * width + x];
    }
}

void RotateBmp(const uint8_t* srcData, uint8_t* destData, uint32_t width, float angle)
{
    // Pixels framing the float point found in source image, used to interpolate destination pixel value
    uint32_t floorX;
    uint32_t floorY;
    uint32_t ceilX;
    uint32_t ceilY;

    // The interpolation factor on X and Y
    float ratioX;
    float ratioY;

    // Interpolation of the two bottom pixel from the hit impact, two top pixels, and interpolation between the 2 previous interpolation
    uint8_t botInterpolation;
    uint8_t upInterpolation;
    uint8_t biInterpolation;

    // Actual pixels framing the floating hit point (upper left, upper right, bottom left, bottom right)
    uint8_t ul;
    uint8_t ur;
    uint8_t bl;
    uint8_t br;

    float center = (float)width / 2;

    float dxCol = (float)sin(angle);
    float dyCol = (float)cos(angle);
    float dxRow = dyCol;
    float dyRow = -dxCol;

    float topLeftX = center - (center * dyCol + center * dxCol);
    float topLeftY = center - (center * dyRow + center * dxRow);

    float rowX = topLeftX;
    float rowY = topLeftY;

    // Destination indices
    uint32_t x = 0, y = 0;

    // Source indices that maps destination indices
    float srcX, srcY;

    uint32_t subImageWidth = (sqrt(2) * width) / 2;
    uint32_t offset = (width - subImageWidth) / 2;
    uint32_t topLeft = offset * width + offset;

    // Rotate top part of the image
    for (y = 0; y < offset; ++y)
    {
        srcX = rowX;
        srcY = rowY;

        for (x = 0; x < width; ++x)
        {
            floorX = floor(srcX);
            ceilX = ceil(srcX);
            floorY = floor(srcY);
            ceilY = ceil(srcY);
            if (!IsInbound(floorX, floorY, width)
                || !IsInbound(floorX, ceilY, width)
                || !IsInbound(ceilX, floorY, width)
                || !IsInbound(ceilX, ceilY, width))
            {
                destData[x + width * y] = 0;
                srcX += dxRow;
                srcY += dyRow;
                continue;
            }

            ul = srcData[ceilY * width + floorX];
            ur = srcData[ceilY * width + ceilX];
            bl = srcData[floorY * width + floorX];
            br = srcData[floorY * width + ceilX];

            ratioX = srcX - floorX;
            ratioY = srcY - floorY;

            botInterpolation = (1 - ratioX) * bl + ratioX * br;
            upInterpolation = (1 - ratioX) * ul + ratioX * ur;
            biInterpolation = (1 - ratioY) * botInterpolation + ratioY * upInterpolation;

            destData[x + width * y] = biInterpolation;

            srcX += dxRow;
            srcY += dyRow;
        }
        rowX += dxCol;
        rowY += dyCol;

    }

    srcX = rowX;
    srcY = rowY;

    // Rotate line linking left side of image and top left pixel of the sub image
    for (x = 0; x < offset; ++x)
    {
        floorX = floor(srcX);
        ceilX = ceil(srcX);
        floorY = floor(srcY);
        ceilY = ceil(srcY);
        if (!IsInbound(floorX, floorY, width)
            || !IsInbound(floorX, ceilY, width)
            || !IsInbound(ceilX, floorY, width)
            || !IsInbound(ceilX, ceilY, width))
        {
            destData[x + width * y] = 0;
            srcX += dxRow;
            srcY += dyRow;
            continue;
        }

        ul = srcData[ceilY * width + floorX];
        ur = srcData[ceilY * width + ceilX];
        bl = srcData[floorY * width + floorX];
        br = srcData[floorY * width + ceilX];

        ratioX = srcX - floorX;
        ratioY = srcY - floorY;

        botInterpolation = (1 - ratioX) * bl + ratioX * br;
        upInterpolation = (1 - ratioX) * ul + ratioX * ur;
        biInterpolation = (1 - ratioY) * botInterpolation + ratioY * upInterpolation;

        destData[x + width * y] = biInterpolation;

        srcX += dxRow;
        srcY += dyRow;

    }

    // Rotate the sub image and its left and right sides
    for (y = y; y < offset + subImageWidth; ++y)
    {
        // Center (no need to check pixels validity)
        for (x = offset; x < offset + subImageWidth; ++x)
        {
            floorX = floor(srcX);
            ceilX = ceil(srcX);
            floorY = floor(srcY);
            ceilY = ceil(srcY);

            ul = srcData[ceilY * width + floorX];
            ur = srcData[ceilY * width + ceilX];
            bl = srcData[floorY * width + floorX];
            br = srcData[floorY * width + ceilX];

            ratioX = srcX - floorX;
            ratioY = srcY - floorY;

            botInterpolation = (1 - ratioX) * bl + ratioX * br;
            upInterpolation = (1 - ratioX) * ul + ratioX * ur;
            biInterpolation = (1 - ratioY) * botInterpolation + ratioY * upInterpolation;

            destData[x + width * y] = biInterpolation;
            srcX += dxRow;
            srcY += dyRow;
        }

        // Right side
        for (x = offset + subImageWidth; x < width; ++x)
        {
            floorX = floor(srcX);
            ceilX = ceil(srcX);
            floorY = floor(srcY);
            ceilY = ceil(srcY);
            if (!IsInbound(floorX, floorY, width)
                || !IsInbound(floorX, ceilY, width)
                || !IsInbound(ceilX, floorY, width)
                || !IsInbound(ceilX, ceilY, width))
            {
                destData[x + width * y] = 0;
                srcX += dxRow;
                srcY += dyRow;
                continue;
            }

            ul = srcData[ceilY * width + floorX];
            ur = srcData[ceilY * width + ceilX];
            bl = srcData[floorY * width + floorX];
            br = srcData[floorY * width + ceilX];

            ratioX = srcX - floorX;
            ratioY = srcY - floorY;

            botInterpolation = (1 - ratioX) * bl + ratioX * br;
            upInterpolation = (1 - ratioX) * ul + ratioX * ur;
            biInterpolation = (1 - ratioY) * botInterpolation + ratioY * upInterpolation;

            destData[x + width * y] = biInterpolation;

            srcX += dxRow;
            srcY += dyRow;
        }
        rowX += dxCol;
        rowY += dyCol;

        srcX = rowX;
        srcY = rowY;

        // Left side
        for (x = 0; x < offset; ++x)
        {
            floorX = floor(srcX);
            ceilX = ceil(srcX);
            floorY = floor(srcY);
            ceilY = ceil(srcY);
            if (!IsInbound(floorX, floorY, width)
                || !IsInbound(floorX, ceilY, width)
                || !IsInbound(ceilX, floorY, width)
                || !IsInbound(ceilX, ceilY, width))
            {
                destData[x + width * y] = 0;
                srcX += dxRow;
                srcY += dyRow;
                continue;
            }

            ul = srcData[ceilY * width + floorX];
            ur = srcData[ceilY * width + ceilX];
            bl = srcData[floorY * width + floorX];
            br = srcData[floorY * width + ceilX];

            ratioX = srcX - floorX;
            ratioY = srcY - floorY;

            botInterpolation = (1 - ratioX) * bl + ratioX * br;
            upInterpolation = (1 - ratioX) * ul + ratioX * ur;
            biInterpolation = (1 - ratioY) * botInterpolation + ratioY * upInterpolation;

            destData[x + width * y] = biInterpolation;
            srcX += dxRow;
            srcY += dyRow;
        }

    }

    // Rotate bottom part of the image
    for (y = y; y < width; y++)
    {
        srcX = rowX;
        srcY = rowY;

        for (x = 0; x < width; ++x)
        {
            floorX = floor(srcX);
            ceilX = ceil(srcX);
            floorY = floor(srcY);
            ceilY = ceil(srcY);
            if (!IsInbound(floorX, floorY, width)
                || !IsInbound(floorX, ceilY, width)
                || !IsInbound(ceilX, floorY, width)
                || !IsInbound(ceilX, ceilY, width))
            {
                destData[x + width * y] = 0;
                srcX += dxRow;
                srcY += dyRow;
                continue;
            }

            ul = srcData[ceilY * width + floorX];
            ur = srcData[ceilY * width + ceilX];
            bl = srcData[floorY * width + floorX];
            br = srcData[floorY * width + ceilX];

            ratioX = srcX - floorX;
            ratioY = srcY - floorY;

            botInterpolation = (1 - ratioX) * bl + ratioX * br;
            upInterpolation = (1 - ratioX) * ul + ratioX * ur;
            biInterpolation = (1 - ratioY) * botInterpolation + ratioY * upInterpolation;

            destData[x + width * y] = biInterpolation;

            srcX += dxRow;
            srcY += dyRow;
        }
        rowX += dxCol;
        rowY += dyCol;
    }
}