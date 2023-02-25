#include "../includes/FramePreprocessor.h"

#define DEBUG_SAVE_IMAGE

namespace
{
    constexpr auto MNISTHeight  = 28;
    constexpr auto MNISTWidth   = 28;
}

static cv::Point2f GetImageCenterOfMass(const cv::Mat& image)
{
    std::vector<int> mass_rows(image.rows, 0);
    std::vector<int> mass_cols(image.cols, 0);

    for (size_t row = 0; row < image.rows; row++)
    {
        for (size_t col = 0; col < image.cols; col++)
        {
            mass_cols[col] += image.at<uchar>(row, col);
            mass_rows[row] += image.at<uchar>(row, col);
        }
    }

    for (auto &mass_col : mass_cols) mass_col /= image.cols;
    for (auto &mass_row : mass_rows) mass_row /= image.rows;

    auto lambda = [] (const std::vector<int> &mass_coord) -> float {
        uint64_t mass   = 0;
        uint64_t moment = 0;
        size_t len      = mass_coord.size();

        if (len == 0) return 0;

        for (size_t i = 0; i < len; i++)
        {
            moment  += (mass_coord[i] * i);
            mass    += (mass_coord[i] * 1);
        }
        return moment / float(mass);
    };

    cv::Point2f centerOfMass;
    centerOfMass.x = lambda(mass_cols);
    centerOfMass.y = lambda(mass_rows);

    return centerOfMass;
}

static void ShiftImageByCenterOfMass(cv::Mat& image)
{
    cv::Point2f centerOfMass = GetImageCenterOfMass(image);
    int rows    = image.rows;
    int cols    = image.cols;

    int shiftX  = round((cols / 2.0) - centerOfMass.x);
    int shiftY  = round((rows / 2.0) - centerOfMass.y);

    /// [[1 0 shiftX]
    ///  [0 1 shiftY]]
    cv::Matx23f warpMat(1, 0, shiftX, 0, 1, shiftY);

    cv::Mat shifted;
    cv::warpAffine(image, shifted, warpMat, cv::Size(cols, rows));
    swap(image, shifted);
}

bool MNISTRequirePreprocessing(const cv::Mat& input, cv::Mat& output)
{
    cv::Mat image = input;

    int imageChannels = image.channels();
    if (imageChannels != 1)
    {
        if (imageChannels == 3)
            cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
        else
        {
            std::cout << "[ERROR]\t" << __FUNCTION__ << " Input image has " << imageChannels << " channels. Support only 1 or 3" << std::endl;
            return false;
        }
    }
    cv::threshold(image,image, 255. * 0.2, 255, cv::THRESH_BINARY_INV);
    image = 255 - image;

    cv::Point2f leftUpPointer, rightDownPointer;
    {
        for(size_t row = 0, col_sum = 0; row < image.rows && col_sum == 0; row++)
        {
            leftUpPointer.y = row;
            for(size_t col = 0; col < image.cols && col_sum == 0; col++)
                col_sum += image.at<uchar>(row, col);
        }

        for(size_t row = image.rows, col_sum = 0; row > 0 && col_sum == 0; row--)
        {
            rightDownPointer.y = row - 1;
            for(size_t col = 0; col < image.cols && col_sum == 0; col++)
                col_sum += image.at<uchar>(row - 1, col);
        }

        for(size_t col = 0, row_sum = 0; col < image.cols && row_sum == 0; col++)
        {
            leftUpPointer.x = col;
            for(size_t row = 0; row < image.rows; row++)
                row_sum += image.at<uchar>(row, col);
        }

        for(size_t col = image.cols, row_sum = 0; col > 0 && row_sum == 0; col--)
        {
            rightDownPointer.x = col - 1;
            for(size_t row = image.rows; row > 0; row--)
                row_sum += image.at<uchar>(row - 1, col - 1);
        }

        if (rightDownPointer.x < leftUpPointer.x || rightDownPointer.y < leftUpPointer.y)
        {
            std::cout << "[ERROR]\t" << __FUNCTION__ << " Input image is wrong" << std::endl;
            return false;
        }
    }

    {
        float boundingBoxWidth  = rightDownPointer.x - leftUpPointer.x;
        float boundingBoxHeight = rightDownPointer.y - leftUpPointer.y;

        cv::Mat ROI(image, cv::Rect(leftUpPointer.x, leftUpPointer.y, boundingBoxWidth, boundingBoxHeight));
        cv::Mat croppedImage;
        ROI.copyTo(croppedImage);
        swap(image, croppedImage);
    }

    {
        int rows = image.rows;
        int cols = image.cols;

        bool bNeedBlur = false;
        if (rows > 103 && cols > 103)
            bNeedBlur = true;

        if (rows > cols)
        {
            double factor = 20.0 / rows;
            rows = 20;
            cols = round(cols * factor);
        }
        else
        {
            auto factor = 20.0 / cols;
            cols = 20;
            rows = int(round(rows * factor));
        }

        if (bNeedBlur)
        {
            std::cout << "[INFO]\tUsing Gaussian blur..." << std::endl;
            cv::resize(image, image, cv::Size(image.cols / 3, image.rows / 3), cv::INTER_AREA);
            GaussianBlur(image, image, cv::Size(3, 3), 0);
        }
        cv::resize(image, image, cv::Size(cols, rows), cv::INTER_AREA);
        cv::threshold(image,image, 0.01, 255, cv::THRESH_BINARY);
    }

    {
        auto rows = image.rows;
        auto cols = image.cols;

        int colPadLeft  = ceil((MNISTWidth      - cols) / 2.0);
        int colPadRight = floor((MNISTWidth     - cols) / 2.0);
        int rowPadUp    = ceil((MNISTHeight     - rows) / 2.0);
        int rowPadDown  = floor((MNISTHeight    - rows) / 2.0);

        cv::copyMakeBorder(image, image,
                           rowPadUp, rowPadDown,
                           colPadLeft, colPadRight,
                           cv::BORDER_CONSTANT);
    }

    ShiftImageByCenterOfMass(image);
    swap(output, image);

#ifdef DEBUG_SAVE_IMAGE
    cv::imwrite("prepared.png", output);
#endif

    return true;
}
