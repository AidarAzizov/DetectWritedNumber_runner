#ifndef OPENCV_TESTS_FRAMEANALYZER_H
#define OPENCV_TESTS_FRAMEANALYZER_H

#include <iostream>

/****  OpenCV includes  ****/
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
/****  OpenCV includes  ****/


// torch libs will be invisible for include this module
class ShellCNN;

class CFrameAnalyzer {

public:
    CFrameAnalyzer(const std::string& sPath);
    ~CFrameAnalyzer();

    /// For number classification cvImage must have [28,28] size
    /// like MNISTNet images
    std::vector<int64_t> Analyze(const std::vector<cv::Mat>& cvImages) const noexcept;

private:
    std::unique_ptr<ShellCNN> m_pModel;
};


#endif //OPENCV_TESTS_FRAMEANALYZER_H
