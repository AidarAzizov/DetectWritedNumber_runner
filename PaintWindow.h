#ifndef OPENCV_TESTS_PAINTWINDOW_H
#define OPENCV_TESTS_PAINTWINDOW_H

#include <iostream>

/****  OpenCV includes  ****/
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
/****  OpenCV includes  ****/

#define SUCCESS_PAINTED 0
#define ERROR_PAINTED 1

enum ePaintedImageAction : uint8_t {
    eNotSave = 0,
    eSave,
    eCount
};

class CFrameAnalyzer;

class CPaintWindow {

private:
    static const cv::Mat backgroundImage;

    static void mouseCallback(int event, int x, int y, int flags, void* param) noexcept;
    void mouseHandler(int event, int x, int y) noexcept;
public:

    CPaintWindow(const std::string &sScriptedModelPath, const ePaintedImageAction eAction = eNotSave);
    ~CPaintWindow();

    int Run() noexcept;

private:
    std::unique_ptr<CFrameAnalyzer> m_analyzer;
    const ePaintedImageAction m_eImageAction;

    bool    m_bDrawMode;
    cv::Mat m_showingImage;
};


#endif //OPENCV_TESTS_PAINTWINDOW_H
