#include "../includes/PaintWindow.h"
#include "../includes/FrameAnalyzer.h"
#include "../includes/FramePreprocessor.h"

#include <filesystem>

namespace
{
    const std::string sWindowName           = "Detect Writed Number";
    const std::string sSavedImageFolder     = "predicted";
    const std::string sSavedImageExtension  = ".png";

    constexpr auto kWindowRows  = 512;
    constexpr auto kWindowCols  = 512;
    constexpr auto kWindowType  = CV_8UC1;

    constexpr auto kWindowPaintPixelFat = (kWindowRows >= kWindowCols ? kWindowCols : kWindowRows) / 64;

    const auto kWindowFillPixel     = cv::Scalar(0);
    const auto kWindowPaintPixel    = cv::Scalar(255);

    static_assert(kWindowPaintPixelFat > 0);

    std::string GetStringCurrTime()
    {
        auto t = std::time(nullptr);
        auto tm = *std::localtime(&t);

        std::ostringstream oss;
        oss << std::put_time(&tm, "%Y-%m-%d_%H:%M:%S");
        return oss.str();
    }
}


/////////////////////////////////////
///**** Static methods/fields ****///
/////////////////////////////////////
const cv::Mat CPaintWindow::backgroundImage = cv::Mat(kWindowRows,
                                                      kWindowCols,
                                                      kWindowType,
                                                      kWindowFillPixel);

void CPaintWindow::mouseCallback(int event, int x, int y, int flags, void* param) noexcept
{
    CPaintWindow *pPaintWindow = reinterpret_cast<CPaintWindow*>(param);
    if (pPaintWindow)
        pPaintWindow->mouseHandler(event, x, y);
}



//////////////////////////////////////
///**** Constructor/destructor ****///
//////////////////////////////////////
CPaintWindow::CPaintWindow(const std::string &sScriptedModelPath, const ePaintedImageAction eAction)
    : m_showingImage(backgroundImage.clone())
    , m_bDrawMode(false)
    , m_analyzer()
    , m_eImageAction(eAction)
{
    m_analyzer = std::make_unique<CFrameAnalyzer>(sScriptedModelPath);
}

CPaintWindow::~CPaintWindow() = default;



///////////////////////
///**** Methods ****///
///////////////////////
void CPaintWindow::mouseHandler(int event, int x, int y) noexcept
{
    try
    {
        if (event == cv::EVENT_LBUTTONDOWN)
            m_bDrawMode = true;
        else if(event == cv::EVENT_LBUTTONUP)
            m_bDrawMode = false;
        else if(event == cv::EVENT_RBUTTONDOWN)
        {
            std::cout << "[INFO]\tStart Analyze process..." << std::endl;

            cv::Mat prepared_image;
            std::string sFilename = "not_preprocessed";
            if (MNISTRequirePreprocessing(m_showingImage, prepared_image))
            {
                std::vector<int64_t> predicted = m_analyzer->Analyze({prepared_image});

                if (!predicted.empty())
                {
                    const int predictedNumber = predicted[0];
                    sFilename = std::to_string(predictedNumber);

                    std::cout << "[INFO]\tUser painted '" << predictedNumber << "'\n";
                }
                else
                {
                    sFilename = "not_predicted";
                }
            }

            switch (m_eImageAction)
            {
            case eSave:
            {
                std::filesystem::create_directories(sSavedImageFolder);

                const std::string sCurrTime     = GetStringCurrTime();
                const std::string sPathToSave   = sSavedImageFolder + "/" + sFilename + "___" + sCurrTime + sSavedImageExtension;
                cv::imwrite(sPathToSave, m_showingImage);
            } break;

            default:
            {
                /// TODO:
            } break;
            }

            m_showingImage = backgroundImage.clone();
            std::cout << "[INFO]\t...End Analyze process\n" << std::endl;
        }

        if (m_bDrawMode)
            cv::circle(m_showingImage, {x, y}, 9, kWindowPaintPixel, cv::FILLED);
    }
    catch (cv::Exception &except)
    {
        std::cout << "[ERROR]\t" << __FUNCTION__ << " cv::Exception: " << except.what() << std::endl;
    }
    catch (std::exception &except)
    {
        std::cout << "[ERROR]\t" << __FUNCTION__ << " std::exception: " << except.what() << std::endl;
    }
    catch (...)
    {
        std::cout << "[ERROR]\t" << __FUNCTION__ << " Unknown exception" << std::endl;
    }
}

int CPaintWindow::Run() noexcept
{
    int exitCode = ERROR_PAINTED;

    try
    {
        m_showingImage = backgroundImage.clone();
        cv::namedWindow(sWindowName);
        cv::setMouseCallback(sWindowName, CPaintWindow::mouseCallback, this);

        int key = 0;
        while (key != 'q')
        {
            cv::imshow(sWindowName, m_showingImage);
            key = cv::waitKey(1) & 0xFF;
        }

        cv::destroyWindow(sWindowName);
        exitCode = SUCCESS_PAINTED;
    }
    catch(cv::Exception &except)
    {
        std::cout << "[ERROR]\t" << __FUNCTION__ << " cv::Exception: " << except.what() << std::endl;
    }
    catch(std::exception &except)
    {
        std::cout << "[ERROR]\t" << __FUNCTION__ << " std::exception: " << except.what() << std::endl;
    }
    catch(...)
    {
        std::cout << "[ERROR]\t" << __FUNCTION__ << " Unknown exception" << std::endl;
    }

    return exitCode;
}
