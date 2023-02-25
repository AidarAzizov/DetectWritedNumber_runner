#include "FrameAnalyzer.h"

/****  Torch includes  ****/
#include <torch/torch.h>
#pragma warning( push )
#pragma warning( disable : 4146 )
#include <torch/script.h>
#pragma warning( pop )
/****  Torch includes  ****/

namespace
{
    torch::NoGradGuard noGrad;

    const auto deviceType = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
}

///////////////////////////
///**** Model shell ****///
///////////////////////////
class ShellCNN : public torch::jit::Module
{
public:
    ShellCNN(Module module) : torch::jit::Module(module) {}

    ShellCNN()          = default;
    virtual ~ShellCNN() = default;

    ShellCNN& operator=(const ShellCNN &other)  = delete;
    ShellCNN& operator=(ShellCNN &&other)       = delete;
    ShellCNN(const ShellCNN &other)             = delete;
    ShellCNN(ShellCNN &&other)                  = delete;
};



//////////////////////////////////////
///**** Constructor/destructor ****///
//////////////////////////////////////
CFrameAnalyzer::CFrameAnalyzer(const std::string& sPath) : m_pModel()
{
    m_pModel = std::make_unique<ShellCNN>(torch::jit::load(sPath, deviceType));
}

CFrameAnalyzer::~CFrameAnalyzer() = default;



///////////////////////
///**** Methods ****///
///////////////////////
std::vector<int64_t> CFrameAnalyzer::Analyze(const std::vector<cv::Mat>& cvImages) const noexcept
{
    std::vector<int64_t> predicted;

    try
    {
        torch::Tensor inputs;
        for (auto &cvImage : cvImages)
        {
            torch::Tensor tensor_image = torch::from_blob(cvImage.data, { 1, 1, cvImage.rows, cvImage.cols },torch::kByte).to(deviceType);

            inputs = torch::cat({ tensor_image, inputs });
        }
        inputs = inputs.toType(torch::kFloat);

        auto outputs = m_pModel->forward({ inputs }).toTensor();
        std::cout << "[INFO]\tPredicted - " << outputs << std::endl;

        outputs = outputs.argmax(1).toType(torch::ScalarType::Long);

        int64_t *outputData = static_cast<int64_t*>(outputs.data_ptr());
        int64_t outputCount = outputs.size(0);
        size_t imageCount   = cvImages.size();

        assert(outputCount == imageCount);

        predicted.resize(outputCount);
        for (size_t i = 0; i < outputCount; i++)
            predicted[i] = outputData[i];
    }
    catch (c10::Error &except)
    {
        std::cout << "[ERROR]\t" << __FUNCTION__ << " c10::Error: " << except.what() << std::endl;
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

    return predicted;
}

