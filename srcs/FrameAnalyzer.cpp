#include "../includes/FrameAnalyzer.h"

#include "../includes/FramePreprocessor.h"


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

    constexpr auto fTrustedThreshold = 10.8028f;
    const cv::String sFolderWithNumbers    = "../test/numbers";
    const cv::String sFolderWithNotNumbers = "../test/not numbers";

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
void CFrameAnalyzer::Test() const noexcept
{
    auto testImpl = [&](const std::string sFolder)
    {
        std::vector<cv::String> sFilenames;
        cv::glob(sFolder, sFilenames);

        for (auto &filename: sFilenames)
        {
            cv::Mat number = cv::imread(filename);

            cv::Mat preparedImage;
            if (MNISTRequirePreprocessing(number, preparedImage))
            {
                std::cout << filename << std::endl;
                Analyze({ preparedImage });
            }
        }
    };

    std::cout << "***** Test start *****" << std::endl;

    testImpl(sFolderWithNumbers);
    testImpl(sFolderWithNotNumbers);

    std::cout << "***** Test end *****" << std::endl;
}

std::vector<int64_t> CFrameAnalyzer::Analyze(const std::vector<cv::Mat>& cvImages) const noexcept
{
    std::vector<int64_t> predicteds;

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
        std::cout << outputs << std::endl;

        auto outputs_argmax     = outputs.argmax(1).toType(torch::ScalarType::Long);
        const int64_t *outputData   = static_cast<int64_t*>(outputs_argmax.data_ptr());
        if (outputData)
        {
            const int64_t outputCount = outputs_argmax.size(0);
            const size_t imageCount   = cvImages.size();
            assert(outputCount == imageCount);

            predicteds.resize(outputCount);
            for (size_t i = 0; i < outputCount; i++)
            {
                int64_t predictedNumber = outputData[i];
                auto tnsrProbability    = outputs[i][predictedNumber].toType(torch::ScalarType::Float);

                const float *ptrProbability = static_cast<const float*>(tnsrProbability.data_ptr());
                if (ptrProbability && ptrProbability[0] > fTrustedThreshold)
                    predicteds[i] = predictedNumber;
                else
                    predicteds[i] = kErrorPredict;
            }
        }
        else
        {
            std::cout << "[WARNING]\t" << __FUNCTION__ << " output data from model is empty" << std::endl;
        }
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

    return predicteds;
}
