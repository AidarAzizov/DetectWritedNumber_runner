#include "../includes/PaintWindow.h"

//#define TEST_MODEL

int main() {

    const std::string sScriptedModelPath = "/Users/aidarazizov/CLionProjects/OpenCV_tests/model/trace_model.pt";

    CPaintWindow window(sScriptedModelPath, eSave);

#ifdef TEST_MODEL
    window.Test();
#else
    window.Run();
#endif

    return 0;
}
