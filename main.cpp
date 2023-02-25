#include "PaintWindow.h"

int main() {

    const std::string sScriptedModelPath = "/Users/aidarazizov/CLionProjects/OpenCV_tests/trace_model.pt";

    CPaintWindow window(sScriptedModelPath, eSave);
    window.Run();

    return 0;
}
