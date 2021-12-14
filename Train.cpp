#include "TrainUtils.hpp"

using namespace MNN::Express;

class MobilenetV2PostTrain{
public:
    int run(int argc, const char* argv[]) {
        if (argc < 1) {
            std::cout << "usage: ./runTrainDemo.out MobilentV2PostTrain /path/to/mobilenetV2Model"<< std::endl;
            return 0;
        }
        auto varMap = Variable::loadMap(argv[1]);
        if (varMap.empty()) {
            printf("Can not load model %s\n", argv[1]);
            return 0;
        }

        auto inputOutputs = Variable::getInputAndOutput(varMap);
        auto inputs       = Variable::mapToSequence(inputOutputs.first);
        auto outputs      = Variable::mapToSequence(inputOutputs.second);
        std::shared_ptr<Module> model(NN::extract(inputs, outputs, true));

        return 0;
    }
};