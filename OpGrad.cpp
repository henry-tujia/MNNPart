//
//  OpGrad.cpp
//  MNN
//
//  Created by MNN on 2019/05/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OpGrad.hpp"
#include "MNN/MNNDefine.h"
using namespace std;
using namespace MNN::Express;
namespace MNN {
static std::map<int, OpGrad*>& getConverter() {
    static std::map<int, OpGrad*> gConverterMap;
    return gConverterMap;
}

OpGrad* OpGrad::get(int type) {
    auto& converterMap = getConverter();
    auto iter          = converterMap.find(type);
    if (iter != converterMap.end()) {
        return iter->second;
    }
    return nullptr;
}

void OpGrad::insert(int type, OpGrad* converter) {
    auto& converterMap = getConverter();
    converterMap.insert(std::make_pair(type, converter));
}

std::vector<Express::VARP> OpGrad::gradLinear(Express::VARP loss, const std::vector<Express::VARP>& parameters, const std::vector<Express::VARP>& outputDiff, const std::string& blockExpr) {
    std::map<EXPRP, std::vector<VARP>> backwardMap;
    auto outputSize = loss->expr().first->outputSize();
    if (outputSize != outputDiff.size()) {
        MNN_ERROR("The expr output %d, but diff size is %d\n", outputSize, (int)outputDiff.size());
        return {};
    }
    backwardMap[loss->expr().first] = outputDiff;
    std::set<VARP> parameterSet;
    for (auto p : parameters) {
        parameterSet.insert(p);
    }
    auto res = gradCommon(loss, parameterSet, backwardMap, blockExpr);
    std::vector<VARP> linearRes(parameters.size(), nullptr);
    for (int i=0; i<parameters.size(); ++i) {
        auto iter = res.find(parameters[i]);
        if (iter != res.end()) {
            linearRes[i] = iter->second;
        }
    }
    return linearRes;
}

std::map<Express::VARP, Express::VARP> OpGrad::grad(VARP loss, const std::set<Express::VARP>& parameters, const std::string& blockName) {
    std::map<EXPRP, std::vector<VARP>> backwardMap;
    {
        auto shape = loss->getInfo();
        MNN_ASSERT(shape->size == 1);
        auto init                       = _Const(1.0f, shape->dim, shape->order);
        backwardMap[loss->expr().first] = std::vector<VARP>{init};
    }
    return gradCommon(loss, parameters, backwardMap, blockName);
}
std::map<Express::VARP, Express::VARP> OpGrad::gradCommon(Express::VARP loss, const std::set<Express::VARP>& parameters, std::map<EXPRP, std::vector<VARP>>& backwardMap, const std::string& blockName) {
    printf("--------------------get in OpGrad-------------------\n");
    printf("-------------------ExecuteOrder is :--------------------\n");
    auto executeOrder = Variable::getExecuteOrder({loss});
    int index_out = 0;
    for (auto iter = executeOrder.rbegin(); iter != executeOrder.rend(); iter++) {
        index_out ++;
        printf("---------------------------------%dth excute is comming!-----------------------------\n",index_out);
        
        auto expr    = *iter;
        //debug
        auto& inputs = expr->inputs();
        if (backwardMap.find(expr) == backwardMap.end()) {
            continue;
        }
        if (nullptr == expr->get()) {
            continue;
        }
        if (!blockName.empty()) {
            if (blockName == expr->name()) {
                break;
            }
        }
        auto grad = OpGrad::get(expr->get()->type());
        if (nullptr == grad) {
            // MNN_PRINT("Can't grad for %s, %d\n", expr->name().c_str(), expr->get()->type());
            continue;
        }
        MNN_PRINT("grad for %s\n",MNN::EnumNameOpType(expr->get()->type()));
        if(index_out == 91){
            printf("conv is here！\n");
        }
        auto inputGrad = grad->onGrad(expr, backwardMap[expr]);
        auto empty     = true;
        for (auto grad : inputGrad) {
            if (nullptr != grad) {
                empty = false;
                break;
            }
        }
        if (empty) {
            // MNN_PRINT("Can't grad for %s, %d\n", expr->name().c_str(), expr->get()->type());
            continue;
        }
// #ifdef MNN_TRAIN_DEBUG
//         for (int i = 0; i < inputGrad.size(); ++i) {
//             if (nullptr == inputGrad[i]) {
//                 continue;
//             }
//             auto info = inputGrad[i]->getInfo();
//             if (nullptr == info) {
//                 MNN_ERROR("Grad error for %s, %d\n", expr->name().c_str(), expr->get()->type());
//                 break;
//             }
//         }
// #endif
        MNN_ASSERT(inputGrad.size() <= inputs.size());
        for (int i = 0; i < inputGrad.size(); ++i) {
            auto inputExpr = inputs[i]->expr().first;
            auto index     = inputs[i]->expr().second;
            auto backward  = inputGrad[i];
            printf("%dth inputgrad is %f\n",index,backward->readMap<float>()[0]);

            // for (int i = 0;i<10;i++){
            //     printf("%f\n",backward->readMap<float>()[i]);
            // }
            if (nullptr == backward) {
                continue;
            }
            if (backwardMap.find(inputExpr) == backwardMap.end()) {
                backwardMap.insert(std::make_pair(inputExpr, std::vector<VARP>(inputExpr->outputSize())));
            }
            auto& inputVarMap = backwardMap[inputExpr];
            if (nullptr == inputVarMap[index]) {
                inputVarMap[index] = backward;
            } else {
                inputVarMap[index] = _Add(inputVarMap[index], backward);
            }
        }
        
    }
    std::map<Express::VARP, Express::VARP> grads;
    std::map<Expr*, VARP> parametersExpr;
    for (auto p : parameters) {
        parametersExpr.insert(std::make_pair(p->expr().first.get(), p));
    }
    for (auto iter : backwardMap) {
        auto expr = iter.first.get();
        if (parametersExpr.find(expr) != parametersExpr.end()) {
            auto parameter   = parametersExpr[expr];
            grads[parameter] = iter.second[parameter->expr().second];
        }
    }

    //debug

    for (auto iter : grads){
        auto item1 = iter.first->readMap<float>()[0];
        // first is para,second is grad
        printf("grad param is %f\n",item1);
        
        auto item2 = iter.second->readMap<float>()[0];
        printf("grad is %f\n",item2);

    }
    


    // MNN_PRINT("Grad: %d <- %d\n", grads.size(), parameters.size());
    printf("-------------------leave OpGrad-------------------\n");
    return grads;
}

} // namespace MNN
