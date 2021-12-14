#include "PipelineModule.hpp"

PipelineModule::PipelineModule(std::vector<VARP> inputs, std::vector<VARP> outputs, const Transformer& transformFunction) {
    setType(PIPELINE_MODULE);
    std::vector<EXPRP> executeOrder;
    std::set<EXPRP> inputExpr;
    for (auto v : inputs) {
        inputExpr.insert(v->expr().first);
    }
    for (auto output : outputs) {
        Expr::visit(output->expr().first,
        [&executeOrder, &inputExpr](EXPRP expr) {
            if (expr->visited()) {
                return false;
            }
            if (inputExpr.find(expr)!= inputExpr.end()) {
                expr->setVisited(true);
                executeOrder.emplace_back(expr);
                return false;
            }
            return true;
        },
        [&executeOrder](EXPRP expr) {
            //FUNC_PRINT_ALL(var->name().c_str(), s);
            if (!expr->visited()) {
                executeOrder.emplace_back(expr);
                expr->setVisited(true);
            }
            return true;
        });
    }
    for (auto expr : executeOrder) {
        expr->setVisited(false);
    }
    // Set Indexes
    std::map<EXPRP, int> indexes;
    int currentIndexes = 0;
    for (auto expr : executeOrder) {
        indexes[expr] = currentIndexes;
        currentIndexes += expr->outputSize();
    }
    std::set<EXPRP> inputSets;
    mInputIndexes.clear();
    mStackSize = currentIndexes;
    for (auto v : inputs) {
        auto inputExpr = v->expr();
        mInputIndexes.emplace_back(indexes[inputExpr.first] + inputExpr.second);
        inputSets.insert(inputExpr.first);
    }

    // Create All SubModule
    for (auto expr : executeOrder) {
        if (inputSets.find(expr) != inputSets.end()) {
            continue;
        }
        std::pair<std::vector<int>, std::shared_ptr<Module> > moduleResult;
        bool extracted = false;
        if (!transformFunction) {
            moduleResult = std::make_pair(std::vector<int>{}, std::shared_ptr<Module>(nullptr));
        } else {
            moduleResult = transformFunction(expr);
        }
        if (moduleResult.second == nullptr) {
            std::shared_ptr<Module> module(new ExprModule(expr));
            moduleResult.first  = ((ExprModule*)module.get())->inputIndexes();
            moduleResult.second = module;
        } else {
            extracted = true;
        }
        auto subInputs        = expr->inputs();
        auto& exprInputIndexes = moduleResult.first;
        std::vector<int> inputIndexes;
        if (exprInputIndexes.empty() && extracted) {
            inputIndexes.resize(subInputs.size());
            for (int i = 0; i < inputIndexes.size(); ++i) {
                auto inputExpr  = subInputs[i]->expr();
                inputIndexes[i] = indexes[inputExpr.first] + inputExpr.second;
            }
        } else {
            inputIndexes.resize(exprInputIndexes.size());
            for (int i = 0; i < inputIndexes.size(); ++i) {
                auto inputExpr  = subInputs[exprInputIndexes[i]]->expr();
                inputIndexes[i] = indexes[inputExpr.first] + inputExpr.second;
            }
        }
        std::vector<int> outputIndexes(expr->outputSize());
        for (int i = 0; i < outputIndexes.size(); ++i) {
            outputIndexes[i] = indexes[expr] + i;
        }
        mSubModules.emplace_back(std::make_tuple(moduleResult.second, inputIndexes, outputIndexes));
        registerModel({moduleResult.second});
    }
    mOutputIndexes.clear();
    for (auto output : outputs) {
        auto outputExpr = output->expr();
        mOutputIndexes.emplace_back(indexes[outputExpr.first] + outputExpr.second);
    }
}
