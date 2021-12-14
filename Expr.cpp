#include <iostream>

std::vector<VARP> Variable::load(const char* fileName) {
    AutoStorage<uint8_t> buffer;
    {
        FileLoader loader(fileName);
        if (!loader.valid()) {
            MNN_ERROR("Error for open %s\n", fileName);
            return {};
        }
        loader.read();
        if (!loader.valid()) {
            return {};
        }
        loader.merge(buffer);
        if (buffer.get() == nullptr) {
            return {};
        }
    }
    return load(buffer.get(), buffer.size());
}
std::vector<VARP> Variable::load(const uint8_t* buffer, size_t length) {
    AUTOTIME;
    flatbuffers::Verifier verify((const uint8_t*)(buffer), length);
    if (false == VerifyNetBuffer(verify)) {
        MNN_PRINT("Invalidate buffer to create variable\n");
        return {};
    }
    std::unique_ptr<NetT> source(UnPackNet(buffer));
    if (nullptr == source) {
        return {};
    }
    if (source->oplists.empty()) {
        MNN_ERROR("Invalid net\n");
        return {};
    }
    // FUNC_PRINT(source->oplists.size());

    auto opSize      = source->oplists.size();
    auto tensorCount = source->tensorName.size();
    if (tensorCount == 0) {
        tensorCount = source->tensorNumber;
    }
    std::vector<VARP> variable;
    variable.reserve(tensorCount);
    std::map<int, VARP> variableMap;

    // Generate All Exprs by order of net
    for (int i = 0; i < opSize; ++i) {
        std::vector<VARP> inputs;
        auto op = source->oplists[i].get();
        for (int index = 0; index < op->inputIndexes.size(); ++index) {
            auto inputIndex = op->inputIndexes[index];
            if (variableMap.find(inputIndex) == variableMap.end()) {
                MNN_ERROR("Can't find variable for %s, the graph is error\n", op->name.c_str());
                break;
            }
            inputs.emplace_back(variableMap[inputIndex]);
        }
        EXPRP expr = Expr::create(source->oplists[i].get(), inputs, (int)op->outputIndexes.size());
        expr->setName(source->oplists[i]->name);

        for (int index = 0; index < op->outputIndexes.size(); ++index) {
            auto outputIndex = op->outputIndexes[index];
            if (variableMap.find(outputIndex) == variableMap.end()) {
                auto newVariable = Variable::create(expr, index);
                if (source->tensorName.size() > outputIndex) {
                    newVariable->setName(source->tensorName[outputIndex]);
                }
                variableMap[outputIndex] = newVariable;
                variable.emplace_back(newVariable);
            }
        }
    }
    return variable;
}

std::map<std::string, VARP> Variable::loadMap(const uint8_t* buffer, size_t length) {
    AUTOTIME;
    auto variables = load(buffer, length);
    std::map<std::string, VARP> varMap;
    for (auto v : variables) {
        varMap[v->name()] = v;
    }
    return varMap;
}

std::map<std::string, VARP> Variable::loadMap(const char* fileName) {
    AUTOTIME;
    auto variables = load(fileName);
    std::map<std::string, VARP> varMap;
    for (auto v : variables) {
        varMap[v->name()] = v;
    }
    return varMap;
}


std::pair<std::map<std::string, VARP>, std::map<std::string, VARP>> Variable::getInputAndOutput(const std::map<std::string, VARP>& allVariable) {
    std::pair<std::map<std::string, VARP>, std::map<std::string, VARP>> res;
    for (auto& iter : allVariable) {
        auto var = iter.second;
        if (var->expr().first->get() == nullptr && var->expr().first->mType == VARP::INPUT) {
            res.first[var->name()] = var;
        }
        if (var->linkNumber() == 0) {
            res.second[var->name()] = var;
        }
    }
    return res;
}

std::vector<VARP> Variable::mapToSequence(const std::map<std::string, VARP>& source) {
    std::vector<VARP> outputs;
    outputs.reserve(source.size());
    for (auto& iter : source) {
        outputs.emplace_back(iter.second);
    }
    return outputs;
}

std::vector<EXPRP> Variable::getExecuteOrder(const std::vector<VARP>& outputs) {
    std::vector<EXPRP> sequence;
    for (auto output : outputs) {
        Expr::visit(
                        output->mFrom, [](EXPRP expr) { return !expr->visited(); },
                        [&sequence](EXPRP expr) {
                            //FUNC_PRINT_ALL(var->name().c_str(), s);
                            if (!expr->visited()) {
                                sequence.emplace_back(expr);
                                expr->setVisited(true);
                            }
                            return true;
                        });
    }
    for (auto expr : sequence) {
        expr->setVisited(false);
    }
    return sequence;
}

void Expr::visit(EXPRP expr, const std::function<bool(EXPRP)>& before, const std::function<bool(EXPRP)>& after) {
    
    bool next = before(expr);
    if (!next) {
        
        return;
    }
    for (int i = 0; i < expr->inputs().size(); ++i) {
        printf("expr id is %s\n",MNN::EnumNameOpType(expr->get()->type()));
        visit(expr->inputs()[i]->mFrom, before, after);
    }
    after(expr);
}