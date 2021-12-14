Module* NN::extract(std::vector<Express::VARP> inputs, std::vector<Express::VARP> outputs, bool fortrain, const std::map<std::string, SubGraph>& subGraph) {
    std::function<std::pair<std::vector<int>, std::shared_ptr<Module>>(EXPRP)> transformFunction;
    if (fortrain) {
        transformFunction =
        [&subGraph](EXPRP source) {
            if (source->get() == nullptr) {
                return std::make_pair(std::vector<int>{}, std::shared_ptr<Module>(nullptr));
            }
            std::shared_ptr<Module> m(NN::Utils::ExtractNotRunableOp(source, subGraph));
            if (nullptr != m) {
                m->setName(source->name());
                return std::make_pair(std::vector<int>{}, m);
            }
            auto convExtracted = NN::Utils::ExtractConvolution(source);
            if (convExtracted.weight == nullptr) {
                return std::make_pair(std::vector<int>{}, std::shared_ptr<Module>(nullptr));
            }
            std::shared_ptr<Module> module(NN::Conv(convExtracted));
            module->setName(source->name());
            return std::make_pair(std::vector<int>{0}, module);
        };
    } else {
        transformFunction = [&subGraph](EXPRP source) {
            if (source->get() == nullptr) {
                return std::make_pair(std::vector<int>{}, std::shared_ptr<Module>(nullptr));
            }
            std::shared_ptr<Module> m(NN::Utils::ExtractNotRunableOp(source, subGraph));
            if (nullptr != m) {
                m->setName(source->name());
                return std::make_pair(std::vector<int>{}, m);
            }
            return std::make_pair(std::vector<int>{}, std::shared_ptr<Module>(nullptr));
        };
    }
    return new PipelineModule(inputs, outputs, transformFunction);
}