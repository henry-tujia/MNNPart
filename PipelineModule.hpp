class PipelineModule : public Module {
public:
    typedef std::function<std::pair<std::vector<int>, std::shared_ptr<Module>>(Express::EXPRP)> Transformer;
    MNN_PUBLIC static Module* load(const std::vector<std::string>& inputs, const std::vector<std::string>& outputs, const uint8_t* buffer, size_t length, const std::shared_ptr<MNN::Express::Executor::RuntimeManager> rtMgr, const Module::Config* config = nullptr);
    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP>& inputs) override;
    virtual void onClearCache() override;
    MNN_PUBLIC std::vector<int> countOutputReference(std::vector<int> outputIndices);

    MNN_PUBLIC PipelineModule(std::vector<Express::VARP> inputs, std::vector<Express::VARP> outputs,
                   const Transformer& transformFunction = {});
private:
    static Module* load(const std::vector<std::string>& inputs, const std::vector<std::string>& outputs, const uint8_t* buffer, size_t length, const std::shared_ptr<MNN::Express::Executor::RuntimeManager> rtMgr, const Module::Config* config, std::map<std::string, SubGraph>& subGraphMap, bool inRecurce = false);
    static void _createSubGraph(const MNN::Net* net, std::shared_ptr<MNN::Express::Executor::RuntimeManager> rtMgr, const Module::Config* config, std::map<std::string, SubGraph>& subGraphMap);

    PipelineModule(){}

    Module* clone(CloneContext* ctx) const override;

    std::vector<std::tuple<std::shared_ptr<Module>, std::vector<int>, std::vector<int>>> mSubModules;
    std::vector<int> mInputIndexes;
    std::vector<int> mOutputIndexes;
    int mStackSize = 0;
    friend class NN;
    std::vector<VARP> mInitVars;
};