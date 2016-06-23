#include <cstdatomic>
#include <limits>
#include "base/common/logging.h"
#include "base/strings/string_split.h"
#include "base/strings/string_number_conversions.h"
#include "../include/mgbrt_config.h"
#include "../include/mgbrt_tree.h"
#include "../include/mgbrt_comm.h"
namespace mgbrt {
namespace tree {
mgbrt::config::Config *gConfTree = mgbrt::config::Config::GetConfig();

void Tree::SplitNodes(FinalSplits* info) {
  VLOG(3) << "split nodes in tree, final splits:" << DumpFinalSplits(*info);
  for (int i = 0; i < info->num; ++i) {
    // DEBUG
    // VLOG(3) << tree::DumpFinalSplits(*info);
    SplitNode(&(info->finalSplits[i]));
  }
}

void Tree::Reset() {
  VLOG(2) << "Tree Reset !";
  nodes.clear();
  nodes.resize(kNodeNum);
  nodes[0].splitValid = 1;
  currentDepth_ = 0;
  nodes[0].loss = 0;
  for (int i = 1; i < kNodeNum; ++i)
    nodes[i].splitValid = 0;
}

int32 Tree::DumpTreeLayer(int32 layer) {
  int32 ret = 1;
  auto conf = config::Config::GetConfig();
  VLOG(2) << "Dump tree layer:" << layer;
  std::ofstream ofile(kFile.c_str(), std::ios_base::app);
  int32 stNode = (std::pow(kBranch, layer) - 1) / (kBranch - 1);
  int32 edNode = (std::pow(kBranch, layer + 1) - 1) / (kBranch - 1) - 1;
  nodes[stNode].label *= conf->learningRate;
  if (nodes[stNode].splitValid) ret = 0;
  if ( 1==1 || 1 == nodes[stNode].isValid) {
    ofile << stNode << ":" << DumpTreeNode(nodes[stNode]);
  }
  for (int i = stNode + 1; i <= edNode; ++i) {
    if ( 1 != 1 && nodes[i].isValid == 0) {
      continue;
    }
    nodes[i].label *= conf->learningRate;
    if (nodes[i].splitValid) ret = 0;
    ofile << "\t" << i << ":" << DumpTreeNode(nodes[i]);
  }
  if (layer != (kDepth - 1)) {
    if (ret == 0) {
      ofile << '\t';
    } else {
      ofile << '\n';
      return 1;
    }
  } else {
    ofile << '\n';
    return 1;
  }
  return ret;
}

int32 Tree::DumpTree() {
  VLOG(2) << "DumpTree Layer:" << currentDepth_;
  // std::cout << "DumpTree Layer:" << currentDepth_;
  int32 ret = DumpTreeLayer(currentDepth_);
  if (ret == 1) {
    currentDepth_ = 0;
    return 1;
  }
  
  int32 depth = currentDepth_ + 1;
  SetCurrentDepth(depth);
  return ret;
}

bool Tree::BuildNode(const std::string& str, TreeNode* node) const {
  if (str.empty()) return false;
  CHECK(node) << "node is empty, tree: ";
  auto& workNode = *node;
  std::vector<std::string> terms;
  base::SplitStringWithOptions(str, ":", true, false, &terms);
  int num = 0;
  int nodId = atoi(terms[num++].c_str());
  workNode.leftChildIdx = nodId * 3 + 1;
  workNode.rightChildIdx = nodId * 3 + 2;
  workNode.unknownChildIdx = nodId * 3 + 3;
  workNode.feaDimIdx = atoi(terms[num++].c_str());
  workNode.splitValue = atof(terms[num++].c_str());
  workNode.label = atof(terms[num++].c_str());
  workNode.splitValid = atol(terms[num++].c_str());
  VLOG(2) << "BUILDNODE:" << DumpTreeNode(*node);
  return true;
}

bool Tree::BuildTree(const std::string& Serialization) {
  if (Serialization.empty()) return false;
  std::vector<std::string> terms;
  base::SplitStringWithOptions(Serialization, "\t", true, false, &terms);
  CHECK(nodes.size() >= (terms.size()))
      << "kNodeNum:" << kNodeNum << " Serizalization node nums:" << terms.size();
  for (size_t i = 0; i < terms.size(); ++i)
    if (!BuildNode(terms[i], &(nodes[i])))
      return false;
  return true;
}

int32 Tree::MatchStopCondition(const FinalSplitInfo& splitInfo) const {
  // VLOG(3) << tree::DumpFinalSplitInfo(splitInfo);
  // 非可分裂节点
  if (!(nodes[splitInfo.nodeIdx].splitValid)) {
    VLOG(3) << base::StringPrintf("splitEnded! nodes[%d] cannot be splited, splitInfo:%s",
                                  splitInfo.nodeIdx, tree::DumpFinalSplitInfo(splitInfo).c_str());
    return 1;
  }
  // 加入新的分裂条件
  if (splitInfo.insNum + splitInfo.unknownInsNum
      <= splitConditions.minNodes) {
    VLOG(3) << base::StringPrintf("splitEnded! nodes[%d].insnum=%ld is too small, less than minNodes:%ld",
                                  splitInfo.nodeIdx,
                                  splitInfo.insNum + splitInfo.unknownInsNum,
                                  splitConditions.minNodes);
    return 2;  // 不小于预定节点数
  }
  // if (nodes[splitInfo.nodeIdx].)
  // 最后一层
  if (splitInfo.nodeIdx >= (kNodeNum - 1) / kBranch) {
    VLOG(3) << base::StringPrintf("splitEnded! splitInfo's nodeIdx:%d, in the last tree Layer",
                                  splitInfo.nodeIdx);
    return 3;
  }

  return 0;
}

void Tree::SplitNode(FinalSplitInfo* splitInfo_t) {
  // DEBUG
  int32 result = 0;
  const FinalSplitInfo& splitInfo = *splitInfo_t;
  CHECK(splitInfo.nodeIdx < (int)(nodes.size()));
  if ((result = MatchStopCondition(splitInfo))) {
    auto& workNode = nodes[splitInfo.nodeIdx];
    workNode.insNum = splitInfo.insNum + splitInfo.unknownInsNum;
    workNode.unknownInsNum = splitInfo.unknownInsNum;
    workNode.splitValid = 0;
    workNode.loss = std::numeric_limits<double>::max();
    LOG(INFO) << "Node Id:" << splitInfo.nodeIdx << " split finished.";
    return;
  }
  // DEBUG
  auto& workNode = nodes[splitInfo.nodeIdx];
  workNode.isValid = 1;

  workNode.leftChildIdx = splitInfo.nodeIdx * kBranch + 1;
  workNode.rightChildIdx = splitInfo.nodeIdx * kBranch + 2;
  workNode.unknownChildIdx = splitInfo.nodeIdx * kBranch + 3;
  workNode.feaDimIdx = splitInfo.feaDimIdx;
  workNode.splitValue = splitInfo.splitValue;
  workNode.insNum = splitInfo.insNum + splitInfo.unknownInsNum;
  workNode.unknownInsNum = splitInfo.unknownInsNum;
  // VLOG(3) << "SplitNode, workNode test " << "workNode insNum:"
  //    << workNode.insNum;
  workNode.sumInf = splitInfo.unknownLabelSum + splitInfo.sumLabel;
  // VLOG(3) << "SplitNode, workNode test " << "workNode suminf:"
  //    << workNode.sumInf;
  
  workNode.label = (double)(workNode.sumInf) / (double)(workNode.insNum);
  // VLOG(3) << "SplitNode, workNode test " << "workNode label:"
  //     << workNode.label;

  workNode.loss = - std::pow(splitInfo.la, 2) / splitInfo.ma
      - std::pow(splitInfo.sumLabel - splitInfo.la, 2) / (splitInfo.insNum - splitInfo.ma);
  workNode.gain = (double)splitInfo.ma * (double)(splitInfo.insNum - splitInfo.ma) / (double)splitInfo.insNum;
  workNode.gain *= pow((splitInfo.la / splitInfo.ma)
                       - (splitInfo.sumLabel - splitInfo.la) / (splitInfo.insNum - splitInfo.ma), 2);
  if (workNode.unknownChildIdx < kNodeNum) {
    nodes[workNode.leftChildIdx].splitValid = 1;
    if (splitInfo.ma != 0)
      nodes[workNode.leftChildIdx].label =
          splitInfo.la / (double)(splitInfo.ma);
    else 
      nodes[workNode.leftChildIdx].label = workNode.label;

    nodes[workNode.rightChildIdx].splitValid = 1;
    if (splitInfo.insNum != splitInfo.ma)
      nodes[workNode.rightChildIdx].label =
          (splitInfo.sumLabel - splitInfo.la) / (double)(splitInfo.insNum - splitInfo.ma);
    else
      nodes[workNode.rightChildIdx].label = workNode.label;

    nodes[workNode.unknownChildIdx].splitValid = 1;
    if (splitInfo.unknownInsNum != 0)
      nodes[workNode.unknownChildIdx].label =
          splitInfo.unknownLabelSum / (double)(splitInfo.unknownInsNum);
    else
      nodes[workNode.unknownChildIdx].label = workNode.label;
    if (currentDepth_ == kDepth - 1) {
      workNode.splitValid = 0;
      nodes[workNode.leftChildIdx].splitValid = 0;
      nodes[workNode.rightChildIdx].splitValid = 0;
      nodes[workNode.unknownChildIdx].splitValid = 0;
    }
  }
}

void Tree::UpdateInstanceNode(comm::Instance* instance) const {
  if (currentDepth_ == 0) {
    instance->nodeIdx = 0;
    VLOG(3) << "Tree Construct Start, RESET NODE TO 0";
    return;
  }
  int32 nodeIdx = instance->nodeIdx;
  auto& workNode = nodes[nodeIdx];
  // DEBUG
  if (workNode.splitValid) {
    int32 splitFea = workNode.feaDimIdx;
    double splitVal = workNode.splitValue;
    // xiugai
    //if (instance->points[splitFea - FEATOFFSET].exist == 0) {
    VLOG(3) << "instance feature value:"
            << instance->points[splitFea - FEATOFFSET].feaValue
            << ", nodeIdx:" << instance->nodeIdx
            << ", feaDimIdx:" << splitFea
            << ", max double:" << std::numeric_limits<double>::min();
    if (instance->points[splitFea - FEATOFFSET].feaValue
        == std::numeric_limits<double>::min()) {
      instance->nodeIdx = workNode.unknownChildIdx;
    } else if (instance->points[splitFea - FEATOFFSET].feaValue < splitVal) {
      instance->nodeIdx = workNode.leftChildIdx;
    } else {
      instance->nodeIdx = workNode.rightChildIdx;
    }
  }
  VLOG(3) << "instance idx:" << instance->nodeIdx;
}

void Tree::UpdateFx(comm::Instance* instance) const {
  int32 nodeIdx = instance->nodeIdx;
  instance->fx += nodes[nodeIdx].label;
}

double Tree::Predict(const comm::Instance& instance) const {
  int deep = 0;
  int nod = 0;
  VLOG(2) << "--------predict------------" << std::endl;
  VLOG(2) << nod << std::endl;
  while (deep < this->kDepth) {
    ++deep;
    auto& workNode = nodes[nod];
    int32 feaId = workNode.feaDimIdx;
    double feaVal = workNode.splitValue;
    if (workNode.splitValid == 0) return workNode.label;
    if (instance.points[feaId - FEATOFFSET].feaValue
        == std::numeric_limits<double>::min()) {
      nod = workNode.unknownChildIdx;
      VLOG(2) << nod << std::endl;
      continue;
    }
    if (instance.points[feaId - FEATOFFSET].feaValue <= feaVal) {
      nod = workNode.leftChildIdx;
    } else {
      nod = workNode.rightChildIdx;
    }
    // DEBUG
    VLOG(2) << nod << std::endl;
  }
  return nodes[nod].label;
}

double Tree::PredictTrace1(const comm::Instance& instance, int *node) const {
  int deep = 0;
  int nod = 0;
  while (deep < this->kDepth) {
    ++deep;
    auto& workNode = nodes[nod];
    int32 feaId = workNode.feaDimIdx;
    double feaVal = workNode.splitValue;
    if (workNode.splitValid == 0) {
      *node = nod;
      return workNode.label;
    }
    if (instance.points[feaId - FEATOFFSET].feaValue
        == std::numeric_limits<double>::min()) {
      nod = workNode.unknownChildIdx;
      VLOG(2) << nod << std::endl;
      continue;
    }
    if (instance.points[feaId - FEATOFFSET].feaValue <= feaVal) {
      nod = workNode.leftChildIdx;
    } else {
      nod = workNode.rightChildIdx;
    }
  }
  *node = nod;
  return nodes[nod].label;
}


double Tree::Predict(const std::vector<FeaInfo>& feature, int32* node) const {
  int deep = 0;
  int nod = 0;
  while (deep < this->kDepth) {
    ++deep;
    auto& workNode = nodes[nod];
    int32 feaId = workNode.feaDimIdx;
    double feaVal = workNode.splitValue;
    if (workNode.splitValid == 0) {
      *node = nod;
      return workNode.label;
    }
    if (feature[feaId - FEATOFFSET].feaDimIdx < 0) {
      nod = workNode.unknownChildIdx;
      VLOG(2) << nod << std::endl;
      continue;
    }
    if (feature[feaId - FEATOFFSET].feaVal <= feaVal) {
      nod = workNode.leftChildIdx;
    } else {
      nod = workNode.rightChildIdx;
    }
  }
  *node = nod;
  return nodes[nod].label;
}

void LS_Tree::CalculateGradient(comm::Instance* instance) const {
  instance->gx = instance->label - instance->fx;
}

void LS_Tree::CalculateCoefficient(double kLearningRate) {
  for (int i = 0; i < kNodeNum; ++i)
    nodes[i].label *= kLearningRate;
}

void L2_Tree::CalculateGradient(comm::Instance* instance) const {
  const int& y_i = instance->gx;
  instance->gx = 2 * y_i / (1 + exp(2 * y_i * instance->fx));
}

void L2_Tree::CalculateCoefficient(double kLearningRate) {
  CHECK(false) << "do not use this calculatecoeffient function in l2 tree";
  return;
}

void L2_Tree::CalculateCoefficient(double kLearningRate, const AllGammaStatInfo& gammaInfo) {
  for (int i = 0; i < gammaInfo.infoNum; ++i) {
    auto& gamma = gammaInfo.infos[i];
    nodes[gamma.nodeIdx].label = kLearningRate * gamma.ySum / gamma.absySum;
  }
}

Booster::Booster() {
  auto config = config::Config::GetConfig();
  treeNum_ = config->treeNum;
  trees_.resize(treeNum_);
  treeBranch_ = config->branchNum;
  treeDepth_ = config->maxTreeDepth;
  splitCond_.minNodes = config->treeNodeMinInsNum;
  pool_.reset(new thread::ThreadPool(config->calcThreadNum));
  for (uint32 i = 0; i < config->treeNum; ++i) {
    trees_[i].reset(new tree::LS_Tree(treeDepth_, treeBranch_, splitCond_));
  }
}

Booster::~Booster() {
  pool_->JoinAll();
}

bool Booster::LoadTree(const std::string& filename) {
  std::ifstream ifile(filename);
  if (!(ifile.is_open())) {
    LOG(ERROR) << "load tree failed, tree filename path:" << filename;
    return false;
  }
  std::string treeSeri;
  int32 treeIdx = 0;
  while (std::getline(ifile, treeSeri)) {
    CHECK(this->trees_[treeIdx]->BuildTree(treeSeri));
    treeIdx += 1;
    VLOG(1) << "load tree success, tree:" << treeIdx;
  }
  ifile.close();
  return true;
}

void Booster::ThreadPredict(comm::Instance* instance, int32 n, double* sum, std::atomic<int32>* counter) const {
  // std::cout << "predicting..." << n << std::endl;
  if (trees_[n]) {
    // std::cout << "predict tree:" << n << std::endl;
    *sum += trees_[n]->Predict(*instance);
  }
  ++(*counter);
}

double Booster::Predict(comm::Instance* instance) const {
  double sum = 0;
  std::atomic<int32> counter(0);
  // std::cout << "predict :" << comm::DumpInstance(*instance) << std::endl;
  for (int32 i = 0; i < treeNum_; ++i) {
    ThreadPredict(instance, i, &sum, &counter);
  }
  // std::cout << "predict done!" << std::endl;
  return sum;
}

//////////////////add by ycn/////////////////////
void Booster::ThreadPredictTrace(
    comm::Instance* instance,
    int32 n,
    double* label,
    std::atomic<int32>* counter,
    int *node) const {
  // std::cout << "predicting..." << n << std::endl;
  if (trees_[n]) {
    // std::cout << "predict tree:" << n << std::endl;
    *label = trees_[n]->PredictTrace1(*instance, node);
  }
  ++(*counter);
}

double Booster::PredictTrace(comm::Instance* instance) const {
  double sum = 0;
  std::atomic<int32> counter(0);
  // std::cout << "predict :" << comm::DumpInstance(*instance) << std::endl;
  for (int32 i = 0; i < treeNum_; ++i) {
    int node = -1;
    double label = 0.0;
    ThreadPredictTrace(instance, i, &label, &counter, &node);
    node = node;
    sum += label;
#ifdef DUMP_PRED_INFO
    if (true == gConfTree->isDumpPredInfo) {
      mgbrt::comm::PredInfo predInfo = {-1,0.0,0.0};
      predInfo.nodeId = node;
      predInfo.predLabel = label;
      predInfo.sumLabel = sum;
      instance->predInfo.push_back(predInfo);
      //instance->predNodeIds[i] = node;
      //instance->predNodeLabels[i] = label;
      //instance->predNodeLabels[i] = sum; // 保存i棵树时预测的label,便于计算最优树的棵数
    }
#endif
  }
  // std::cout << "predict done!" << std::endl;
  return sum;
}
//////////////////add by ycn/////////////////////

double Booster::Predict(const std::vector<FeaInfo>& feaInfos, std::vector<int32>* nods) const {
  nods->resize(treeNum_);
  double sum = 0;
  for (int32 i = 0; i < treeNum_; ++i) {
    sum += trees_[i]->Predict(feaInfos, &(nods->at(i)));
  }
  return sum;
}

}
}
