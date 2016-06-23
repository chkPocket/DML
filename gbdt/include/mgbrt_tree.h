#pragma once

#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <boost/shared_ptr.hpp>
#include "base/common/logging.h"
#include "base/common/base.h"
#include "base/strings/string_printf.h"
#include "base/thread/sync.h"
#include "base/thread/thread_pool.h"
#include "base/thread/thread.h"
#include "../include/mgbrt_config.h"

namespace mgbrt {
namespace comm {
class Instance;
} 
namespace tree {
struct FeaInfo {
 int32 feaDimIdx;
 double feaVal;
};

typedef struct TreeNode {
  int32 leftChildIdx;
  int32 rightChildIdx;
  int32 unknownChildIdx;  // 特征缺失节点 id
  int32 feaDimIdx;  // 分割点的特征维度 id
  double splitValue;  // 分割点的特征值
  double label;  // 叶子节点的 label 值
  int64 insNum;  // 当前叶子节点下的样本个数
  int64 unknownInsNum;  // 当前叶子节点下的样本个数
  double sumInf;  // 当前叶子节点上样本的y值之和
  int64 splitValid;  // 是否可以分割
  double loss;  // 当前叶子节点上的数据 loss
  double gain;
  int64 isValid; //当前节点中的数据是否有效-0:无效;1:有效
  TreeNode() {
    leftChildIdx = 0;
    rightChildIdx = 0;
    unknownChildIdx = 0;
    feaDimIdx = 0;
    splitValue = 0.0;
    label = 0.0;
    insNum = 0;
    unknownInsNum = 0;
    splitValid = 0;
    loss = 0.0;
    gain = 0.0;
    isValid = 0;
  }
} TreeNode;

inline std::string DumpTreeNode(const TreeNode& treeNode) {
#if 0
  return base::StringPrintf("%d:%lf:%lf:%ld",
                            treeNode.feaDimIdx, treeNode.splitValue,
#endif
#if 1
  return base::StringPrintf("%d:%lf:%lf:%ld:%ld:%ld:%lf",
                            treeNode.feaDimIdx, treeNode.splitValue,
                            treeNode.label, treeNode.splitValid,
                            treeNode.insNum,treeNode.unknownInsNum,
                            treeNode.sumInf);
#endif
#if 0
  return base::StringPrintf("%d:%lf:%lf:%ld:%ld:%ld",
                            treeNode.feaDimIdx, treeNode.splitValue,
                            treeNode.label, treeNode.splitValid,
                            treeNode.insNum,treeNode.unknownInsNum);
#endif

  /*
  return base::StringPrintf(
      "%d:%d:%d:%d:%lf:%lf:%ld:%lf:%ld:%lf",
      treeNode.leftChildIdx, treeNode.rightChildIdx,
      treeNode.unknownChildIdx, treeNode.feaDimIdx,
      treeNode.splitValue, treeNode.label, treeNode.insNum,
      treeNode.sumInf, treeNode.splitValid, treeNode.loss
      );
  return base::StringPrintf(
      "leftChild:%d, rightChild:%d, unknownChild:%d, "
      "feaDimIdx:%d, splitValue:%lf, label:%lf, insNum:%ld, "
      "sumInf:%lf, splitValid?:%ld, split gain:%lf",
      treeNode.leftChildIdx, treeNode.rightChildIdx,
      treeNode.unknownChildIdx, treeNode.feaDimIdx,
      treeNode.splitValue, treeNode.label, treeNode.insNum,
      treeNode.sumInf, treeNode.splitValid, treeNode.gain
      );
  */
}

typedef struct SplitInfo {
  int32 nodeIdx;  // 当前 Split 的节点 id
  int32 feaDimIdx;  // 当前 Split 点的特征维度 id
  double splitValue;  // 分割点的特征值
  double sumLabel;  // 当前叶子节点下所有样本的 label 之和 ( 不包括未知节点 )
  double unknownLabelSum;  // 当前叶子节点下未知节点的 label 和
  double la;  // 候选值左侧样本的标签之和
  int64 totalInsNum;  // 当前特征叶子节点下所有样本的条数(除 unknown 节点外)
  int64 unknownInsNum;  // unknown 的样本条数
  int64 ma;  // 候选值左侧样本数
  double diff;
  double loss;
} SplitInfo;

inline std::string DumpSplitInfo(const SplitInfo& splitInfo) {
  return base::StringPrintf("nodeIdx:%d, feaDimIdx:%d, splitValue:%lf, sumLabel:%lf,"
                            "unknownLabelSum:%lf, la:%lf, totalInsNum:%ld, unknownInsNum:%ld"
                            "ma:%ld, diff:%lf", splitInfo.nodeIdx, splitInfo.feaDimIdx, splitInfo.splitValue,
                            splitInfo.sumLabel, splitInfo.unknownLabelSum, splitInfo.la,
                            splitInfo.totalInsNum, splitInfo.unknownInsNum, splitInfo.ma, splitInfo.diff);
}

typedef struct SplitCandidates {
  int64 candidatesNum; // 存储 SplitInfo 数量
  SplitInfo splitInfos[0];  // 保存当前 MixWorker 产出的所有 split 点的信息
} SplitCandidates;

inline std::string DumpSplitCandidates(const SplitCandidates& candidates) {
  std::string ret;
  ret += base::StringPrintf("CandiNums: %ld ** ", candidates.candidatesNum);
  for (int i = 0; i < candidates.candidatesNum; ++i) {
    ret += "\t" + DumpSplitInfo(candidates.splitInfos[i]);
  }
  return ret;
}

typedef struct FinalSplitInfo {
  int32 feaDimIdx;  // 候选值所属的特征维度 id
  int32 nodeIdx;  // 候选值所属的叶子节点
  double splitValue;  // 候选值
  double sumLabel;  // 分裂前当前叶子节点下所有样本的 label 之和
  double unknownLabelSum;  // 当前叶子节点下未知节点的 label 和
  double la;  // 分割点左侧样本的标签之和
  int64 insNum;  // 分裂当前节点的所有样本条数
  int64 unknownInsNum;
  int64 ma;  // 分割点左侧样本个数
  int64 init;
  double diff;
  double loss;
  int64 isStopSplit;
} FinalSplitInfo;

inline std::string DumpFinalSplitInfo(const FinalSplitInfo& finalSplitInfo) {
  return base::StringPrintf("feaDimIdx: %d, nodeIdx: %d, splitValue: %lf, sumLabel: %lf,"
                            "la: %lf, insNum: %ld, ma: %ld, diff:\t%f",
                            finalSplitInfo.feaDimIdx, finalSplitInfo.nodeIdx,
                            finalSplitInfo.splitValue, finalSplitInfo.sumLabel, finalSplitInfo.la,
                            finalSplitInfo.insNum, finalSplitInfo.ma, finalSplitInfo.diff);
}

typedef struct FinalSplits {
  int64 num;  // final splits 的个数
  FinalSplitInfo finalSplits[0];
} FinalSplits;

inline std::string DumpFinalSplits(const FinalSplits& finalSplits) {
  std::string ret;
  ret += base::StringPrintf("num: %ld", finalSplits.num);
  for (int i = 0; i < finalSplits.num; ++i)
    ret += "\t" + DumpFinalSplitInfo(finalSplits.finalSplits[i]);
  return ret;
}

// 用于 LS 树的统计结果
typedef struct GammaStatInfo {
  double ySum;  // y 值之和
  double absySum;  // |y|(2-|y|) 值之和
  int32 nodeIdx;  // 叶子节点的 id
  int32 align;  // 内存对齐
} GammaStatInfo;

inline std::string DumpGammaStatInfo(const GammaStatInfo& gammaStatInfo) {
  return base::StringPrintf("ySum: %lf, absySum: %lf, nodeIdx: %d",
                            gammaStatInfo.ySum, gammaStatInfo.absySum, gammaStatInfo.nodeIdx);
}

typedef struct AllGammaStatInfo {
  int32 infoNum;
  int32 align;  // 内存对齐
  GammaStatInfo infos[0];  // stat info
} AllGammaStatInfo;

inline std::string DumpAllGammaStatInfo(const AllGammaStatInfo& info) {
  std::string ret;
  ret = base::StringPrintf("infoNum: %d", info.infoNum);
  for (int i = 0; i < info.infoNum; ++i)
    ret += "\t" + DumpGammaStatInfo(info.infos[i]);
  return ret;
};

struct SplitCondition {
  // double loss;
  int64 minNodes;  // 确保每个分裂节点不少于这个数
};

class Tree {
 public:
  Tree(int32 depth, int32 branch, const SplitCondition& splitCond)
      : kDepth(depth), kBranch(branch), kNodeNum((powl(branch, depth + 1) - 1) / (branch - 1)),
    splitConditions(splitCond) {
    nodes.resize(kNodeNum);
    nodes[0].splitValid = 1;
    currentDepth_ = 0;
    auto config = config::Config::GetConfig();
    for (int i = 1; i < kNodeNum; ++i) {
      nodes[i].splitValid = 0;
    }
    auto filename = config->treeName;
    //auto round = config->treeNum;
    //char s[4];
    //sprintf(s, "%ld", round);
    //filename.append(".");
    //filename.append(s);
    std::ofstream file(filename);
    file.close();
    kFile = filename;
  };
  virtual ~Tree() {
    /*
    */
  };
  // 重置树
  void Reset();
  void SplitNodes(FinalSplits* info);
  int32 MatchStopCondition(const FinalSplitInfo& info) const;
  // Only for branch = 3;
  void UpdateInstanceNode(comm::Instance* instance) const;
  void UpdateFx(comm::Instance* instance) const;
  virtual void CalculateGradient(comm::Instance* instance) const = 0;

  // L2 tree 请先转指针，然后再使用 L2 tree 中的这个函数, 注意接口有变化
  virtual void CalculateCoefficient(double kLearningRate) = 0;

  // 当前层起始节点 id
  virtual int32 TreeLayerStNode() const {
    return (std::pow(kBranch, currentDepth_) - 1) / (kBranch - 1);
  }

  // 当前层结束节点 id
  virtual int32 TreeLayerEdNode() const {
    return (std::pow(kBranch, currentDepth_ + 1) - 1) / (kBranch - 1) - 1;
  }

  // 当前层的节点个数
  virtual int32 TreeLayerNodeNums() const {
    return std::pow(kBranch, currentDepth_);
  }

  // 满树的叶子节点个数
  virtual int32 TreeLeafNodeNums() const {
    return std::pow(kBranch, kDepth);
  }

  virtual void SetCurrentDepth(int32 depth) {
    // CHECK(depth <= kDepth) << "depth larger than tree's depth";
    if (depth > kDepth) depth = kDepth;
    currentDepth_ = depth;
  }

  virtual int32 GetCurrentDepth() const {
    return currentDepth_;
  }

  const int32 kDepth;
  int32 kBranch;
  int64 kNodeNum;
  std::string kFile;
  SplitCondition splitConditions;
  // 更新数的时候记得更新深度
  std::vector<TreeNode> nodes;
  // std::vector<double> fea_influence;
  int32 DumpTree();
  // std::string SerializeTree(const Tree& tree);
  bool BuildTree(const std::string& Serialization);
  // void SaveTreeModel(const std::string& file);
  // void LoadTreeModel(const std::string& file, int32 n);
  double Predict(const comm::Instance& instance) const;
  double PredictTrace1(const comm::Instance& instance, int *node) const;
  double Predict(const std::vector<FeaInfo>& feaInfos, int32* node) const;
 protected:
  int32 DumpTreeLayer(int32 layer);
  bool BuildNode(const std::string& str, TreeNode* node) const;
  void SplitNode(FinalSplitInfo* info);
  int32 currentDepth_;
  DISALLOW_COPY_AND_ASSIGN(Tree);
};

class LS_Tree : public Tree {
 public:
  LS_Tree(uint32 depth, int32 branch, const SplitCondition& splitCond)
      : Tree(depth, branch, splitCond) {}
  virtual ~LS_Tree() {
  }
  virtual void CalculateGradient(comm::Instance* instance) const;
  virtual void CalculateCoefficient(double kLearningRate);
};

class L2_Tree : public Tree {
 public:
  L2_Tree(uint32 depth, int32 branch, const SplitCondition& splitCond)
      : Tree(depth, branch, splitCond) {}
  virtual ~L2_Tree() {
  }
  virtual void CalculateGradient(comm::Instance* instance) const;
  virtual void CalculateCoefficient(double kLearningRate);
  void CalculateCoefficient(double kLearningRate, const AllGammaStatInfo& gammaInfo);
};

class Booster {
 public:
  Booster();
  ~Booster();
  bool LoadTree(const std::string& filename);
  double Predict(comm::Instance* instance) const;
  //double PredictTrace(comm::Instance* instance, std::vector<int> &vecNodeList) const;
  double PredictTrace(comm::Instance* instance) const;
  double Predict(const std::vector<FeaInfo>& feaInfos, std::vector<int32>* nods) const;
 private:
  void ThreadPredict(comm::Instance* instance, int32 n, double* sum, std::atomic<int32>* counter) const;
  void ThreadPredictTrace(
      comm::Instance* instance,
      int32 n,
      double* sum,
      std::atomic<int32>* counter,
      int *node) const;
  int32 treeNum_;
  int32 treeDepth_;
  int32 treeBranch_;
  SplitCondition splitCond_;
  std::vector<boost::shared_ptr<Tree> > trees_;
  thread::Mutex lock_;
  boost::shared_ptr<thread::ThreadPool> pool_;
};

}
}
