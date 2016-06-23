#pragma once

#include <iostream>
#include <cstdio>
#include <cstdatomic>
#include <shared_ptr>

#include "../../common/include/base/logging.h"
#include "../../common/include/base/threadpool.h"

#define FREE(p)    \
    do {                    \
      if (NULL != p) {    \
        free(p);        \
        p = NULL;       \
      }                   \
    } while (0);
#define DELETE(p)  \
  do {                    \
    if (NULL != p) {    \
      delete p;       \
      p = NULL;       \
    }                   \
  } while(0);


#define MALLOC(num,type) (type *)alloc( (num) * sizeof(type) )

namespace DML {
namespace gbdt {

class Tree;

struct Point {
  float feaValue;
}

inline const std::string& DumpPoint(const Point& point) {
  char tmpStr[20];
  sprintf(tmpStr, "%s", point.feaValue);
  return std::string(tmpStr);
}

}
}
