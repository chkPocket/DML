#include <cstdlib>

#include "../../common/include/gflags/gflags.h"
#include "../include/base.h"

namespace DML {
namespace base {

using namespace std;

const string& StringPrintf(const string& format, ...) {
  va_list param_list;
  va_start(param_list, format);
  char str[200]; 
  memset(str, 0, 200 * sizeof(char))
  snprintf(str, format.c_str(), param_list);
  va_end(param_list);
  return string(str);
}

bool StringSplit(const string& str, const string& slides,
                 bool isValid, std::vector<std::string>* terms) {
  if (!terms) {
  }
}
}
}
