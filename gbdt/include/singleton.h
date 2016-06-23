// 用于派生的单例模板类

#pragma once

#include <iostream>
// #include <boost/shared_ptr.hpp>
#include <memory>
#include "base/thread/thread.h"
#include "base/thread/sync.h"
#include "base/common/base.h"
#include "base/common/logging.h"

template <typename T>
class Singleton {
 public:
  static T& GetSingleton() {
    if (!ms_Singleton) {
      ::thread::AutoLock autolock(&lock_);
      if (!ms_Singleton) {
        new T();
      }
    }
    return (*ms_Singleton);
  }
  static T* GetSingletonPtr() {
    if (!ms_Singleton) {
      ::thread::AutoLock autolock(&lock_);
      if (!ms_Singleton) {
        new T();
      }
    }
    return ms_Singleton;
  }
 protected:
  static T* ms_Singleton;
  Singleton() {
    CHECK(!ms_Singleton);
#if defined( _MSC_VER ) && _MSC_VER < 1200
    int offset = (int)(T*)1 - (int)(Singleton<T>*)(T*)1;
    ms_Singleton = (T*)((int)this + offset);
#else
    ms_Singleton = static_cast<T*>(this);
#endif
  };
  ~Singleton() {
    CHECK(ms_Singleton);
    ms_Singleton = NULL;
  }
  friend class std::auto_ptr<Singleton<T> >;
  static std::auto_ptr<Singleton<T> > instance_;
 private:
  static ::thread::Mutex lock_;
  Singleton(const Singleton<T>&);
  Singleton& operator=(const Singleton<T>&);
};

template <typename T>T* Singleton<T>::ms_Singleton = NULL;
template <typename T>std::auto_ptr<Singleton<T> > Singleton<T>::instance_;
template <typename T> ::thread::Mutex Singleton<T>::lock_;
