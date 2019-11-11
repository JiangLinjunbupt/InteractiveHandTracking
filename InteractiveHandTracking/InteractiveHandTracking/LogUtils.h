#ifndef _LOG_UTILS_H_
#define _LOG_UTILS_H_

#include <stdio.h>
#include <string.h>

#define DEBUG // log开关

#define __FILENAME__ (strrchr(__FILE__, '/') + 1) // 文件名

#ifdef DEBUG
#define LOGD(format, ...) printf_s("[%s][%s][%d]: " format "\n", __FILENAME__, __FUNCTION__,\
                            __LINE__, ##__VA_ARGS__)
#else
#define LOGD(format, ...)
#endif

#endif // _LOG_UTILS_H_