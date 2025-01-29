#pragma once
#ifndef CREATEDIRECTORY_H
#define CREATEDIRECTORY_H

#include <string>
#include <windows.h>

// 创建目录函数
inline bool createDirectory(const std::string& path)
{
    std::wstring widePath(path.begin(), path.end());
    return CreateDirectory(widePath.c_str(), NULL) || GetLastError() == ERROR_ALREADY_EXISTS;
}

#endif // CREATEDIRECTORY_H
