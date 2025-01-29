#pragma once
#ifndef HAAR_FACE_DETECTOR_H
#define HAAR_FACE_DETECTOR_H

#include <string>


void detectAndSaveFaces(const std::string& haarCascadePath, const std::string& savePath);

#endif // HAAR_FACE_DETECTOR_H
