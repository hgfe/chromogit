#pragma once
#include "conn.h"
#include "main.h"

extern "C" __declspec(dllexport) int moduleScoring(Mat originPicture, String pictureType,
	float & pictureScore, int & singleNum);