float ChromoScore(float avgLength, int singleNum) {
	float number = (float)singleNum;
	// 配置多项式系数
	float score = 0;
	float a = 1.279;
	float b1 = 0.774, b2 = 0.221;
	float c1 = 3.324, c2 = -2.865;
	//输入数据归一化
	float x = avgLength / 3178.7;
	float y = number / 48;
	//计算打分
	score = a + b1*x + b2*x*x + c1*y + c2*y*y;
	return score;
}