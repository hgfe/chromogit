数模研赛招募
========
已有队员
----
两名电院自动化系18级直博生，专业为模式识别/图像处理相关，有数模参赛经历，其中一人CUMCM2016国二，熟悉Python、Matlab、Latex，可负责编程和写作

招募队员
-----
希望能找一名最好是数学相关专业的，可以帮助建模，有时间、好沟通的队友，有数模竞赛经历者更佳<br>
（微信号big_hamster）



染色体项目代码
=====

0126
----
1.增加了十字分割moduleCrossSeg的异常处理<br>
2.增加了dll的版本信息<br>
3.对流程图进行了更新<br>

0123
----
1.修改了module0、module1、moduleOpti的部分算法<br>
2.增加了代码中的部分异常处理<br>
3.染色体chromo中的position结构体增加了字段：int cImgBoundingBoxOffset[4]<br>
作用：cImg旋转并填充之后，需要进行截取才能得到cImgRotated，这个字段表征cImgRotated在截取前图像中的位置。<br>
4.染色体cImgType字段bool类型改为了int类型，其中0表示交叉染色体cluster，1表示单条染色体single，2表示possible_cluster，顶层可以将2视作1，即为单条染色体<br>

0105
----
增加了moduleChooseROI模块

1226
----
修改了module1和module0的接口，修改了module1的去噪部分，针对去核做了相应修改

1204
----
修改了module1自动分割模块，针对十字交叉型染色体尽可能按照对角线的形式切割；
修改了moduleCrossSeg模块中的函数，不影响使用和切割效果

1126
----
工程中的源代码保存在code文件夹中，包括moduleOpti优化模块、module0打分模块、module1自动分割模块、
module2手动拼接模块、module3手动分割模块、moduleCrossSeg十字分割模块。<br>
opencv版本为3.4.0<br>
dll包括release版本和debug版本<br>

1122
----
1.修复了优化模块中多次执行结果不同的问题<br>
2.加入了自动分割的异常处理<br>

1115
----
1.对Module优化模块进行了转码；效果有较好提升<br>
2.修复了自动分割和打分模块的bug<br>

1108
----
解决了模块0打分针对AIpng报错的问题<br>

1107
----
模块3多骨架分割的部分进行了较大改进<br>

1106
----
1.模块0和模块1的预处理部分进行了优化<br>

1105
----
1.十字交叉分割问题修复bug<br>

1102
----
模块3手动分割增加了多骨架分割的部分<br>
接口函数进行了相应修改<br>
int ManualSegment(chromo chromoData, vector<chromo>& newChromoDataList, vector<Mat> allLines, Point clickPosi, int segType, int newCutNum);<br>


1031
----
模块1自动分割修复了最后将cluster分割后mask显示不全的bug<br>

1030
----
十字交叉分割问题修复<br>
1.修复cImgType的问题<br>
2.修复部分染色体切割后没有旋转的问题<br>
3.更新正确坐标<br>

1029
----
1.打分模块重新拟合<br>
2.十字交叉分割测试完毕<br>

各个模块接口参数
-----
模块0接口函数：打分<br>
int moduleScoring(Mat originPicture, String pictureType, float & pictureScore, int & singleNum);<br>
@param<br>
originPicture 原始图像，Mat类型<br>
pictureType 图像类型，raw或tif<br>
pictureScore 输出打分，float类型<br>
singleNum 单条染色体数量，int类型<br>

模块1接口函数：自动分割<br>
int moduleSeg(Mat originPicture, String pictureType, String patientId, String glassId, String karyoId,
	Mat& optiPicture, String& optiPictureType, vector<chromo>& chromoData);<br>
@param<br>
originPicture 原始图像，Mat类型<br>
pictureType 图像类型，raw或tif<br>
patientId 患者编号<br>
glassId 玻片编号<br>
karyoId 核型编号<br>
optiPicture 优化后输出图像，Mat类型<br>
optiPictureType 优化后输出图像类型，raw或tif<br>
chromoData 染色体数据结构体列表，vector<chromo>类型<br>


模块2接口函数：手动拼接<br>
int moduleSplit(Mat originPicture, String pictureType, vector<chromo> chromoDataArray, int newCutIndex,
	chromo& chromoData);<br>
@param<br>
originPicture 原始图像，Mat类型<br>
pictureType 图像类型，raw或tif<br>
chromoDataArray 输入染色体数据结构体列表，vector<chromo>类型<br>
newCutIndex 新染色体的起始编号，int类型<br>
chromoData 输出染色体数据结构体，chromo类型<br>

模块3接口函数：手动分割<br>
int ManualSegment(chromo chromoData, vector<chromo>& newChromoDataList, vector<Mat> allLines, Point clickPosi, int segType, int newCutNum);<br>
@param<br>
chromoData 输入染色体数据结构体，chromo类型<br>
newChromoDataList 输出染色体数据结构体列表，vector<chromo>类型（每条切出的染色体是一个chromo）<br>
allLines 输入的画线列表，vector<Mat>类型（边缘切割时只有一个元素allLines[0]，骨架切割时每条画线独立为一个allLines[i]元素）<br>
clickPosi 鼠标点击坐标，Point类型<br>
segType 分割方式，0为边缘，1为骨架<br>
newCutNum 分割出的染色体起始编号<br>

模块CrossSeg接口函数：十字交叉分割<br>
int moduleCrossSeg(chromo chromoData, int newCutNum, vector<vector<chromo>>& chromoDataList);<br>

@param<br>
chromoData 输入染色体数据结构体，chromo类型<br>
newCutNum 分割出的染色体起始编号<br>
chromoDataList 输出的分割结果<br>

模块Opti接口函数：优化<br>
Mat moduleOpti(Mat originPicture, String pictureType);<br>

所有输出为int的函数均为errorCode，0为正常，1为异常<br>
