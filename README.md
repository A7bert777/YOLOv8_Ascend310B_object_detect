# YOLOv8_Ascend310B_object_detect

CSDN地址：[【YOLOv8部署至Ascend 310B】模型训练→转换om→Atlas部署](https://blog.csdn.net/A_l_b_ert/article/details/156068229?spm=1001.2014.3001.5502)

QQ咨询（not free）：2506245294

# 目标检测仓库

切记：一定要在Ascend系列开发板上运行，不要在虚拟机上跑，ARM64和X86不一样！

1.项目代码介绍

src/sampleYOLOV8.cpp ：主程序运行文件

src/label.h: 标签名设置

data 中是第三方库

model 是模型位置


3.编译运行


**①cd src**

**②cmake .**

**③make**

**④cd ../out**

**⑤./main**

此处统一说明：加QQ后直接说问题和小星星截图，对于常见的相同问题，很多都已在CSDN博客中提到了，已在评论中详细解释过的问题，不予回复。
