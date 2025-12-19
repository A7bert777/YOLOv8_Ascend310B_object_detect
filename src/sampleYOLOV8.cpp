#include <dirent.h>
#include <opencv2/opencv.hpp>
#include "AclLiteUtils.h"
#include "AclLiteImageProc.h"
#include "AclLiteResource.h"
#include "AclLiteError.h"
#include "AclLiteModel.h"
#include "label.h"
#include <chrono>

using namespace std;
using namespace cv;
typedef enum Result
{
    SUCCESS = 0,
    FAILED = 1
} Result;

typedef struct BoundBox
{
    float x;
    float y;
    float width;
    float height;
    float score;
    size_t classIndex;
    size_t index;
} BoundBox;

bool sortScore(BoundBox box1, BoundBox box2)
{
    return box1.score > box2.score;
}

class SampleYOLOV8
{
public:
    SampleYOLOV8(const char *modelPath, const int32_t modelWidth, const int32_t modelHeight);
    Result InitResource();
    Result ProcessInput(string testImgPath);
    Result Inference(std::vector<InferenceOutput> &inferOutputs);
    Result GetResult(std::vector<InferenceOutput> &inferOutputs, string imagePath, size_t imageIndex, bool release);
    ~SampleYOLOV8();

private:
    void ReleaseResource();
    AclLiteResource aclResource_;
    AclLiteImageProc imageProcess_;
    AclLiteModel model_;
    aclrtRunMode runMode_;
    ImageData resizedImage_;
    const char *modelPath_;
    int32_t modelWidth_;
    int32_t modelHeight_;
};

SampleYOLOV8::SampleYOLOV8(const char *modelPath, const int32_t modelWidth, const int32_t modelHeight) : modelPath_(modelPath), modelWidth_(modelWidth), modelHeight_(modelHeight)
{
}

SampleYOLOV8::~SampleYOLOV8()
{
    ReleaseResource();
}

Result SampleYOLOV8::InitResource()
{
    // init acl resource
    AclLiteError ret = aclResource_.Init();
    if (ret == FAILED)
    {
        ACLLITE_LOG_ERROR("resource init failed, errorCode is %d", ret);
        return FAILED;
    }

    ret = aclrtGetRunMode(&runMode_);
    if (ret == FAILED)
    {
        ACLLITE_LOG_ERROR("get runMode failed, errorCode is %d", ret);
        return FAILED;
    }

    // init dvpp resource
    ret = imageProcess_.Init();
    if (ret == FAILED)
    {
        ACLLITE_LOG_ERROR("imageProcess init failed, errorCode is %d", ret);
        return FAILED;
    }

    // load model from file
    ret = model_.Init(modelPath_);
    if (ret == FAILED)
    {
        ACLLITE_LOG_ERROR("model init failed, errorCode is %d", ret);
        return FAILED;
    }
    return SUCCESS;
}

Result SampleYOLOV8::ProcessInput(string testImgPath)
{
    // read image from file
    ImageData image;
    AclLiteError ret = ReadJpeg(image, testImgPath);
    if (ret == FAILED)
    {
        ACLLITE_LOG_ERROR("ReadJpeg failed, errorCode is %d", ret);
        return FAILED;
    }

    // copy image from host to dvpp
    ImageData imageDevice;
    ret = CopyImageToDevice(imageDevice, image, runMode_, MEMORY_DVPP);
    if (ret == FAILED)
    {
        ACLLITE_LOG_ERROR("CopyImageToDevice failed, errorCode is %d", ret);
        return FAILED;
    }

    // image decoded from JPEG format to YUV
    ImageData yuvImage;
    ret = imageProcess_.JpegD(yuvImage, imageDevice);
    if (ret == FAILED)
    {
        ACLLITE_LOG_ERROR("Convert jpeg to yuv failed, errorCode is %d", ret);
        return FAILED;
    }

    // zoom image to modelWidth_ * modelHeight_
    ret = imageProcess_.Resize(resizedImage_, yuvImage, modelWidth_, modelHeight_);
    if (ret == FAILED)
    {
        ACLLITE_LOG_ERROR("Resize image failed, errorCode is %d", ret);
        return FAILED;
    }
    return SUCCESS;
}

Result SampleYOLOV8::Inference(std::vector<InferenceOutput> &inferOutputs)
{
    // create input data set of model
    AclLiteError ret = model_.CreateInput(static_cast<void *>(resizedImage_.data.get()), resizedImage_.size);
    if (ret == FAILED)
    {
        ACLLITE_LOG_ERROR("CreateInput failed, errorCode is %d", ret);
        return FAILED;
    }

    // inference
    ret = model_.Execute(inferOutputs);
    if (ret != ACL_SUCCESS)
    {
        ACLLITE_LOG_ERROR("execute model failed, errorCode is %d", ret);
        return FAILED;
    }

    return SUCCESS;
}

Result SampleYOLOV8::GetResult(std::vector<InferenceOutput> &inferOutputs,
                               string imagePath, size_t imageIndex, bool release)
{
    uint32_t outputDataBufId = 0;
    float *classBuff = static_cast<float *>(inferOutputs[outputDataBufId].data.get());
    // confidence threshold
    float confidenceThreshold = 0.35;

    // class number
    size_t classNum = 7;

    //// number of (x, y, width, hight)
    size_t offset = 4;

    // total number of boxs yolov8 [1,84,8400]
    size_t modelOutputBoxNum = 8400; 

    // read source image from file
    cv::Mat srcImage = cv::imread(imagePath);
    int srcWidth = srcImage.cols;
    int srcHeight = srcImage.rows;

    // filter boxes by confidence threshold
    vector<BoundBox> boxes;
    size_t yIndex = 1;
    size_t widthIndex = 2;
    size_t heightIndex = 3;

    // size_t all_num = 1 * 84 * 8400 ; // 705,600

    for (size_t i = 0; i < modelOutputBoxNum; ++i)
    {

        float maxValue = 0;
        size_t maxIndex = 0;
        for (size_t j = 0; j < classNum; ++j)
        {

            float value = classBuff[(offset + j) * modelOutputBoxNum + i];
            if (value > maxValue)
            {
                // index of class
                maxIndex = j;
                maxValue = value;
            }
        }

        if (maxValue > confidenceThreshold)
        {
            BoundBox box;
            box.x = classBuff[i] * srcWidth / modelWidth_;
            box.y = classBuff[yIndex * modelOutputBoxNum + i] * srcHeight / modelHeight_;
            box.width = classBuff[widthIndex * modelOutputBoxNum + i] * srcWidth / modelWidth_;
            box.height = classBuff[heightIndex * modelOutputBoxNum + i] * srcHeight / modelHeight_;
            box.score = maxValue;
            box.classIndex = maxIndex;
            box.index = i;
            if (maxIndex < classNum)
            {
                boxes.push_back(box);
            }
        }
    }

    ACLLITE_LOG_INFO("filter boxes by confidence threshold > %f success, boxes size is %ld", confidenceThreshold,boxes.size());

    // filter boxes by NMS
    vector<BoundBox> result;
    result.clear();
    float NMSThreshold = 0.45;
    int32_t maxLength = modelWidth_ > modelHeight_ ? modelWidth_ : modelHeight_;
    std::sort(boxes.begin(), boxes.end(), sortScore);
    BoundBox boxMax;
    BoundBox boxCompare;
    while (boxes.size() != 0)
    {
        size_t index = 1;
        result.push_back(boxes[0]);
        while (boxes.size() > index)
        {
            boxMax.score = boxes[0].score;
            boxMax.classIndex = boxes[0].classIndex;
            boxMax.index = boxes[0].index;

            // translate point by maxLength * boxes[0].classIndex to
            // avoid bumping into two boxes of different classes
            boxMax.x = boxes[0].x + maxLength * boxes[0].classIndex;
            boxMax.y = boxes[0].y + maxLength * boxes[0].classIndex;
            boxMax.width = boxes[0].width;
            boxMax.height = boxes[0].height;

            boxCompare.score = boxes[index].score;
            boxCompare.classIndex = boxes[index].classIndex;
            boxCompare.index = boxes[index].index;

            // translate point by maxLength * boxes[0].classIndex to
            // avoid bumping into two boxes of different classes
            boxCompare.x = boxes[index].x + boxes[index].classIndex * maxLength;
            boxCompare.y = boxes[index].y + boxes[index].classIndex * maxLength;
            boxCompare.width = boxes[index].width;
            boxCompare.height = boxes[index].height;

            // the overlapping part of the two boxes
            float xLeft = max(boxMax.x, boxCompare.x);
            float yTop = max(boxMax.y, boxCompare.y);
            float xRight = min(boxMax.x + boxMax.width, boxCompare.x + boxCompare.width);
            float yBottom = min(boxMax.y + boxMax.height, boxCompare.y + boxCompare.height);
            float width = max(0.0f, xRight - xLeft);
            float hight = max(0.0f, yBottom - yTop);
            float area = width * hight;
            float iou = area / (boxMax.width * boxMax.height + boxCompare.width * boxCompare.height - area);

            // filter boxes by NMS threshold
            if (iou > NMSThreshold)
            {
                boxes.erase(boxes.begin() + index);
                continue;
            }
            ++index;
        }
        boxes.erase(boxes.begin());
    }

    ACLLITE_LOG_INFO("filter boxes by NMS threshold > %f success, result size is %ld", NMSThreshold,result.size());
 
    // opencv draw label params
    const double fountScale = 0.5;
    const uint32_t lineSolid = 2;
    const uint32_t labelOffset = 11;
    const cv::Scalar fountColor(0, 0, 255); // BGR
    const vector<cv::Scalar> colors{
        cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0),
        cv::Scalar(0, 0, 255)};

    int half = 2;
    for (size_t i = 0; i < result.size(); ++i)
    {
        cv::Point leftUpPoint, rightBottomPoint;
        leftUpPoint.x = result[i].x - result[i].width / half;
        leftUpPoint.y = result[i].y - result[i].height / half;
        rightBottomPoint.x = result[i].x + result[i].width / half;
        rightBottomPoint.y = result[i].y + result[i].height / half;
        cv::rectangle(srcImage, leftUpPoint, rightBottomPoint, colors[i % colors.size()], lineSolid);
        string className = label[result[i].classIndex];
        string markString = to_string(result[i].score) + ":" + className;

        ACLLITE_LOG_INFO("object detect [%s] success", markString.c_str());

        cv::putText(srcImage, markString, cv::Point(leftUpPoint.x, leftUpPoint.y + labelOffset),
                    cv::FONT_HERSHEY_COMPLEX, fountScale, fountColor);
    }
    string savePath = "out_" + to_string(imageIndex) + ".jpg";
    cv::imwrite(savePath, srcImage);
    if (release)
    {
        free(classBuff);
        classBuff = nullptr;
    }
    return SUCCESS;
}

void SampleYOLOV8::ReleaseResource()
{
    model_.DestroyResource();
    imageProcess_.DestroyResource();
    aclResource_.Release();
}

int main()
{
    const char *modelPath = "../model/construction_yolov8n_best.om";
    const string imagePath = "../data";
    const int32_t modelWidth = 640;
    const int32_t modelHeight = 640;

    // all images in dir
    DIR *dir = opendir(imagePath.c_str());
    if (dir == nullptr)
    {
        ACLLITE_LOG_ERROR("file folder does no exist, please create folder %s", imagePath.c_str());
        return FAILED;
    }
    vector<string> allPath;
    struct dirent *entry;
    while ((entry = readdir(dir)) != nullptr)
    {
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0 || strcmp(entry->d_name, ".keep") == 0)
        {
            continue;
        }
        else
        {
            string name = entry->d_name;
            string imgDir = imagePath + "/" + name;
            allPath.push_back(imgDir);
        }
    }
    closedir(dir);

    if (allPath.size() == 0)
    {
        ACLLITE_LOG_ERROR("the directory is empty, please download image to %s", imagePath.c_str());
        return FAILED;
    }

    // inference
    string fileName;
    bool release = false;
    SampleYOLOV8 sampleYOLO(modelPath, modelWidth, modelHeight);
    Result ret = sampleYOLO.InitResource();
    if (ret == FAILED)
    {
        ACLLITE_LOG_ERROR("InitResource failed, errorCode is %d", ret);
        return FAILED;
    }

    for (size_t i = 0; i < allPath.size(); i++)
    {
        if (allPath.size() == i)
        {
            release = true;
        }
        std::vector<InferenceOutput> inferOutputs;
        fileName = allPath.at(i).c_str();
        std::chrono::time_point<std::chrono::steady_clock> start = std::chrono::steady_clock::now();
        ret = sampleYOLO.ProcessInput(fileName);
        if (ret == FAILED)
        {
            ACLLITE_LOG_ERROR("ProcessInput image failed, errorCode is %d", ret);
            return FAILED;
        }
        
        ret = sampleYOLO.Inference(inferOutputs);
        if (ret == FAILED)
        {
            ACLLITE_LOG_ERROR("Inference failed, errorCode is %d", ret);
            return FAILED;
        }

        ret = sampleYOLO.GetResult(inferOutputs, fileName, i, release);
        if (ret == FAILED)
        {
            ACLLITE_LOG_ERROR("GetResult failed, errorCode is %d", ret);
            return FAILED;
        }
        std::chrono::time_point<std::chrono::steady_clock> end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        ACLLITE_LOG_INFO("Inference elapsed time : %f s , fps is %f", elapsed.count(), 1 / elapsed.count());
    }
    return SUCCESS;
}