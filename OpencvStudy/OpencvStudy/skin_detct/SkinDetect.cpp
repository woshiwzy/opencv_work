#include "headers.hpp"
#include "Tool.hpp"

Mat input_image;
Mat output_mask;
Mat output_image;
Mat mask;


map<string, vector<vector<float>>> getMLdata(){
    
    //    int MAX_PATH=120;
    //    char   buffer[MAX_PATH];
    //    getcwd(buffer, MAX_PATH);
    //    cout<<string(buffer)<<endl;
    //
    int resizeRect=240;
    string rootdir="/Users/wangzy/workspace/opencv_work/OpencvStudy/OpencvStudy/ml_data";
    
    vector<string> files=Tool::listFiles(rootdir);
    map<string, vector<vector<float>>> dataMap;
    int sacnWidnowSize=resizeRect;
    
    HOGDescriptor *hog=new HOGDescriptor(Size(sacnWidnowSize,sacnWidnowSize),
                                         Size(sacnWidnowSize/2,sacnWidnowSize/2),
                                         Size(sacnWidnowSize/4,sacnWidnowSize/4),
                                         Size(sacnWidnowSize/4,sacnWidnowSize/4),9);
    
    for(size_t i=0,isize=files.size();i<isize;i++){
        string file=files.at(i);
        string absPath=string(rootdir+"/"+file);
        
        cout<<absPath<<endl;
        
        vector<vector<float>> descriptersVector;
        
        vector<string> subFiles=Tool::listFiles(absPath);
        if(!subFiles.empty()){
            cout<<"subfile count:"<<subFiles.size()<<endl;
            
            for(size_t j=0,jsize=subFiles.size();j<jsize;j++){
                string imgAbsPath=absPath+"/"+subFiles.at(j);
                Mat mat=imread(imgAbsPath);
                //             cout<<imgAbsPath<<" "<<mat.rows<<","<<mat.cols;
                Mat newMat(resizeRect,resizeRect,CV_8UC1);
                
                resize(mat, newMat, Size(resizeRect,resizeRect));//归一化，都变成240X240大小
                
                vector<float> descripters;
                hog->compute(newMat, descripters);
                
                descriptersVector.push_back(descripters);
                
            }
            
            dataMap[file]=descriptersVector;//存放到Map 中去
            
            
        }
    }
    
    delete hog;
    return dataMap;
}



int main()
{
    VideoCapture cam(0);
    if(!cam.isOpened())
        return 0;




    double fps;
    char fpst[32];  // 用于存放帧率的字符串
    double t = 0;

    RNG rng(12345);

    Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
    
   Ptr<KNearest> knn= KNearest::create();
    knn->setDefaultK(3);
    knn->setIsClassifier(true);
    
    Ptr<TrainData> tData= TrainData::create(dataMat, ROW_SAMPLE, labelMat);

    while(true)
    {

        t = (double)cv::getTickCount();//帧率开始

        cam >> input_image;

        flip(input_image, input_image, 1);

        if(input_image.empty())
            return 0;



        output_mask = Mat::zeros(input_image.size(), CV_8UC1);

        Tool::getSkinOutput(input_image, output_mask);


        vector< vector<Point> > contours;   // 轮廓
        vector< vector<Point> > filterContours; // 筛选后的轮廓
        vector< Vec4i > hierarchy;    // 轮廓的结构信息
        contours.clear();
        hierarchy.clear();
        filterContours.clear();

        findContours(output_mask, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

        vector<vector<Point>> hull(contours.size());
        // 去除伪轮廓
        for (size_t i = 0; i < contours.size(); i++)
        {

            Rect r=boundingRect(contours[i]);

            float whs=r.width*1.0f/r.height;
            float hws=r.height*1.0/r.width;

            long abs=fabs(contourArea(Mat(contours[i]))) ;

            int ow=output_mask.cols;
            int oh=output_mask.rows;


            if((r.width>100 && r.height>100)/*&&(whs>=0.6f && whs<=1.6f)*/&&(abs> 1000) && hws>0.9f ){
                
                filterContours.push_back(contours[i]);
                
                Tool::drawRectangle(output_image, r);
                Tool::drawRectangle(output_mask, r);
                
                int resizeRect=240;
                Mat newMat(resizeRect,resizeRect,CV_8UC1);
            
                resize(output_mask(r), newMat, Size(resizeRect,resizeRect));//归一化，都变成240X240大小
                
                
            
                
            }

        }

        output_mask.setTo(0);

        drawContours(output_mask, filterContours, -1, Scalar(255,0,0), CV_FILLED); //8, hierarchy);

        input_image.copyTo(output_image, output_mask);


        //====================
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        fps = 1.0 / t;
        sprintf(fpst, "fps:%.2f", fps);      // 帧率保留两位小数
               //帧率结束

        Tool::drawText(Point(100,100), string(fpst), output_mask);


        imshow("input image msk", output_mask);
//        imshow("input image", input_image);
        imshow("output image", output_image);

        output_image.setTo(0);
        if(27 == waitKey(30))
            return 0;
    }
    return 0;
}
