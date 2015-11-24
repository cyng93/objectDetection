/**
 * @file objectDetection.cpp
 * @author A. Huaman ( based in the classic facedetect.cpp in samples/c )
 * @brief A simplified version of facedetect.cpp, show how to load a cascade classifier and how to find objects (Face + eyes) in a video stream
 */
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>
#include <iomanip>

#include <pthread.h>
#include <sys/sysinfo.h>

#define scaleInput 40        // WIDTH=16*n    HEIGHT=9*n
#define jumpFrame 600        // uncomment to detect all frame
#define outputFrame         // uncomment to disable writing frame to directory
#define numOfTolerant 2
#define multiDetect         // uncomment to detect with only one classifier

#define parallel            // uncomment to run objectDetection in sequential (PENDING..)
#define classifier_read     // uncomment to load classifier in each thread instead of read

#define taskset 16          // correct the numOfCore when program is run with taskset


using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay( Mat frame );
void *handler(void* parameters);

/** Global variables */
//-- Note, either copy these two files from opencv/data/haarscascades to your current folder, or change these locations
String obj_cascade_name = "classifier/frontal_pos3000_stg14.xml";
String obj2_cascade_name = "classifier/frontal_pos3000_stg20.xml";
String obj3_cascade_name = "classifier/frontal_pos3000_stg30.xml";


string window_name = "Object detection";
RNG rng(12345);

unsigned long frameCount = 0;
unsigned long numOfFrame = 0;
unsigned long numOfCorrectFrame = 0;
#ifdef multiDetect
unsigned long numOfCorrectFrame_2 = 0;
unsigned long numOfCorrectFrame_3 = 0;

#endif
String videoFilename = "testVideo/oriVideo.mov";

#ifdef outputFrame
String outputFilePrefix  = "./outputFrame/frame_";
String outputFileType = ".png";
stringstream ss;
String outputFilename;
#endif

#ifdef parallel
typedef struct partialCorrect_padding_t{
    unsigned long threadPartialCorrect;     // 8 bytes
 #ifdef multiDetect
    unsigned long threadPartialCorrect_2;   // 8 bytes
    unsigned long threadPartialCorrect_3;   // 8 bytes
    int padding[10];                        // 40 bytes
 #else
    int padding[14];                        // 56 bytes
 #endif
}threadPartialCorrect_t;

int numOfCores = 0;
int curThreadIndex = 0;
pthread_mutex_t threadIndexLock;
threadPartialCorrect_t *threadPartialCorrect;
int *retVal;
#endif

#ifdef classifier_read
FileStorage fs1(obj_cascade_name, cv::FileStorage::READ);
FileStorage fs2(obj2_cascade_name, cv::FileStorage::READ);
FileStorage fs3(obj3_cascade_name, cv::FileStorage::READ);
#endif

/**
 * @function main
 */
int main( int argc, char **argv  )
{
    int i;
    pthread_t *threadPool;

    numOfCores = get_nprocs();
#ifdef taskset
    numOfCores = (numOfCores < taskset) ? numOfCores : taskset;
#endif

    //-- 0. Get Num Of Frames
    VideoCapture capture( videoFilename );
    if( !capture.isOpened() ){
        cout << "Fail to open video file" << endl;
        return -1;
    }
    numOfFrame = capture.get(CV_CAP_PROP_FRAME_COUNT);
    capture.release();

#ifdef jumpFrame
    unsigned long numOfDetectFrame = numOfFrame/jumpFrame;
    cout << " Frame Jumping Rate : " << jumpFrame << endl;
#else
    unsigned long numOfDetectFrame = numOfFrame;
#endif
    cout << " Detecting [ " << numOfDetectFrame << " / " << numOfFrame << " ] frames ..." <<  endl;

    pthread_mutex_init(&threadIndexLock, NULL);
    threadPartialCorrect = (threadPartialCorrect_t *)malloc(numOfCores * sizeof(threadPartialCorrect_t));
    threadPool = (pthread_t *)malloc(numOfCores * sizeof(pthread_t));

    for(i=0; i<numOfCores; i++){
        threadPartialCorrect[i].threadPartialCorrect = 0;
    #ifdef multiDetect
        threadPartialCorrect[i].threadPartialCorrect_2 = 0;
        threadPartialCorrect[i].threadPartialCorrect_3 = 0;
    #endif

    }

    // thread creation
    for(i=0; i<numOfCores; i++)
        pthread_create(&threadPool[i], NULL, handler, NULL);
    // thread join
    for(i=0; i<numOfCores; i++)
            pthread_join(threadPool[i], NULL);

    for(i=0; i<numOfCores; i++){
        numOfCorrectFrame += threadPartialCorrect[i].threadPartialCorrect;
    #ifdef multiDetect
        numOfCorrectFrame_2 += threadPartialCorrect[i].threadPartialCorrect_2;
        numOfCorrectFrame_3 += threadPartialCorrect[i].threadPartialCorrect_3;
    #endif
    }


    cout << "[Classifier 1] #Correct : [ " << numOfCorrectFrame << " / " << numOfDetectFrame << " ]" << endl;
#ifdef multiDetect
    cout << "[Classifier 2] #Correct : [ " << numOfCorrectFrame_2 << " / " << numOfDetectFrame << " ]" << endl;
    cout << "[Classifier 3] #Correct : [ " << numOfCorrectFrame_3 << " / " << numOfDetectFrame << " ]" << endl;
#endif

    pthread_mutex_destroy(&threadIndexLock);
    free(threadPool);
    free(threadPartialCorrect);

    return 0;
}



void * handler(void* parameters)
{
    int myThreadIndex;
    unsigned long framePerThread = numOfFrame/numOfCores;
    unsigned long frameCount;
    unsigned long partialCorrectFrame;

    VideoCapture capture;
    Mat frame;
    CascadeClassifier obj_cascade, obj2_cascade, obj3_cascade;

    // each thread getting their own index
    pthread_mutex_lock(&threadIndexLock);
    myThreadIndex = curThreadIndex;
    curThreadIndex++;
    pthread_mutex_unlock(&threadIndexLock);

    frameCount = (myThreadIndex * framePerThread);

    //-- 1. Read / Load the cascades
#ifdef classifier_read
    if( !obj_cascade.read( fs1.getFirstTopLevelNode() ) ){ printf("--(!)Error loading\n"); *retVal=-1; return (void*)retVal; }
        if( !obj2_cascade.read( fs2.getFirstTopLevelNode() ) ){ printf("--(!)Error loading\n"); *retVal=-1; return (void*)retVal; }
            if( !obj3_cascade.read( fs3.getFirstTopLevelNode() ) ){ printf("--(!)Error loading\n"); *retVal=-1; return (void*)retVal; }
#else
    if( !obj_cascade.load( obj_cascade_name ) ){ printf("--(!)Error loading\n"); *retVal=-1; return (void*)retVal; }
    if( !obj2_cascade.load( obj2_cascade_name ) ){ printf("--(!)Error loading\n"); *retVal=-1; return (void*)retVal; }
    if( !obj3_cascade.load( obj3_cascade_name ) ){ printf("--(!)Error loading\n"); *retVal=-1; return (void*)retVal; }
#endif

    //-- 2. Read the video stream
    capture.open( videoFilename );

    if( !capture.isOpened() ){
        cout << "Fail to open video file" << endl;
        *retVal = -1;
        return (void *)retVal;
    }

    numOfFrame = capture.get(CV_CAP_PROP_FRAME_COUNT);

#ifdef scaleInput
    capture.set(CV_CAP_PROP_FRAME_WIDTH, 16 * scaleInput);      // Ratio = 16 : 9
    capture.set(CV_CAP_PROP_FRAME_HEIGHT, 9 * scaleInput);
#endif

    for(;;){
    #ifdef jumpFrame
	if(! capture.set(CV_CAP_PROP_POS_FRAMES, frameCount += jumpFrame)) { cout << "error jumpFrame"; *retVal=-1; return (void*)retVal; }
    #else
        frameCount++;
    #endif
	if( frameCount >= (myThreadIndex+1)*framePerThread || !capture.read(frame) )  // frameCount check
            break;
    #ifdef scaleInput
	resize(frame, frame, Size( 16 * scaleInput, 9 * scaleInput), 0, 0, INTER_CUBIC);
    #endif
	//-- 3. Apply the classifier to the frame
	if( frame.empty() )
        { printf(" --(!) No captured frame -- Break!"); break; }
        else
        {
            std::vector<Rect> objs, objs2, objs3;
            Mat frame_gray;

            cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
            equalizeHist( frame_gray, frame_gray );
            //-- Detect objs
            obj_cascade.detectMultiScale( frame_gray, objs, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
        #ifdef multiDetect
            obj2_cascade.detectMultiScale( frame_gray, objs2, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
            obj3_cascade.detectMultiScale( frame_gray, objs3, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
        #endif


            cout << "[ frame #" << frameCount << " ]"
            << endl << "\t" << obj_cascade_name << ": " << objs.size()
        #ifdef multiDetect
            << endl << "\t" << obj2_cascade_name << ": " << objs2.size()
            << endl << "\t" << obj3_cascade_name << ": " << objs3.size()
        #endif
            << endl;

            for( size_t i = 0; i < objs.size() ; i++ )
            {
        	Point upperLeft_1( objs[i].x, objs[i].y );
        	Point bottomRight_1( objs[i].x + objs[i].width, objs[i].y + objs[i].height );
        	rectangle( frame, upperLeft_1, bottomRight_1, Scalar( 255, 0, 255 ), 2, 8, 0 );
            }
            if( objs.size() <= numOfTolerant)
                threadPartialCorrect[myThreadIndex].threadPartialCorrect ++;

        #ifdef multiDetect
            for( size_t i = 0; i < objs2.size() ; i++ )
            {
        	Point upperLeft_2( objs2[i].x, objs2[i].y );
        	Point bottomRight_2( objs2[i].x + objs2[i].width, objs2[i].y + objs2[i].height );
        	rectangle( frame, upperLeft_2, bottomRight_2, Scalar( 255, 255, 0 ), 2, 8, 0 );
            }
            if(objs2.size() <= numOfTolerant)
                threadPartialCorrect[myThreadIndex].threadPartialCorrect_2 ++;

            for( size_t i = 0; i < objs3.size() ; i++ )
            {
        	Point upperLeft_3( objs3[i].x, objs3[i].y );
        	Point bottomRight_3( objs3[i].x + objs3[i].width, objs3[i].y + objs3[i].height );
        	rectangle( frame, upperLeft_3, bottomRight_3, Scalar( 0, 255, 255 ), 2, 8, 0 );
            }
            if(objs3.size() <= numOfTolerant)
                threadPartialCorrect[myThreadIndex].threadPartialCorrect_3 ++;

        #endif
            //-- Show what you got
            //imshow( window_name, frame );

        #ifdef outputFrame
            ss << outputFilePrefix << setfill('0') << setw(5) << frameCount << outputFileType;
            outputFilename = ss.str();
            ss.str("");
            imwrite(outputFilename, frame);
        #endif

        }

	int c = waitKey(10);
	if( (char)c == 'c' ) { break; }
    }


}
