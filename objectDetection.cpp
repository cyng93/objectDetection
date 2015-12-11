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
#define jumpFrame 30        // uncomment to detect all frame
#define outputFrame         // uncomment to disable writing frame to directory
#define numOfTolerant 2
#define multiDetect         // uncomment to detect with only one classifier

#define parallel            // uncomment to run objectDetection in sequential (PENDING..)
#define classifier_read     // uncomment to load classifier in each thread instead of read

#define taskset 4          // correct the numOfCore when program is run with taskset


using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay( Mat frame );
void *handler(void* parameters);

/** Global variables */
//-- Note, either copy these two files from opencv/data/haarscascades to your current folder, or change these locations
String obj_cascade_name_1 = "classifier/frontal_negSet1_pos7000_stg14.xml";
String obj_cascade_name_2 = "classifier/frontal_negSet2_pos7000_stg14.xml";
String obj_cascade_name_3 = "classifier/frontal_negSet1+negScene_pos7000_stg14.xml";
String obj_cascade_name_4 = "classifier/frontal_negSet1+negScene_pos7000_stg14.xml";

string window_name = "Object detection";
RNG rng(12345);

unsigned long long frameCount = 0;
unsigned long long numOfFrame = 0;
unsigned long long numOfObject = 0;

unsigned long long numOfHit_1 = 0;
unsigned long long numOfFalseDetect_1 = 0;
#ifdef multiDetect
unsigned long long numOfHit_2 = 0;
unsigned long long numOfFalseDetect_2 = 0;
unsigned long long numOfHit_3 = 0;
unsigned long long numOfFalseDetect_3 = 0;
unsigned long long numOfHit_4 = 0;
unsigned long long numOfFalseDetect_4 = 0;
#endif

String videoFilename = "testVideo/oriVideo.mov";
String answerFilename = "answer.txt";

#ifdef outputFrame
String outputFilePrefix  = "./outputFrame/frame_";
String outputFileType = ".png";
stringstream ss;
String outputFilename;
#endif

#ifdef parallel
typedef struct partialResult_padding_t{
    unsigned long long threadPartialNumOfObject;   // 8 bytes
    unsigned long long threadPartialHit_1;            // 8 bytes
    unsigned long long threadPartialFalseDetect_1;    // 8 bytes
 #ifdef multiDetect
    unsigned long long threadPartialHit_2;          // 8 bytes
    unsigned long long threadPartialFalseDetect_2;  // 8 bytes
    unsigned long long threadPartialHit_3;          // 8 bytes
    unsigned long long threadPartialFalseDetect_3;  // 8 bytes
    unsigned long long threadPartialHit_4;          // 8 bytes
    unsigned long long threadPartialFalseDetect_4;  // 8 bytes
    int padding[14];                                // 16 bytes
 #else
    int padding[10];                                // 40 bytes
 #endif
}threadPartialResult_t;

int numOfCores = 0;
int curThreadIndex = 0;
pthread_mutex_t threadIndexLock;
threadPartialResult_t *threadPartialResult;
int *retVal;
#endif

#ifdef classifier_read
FileStorage fs1(obj_cascade_name_1, cv::FileStorage::READ);
FileStorage fs2(obj_cascade_name_2, cv::FileStorage::READ);
FileStorage fs3(obj_cascade_name_3, cv::FileStorage::READ);
FileStorage fs4(obj_cascade_name_4, cv::FileStorage::READ);
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
    cout << " [SAMPLING RATE]  1 : " << jumpFrame << " frames" << endl;
#endif

    pthread_mutex_init(&threadIndexLock, NULL);
    threadPartialResult = (threadPartialResult_t *)malloc(numOfCores * sizeof(threadPartialResult_t));
    threadPool = (pthread_t *)malloc(numOfCores * sizeof(pthread_t));

    for(i=0; i<numOfCores; i++){
        threadPartialResult[i].threadPartialNumOfObject = 0;
        threadPartialResult[i].threadPartialHit_1 = 0;
        threadPartialResult[i].threadPartialFalseDetect_1 = 0;
    #ifdef multiDetect
        threadPartialResult[i].threadPartialHit_2 = 0;
        threadPartialResult[i].threadPartialFalseDetect_2 = 0;
        threadPartialResult[i].threadPartialHit_3 = 0;
        threadPartialResult[i].threadPartialFalseDetect_3 = 0;
        threadPartialResult[i].threadPartialHit_4 = 0;
        threadPartialResult[i].threadPartialFalseDetect_4 = 0;
    #endif

    }

    // thread creation
    for(i=0; i<numOfCores; i++)
        pthread_create(&threadPool[i], NULL, handler, NULL);
    // thread join
    for(i=0; i<numOfCores; i++)
            pthread_join(threadPool[i], NULL);

    for(i=0; i<numOfCores; i++){
        numOfObject += threadPartialResult[i].threadPartialNumOfObject;
        numOfHit_1 += threadPartialResult[i].threadPartialHit_1;
        numOfFalseDetect_1 += threadPartialResult[i].threadPartialFalseDetect_1;
    #ifdef multiDetect
        numOfHit_2 += threadPartialResult[i].threadPartialHit_2;
        numOfFalseDetect_2 += threadPartialResult[i].threadPartialFalseDetect_2;
        numOfHit_3 += threadPartialResult[i].threadPartialHit_3;
        numOfFalseDetect_3 += threadPartialResult[i].threadPartialFalseDetect_3;
        numOfHit_4 += threadPartialResult[i].threadPartialHit_4;
        numOfFalseDetect_4 += threadPartialResult[i].threadPartialFalseDetect_4;
    #endif
    }


    cout << "\n\n==================== DETECTION RESULT ====================" << endl;
    cout << "\n[Classifier 1] - PINK"
    << "\n  Classifer:\t" << obj_cascade_name_1
    << "\n  Hit Rate:\t" << numOfHit_1 << " / " << numOfObject
    << "\n  False Detect:\t" << numOfFalseDetect_1
    << "\n  Accuracy:\t" << (float)numOfHit_1/(numOfObject+numOfFalseDetect_1) << endl;

#ifdef multiDetect
    cout << "\n[Classifier 2] - BLUE"
    << "\n  Classifer:\t" << obj_cascade_name_2
    << "\n  Hit Rate:\t" << numOfHit_2 << " / " << numOfObject
    << "\n  False Detect:\t" << numOfFalseDetect_2
    << "\n  Accuracy:\t" << (float)numOfHit_2/(numOfObject+numOfFalseDetect_2) << endl;

    cout << "\n[Classifier 3] - YELLOW"
    << "\n  Classifer:\t" << obj_cascade_name_3
    << "\n  Hit Rate:\t" << numOfHit_3 << " / " << numOfObject
    << "\n  False Detect:\t" << numOfFalseDetect_3
    << "\n  Accuracy:\t" << (float)numOfHit_3/(numOfObject+numOfFalseDetect_3) << endl;

    cout << "\n[Classifier 4] - GREEN"
    << "\n  Classifer:\t" << obj_cascade_name_4
    << "\n  Hit Rate:\t" << numOfHit_4 << " / " << numOfObject
    << "\n  False Detect:\t" << numOfFalseDetect_4
    << "\n  Accuracy:\t" << (float)numOfHit_4/(numOfObject+numOfFalseDetect_4) << endl;
#endif

    pthread_mutex_destroy(&threadIndexLock);
    free(threadPool);
    free(threadPartialResult);

    return 0;
}



void * handler(void* parameters)
{
    int myThreadIndex;
    unsigned long long framePerThread = numOfFrame/numOfCores;
    unsigned long long frameCount;
    unsigned long long partialCorrectFrame;

    int *answer;
    unsigned long long i;       // for array purpose

    VideoCapture capture;
    Mat frame;
    CascadeClassifier obj_cascade_1, obj_cascade_2, obj_cascade_3, obj_cascade_4;

    // each thread getting their own index
    pthread_mutex_lock(&threadIndexLock);
    myThreadIndex = curThreadIndex;
    curThreadIndex++;
    pthread_mutex_unlock(&threadIndexLock);

    frameCount = (myThreadIndex * framePerThread);

    //-- 1. Read / Load the cascades
#ifdef classifier_read
    if( !obj_cascade_1.read( fs1.getFirstTopLevelNode() ) ){ printf("--(!)Error loading\n"); *retVal=-1; return (void*)retVal; }
    if( !obj_cascade_2.read( fs2.getFirstTopLevelNode() ) ){ printf("--(!)Error loading\n"); *retVal=-1; return (void*)retVal; }
    if( !obj_cascade_3.read( fs3.getFirstTopLevelNode() ) ){ printf("--(!)Error loading\n"); *retVal=-1; return (void*)retVal; }
    if( !obj_cascade_4.read( fs4.getFirstTopLevelNode() ) ){ printf("--(!)Error loading\n"); *retVal=-1; return (void*)retVal; }
#else
    if( !obj_cascade_1.load( obj_cascade_name ) ){ printf("--(!)Error loading\n"); *retVal=-1; return (void*)retVal; }
    if( !obj_cascade_2.load( obj2_cascade_name ) ){ printf("--(!)Error loading\n"); *retVal=-1; return (void*)retVal; }
    if( !obj_cascade_3.load( obj3_cascade_name ) ){ printf("--(!)Error loading\n"); *retVal=-1; return (void*)retVal; }
    if( !obj_cascade_4.load( obj3_cascade_name ) ){ printf("--(!)Error loading\n"); *retVal=-1; return (void*)retVal; }
#endif

    //-- 2. Read the video stream
    capture.open( videoFilename );

    if( !capture.isOpened() ){
        cout << "Fail to open video file" << endl;
        *retVal = -1;
        return (void *)retVal;
    }

    //-- 3. Load "objPerFrame"
    answer = new int[numOfFrame];

    FILE *fp = fopen(answerFilename.c_str(), "r");
    if(!fp){ cout << "--(!)Error loading objPerFrame_file" << endl; *retVal=-1; return (void*)retVal;}

    for(i=0; i<numOfFrame; i++){
        if(feof(fp))
            break;

        fscanf(fp, "%d", &answer[i]);
    }

    fclose(fp);

#ifdef scaleInput
    capture.set(CV_CAP_PROP_FRAME_WIDTH, 16 * scaleInput);      // Ratio = 16 : 9
    capture.set(CV_CAP_PROP_FRAME_HEIGHT, 9 * scaleInput);
#endif

    #ifdef jumpFrame
    for(;;frameCount += jumpFrame)
    #else
    for(;;frameCount ++)
    #endif
    {
	if(! capture.set(CV_CAP_PROP_POS_FRAMES, frameCount)) { cout << "error jumpFrame"; *retVal=-1; return (void*)retVal; }

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
            std::vector<Rect> objs1, objs2, objs3, objs4;
            Mat frame_gray;

            cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
            equalizeHist( frame_gray, frame_gray );
            //-- Detect objs
            obj_cascade_1.detectMultiScale( frame_gray, objs1, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
        #ifdef multiDetect
            obj_cascade_2.detectMultiScale( frame_gray, objs2, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
            obj_cascade_3.detectMultiScale( frame_gray, objs3, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
            obj_cascade_4.detectMultiScale( frame_gray, objs4, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
        #endif


            cout << " [" << myThreadIndex << "] frame #" << frameCount << " , #ans: " << answer[frameCount]
            << endl << "\t" << obj_cascade_name_1 << ": " << objs1.size()
        #ifdef multiDetect
            << endl << "\t" << obj_cascade_name_2 << ": " << objs2.size()
            << endl << "\t" << obj_cascade_name_3 << ": " << objs3.size()
            << endl << "\t" << obj_cascade_name_4 << ": " << objs3.size()
        #endif
            << endl;

            threadPartialResult[myThreadIndex].threadPartialNumOfObject += answer[frameCount];

            for( size_t i = 0; i < objs1.size() ; i++ )
            {
        	Point upperLeft_1( objs1[i].x, objs1[i].y );
        	Point bottomRight_1( objs1[i].x + objs1[i].width, objs1[i].y + objs1[i].height );
        	rectangle( frame, upperLeft_1, bottomRight_1, Scalar( 255, 0, 255 ), 2, 8, 0 );
            }
            if( objs1.size() == answer[frameCount])
                threadPartialResult[myThreadIndex].threadPartialHit_1 ++;
            threadPartialResult[myThreadIndex].threadPartialFalseDetect_1 += ( objs1.size() - answer[frameCount] );


        #ifdef multiDetect
            for( size_t i = 0; i < objs2.size() ; i++ )
            {
        	Point upperLeft_2( objs2[i].x, objs2[i].y );
        	Point bottomRight_2( objs2[i].x + objs2[i].width, objs2[i].y + objs2[i].height );
        	rectangle( frame, upperLeft_2, bottomRight_2, Scalar( 255, 0, 0 ), 2, 8, 0 );
            }
            if( objs2.size() == answer[frameCount])
                threadPartialResult[myThreadIndex].threadPartialHit_2 ++;
            threadPartialResult[myThreadIndex].threadPartialFalseDetect_2 += ( objs2.size() - answer[frameCount] );

            for( size_t i = 0; i < objs3.size() ; i++ )
            {
        	Point upperLeft_3( objs3[i].x, objs3[i].y );
        	Point bottomRight_3( objs3[i].x + objs3[i].width, objs3[i].y + objs3[i].height );
        	rectangle( frame, upperLeft_3, bottomRight_3, Scalar( 0, 255, 255 ), 2, 8, 0 );
            }
            if( objs3.size() == answer[frameCount])
                threadPartialResult[myThreadIndex].threadPartialHit_3 ++;
            threadPartialResult[myThreadIndex].threadPartialFalseDetect_3 += ( objs3.size() - answer[frameCount] );

            for( size_t i = 0; i < objs4.size() ; i++ )
            {
        	Point upperLeft_4( objs4[i].x, objs4[i].y );
        	Point bottomRight_4( objs4[i].x + objs4[i].width, objs4[i].y + objs4[i].height );
        	rectangle( frame, upperLeft_4, bottomRight_4, Scalar( 0, 255, 0 ), 2, 8, 0 );
            }
            if( objs4.size() == answer[frameCount])
                threadPartialResult[myThreadIndex].threadPartialHit_4 ++;
            threadPartialResult[myThreadIndex].threadPartialFalseDetect_4 += ( objs4.size() - answer[frameCount] );


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
