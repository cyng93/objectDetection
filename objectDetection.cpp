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

//#define sourceWebcam
#define userInput
#define scaleInputOn        // WIDTH=16*n    HEIGHT=9*n
#define jumpFrame 30
#define outputFrame
#define numOfTolerant 2
#define multiDetect
#define parallel

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

double frameCount = 0;
double numOfFrame = 0;
double numOfCorrectFrame = 0;
String videoFilename = "testVideo/oriVideo.mov";
int scaleInput = 40;
#ifdef outputFrame
String outputFilePrefix  = "./outputFrame/frame_";
String outputFileType = ".png";
stringstream ss;
String outputFilename;
#endif
#ifdef parallel
typedef struct partialCorrect_padding_t{
    unsigned long threadPartialCorrect;     // 8 bytes
    int padding[14];                    // 56 bytes
}threadPartialCorrect_t;

int numOfCores = 0;
int curThreadIndex = 0;
pthread_mutex_t threadIndexLock;
threadPartialCorrect_t *threadPartialCorrect;
int *retVal;
#endif

/**
 * @function main
 */
int main( int argc, char **argv  )
{
    VideoCapture capture;
    int i;
    pthread_t *threadPool;

    numOfCores = get_nprocs();

    //-- 0. Get Num Of Frames
    capture.open( videoFilename );
    if( !capture.isOpened() ){
        cout << "Fail to open video file" << endl;
        return -1;
    }
    numOfFrame = capture.get(CV_CAP_PROP_FRAME_COUNT);
    cout << "Total Frame # = " << numOfFrame << endl;
    capture.release();

    pthread_mutex_init(&threadIndexLock, NULL);
    threadPartialCorrect = (threadPartialCorrect_t *)malloc(numOfCores * sizeof(threadPartialCorrect_t));
    threadPool = (pthread_t *)malloc(numOfCores * sizeof(pthread_t));

    // thread creation
    for(i=0; i<numOfCores; i++)
        pthread_create(&threadPool[i], NULL, handler, NULL);
    // thread join
    for(i=0; i<numOfCores; i++)
            pthread_join(threadPool[i], NULL);

    for(i=0; i<numOfCores; i++)
        numOfCorrectFrame += threadPartialCorrect[i].threadPartialCorrect;

    cout << "#Correct Frame: [ " << numOfCorrectFrame << " / " <<
#ifdef jumpFrame
    frameCount/jumpFrame
#else
    frameCount
#endif
    << " ]" << endl;

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

    //-- 1. Load the cascades
    if( !obj_cascade.load( obj_cascade_name ) ){ printf("--(!)Error loading\n"); *retVal=-1; return (void*)retVal; }
    if( !obj2_cascade.load( obj2_cascade_name ) ){ printf("--(!)Error loading\n"); *retVal=-1; return (void*)retVal; }
    if( !obj3_cascade.load( obj3_cascade_name ) ){ printf("--(!)Error loading\n"); *retVal=-1; return (void*)retVal; }

    //-- 2. Read the video stream
    capture.open( videoFilename );

    if( !capture.isOpened() ){
        cout << "Fail to open video file" << endl;
        *retVal = -1;
        return (void *)retVal;
    }

    numOfFrame = capture.get(CV_CAP_PROP_FRAME_COUNT);

#ifdef scaleInputOn
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
    #ifdef scaleInputOn
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
        #ifdef multiDetect
            for( size_t i = 0; i < objs2.size() ; i++ )
            {
        	Point upperLeft_2( objs2[i].x, objs2[i].y );
        	Point bottomRight_2( objs2[i].x + objs2[i].width, objs2[i].y + objs2[i].height );
        	rectangle( frame, upperLeft_2, bottomRight_2, Scalar( 255, 255, 0 ), 2, 8, 0 );
            }
            for( size_t i = 0; i < objs3.size() ; i++ )
            {
        	Point upperLeft_3( objs3[i].x, objs3[i].y );
        	Point bottomRight_3( objs3[i].x + objs3[i].width, objs3[i].y + objs3[i].height );
        	rectangle( frame, upperLeft_3, bottomRight_3, Scalar( 0, 255, 255 ), 2, 8, 0 );
            }
        #endif
            //-- Show what you got
            //imshow( window_name, frame );

            if( objs.size() <= numOfTolerant
            #ifdef multiDetect
                && objs2.size() <= numOfTolerant
                && objs3.size() <= numOfTolerant
            #endif
            )
                threadPartialCorrect[myThreadIndex].threadPartialCorrect ++;
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
