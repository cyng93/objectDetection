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
#define multiDetect 4       // uncomment to detect with only one classifier

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
#ifdef multiDetect
String obj_cascade_name[multiDetect] =	{ "classifier/frontal_negSet1_pos7000_stg14.xml"
					#if multiDetect > 1
										, "classifier/frontal_negSet2_pos7000_stg14.xml"
					#endif
					#if multiDetect > 2
										, "classifier/frontal_negSet1+negScene_pos7000_stg14.xml"
					#endif
					#if multiDetect > 3
										, "classifier/frontal_negSet2+negScene_pos7000_stg14.xml"
					#endif
										};

String obj_rect_color[multiDetect] =	{ "PINK"
					#if multiDetect > 1
										, "BLUE"
					#endif
					#if multiDetect > 2
										, "YELLOW"
					#endif
					#if multiDetect > 3
										, "GREEN"
					#endif
										};
									
const Scalar obj_scalar[multiDetect] =	{ Scalar(255, 0 ,255)
					#if multiDetect > 1
										, Scalar(255, 0 , 0)
					#endif
					#if multiDetect > 2
										, Scalar(0, 255, 255)
					#endif
					#if multiDetect > 3
										, Scalar(0, 255, 0)
					#endif
										};
#else
String obj_cascade_name = "classifier/frontal_negSet1_pos7000_stg14.xml";
String obj_rect_color = "PINK";
#endif

string window_name = "Object detection";
RNG rng(12345);

unsigned long long frameCount = 0;
unsigned long long numOfFrame = 0;
unsigned long long numOfObject = 0;


#ifdef multiDetect
unsigned long long numOfHit[multiDetect] = {0};
unsigned long long numOfFalseDetect[multiDetect] = {0};
#else
unsigned long long numOfHit = 0;
unsigned long long numOfFalseDetect = 0;
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
 #ifdef multiDetect
    unsigned long long threadPartialNumOfObject;				// 8 bytes
	unsigned long long threadPartialHit[multiDetect];			// 8 bytes * n
	unsigned long long threadPartialFalseDetect[multiDetect];	// 8 bytes * n
	int padding[(64-((multiDetect*16 + 8)%64))/4];
 #else
    unsigned long long threadPartialNumOfObject;   // 8 bytes
    unsigned long long threadPartialHit;         // 8 bytes
    unsigned long long threadPartialFalseDetect; // 8 bytes
    int padding[10];                               // 40 bytes
 #endif
}threadPartialResult_t;

int numOfCores = 0;
int curThreadIndex = 0;
pthread_mutex_t threadIndexLock;
threadPartialResult_t *threadPartialResult;
int *retVal;
#endif

#ifdef classifier_read
  #ifdef multiDetect
	FileStorage fs0(obj_cascade_name[0], FileStorage::READ);
	#if multiDetect > 1
	FileStorage fs1(obj_cascade_name[1], FileStorage::READ);
	#endif
	#if multiDetect > 2
	FileStorage fs2(obj_cascade_name[2], FileStorage::READ);
	#endif
	#if multiDetect > 3
	FileStorage fs3(obj_cascade_name[3], FileStorage::READ);
	#endif
	// #if multiDetect > N 
	// FileStorage tmpN(obj_cascade_name[N], FileStorage::READ);
	// #endif
  #else
  FileStorage fs(obj_cascade_name, FileStorage::READ);
  #endif
#endif

/**
 * @function main
 */
int main( int argc, char **argv  )
{
    int i, j;
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
		#ifdef multiDetect
		threadPartialResult[i].threadPartialNumOfObject = 0;
		for(j=0; j<multiDetect; j++){
    	    threadPartialResult[i].threadPartialHit[j] = 0;
    	    threadPartialResult[i].threadPartialFalseDetect[j] = 0;
		}
		#else
		threadPartialResult[i].threadPartialNumOfObject = 0;
    	threadPartialResult[i].threadPartialHit = 0;
    	threadPartialResult[i].threadPartialFalseDetect = 0;
    	#endif
    }

    // thread creation
    for(i=0; i<numOfCores; i++)
        pthread_create(&threadPool[i], NULL, handler, NULL);
    // thread join
    for(i=0; i<numOfCores; i++)
            pthread_join(threadPool[i], NULL);

    for(i=0; i<numOfCores; i++){
	#ifdef multiDetect
        numOfObject += threadPartialResult[i].threadPartialNumOfObject;
		for(j=0; j<multiDetect; j++){
			numOfHit[j] += threadPartialResult[i].threadPartialHit[j];
			numOfFalseDetect[j] += threadPartialResult[i].threadPartialFalseDetect[j];
        }
    #else
        numOfObject += threadPartialResult[i].threadPartialNumOfObject;
        numOfHit += threadPartialResult[i].threadPartialHit;
        numOfFalseDetect += threadPartialResult[i].threadPartialFalseDetect;
    #endif
    }

    cout << "\n\n==================== DETECTION RESULT ====================" << endl;
#ifdef multiDetect
	for(j=0; j<multiDetect; j++){
		cout << "\n[Classifier " << j+1 << "] - " << obj_rect_color[j]
    	<< "\n  Classifer:\t" << obj_cascade_name[j]
    	<< "\n  Hit Rate:\t" << numOfHit[j] << " / " << numOfObject
    	<< "\n  False Detect:\t" << numOfFalseDetect[j]
    	<< "\n  Accuracy:\t" << (float)numOfHit[j]/(numOfObject+numOfFalseDetect[j]) << endl;
	}
#else
    cout << "\n[Classifier 1] - " << obj_rect_color
    << "\n  Classifer:\t" << obj_cascade_name
    << "\n  Hit Rate:\t" << numOfHit << " / " << numOfObject
    << "\n  False Detect:\t" << numOfFalseDetect
    << "\n  Accuracy:\t" << (float)numOfHit/(numOfObject+numOfFalseDetect) << endl;
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
    unsigned long long i, j;       // for array purpose

    VideoCapture capture;
    Mat frame;

    // each thread getting their own index
    pthread_mutex_lock(&threadIndexLock);
    myThreadIndex = curThreadIndex;
    curThreadIndex++;
    pthread_mutex_unlock(&threadIndexLock);

    frameCount = (myThreadIndex * framePerThread);

    //-- 1. Read / Load the cascades
#ifdef multiDetect
	CascadeClassifier obj_cascade[multiDetect];
  #ifdef classifier_read

	if( !obj_cascade[0].read( fs0.getFirstTopLevelNode() ) ){ printf("--(!)Error loading\n"); *retVal=-1; return (void*)retVal; }
    #if multiDetect > 1
	if( !obj_cascade[1].read( fs1.getFirstTopLevelNode() ) ){ printf("--(!)Error loading\n"); *retVal=-1; return (void*)retVal; }
	#endif
	#if multiDetect > 2
	if( !obj_cascade[2].read( fs2.getFirstTopLevelNode() ) ){ printf("--(!)Error loading\n"); *retVal=-1; return (void*)retVal; }
	#endif
	#if multiDetect > 3
	if( !obj_cascade[3].read( fs3.getFirstTopLevelNode() ) ){ printf("--(!)Error loading\n"); *retVal=-1; return (void*)retVal; }
	#endif
    
  #else
	for(j=0; j<multiDetect; j++)
		if( !obj_cascade[j].load( obj_cascade_name[j] ) ){ printf("--(!)Error loading\n"); *retVal=-1; return (void*)retVal; }
  #endif
#else
    CascadeClassifier obj_cascade;
  #ifdef classifier_read
	fs.open(obj_cascade_name, cv::FileStorage::READ);
    if( !obj_cascade.read( fs.getFirstTopLevelNode() ) ){ printf("--(!)Error loading\n"); *retVal=-1; return (void*)retVal; }
  #else
	if( !obj_cascade.load( obj_cascade_name ) ){ printf("--(!)Error loading\n"); *retVal=-1; return (void*)retVal; }
  #endif
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
	if( frame.empty() ) { printf(" --(!) No captured frame -- Break!"); break; }
	else
    {
	#ifdef multiDetect
        std::vector<Rect> objs[multiDetect];
	#else
        std::vector<Rect> objs;
	#endif
        Mat frame_gray;

        cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
        equalizeHist( frame_gray, frame_gray );
        //-- Detect objs

	#ifdef multiDetect
		for(j=0; j<multiDetect; j++)
			obj_cascade[j].detectMultiScale( frame_gray, objs[j], 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
    #else
		obj_cascade.detectMultiScale( frame_gray, objs, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
	#endif


        cout << " [" << myThreadIndex << "] frame #" << frameCount << " , #ans: " << answer[frameCount] << endl;
	#ifdef multiDetect
		for(j=0; j<multiDetect; j++)
			cout << endl << "\t" << obj_cascade_name[j] << ": " << objs[j].size();
	#else
        cout << endl << "\t" << obj_cascade_name << ": " << objs.size();
    #endif
        cout << endl;

        threadPartialResult[myThreadIndex].threadPartialNumOfObject += answer[frameCount];

    #ifdef multiDetect
		for(j=0; j<multiDetect; j++){
			for( size_t i = 0; i < objs[j].size() ; i++ )
        	{
    		Point upperLeft( objs[j][i].x, objs[j][i].y );
    		Point bottomRight( objs[j][i].x + objs[j][i].width, objs[j][i].y + objs[j][i].height );
    		rectangle( frame, upperLeft, bottomRight, obj_scalar[j], 2, 8, 0 );
        	}
        	if( objs[j].size() == answer[frameCount])
        	    threadPartialResult[myThreadIndex].threadPartialHit[j] ++;
        	threadPartialResult[myThreadIndex].threadPartialFalseDetect[j] += ( objs[j].size() - answer[frameCount] );
		}
	#else
        for( size_t i = 0; i < objs.size() ; i++ )
        {
    	Point upperLeft( objs[i].x, objs[i].y );
    	Point bottomRight( objs[i].x + objs[i].width, objs[i].y + objs[i].height );
    	rectangle( frame, upperLeft_1, bottomRight_1, Scalar( 255, 0, 255 ), 2, 8, 0 );
        }
        if( objs.size() == answer[frameCount])
            threadPartialResult[myThreadIndex].threadPartialHit ++;
        threadPartialResult[myThreadIndex].threadPartialFalseDetect += ( objs.size() - answer[frameCount] );
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
