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

//#define sourceWebcam
#define userInput
#define scaleInputOn        // WIDTH=16*n    HEIGHT=9*n
#define jumpFrame 90
#define outputCorrect
#define numOfTolerant 3


using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay( Mat frame );

/** Global variables */
//-- Note, either copy these two files from opencv/data/haarscascades to your current folder, or change these locations
String obj_cascade_name = "classifier/frontal_pos3000_stg14.xml";
String obj2_cascade_name = "classifier/frontal_pos3000_stg20.xml";

CascadeClassifier obj_cascade, obj2_cascade;
string window_name = "Object detection";
RNG rng(12345);

double frameCount = 0;
double totalFrameNum = 0;
double numOfCorrectFrame = 0;
String videoFilename = "testVideo/oriVideo.mov";
int scaleInput = 40;
#ifdef outputCorrect
String outputFilePrefix  = "./outputFrame/frame_";
String outputFileType = ".png";
stringstream ss;
String outputFilename;
#endif

/**
 * @function main
 */
int main( int argc, char **argv  )
{
    VideoCapture capture;
    Mat frame;

    //-- 1. Load the cascades
#ifdef userInput
    if(argc > 4){ cout << "[USAGE] ./objectDetection <frame_scale> <path_to_classifier 1> <path_to_classifier 2>" << endl; return -1;}
    if(argc >= 2)
        scaleInput = atoi(argv[1]);
    if(argc >= 3)
        obj_cascade_name = argv[2];
    if(argc >= 4)
        obj2_cascade_name = argv[3];
#endif
    if( !obj_cascade.load( obj_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
    if( !obj2_cascade.load( obj2_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

    //-- 2. Read the video stream
#ifdef sourceWebcam
    capture.open( -1 );
#else
    capture.open( videoFilename );
#endif

    if( !capture.isOpened() ){
        cout << "Fail to open video file" << endl;
        return -1;
    }

    totalFrameNum = capture.get(CV_CAP_PROP_FRAME_COUNT);
    cout << "Total Frame # = " << totalFrameNum << endl;


    for(;;)
    {
    #ifdef sourceWebcam
        capture >> frame;
    #else
#ifdef scaleInputOn
	capture.set(CV_CAP_PROP_FRAME_WIDTH, 16 * scaleInput);      // Ratio = 16 : 9
	capture.set(CV_CAP_PROP_FRAME_HEIGHT, 9 * scaleInput);
#endif

    #ifdef jumpFrame
	if(! capture.set(CV_CAP_PROP_POS_FRAMES, frameCount += jumpFrame)) { cout << "error jumpFrame"; return -1; }
    #else
        frameCount++;
    #endif
	if( frameCount >= totalFrameNum || !capture.read(frame) )  // frameCount check
            break;
#ifdef scaleInputOn
	resize(frame, frame, Size( 16 * scaleInput, 9 * scaleInput), 0, 0, INTER_CUBIC);
#endif
    #endif
	//-- 3. Apply the classifier to the frame
	if( !frame.empty() )
	{ detectAndDisplay( frame ); }
	else
	{ printf(" --(!) No captured frame -- Break!"); break; }

	int c = waitKey(10);
	if( (char)c == 'c' ) { break; }
    }

    cout << "#Correct Frame: [ " << numOfCorrectFrame << " / " <<
#ifdef jumpFrame
    frameCount/jumpFrame
#else
    frameCount
#endif
    << " ]" << endl;

    return 0;
}

/**
 * @function detectAndDisplay
 */
void detectAndDisplay( Mat frame )
{
    std::vector<Rect> objs, objs2;
    Mat frame_gray;

    cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );
    //-- Detect objs
    obj_cascade.detectMultiScale( frame_gray, objs, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
    obj2_cascade.detectMultiScale( frame_gray, objs2, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );

    cout << "[" <<
#ifdef jumpFrame
    frameCount/jumpFrame
#else
    frameCount
#endif
    << "] frame #" << frameCount << " : " << endl 
	<< " " << obj_cascade_name << ": " << objs.size() << " ; " << obj2_cascade_name << ": " << objs2.size() << endl;

    for( size_t i = 0; i < objs.size() ; i++ )
    {
	Point upperLeft_1( objs[i].x, objs[i].y );
	Point bottomRight_1( objs[i].x + objs[i].width, objs[i].y + objs[i].height );
	rectangle( frame, upperLeft_1, bottomRight_1, Scalar( 255, 0, 255 ), 2, 8, 0 );

    }
    for( size_t i = 0; i < objs2.size() ; i++ )
    {
	Point upperLeft_2( objs2[i].x, objs2[i].y );
	Point bottomRight_2( objs2[i].x + objs2[i].width, objs2[i].y + objs2[i].height );
	rectangle( frame, upperLeft_2, bottomRight_2, Scalar( 255, 255, 0 ), 2, 8, 0 );

    }
    //-- Show what you got
    //imshow( window_name, frame );

    //if(objs.size() <= numOfTolerant)
    if(objs.size())
    {
	numOfCorrectFrame++;
#ifdef outputCorrect
	ss << outputFilePrefix << frameCount << outputFileType;
	outputFilename = ss.str();
	ss.str("");
	imwrite(outputFilename, frame);
#endif
    }

}
