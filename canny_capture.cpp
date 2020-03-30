#include "opencv2/core/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/videoio.hpp"

#include <iostream>
#include <time.h>

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <pthread.h>
#include <sched.h>
#include <time.h>
#include <semaphore.h>
#include <unistd.h>
#include <sys/param.h>
#include <errno.h>
#include <ctype.h>

using namespace cv;
using namespace std;

#define SCHEDULING_POLICY SCHED_FIFO
#define INHERIT_SCHEDULER PTHREAD_EXPLICIT_SCHED
#define ITERATION (10)
#define VALUE_X (1280)
#define VALUE_Y (960)

#define handle_error_en(en, msg) \
  do { errno = en; perror(msg); exit(EXIT_FAILURE); } while (0)

sem_t sem_F, sem_T1, sem_T2, sem_T3;
pthread_attr_t frame_sched_attr, t1_sched_attr, t2_sched_attr, t3_sched_attr;
struct sched_param frame_param, t1_param, t2_param, t3_param;
pthread_t frame_thread_id, t1_thread_id, t2_thread_id, t3_thread_id;

double frame_max = 0, t1_max = 0, t2_max = 0, t3_max = 0;

Mat src, src_gray;
Mat dst, detected_edges;
int lowThreshold = 0;

const int max_lowThreshold = 100;
const int ratio_t = 3;
const int kernel_size = 3;
const char* window_name = "Edge Map";

void *frame_capture_handler (void *arg);
void *t1_handler (void *arg);
void *t2_handler (void *arg);
void *t3_handler (void *arg);

static void CannyThreshold(int, void*)
{
    blur( src_gray, detected_edges, Size(3,3) );
    Canny( detected_edges, detected_edges, lowThreshold, lowThreshold*ratio_t, kernel_size );
    dst = Scalar::all(0);
    src.copyTo( dst, detected_edges);
    //imshow( window_name, dst );
}

static void HoughCircleTranform (void)
{
    // Loads an image
    Mat src = imread("test.jpg", IMREAD_COLOR );
    // Check if image is loaded fine
    if(src.empty()){
        printf(" Error opening image\n");
        return;
    }

    Mat gray;
    
    cvtColor(src, gray, COLOR_BGR2GRAY);
    medianBlur(gray, gray, 5);
    vector<Vec3f> circles;
    HoughCircles(gray, circles, HOUGH_GRADIENT, 1,
                 gray.rows/16,  // change this value to detect circles with different distances to each other
                 100, 30, 1, 30 // change the last two parameters
            // (min_radius & max_radius) to detect larger circles
    );
    
    for(size_t i = 0; i < circles.size(); i++) {

        Vec3i c = circles[i];
        Point center = Point(c[0], c[1]);
        // circle center
        circle( src, center, 1, Scalar(0,100,100), 3, LINE_AA);
        // circle outline
        int radius = c[2];
        circle( src, center, radius, Scalar(255,0,255), 3, LINE_AA);
    }
    //imshow("detected circles", src);
}

static void HoughLineTransform (void)
{

    // Declare the output variables
    Mat dst, cdst, cdstP;

    // Loads an image
    Mat src = imread("test.jpg", IMREAD_GRAYSCALE );
    
    // Check if image is loaded fine
    if(src.empty()) {
        printf(" Error opening image\n");
        return;
    }

    // Edge detection
    Canny(src, dst, 50, 200, 3);
    // Copy edges to the images that will display the results in BGR
    cvtColor(dst, cdst, COLOR_GRAY2BGR);
    cdstP = cdst.clone();
    // Standard Hough Line Transform
    vector<Vec2f> lines; // will hold the results of the detection

    HoughLines(dst, lines, 1, CV_PI/180, 150, 0, 0 ); // runs the actual detection
    
    // Draw the lines
    for( size_t i = 0; i < lines.size(); i++ )
    {
        float rho = lines[i][0], theta = lines[i][1];
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a*rho, y0 = b*rho;
        pt1.x = cvRound(x0 + 1000*(-b));
        pt1.y = cvRound(y0 + 1000*(a));
        pt2.x = cvRound(x0 - 1000*(-b));
        pt2.y = cvRound(y0 - 1000*(a));
        line( cdst, pt1, pt2, Scalar(0,0,255), 3, LINE_AA);
    }
    
    // Probabilistic Line Transform
    vector<Vec4i> linesP; // will hold the results of the detection
    HoughLinesP(dst, linesP, 1, CV_PI/180, 50, 50, 10 ); // runs the actual detection
    
    // // Draw the lines
    // for( size_t i = 0; i < linesP.size(); i++ )
    // {
    //     Vec4i l = linesP[i];
    //     line( cdstP, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, LINE_AA);
    // }
    
    // // Show results
    // imshow("Source", src);
    // imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst);
    // imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP);

}

static void CaptureImage(void)
{
  // Capture the Image from the webcam
  VideoCapture cap(0);

  cap.set( CAP_PROP_FRAME_WIDTH, VALUE_X);
  cap.set( CAP_PROP_FRAME_HEIGHT, VALUE_Y);

  // Get the frame
  Mat save_img; cap >> save_img;

  if(save_img.empty()) {
    std::cerr << "Something is wrong with the webcam, could not get frame." << std::endl;
  }
  // Save the frame into a file
  imwrite("test.jpg", save_img); // A JPG FILE IS BEING SAVED
}

static void CannyTransform(void)
{
  src = imread("test.jpg", IMREAD_COLOR ); // Load an image
  
  if( src.empty() ) {
    std::cout << "Could not open or find the image!\n" << std::endl;
    return;
  }
  
  dst.create( src.size(), src.type() );
  cvtColor( src, src_gray, COLOR_BGR2GRAY );
  //namedWindow( window_name, WINDOW_AUTOSIZE );
  //createTrackbar( "Min Threshold:", window_name, &lowThreshold, max_lowThreshold, CannyThreshold );
  
  CannyThreshold(0, 0);
  
}

int main( int argc, char** argv )
{

  sem_init (&sem_F, 0, 1);
  sem_init (&sem_T1, 0, 0);
  sem_init (&sem_T2, 0, 0);
  sem_init (&sem_T3, 0, 0);

  // pthread_attr_init(&frame_sched_attr);
  // pthread_attr_init(&t1_sched_attr);
  // pthread_attr_init(&t2_sched_attr);
  // pthread_attr_init(&t3_sched_attr);

  // pthread_attr_setinheritsched(&frame_sched_attr, PTHREAD_EXPLICIT_SCHED);
  // pthread_attr_setschedpolicy(&frame_sched_attr, SCHED_FIFO);

  // pthread_attr_setinheritsched(&t1_sched_attr, PTHREAD_EXPLICIT_SCHED);
  // pthread_attr_setschedpolicy(&t1_sched_attr, SCHED_FIFO);

  // pthread_attr_setinheritsched(&t2_sched_attr, PTHREAD_EXPLICIT_SCHED);
  // pthread_attr_setschedpolicy(&t2_sched_attr, SCHED_FIFO);

  // pthread_attr_setinheritsched(&t3_sched_attr, PTHREAD_EXPLICIT_SCHED);
  // pthread_attr_setschedpolicy(&t3_sched_attr, SCHED_FIFO);

  // pthread_attr_setschedparam(&frame_sched_attr, &frame_param);
  // pthread_attr_setschedparam(&t1_sched_attr, &t1_param);
  // pthread_attr_setschedparam(&t2_sched_attr, &t2_param);
  // pthread_attr_setschedparam(&t3_sched_attr, &t3_param);

  if (pthread_create(&frame_thread_id, &frame_sched_attr, frame_capture_handler, NULL) != 0) {
    printf("Error creating Frame Capture thread\n");
  }

  if (pthread_create(&t1_thread_id, &t1_sched_attr, t1_handler, NULL) != 0) {
    printf("Error creating first transform thread\n");
  }

  if (pthread_create(&t2_thread_id, &t2_sched_attr, t2_handler, NULL) != 0) {
    printf("Error creating second transform thread\n");
  }

  if (pthread_create(&t3_thread_id, &t2_sched_attr, t3_handler, NULL) != 0) {
    printf("Error creating third tranform thread\n");
  }


  // CaptureImage();

  // ThreadHandler();

  // HoughCircle();

  // HoughLine();
  pthread_join(frame_thread_id, NULL);
  pthread_join(t1_thread_id, NULL);
  pthread_join(t2_thread_id, NULL);
  pthread_join(t3_thread_id, NULL);

  pthread_attr_destroy(&frame_sched_attr);
  pthread_attr_destroy(&t1_sched_attr);
  pthread_attr_destroy(&t2_sched_attr);
  pthread_attr_destroy(&t3_sched_attr);

  sem_destroy(&sem_F);
  sem_destroy(&sem_T1);
  sem_destroy(&sem_T2);
  sem_destroy(&sem_T3);

  //sched_setscheduler(getpid(), SCHED_OTHER, &nrt_param);

  printf("/nThe Worst case execution time for frame capture thread = %f\n", frame_max*1000);
  printf("The Worst case execution time for first_transform thread = %f\n", t1_max*1000);
  printf("The Worst case execution time for second_transform thread = %f\n", t2_max*1000);
  printf("The Worst case execution time for third_transform thread = %f\n", t3_max*1000);
  
  return 0;
}


void *frame_capture_handler (void *arg)
{
  clock_t start;
  double cpu_time_used;
  int i = 0;

  while (i < ITERATION)
  {
    start = clock();
    sem_wait(&sem_F);

    i++;
    CaptureImage();

    cpu_time_used = ((double)(clock() - start))/CLOCKS_PER_SEC;
    printf("\nTime taken for Frame Capture thread in %d interation = %f ms\n", i, (cpu_time_used*1000));
    frame_max = (cpu_time_used > frame_max ? cpu_time_used : frame_max);
    
    sem_post(&sem_T1);
  } 
}

void *t1_handler (void *arg)
{
  clock_t start;
  double cpu_time_used;
  int i = 0;

  while (i < ITERATION)
  {
    start = clock();
    sem_wait(&sem_T1);

    i++;
    CannyTransform();

    cpu_time_used = ((double)(clock() - start))/CLOCKS_PER_SEC;
    printf("Time taken for first_transformation thread in %d iteration = %f ms\n", i, (cpu_time_used*1000));
    t1_max = (cpu_time_used > t1_max ? cpu_time_used : t1_max);

    sem_post(&sem_T2);
  } 
}

void *t2_handler (void *arg)
{
  clock_t start;
  double cpu_time_used;
  int i = 0;

  while (i < ITERATION)
  {
    start = clock();
    sem_wait(&sem_T2);

    i++;
    HoughCircleTranform();

    cpu_time_used = ((double)(clock() - start))/CLOCKS_PER_SEC;
    printf("Time taken for second_transformation thread in %d iteration = %f ms\n", i, (cpu_time_used*1000));
    t2_max = (cpu_time_used > t2_max ? cpu_time_used : t2_max);

    sem_post(&sem_T3);
  } 
}

void *t3_handler (void *arg)
{
  clock_t start;
  double cpu_time_used;
  int i = 0;
  
  while (i < ITERATION)
  {
    start = clock();
    sem_wait(&sem_T3);

    i++;
    HoughLineTransform();

    cpu_time_used = ((double)(clock() - start))/CLOCKS_PER_SEC;
    printf("Time taken for third_transformation thread in %d iteration = %f ms\n", i, (cpu_time_used*1000));
    t3_max = (cpu_time_used > t3_max ? cpu_time_used : t3_max);

    sem_post(&sem_F);
  } 
}