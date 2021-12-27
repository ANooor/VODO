#include "vodo.h"

using namespace cv;
using namespace std;

#define MAX_FRAME 1000
#define MIN_NUM_FEAT 2000

double getAbsoluteScale(int frame_id, int sequence_id, double z_cal)	{

    string line;
    int i = 0;
    ifstream myfile ("KITTI_VO/00.txt");
    double x =0, y=0, z = 0;
    double x_prev, y_prev, z_prev;
    if (myfile.is_open())
    {
        while (( getline (myfile,line) ) && (i<=frame_id))
        {
            z_prev = z;
            x_prev = x;
            y_prev = y;
            std::istringstream in(line);
            //cout << line << '\n';
            for (int j=0; j<12; j++)  {
                in >> z ;
                if (j==7) y=z;
                if (j==3)  x=z;
            }

            i++;
        }
        myfile.close();
    }

    else {
        cout << "Unable to open file";
        return 0;
    }

    return sqrt((x-x_prev)*(x-x_prev) + (y-y_prev)*(y-y_prev) + (z-z_prev)*(z-z_prev)) ;

}


int main( int argc, char** argv )	{

    Mat img_1, img_2;
    Mat R_f, t_f;
    ofstream myfile;
    myfile.open ("results1_1.txt");

    double scale = 1.00;
    char first_img[200];
    char second_img[200];
    sprintf(first_img, "KITTI_VO/00/image_2/%06d.png", 0);
    sprintf(second_img, "KITTI_VO/00/image_2/%06d.png", 1);

    char text[100];
    int font_face = FONT_HERSHEY_PLAIN;
    double font_scale = 1;
    int thickness = 1;
    cv::Point text_org(10, 50);

    //read the first two frames from the dataset
    Mat img_1_c = imread(first_img);
    Mat img_2_c = imread(second_img);

    if ( !img_1_c.data || !img_2_c.data )
    {
        std::cout<< " --(!) Error reading images " << std::endl; return -1;
    }

    // we work with grayscale images
    cvtColor(img_1_c, img_1, COLOR_BGR2GRAY);
    cvtColor(img_2_c, img_2, COLOR_BGR2GRAY);

    // feature detection, tracking
    vector<Point2f> points1, points2;        //vectors to store the coordinates of the feature points
    featureExtraction(img_1, points1);        //detect features in img_1
    vector<uchar> status;
    featureTracking(img_1,img_2,points1,points2, status); //track those features to img_2

    //TODO: add a fucntion to load these values directly from KITTI's calib files
    // WARNING: different sequences in the KITTI VO dataset have different intrinsic/extrinsic parameters
    double focal = 718.8560;
    cv::Point2d pp(607.1928, 185.2157);
    //recovering the pose and the essential matrix
    Mat E, R, t, mask;
    E = findEssentialMat(points2, points1, focal, pp, RANSAC, 0.999, 1.0, mask);
    recoverPose(E, points2, points1, R, t, focal, pp, mask);

    Mat prevImage = img_2;
    Mat curr_image;
    vector<Point2f> prev_features = points2;
    vector<Point2f> curr_features;

    char filename[100];

    R_f = R.clone();
    t_f = t.clone();

    clock_t begin = clock();

    namedWindow( "Road facing camera", WINDOW_AUTOSIZE );// Create a window for display.
    namedWindow( "Trajectory", WINDOW_AUTOSIZE );// Create a window for display.

    Mat traj = Mat::zeros(600, 600, CV_8UC3);

    for(int num_frame=2; num_frame < MAX_FRAME; num_frame++)
    {
        sprintf(filename, "KITTI_VO/00/image_2/%06d.png", num_frame);
        //cout << numFrame << endl;
        Mat curr_image_cpy = imread(filename);
        cvtColor(curr_image_cpy, curr_image, COLOR_BGR2GRAY);
        vector<uchar> status;
        featureTracking(prevImage, curr_image, prev_features, curr_features, status);

        E = findEssentialMat(curr_features, prev_features, focal, pp, RANSAC, 0.999, 1.0, mask);
        recoverPose(E, curr_features, prev_features, R, t, focal, pp, mask);

        Mat prev_pts(2,prev_features.size(), CV_64F), curr_pts(2,curr_features.size(), CV_64F);


        for(int i=0;i<prev_features.size();i++)
        {
            prev_pts.at<double>(0,i) = prev_features.at(i).x;
            prev_pts.at<double>(1,i) = prev_features.at(i).y;

            curr_pts.at<double>(0,i) = curr_features.at(i).x;
            curr_pts.at<double>(1,i) = curr_features.at(i).y;
        }

        scale = getAbsoluteScale(num_frame, 0, t.at<double>(2));

        //cout << "Scale is " << scale << endl;

        if ((scale>0.1)&&(t.at<double>(2) > t.at<double>(0)) && (t.at<double>(2) > t.at<double>(1)))
        {
            t_f = t_f + scale*(R_f*t);
            R_f = R*R_f;
        }

        else
        {
            //scale below 0.1, or incorrect translation
        }

        // lines for printing results

        if (prev_features.size() < MIN_NUM_FEAT)
        {
            featureExtraction(prevImage, prev_features);
            featureTracking(prevImage,curr_image,prev_features,curr_features, status);
        }

        prevImage = curr_image.clone();
        prev_features = curr_features;

        int x = int(t_f.at<double>(0)) + 300;
        int y = int(t_f.at<double>(2)) + 100;
        circle(traj, Point(x, y) ,1, CV_RGB(255,0,0), 2);

        rectangle( traj, Point(10, 30), Point(550, 50), CV_RGB(0,0,0), cv::FILLED);
        sprintf(text, "Coordinates: x = %02fm y = %02fm z = %02fm", t_f.at<double>(0), t_f.at<double>(1), t_f.at<double>(2));
        putText(traj, text, text_org, font_face, font_scale, Scalar::all(255), thickness, 8);

        imshow( "Road facing camera", curr_image_cpy );
        imshow( "Trajectory", traj );

        waitKey(1);

    }

    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    cout << "Total time taken: " << elapsed_secs << "s" << endl;

    return 0;
}
