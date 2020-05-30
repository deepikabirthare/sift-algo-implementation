#include "Mysift.h"
#include "climits"
#define SIGMA_ANTIALIAS			0.5
#define SIGMA_PREBLUR			1.0
#define CURVATURE_THRESHOLD		5.0
#define CONTRAST_THRESHOLD		0.03		// in terms of 255
#define M_PI					3.1415926535897932384626433832795
#define NUM_BINS				36
#define MAX_KERNEL_SIZE			20
#define FEATURE_WINDOW_SIZE		16
#define DESC_NUM_BINS			8
#define FVSIZE					128
#define	FV_THRESHOLD			0.2

using namespace cv;
mySIFT::mySIFT(Mat src, int octaves, int intervals) {
	srcImage = src.clone();
	m_numIntervals = intervals;
	m_numOctaves = octaves;
}
void mySIFT::DoSift() {
	BuildScaleSpace();
	DetectExtrema();
	AssignOrientations();
	return;
}
void mySIFT::BuildScaleSpace() {
	Mat src_gray;
	Mat imgFloat;  //floating point image
	cvtColor(srcImage, src_gray, COLOR_BGR2GRAY);
	src_gray.convertTo(imgFloat, CV_32F, 1.0 / 255, 0);
	GaussianBlur(imgFloat, imgFloat, Size(0, 0), SIGMA_ANTIALIAS);
	resize(imgFloat, imgFloat, Size(imgFloat.size() * 2));
	m_glist.push_back(vector<Mat>());
	m_glist[0].push_back(imgFloat.clone());
	pyrUp(imgFloat,m_glist[0][0]);
	GaussianBlur(m_glist[0][0], m_glist[0][0], Size(0, 0), SIGMA_PREBLUR);
	double initSigma = sqrt(2.0f);
	m_absSigma.push_back(vector<double>());
	m_absSigma[0].push_back(initSigma * 0.5);
	int i, j;
	for (i = 0; i < m_numOctaves; i++) {
		double sigma = initSigma;
		Size currentSize = m_glist[i][0].size();
		m_dogList.push_back(vector<Mat>());
		for (j = 1; j< int(m_numIntervals + 3); j++) {
			double sigma_f = sqrt(pow(2.0, 2.0 / m_numIntervals) - 1) * sigma; //New sigma
			sigma = pow(2.0, 1.0 / m_numIntervals) * sigma;
			m_absSigma[i].push_back(sigma * 0.5 * pow(2.0f, (float)i));
			//Gaussian Blur
			Mat temp;
			GaussianBlur(m_glist[i][j - 1], temp, Size(0, 0), sigma_f);
			m_glist[i].push_back(temp);
			Mat temp1;
			subtract(m_glist[i][j - 1], m_glist[i][j], temp1, Mat());
			m_dogList[i].push_back(temp1);
			

		}
		//if we are not at last octave
		if (i < m_numOctaves - 1) {
			Mat temp3;
			pyrDown(m_glist[i][0], temp3, currentSize / 2);
			m_absSigma.push_back(vector<double>());
			m_glist.push_back(vector<Mat>());
			m_glist[i + 1].push_back(temp3);
			m_absSigma[i + 1].push_back(m_absSigma[i][m_numIntervals]);
		}

	}

}

//DetectExtrema

void mySIFT::DetectExtrema() {
	double curvature_ratio, curvature_threshold;
	int scale;
	double dxx, dyy, dxy, trH, detH;
	unsigned int num = 0; //number of keypoints detected
	unsigned int numRemoved = 0; //no. of keypoints rejected
	Mat middle, up, down;
	curvature_threshold = (CURVATURE_THRESHOLD + 1) * (CURVATURE_THRESHOLD + 1) / (CURVATURE_THRESHOLD);
	for (int i = 0; i < m_numOctaves; i++) {
		scale = int(pow(2.0, double(i)));
		m_extrema.push_back(vector<Mat>());
		for (int j = 1; j < m_numIntervals+1; j++) {
			Mat temp = Mat::zeros(m_dogList[i][0].size(),CV_8UC1);
			m_extrema[i].push_back(temp);
			middle = m_dogList[i][j];
			up = m_dogList[i][j + 1];
			down = m_dogList[i][j - 1];
			int xi, yi;
			for (xi = 1; xi < m_dogList[i][j].rows-1; xi++) {
				for (yi = 1; yi < m_dogList[i][j].cols - 1; yi++) {
					bool justSet = false;
					float curr = middle.at<float>(Point(yi, xi));
					if(curr>middle.at<float>(Point(yi+1,xi))&&
						curr>middle.at<float>(Point(yi-1,xi))&&
						curr>middle.at<float>(Point(yi,xi+1))&&
						curr>middle.at<float>(Point(yi,xi-1))&&
						curr>middle.at<float>(Point(yi+1,xi+1))&&
						curr>middle.at<float>(Point(yi-1,xi-1))&&
						curr>middle.at<float>(Point(yi-1,xi+1))&&
						curr>middle.at<float>(Point(yi+1,xi-1))&&
						curr>up.at<float>(Point(yi,xi))&&
						curr>up.at<float>(Point(yi+1,xi))&&
						curr>up.at<float>(Point(yi-1,xi))&&
						curr>up.at<float>(Point(yi,xi+1))&&
						curr>up.at<float>(Point(yi, xi-1))&&
						curr>up.at<float>(Point(yi+1,xi+1))&&
						curr>up.at<float>(Point(yi-1,xi-1))&&
						curr>up.at<float>(Point(yi-1,xi+1))&&
						curr>up.at<float>(Point(yi+1,xi-1))&&
						curr>down.at<float>(Point(yi,xi))&&
						curr>down.at<float>(Point(yi+1,xi))&&
						curr>down.at<float>(Point(yi-1,xi))&&
						curr>down.at<float>(Point(yi,xi+1))&&
						curr>down.at<float>(Point(yi, xi-1))&&
						curr>down.at<float>(Point(yi+1,xi+1))&&
						curr>down.at<float>(Point(yi-1,xi-1))&&
						curr>down.at<float>(Point(yi-1,xi+1))&&
						curr > down.at<float>(Point(yi + 1, xi - 1)) ){
						num++;
						justSet = true;
						m_extrema[i][j - 1].at<uchar>(Point(yi, xi)) = 255;

					}
					//contrast check
					if (justSet && fabs(middle.at<float>(Point(yi, xi))) < CONTRAST_THRESHOLD) {
						num--;
						numRemoved++;
						justSet = false;
						m_extrema[i][j - 1].at<uchar>(Point(yi, xi)) = 0;
					}
					//Edge check
					if (justSet) {
						dxx = middle.at<float>(Point(yi - 1, xi)) + middle.at<float>(Point(yi + 1, xi)) - 2 * middle.at<float>(Point(yi, xi));
						dyy = middle.at<float>(Point(yi, xi + 1)) + middle.at<float>(Point(yi, xi - 1)) - 2 * middle.at<float>(Point(yi,xi));
						dxy = (middle.at<float>(Point(yi + 1, xi + 1)) + middle.at<float>(Point(yi - 1, xi - 1)) - middle.at<float>(Point(yi + 1, xi - 1)) - middle.at<float>(Point(yi - 1, xi + 1))) / 4.0;
						trH = dxx + dyy;
						detH = dxx * dyy - dxy * dxy;
						curvature_ratio = (trH * trH) / detH;
						if (detH < 0 || curvature_ratio < CURVATURE_THRESHOLD) {
							justSet = false;
							num--;
							numRemoved++;
							m_extrema[i][j - 1].at<uchar>(Point(yi, xi)) = 0;
						}
					}

				}
			}
		}
	}
	m_numKeypoints = num;
}
//Get Kernel size
int mySIFT::GetKernelSize(double sigma, double cutoff) {
	int i;
	for (i = 0; i < MAX_KERNEL_SIZE; i++)
		if (exp(-((double)(i * i)) / (2.0 * sigma * sigma)) < cutoff)
			break;
	int size = 2 * i - 1;
	return size;
}
// Assign orientation
void mySIFT::AssignOrientations() {
	int i, j,k, xi, yi, kk, tt;
	vector<vector<Mat>> magnitude;   //2D vectors for storing magnitude and

	vector<vector<Mat>> orientation;   //orintation
	for (i = 0; i < m_numOctaves; i++) {
		magnitude.push_back(vector<Mat>());
		orientation.push_back(vector<Mat>());
		for (j = 1; j < m_numIntervals + 1; j++) {
			Mat temp1 = Mat::zeros(m_glist[i][j].size(), CV_32FC1);
			Mat temp2 = Mat::zeros(m_glist[i][j].size(), CV_32FC1);
			magnitude[i].push_back(temp1);
			orientation[i].push_back(temp2);
			//iterate over current octave and interval
			for (xi = 1; xi < m_glist[i][j].rows - 1; xi++) {
				for (yi = 1; yi < m_glist[i][j].cols - 1; yi++) {
					float dx = m_glist[i][j].at<float>(Point(yi, xi + 1)) - m_glist[i][j].at<float>(Point(yi, xi - 1));
					float dy = m_glist[i][j].at<float>(Point(yi + 1, xi)) - m_glist[i][j].at<float>(Point(yi - 1, xi));
					magnitude[i][j - 1].at<float>(Point(yi, xi)) = sqrt(dx * dx + dy * dy);
					orientation[i][j - 1].at<float>(Point(yi, xi)) = atan(dy / dx);


				}
			}
				
		}
    }
	//8 BIN HISTOGRAM
	float* hist_orient = new float[NUM_BINS];
	//iterate thru all the octaves
	for (int i = 0; i < m_numOctaves; i++)
	{
		int scale = int(pow(2.0, double(i)));
		int width = m_glist[i][0].cols;
		int height = m_glist[i][0].rows;
		//iterate thru all the intervals in the current scale
		for (int j = 1; j < m_numIntervals + 1; j++)
		{
			double abs_sigma = m_absSigma[i][j];
			Mat img_weight = Mat::zeros(magnitude[i][j - 1].size(), CV_32FC1);
			GaussianBlur(magnitude[i][j - 1], img_weight, Size(0, 0), 1.5 * abs_sigma);

			// Get the kernel size for the Guassian blur
			int h = mySIFT::GetKernelSize(1.5 * abs_sigma) / 2;
			//temporarily creating a mask of region
			//to calculate the orientation
			Mat img_mask = Mat::zeros(Size(width, height), CV_8UC1);

			//iterate thru all the points in the octave 
			//and the interval
			for (xi = 1; xi < height - 1; xi++)
			{
				for (yi = 1; yi < width - 1; yi++)
				{
					int val = (int)m_extrema[i][j - 1].at<uchar>(Point(yi, xi));
					if (val != 0)
					{
						// Reset the histogram thing
						for (k = 0; k < NUM_BINS; k++)
							hist_orient[k] = 0.0;
						// Go through all pixels in the window around the extrema
						for (kk = -h; kk <= h; kk++)
						{
							for (tt = -h; tt <= h; tt++)
							{
								// Ensure we're within the image
								if (xi + kk < 0 || xi + kk >= height || yi + tt < 0 || yi + tt >= width)
									continue;

								float sample_orient = orientation[i][j - 1].at<float>(Point(yi + tt, xi + kk));

								//if (sampleOrient <= -M_PI || sampleOrient > M_PI)
									//printf("Bad Orientation: %f\n", sampleOrient);      //CHANGEABLE

								sample_orient += (float)M_PI;

								int sample_orient_degrees = int(sample_orient * 180 / M_PI);
								hist_orient[(int)sample_orient_degrees / (360 / NUM_BINS)] += img_weight.at<float>(Point(yi + tt, xi + kk));
								img_mask.at<uchar>(Point(yi + tt, xi + kk)) = 255;

							}
						}

						// We've computed the histogram. Now check for the maximum
						float max_peak = hist_orient[0];
						int max_peak_index = 0;
						for (k = 1; k < NUM_BINS; k++)
						{
							if (hist_orient[k] > max_peak)
							{
								max_peak = hist_orient[k];
								max_peak_index = k;
							}
						}

						//list of magnitudes and orientation at the current extrema
						vector<float> orien;
						vector<float> mag;

						for (int k = 0; k < NUM_BINS; k++)
						{
							// Do we have a good peak?
							if (hist_orient[k] > 0.8 * max_peak)
							{
								// Three points. (x2,y2) is the peak and (x1,y1)
								// and (x3,y3) are the neigbours to the left and right.
								// If the peak occurs at the extreme left, the "left
								// neighbour" is equal to the right most. Similarly for
								// the other case (peak is rightmost)
								float x1 = (float)(k - 1);
								float y1;
								float x2 = (float)k;
								float y2 = hist_orient[k];
								float x3 = (float)(k + 1);
								float y3;
								//cout << x1 << " " << x2 << " " << x3 << endl;
								if (k == 0)
								{
									y1 = hist_orient[NUM_BINS - 1];
									y3 = hist_orient[1];
								}
								else if (k == NUM_BINS - 1)
								{
									y1 = hist_orient[NUM_BINS - 2];             ///CHANGED
									y3 = hist_orient[0];
								}
								else
								{
									y1 = hist_orient[k - 1];
									y3 = hist_orient[k + 1];
								}




								float* b = new float[3];
								Mat X = Mat(Size(3, 3), CV_32FC1);
								Mat matInv = Mat(Size(3, 3), CV_32FC1);
								X.at<float>(Point(0, 0)) = x1 * x1;
								X.at<float>(Point(1, 0)) = x1;
								X.at<float>(Point(2, 0)) = 1;
								X.at<float>(Point(0, 1)) = x2 * x2;
								X.at<float>(Point(1, 1)) = x2;
								X.at<float>(Point(2, 1)) = 1;
								X.at<float>(Point(0, 2)) = x3 * x3;
								X.at<float>(Point(1, 2)) = x3;
								X.at<float>(Point(2, 2)) = 1;

								//finding and storing the inverse
								invert(X, matInv);
								

								b[0] = matInv.at<float>(Point(0, 0)) * y1 + matInv.at<float>(Point(1, 0)) * y2 + matInv.at<float>(Point(2, 0)) * y3;
								b[1] = matInv.at<float>(Point(0, 1)) * y1 + matInv.at<float>(Point(1, 1)) * y2 + matInv.at<float>(Point(2, 1)) * y3;
								b[2] = matInv.at<float>(Point(0, 2)) * y1 + matInv.at<float>(Point(1, 2)) * y2 + matInv.at<float>(Point(2, 2)) * y3;

								float x0 = -b[1] / (2 * b[0]);
								
								
								if (fabs(x0) > 2 * NUM_BINS)
									x0 = x2;
								while (x0 < 0)
									x0 += NUM_BINS;
								while (x0 >= NUM_BINS)
									x0 -= NUM_BINS;

								// Normalize it
								float s = (float)M_PI / NUM_BINS;
								float x0_n = x0 * (2 * s);
								//cout << x0_n << endl;
								assert(x0_n >= 0 && x0_n < 2 * M_PI);
								x0_n -= (float)M_PI;
								assert(x0_n >= -M_PI && x0_n < M_PI);

								orien.push_back(x0_n);
								//cout << orien.back() << " ";
								mag.push_back(hist_orient[k]);



							}
						}
						//cout << endl;
						// Save this keypoint into the list
						float r = float(xi * scale / 2);
						float s = float(yi * scale / 2);
						
						Keypoint k(r, s, mag, orien, i * m_numIntervals + j - 1);
						m_keyPoints.push_back(k);
					}
				}
			}

			//imshow(to_string(10 * j + i), img_mask);
			//waitKey(0);
		}
	}

	// Finally, we're done with all the magnitude and orientation images.
	//assert(m_keyPoints.size() == m_numKeypoints);

}

Mat mySIFT::BuildInterpolatedGaussianTable(int size, double sigma)
{
	int i, j;
	double half_kernel_size = size / 2 - 0.5;

	double sog = 0;
	Mat ret = Mat::zeros(size, size, CV_32FC1);

	assert(size % 2 == 0);

	double temp = 0;
	for (i = 0; i < size; i++)
	{
		for (j = 0; j < size; j++)
		{
			temp = mySIFT::gaussian2D(i - half_kernel_size, j - half_kernel_size, sigma);
			ret.at<float>(Point(j, i)) = (float)temp;
			sog += (float)temp;
		}
	}

	for (i = 0; i < size; i++)
	{
		for (j = 0; j < size; j++)
		{
			float val = ret.at<float>(Point(j, i));
			ret.at<float>(Point(j, i)) = (float)(1.0 / (sog * val));
		}
	}

	return ret;
}
// gaussian2D
// Returns the value of the bell curve at a (x,y) for a given sigma
double mySIFT::gaussian2D(double x, double y, double sigma)
{
	double ret = 1.0 / (2 * M_PI * sigma * sigma) * exp(-(x * x + y * y) / (2.0 * sigma * sigma));


	return ret;
}
void mySIFT::ShowKeypoints()
{
	Mat img = srcImage.clone();

	for (int i = 0; i < m_keyPoints.size(); i++)
	{
		Keypoint kp = m_keyPoints[i];
		circle(img, Point(kp.xi, kp.yi), 6, Scalar(0, 0, 0), 1);
		//line(img, Point(kp.xi, kp.yi), Point(kp.xi, kp.yi), CV_RGB(255, 255, 255), 3);
		if (kp.orien.empty()) continue;
		else
		{
			line(img, Point(kp.xi, kp.yi), Point(kp.xi + 10 * cos(kp.orien[0]), kp.yi + 10 * sin((float)kp.orien[0])), CV_RGB(180, 255, 230), 1);
		}
	}

	imshow("Keypoints", img);
	waitKey(0);
}




