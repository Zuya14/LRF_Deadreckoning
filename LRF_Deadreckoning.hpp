#ifndef LRF_DEADRECKONING_HPP_
#define LRF_DEADRECKONING_HPP_

//#define DEBUG_OPENCV_

#ifdef DEBUG_OPENCV_
#include <opencv2/opencv.hpp>
#endif /* DEBUG_OPENCV_ */

#include <vector>

#include "Eigen/Eigen"
#include"ScanMatcher.hpp"

class LRF_Deadreckoning {
public:
	LRF_Deadreckoning();

	bool update(std::vector<double> pts);

private:
	//std::vector<Point> mPoints;
	//std::vector<Point> mPrevPoints;

	double degToRad(double x);

	ScanMatcher mScanMatcher;
	Eigen::MatrixXd mPoints;
	Eigen::MatrixXd mPrevPoints;

	double m_dx;
	double m_dy;
	double m_dtheta;

	double m_x, m_y, m_theta;

	static const int POINT_NUM;
	static const double ANGLE_RANGE;
	static const double ANGLE_OFFSET;
	static const double PI;


	#ifdef DEBUG_OPENCV_
	void drawPoint(Eigen::MatrixXd pts, int width, int height);
	void drawPoints(int width, int height);
	void drawMatchedPoints(int width, int height);
	#endif /* DEBUG_OPENCV_ */
};

#endif /* LRF_DEADRECKONING_HPP_ */