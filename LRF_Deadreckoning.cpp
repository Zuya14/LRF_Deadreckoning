#include "LRF_Deadreckoning.hpp"

#include <cmath>
#include <iostream>

const int LRF_Deadreckoning::POINT_NUM = 1081;
const double LRF_Deadreckoning::ANGLE_RANGE = 270.0;
const double LRF_Deadreckoning::ANGLE_OFFSET = -45.0;
const double LRF_Deadreckoning::PI = 3.14159265358979323846;

LRF_Deadreckoning::LRF_Deadreckoning() :
	mScanMatcher(),
	m_dx(), m_dy(), m_dtheta(),
	m_x(), m_y(), m_theta()
{}

double LRF_Deadreckoning::degToRad(double x) {
	return ((x)* PI / 180.0);
}

bool LRF_Deadreckoning::update(std::vector<double> pts) {

	mPoints = Eigen::MatrixXd(POINT_NUM, 2);

	for (size_t i = 0; i < pts.size(); i++) {
		double degree = (i / (double)(POINT_NUM - 1)) * ANGLE_RANGE + ANGLE_OFFSET;
		double theta = degToRad(degree);
		mPoints(i, 0) = pts[i] * std::cos(theta);
		mPoints(i, 1) = pts[i] * std::sin(theta);
	}

	#ifdef DEBUG_OPENCV_
	//drawPoint(mPoints, 800, 800);
	#endif /* DEBUG_OPENCV_ */

	//std::cout << mPoints << std::endl << std::endl;
	//std::cout << mPrevPoints << std::endl << std::endl;

	//std::cout << (mPrevPoints.size()) << std::endl << std::endl;

	if (mPrevPoints.size() > 0) {
		mScanMatcher.icp_ransac(mPrevPoints, mPoints);
		//mScanMatcher.icp_ransac(mPoints, mPrevPoints);
		m_dx = mScanMatcher.getT()(0, 2);
		m_dy = mScanMatcher.getT()(1, 2);
		m_dtheta = atan2(mScanMatcher.getT()(1, 0), mScanMatcher.getT()(0, 0));

		m_x += m_dx;
		m_y += m_dy;
		m_theta += m_dtheta;

		//std::cout << mScanMatcher.getT() << std::endl;
		printf("dx: %.1f, dy:%.1f, dtheta:%.3f\n", m_dx, m_dy, m_dtheta);
		printf("x: %.1f, y:%.1f, theta:%.3f\n", m_x, m_y, m_theta);

		#ifdef DEBUG_OPENCV_
		drawPoints(800, 800);
		drawMatchedPoints(800, 800);
		//cv::waitKey(1);
		cv::waitKey(1000);
		#endif /* DEBUG_OPENCV_ */
	}

	mPrevPoints = mPoints;

	return mPrevPoints.size() > 0;
}

#ifdef DEBUG_OPENCV_
void LRF_Deadreckoning::drawPoint(Eigen::MatrixXd pts, int width, int height) {
	cv::Mat src = cv::Mat::zeros(cv::Size(width, height), CV_8UC3);
	cv::line(src, cv::Point(0, height / 2), cv::Point(width, height / 2), cv::Scalar(255, 255, 255));
	cv::line(src, cv::Point(width / 2, 0), cv::Point(width / 2, height), cv::Scalar(255, 255, 255));
	cv::Point center(width / 2, height / 2);
	int row = pts.rows();
	double x, y;
	for (size_t i = 0; i < row; i++) {
		x = pts(i, 0) / 10.0;
		y = pts(i, 1) / 10.0;
		cv::circle(src, cv::Point(x, -y) + center, 1, cv::Scalar(0, 0, 255));
	}
	cv::imshow("Points", src);
	//cv::waitKey(1000);
}

void LRF_Deadreckoning::drawPoints(int width, int height) {
	cv::Mat src = cv::Mat::zeros(cv::Size(width, height), CV_8UC3);
	cv::line(src, cv::Point(0, height / 2), cv::Point(width, height / 2), cv::Scalar(255, 255, 255));
	cv::line(src, cv::Point(width / 2, 0), cv::Point(width / 2, height), cv::Scalar(255, 255, 255));
	cv::Point center(width / 2, height / 2);
	int row = mPoints.rows();
	double x1, y1, x2, y2;
	for (size_t i = 0; i < row; i++){
		x1 = mPrevPoints(i, 0) / 10.0;
		y1 = mPrevPoints(i, 1) / 10.0;
		x2 = mPoints(i, 0) / 10.0;
		y2 = mPoints(i, 1) / 10.0;
		cv::circle(src, cv::Point(x1, -y1) + center, 1, cv::Scalar(0, 255, 255));
		cv::circle(src, cv::Point(x2, -y2) + center, 1, cv::Scalar(0, 0, 255));
	}
	cv::imshow("Points", src);
	//cv::waitKey(1000);
}

void LRF_Deadreckoning::drawMatchedPoints(int width, int height) {
	cv::Mat src = cv::Mat::zeros(cv::Size(width, height), CV_8UC3);
	cv::line(src, cv::Point(0, height / 2), cv::Point(width, height / 2), cv::Scalar(255, 255, 255));
	cv::line(src, cv::Point(width / 2, 0), cv::Point(width / 2, height), cv::Scalar(255, 255, 255));
	cv::Point center(width / 2, height / 2);
	int row = mPoints.rows();
	double x, y, x1, y1, x2, y2;

	Eigen::MatrixXd pt;
	double cos = std::cos(m_dtheta);
	double sin = std::sin(m_dtheta);
	for (size_t i = 0; i < row; i++) {

		x = mPrevPoints(i, 0);
		y = mPrevPoints(i, 1);
		x1 = x * cos - y * sin + m_dx;
		y1 = x * sin + y * cos + m_dy;

		x1 = x1 / 10.0;
		y1 = y1 / 10.0;
		x2 = mPoints(i, 0) / 10.0;
		y2 = mPoints(i, 1) / 10.0;
		cv::circle(src, cv::Point(x1, -y1) + center, 1, cv::Scalar(0, 255, 255));
		cv::circle(src, cv::Point(x2, -y2) + center, 1, cv::Scalar(0, 0, 255));
	}
	cv::imshow("Matched Points", src);
	//cv::waitKey(1000);
}
#endif /* DEBUG_OPENCV_ */
