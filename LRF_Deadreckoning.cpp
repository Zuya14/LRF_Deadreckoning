#include "LRF_Deadreckoning.hpp"
#include "UnionFind.hpp"

#include <cmath>
#include <iostream>
#include <algorithm>

#ifdef DEBUG_OPENCV_
#include <opencv2/opencv.hpp>
#endif /* DEBUG_OPENCV_ */

#ifdef DEBUG_MATPLOT_
#include <matplotlib-cpp/matplotlibcpp.h>
namespace plt = matplotlibcpp; 

#include <string>
#include <unordered_map>
static const double interval = 0.001;
#endif /* DEBUG_MATPLOT_ */

#ifdef _OPENMP
#include <omp.h>
#endif

const double LRF_Deadreckoning::SEGMENTATION_DISTANCE = 100.0;
const double LRF_Deadreckoning::SEGMENTATION_MERGE_RATE = 0.5;
const int LRF_Deadreckoning::POINT_NUM = 1081;// これ動的にしたい...
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

	// 単位変換: [m] -> [mm]
	std::for_each(pts.begin(), pts.end(), [](double& distance) { distance *= 1000.0; });

	mPoints = Eigen::MatrixXd(POINT_NUM, 2);

	#ifdef _OPENMP
	#pragma omp parallel for
	#endif
	for (int i = 0; i < pts.size(); i++) {
		double degree = (i / (double)(POINT_NUM - 1)) * ANGLE_RANGE + ANGLE_OFFSET;
		double theta = degToRad(degree);
		mPoints(i, 0) = pts[i] * std::cos(theta);
		mPoints(i, 1) = pts[i] * std::sin(theta);
	}

	mergeIndices(segmentationIndex(pts, 10), SEGMENTATION_MERGE_RATE);

	mChoicePoints = Eigen::MatrixXd(mChoiceIndices.size(), 2);

	#ifdef _OPENMP
	#pragma omp parallel for
	#endif
	for (int i = 0; i < mChoiceIndices.size(); i++){
		mChoicePoints(i, 0) = mPoints(mChoiceIndices[i], 0);
		mChoicePoints(i, 1) = mPoints(mChoiceIndices[i], 1);
	}

	#ifdef DEBUG_OPENCV_
	//drawPoint(mPoints, 800, 800);
	#endif /* DEBUG_OPENCV_ */

	#ifdef DEBUG_MATPLOT_
	//drawPoint(mPoints);
	#endif /* DEBUG_OPENCV_ */

	//std::cout << mPoints << std::endl << std::endl;
	//std::cout << mPrevPoints << std::endl << std::endl;

	//std::cout << (mPrevPoints.size()) << std::endl << std::endl;

	if (mPrevPoints.size() > 0) {
		//mScanMatcher.icp_ransac(mPrevPoints, mPoints);
		//mScanMatcher.icp_ransac(mPoints, mPrevPoints);
		mScanMatcher.icp_ransac(mChoicePoints, mChoicePrevPoints);

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
		cv::waitKey(1);
		//cv::waitKey(1000);
		#endif /* DEBUG_OPENCV_ */

		#ifdef DEBUG_MATPLOT_
		drawPoints();
		drawMatchedPoints();
		plt::pause(interval);
		#endif /* DEBUG_OPENCV_ */
	}

	mPrevPoints = mPoints;

	mChoicePrevPoints = mChoicePoints;

	return mPrevPoints.size() > 0;
}

std::vector<std::vector<double>> LRF_Deadreckoning::segmentation(std::vector<double> pts, int k) {

	int n = pts.size();
	UnionFind unionFind(n);

	// 0 < k < n
	if (0 > k) k = 1;
	if (n < k) k = n;

	// SEGMENTATION_DISTANCEより近い点を同じ木に
	#ifdef _OPENMP
	#pragma omp parallel for
	#endif
	for (int range = 1; range <= k; range++) {
		for (int i = 0; i + k < n; i++) {
			double degree = (range / (double)(POINT_NUM - 1)) * ANGLE_RANGE;
			double theta = degToRad(degree);
			double r1 = pts[i];
			double r2 = pts[i + range];
			double distance = std::sqrt(r1*r1 + r2*r2 - 2.0*r1*r2*std::cos(theta));
			//std::cout << "distance: " << distance << std::endl;
			#ifdef _OPENMP
			#pragma omp critical
			#endif
			{
				if (distance < SEGMENTATION_DISTANCE) unionFind.unite(i, i + range);
			}
		}
	}

	std::vector<std::vector<double>> segPoints;
	std::vector<int> foundParent; // 各木の根

	#ifdef _OPENMP
	#pragma omp parallel for
	#endif
	for (int i = 0; i < n; i++) {
		// points[i]が所属する木が既知か
		int root = unionFind.find(i);
		std::vector<int>::iterator iter = std::find(foundParent.begin(), foundParent.end(), root);
		int index = std::distance(foundParent.begin(), iter);

		#ifdef _OPENMP
		#pragma omp critical
		#endif
		{
			if (index == foundParent.size()) { // 未知の木を発見
				foundParent.push_back(root);

				std::vector<double> tempPoints;
				tempPoints.push_back(pts[i]);
				segPoints.push_back(tempPoints);
			}
			else {                            // 既知の木を発見
				segPoints[index].push_back(pts[i]);
			}
		}
	}

	// 要素数の多い順にソート
	std::sort(segPoints.begin(), segPoints.end(), [](const std::vector<double>& a, const std::vector<double>& b) { return a.size() > b.size(); });

	return std::move(segPoints);
}

// 
std::vector<std::vector<int>> LRF_Deadreckoning::segmentationIndex(std::vector<double> pts, int k) {

	int n = pts.size();
	UnionFind unionFind(n);

	// 0 < k < n
	if (0 > k) k = 1;
	if (n < k) k = n;

	// SEGMENTATION_DISTANCEより近い点を同じ木に
	#ifdef _OPENMP
	#pragma omp parallel for
	#endif
	for (int range = 1; range <= k; range++) {
		for (int i = 0; i + k < n; i++) {
			double degree = (range / (double)(POINT_NUM - 1)) * ANGLE_RANGE;
			double theta = degToRad(degree);
			double r1 = pts[i];
			double r2 = pts[i + range];
			double distance = std::sqrt(r1 * r1 + r2 * r2 - 2.0 * r1 * r2 * std::cos(theta));
			//std::cout << "distance: " << distance << std::endl;
			#ifdef _OPENMP
			#pragma omp critical
			#endif
			{
				if (distance < SEGMENTATION_DISTANCE) unionFind.unite(i, i + range);
			}
		}
	}

	std::vector<std::vector<int>> segIndices;
	std::vector<int> foundParent; // 各木の根

	#ifdef _OPENMP
	#pragma omp parallel for
	#endif
	for (int i = 0; i < n; i++) {
		// points[i]が所属する木が既知か
		int root = unionFind.find(i);
		std::vector<int>::iterator iter = std::find(foundParent.begin(), foundParent.end(), root);
		int index = std::distance(foundParent.begin(), iter);

		#ifdef _OPENMP
		#pragma omp critical
		#endif
		{
			if (index == foundParent.size()) { // 未知の木を発見
				foundParent.push_back(root);

				std::vector<int> tempIndices;
				tempIndices.push_back(i);
				segIndices.push_back(tempIndices);
			}
			else {                            // 既知の木を発見
				segIndices[index].push_back(i);
			}
		}
	}

	// 要素数の多い順にソート
	std::sort(segIndices.begin(), segIndices.end(), [](const std::vector<int>& a, const std::vector<int>& b) { return a.size() > b.size(); });

	return segIndices;
}

void LRF_Deadreckoning::mergeIndices(std::vector<std::vector<int>> segIndices, double merge_rate) {
	
	if (merge_rate < 0.0) merge_rate = 0.0;
	if (merge_rate > 1.0) merge_rate = 1.0;

	int n = 0;
	std::for_each(segIndices.begin(), segIndices.end(), [&n](std::vector<int>& vec) { n += vec.size(); });

	int k = n * merge_rate;

	std::vector<int> choiceIndices;
	choiceIndices.reserve(n);

	for (std::vector<int>& v : segIndices) {
		std::copy(v.begin(), v.end(), std::back_inserter(choiceIndices));
		if (choiceIndices.size() > k) break;
	}

	std::sort(choiceIndices.begin(), choiceIndices.end());

	std::cout << "segment: " << segIndices.size() << std::endl;
	std::cout << choiceIndices.size() << " > " << k << std::endl;

	//for (int& i : choiceIndices) {
	//	std::cout << i << " ";
	//}
	//std::cout << std::endl;

	//return choiceIndices;
	mChoiceIndices = choiceIndices;
}

#ifdef DEBUG_OPENCV_
void LRF_Deadreckoning::drawPoint(Eigen::MatrixXd pts, int width, int height) {
	cv::Mat src = cv::Mat::zeros(cv::Size(width, height), CV_8UC3);
	cv::line(src, cv::Point(0, height / 2), cv::Point(width, height / 2), cv::Scalar(255, 255, 255));
	cv::line(src, cv::Point(width / 2, 0), cv::Point(width / 2, height), cv::Scalar(255, 255, 255));
	cv::Point center(width / 2, height / 2);
	int row = pts.rows();
	double x, y;
	for (int i = 0; i < row; i++) {
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
	int row1 = mPrevPoints.rows();
	int row2 = mPoints.rows();
	double x1, y1, x2, y2;
	for (int i = 0; i < row1; i++) {
		x1 = mPrevPoints(i, 0) / 10.0;
		y1 = mPrevPoints(i, 1) / 10.0;
		cv::circle(src, cv::Point(x1, -y1) + center, 1, cv::Scalar(255, 0, 0));
	}
	for (int i = 0; i < row2; i++){
		x2 = mPoints(i, 0) / 10.0;
		y2 = mPoints(i, 1) / 10.0;
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
	double cos = std::cos(-m_dtheta);
	double sin = std::sin(-m_dtheta);
	int row1 = mPrevPoints.rows();
	int row2 = mPoints.rows();
	for (int i = 0; i < row1; i++) {
		x = mPrevPoints(i, 0);
		y = mPrevPoints(i, 1);
		x1 = x * cos - y * sin - m_dx;
		y1 = x * sin + y * cos - m_dy;
		x1 = x1 / 10.0;
		y1 = y1 / 10.0;
		cv::circle(src, cv::Point(x1, -y1) + center, 1, cv::Scalar(255, 0, 0));
	}
	for (int i = 0; i < row2; i++) {
		x2 = mPoints(i, 0) / 10.0;
		y2 = mPoints(i, 1) / 10.0;
		cv::circle(src, cv::Point(x2, -y2) + center, 1, cv::Scalar(0, 0, 255));
	}
	cv::imshow("Matched Points", src);
	//cv::waitKey(1000);
}

void LRF_Deadreckoning::drawSegmentation(int num) {

	if (num < 0) num = 1;
	if (num > 5) num = 5;

	static cv::Scalar colors[] = {
		cv::Scalar(0, 0, 255),
		cv::Scalar(255, 0, 0),
		cv::Scalar(255,255, 0),
		cv::Scalar(0 ,255, 0),
		cv::Scalar(128, 0,128)
	};


}
#endif /* DEBUG_OPENCV_ */

#ifdef DEBUG_MATPLOT_
void LRF_Deadreckoning::drawPoint(Eigen::MatrixXd pts) {
	int row = pts.rows();
	std::vector<double> x(row), y(row);
	for (int i = 0; i < row; i++) {
		x.at(i) = pts(i, 0);
		y.at(i) = pts(i, 1);
	}

	std::unordered_map<std::string, std::string> um1{{"c", "red"}, {"marker", "."}, {"linewidths", "0"}};
	plt::clf();
	plt::xlim(-3000.0, 3000.0);
	plt::ylim(-3000.0, 3000.0);
	plt::scatter(x, y, 10, um1);

	std::unordered_map<std::string, std::string> um2{{"c", "black"}, {"marker", "*"}};
	std::vector<double> x0(1,0), y0(1,0);
	plt::scatter(x0, y0, 50, um2);
	//plt::show();	
	plt::pause(interval);
}

void LRF_Deadreckoning::drawPoints() {
	int row1 = mPrevPoints.rows();
	int row2 = mPoints.rows();
	std::vector<double> x1(row1), y1(row1);
	std::vector<double> x2(row2), y2(row2);
	for (int i = 0; i < row1; i++) {
		x1.at(i) = mPrevPoints(i, 0);
		y1.at(i) = mPrevPoints(i, 1);
	}
	for (int i = 0; i < row2; i++) {
		x2.at(i) = mPoints(i, 0);
		y2.at(i) = mPoints(i, 1);
	}

	plt::figure(1);
	plt::clf();
	
	std::unordered_map<std::string, std::string> um0{{"c", "blue"}, {"marker", "."}, {"linewidths", "0"}};
	//plt::subplot(1, 2, 1);
	plt::xlim(-3000.0, 3000.0);
	plt::ylim(-3000.0, 3000.0);
	plt::scatter(x1, y1, 10, um0);

	std::unordered_map<std::string, std::string> um1{{"c", "red"}, {"marker", "."}, {"linewidths", "0"}};
	plt::scatter(x2, y2, 10, um1);

	std::unordered_map<std::string, std::string> um2{{"c", "black"}, {"marker", "*"}};
	std::vector<double> x0(1,0), y0(1,0);
	plt::scatter(x0, y0, 50, um2);	
	//plt::pause(interval);
}

void LRF_Deadreckoning::drawMatchedPoints() {
	int row1 = mPrevPoints.rows();
	int row2 = mPoints.rows();
	std::vector<double> x1(row1), y1(row1);
	std::vector<double> x2(row2), y2(row2);
	double cos = std::cos(-m_dtheta);
	double sin = std::sin(-m_dtheta);
	double x, y;
	for (int i = 0; i < row1; i++) {
		x = mPrevPoints(i, 0);
		y = mPrevPoints(i, 1);
		x1.at(i) = x * cos - y * sin - m_dx;
		y1.at(i) = x * sin + y * cos - m_dy;
	}
	for (int i = 0; i < row2; i++) {
		x2.at(i) = mPoints(i, 0);
		y2.at(i) = mPoints(i, 1);
	}

	plt::figure(2);
	plt::clf();

	std::unordered_map<std::string, std::string> um0{{"c", "blue"}, {"marker", "."}, {"linewidths", "0"}};
	//plt::subplot(1, 2, 2);
	plt::xlim(-3000.0, 3000.0);
	plt::ylim(-3000.0, 3000.0);
	plt::scatter(x1, y1, 10, um0);

	std::unordered_map<std::string, std::string> um1{{"c", "red"}, {"marker", "."}, {"linewidths", "0"}};
	plt::scatter(x2, y2, 10, um1);

	std::unordered_map<std::string, std::string> um2{{"c", "black"}, {"marker", "*"}};
	std::vector<double> x0(1,0), y0(1,0);
	plt::scatter(x0, y0, 50, um2);
	//plt::show();	
	//plt::pause(interval);
}

void LRF_Deadreckoning::drawSegmentation(int num) {
	static std::String colors[] = {
		"red",
		"blue",
		"yellow",
		"green",
		"purple"
	};

}
#endif /* DEBUG_MATPLOT_ */
