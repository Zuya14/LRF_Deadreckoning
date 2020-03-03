#ifndef LRF_DEADRECKONING_HPP_
#define LRF_DEADRECKONING_HPP_

//#define DEBUG_MATPLOT_
//#define DEBUG_OPENCV_

#include <vector>

#include "Eigen/Eigen"
#include"ScanMatcher.hpp"

class LRF_Deadreckoning {
public:
	LRF_Deadreckoning();

	bool update(std::vector<double> pts);

	double getDX(){ return m_dx; }
	double getDY(){ return m_dy; }
	double getDTheta(){ return m_dtheta; }

	double getX(){ return m_x; }
	double getY(){ return m_y; }
	double getTheta(){ return m_theta; }
private:
	double degToRad(double x);
	std::vector<std::vector<double>> segmentation(std::vector<double> pts, int k);
	std::vector<std::vector<int>> segmentationIndex(std::vector<double> pts, int k);
	void mergeIndices(std::vector<std::vector<int>> segIndices, double merge_rate);


	ScanMatcher mScanMatcher;
	Eigen::MatrixXd mPoints;
	Eigen::MatrixXd mPrevPoints;

	Eigen::MatrixXd mChoicePoints;
	Eigen::MatrixXd mChoicePrevPoints;

	double m_dx;
	double m_dy;
	double m_dtheta;

	double m_x, m_y, m_theta;

	std::vector<std::vector<int>> mSegIndices;
	std::vector<int> mChoiceIndices;

	static const double SEGMENTATION_DISTANCE;
	static const double SEGMENTATION_MERGE_RATE;
	static const int POINT_NUM;
	static const double ANGLE_RANGE;
	static const double ANGLE_OFFSET;
	static const double PI;

	#ifdef DEBUG_OPENCV_
	void drawPoint(Eigen::MatrixXd pts, int width, int height);
	void drawPoints(int width, int height);
	void drawMatchedPoints(int width, int height);
	void drawSegmentation(int num = 3);
	#endif /* DEBUG_OPENCV_ */

	#ifdef DEBUG_MATPLOT_
	void drawPoint(Eigen::MatrixXd pts);
	void drawPoints();
	void drawMatchedPoints();
	void drawSegmentation(int num = 3);
	#endif /* DEBUG_MATPLOT_ */
};

#endif /* LRF_DEADRECKONING_HPP_ */
