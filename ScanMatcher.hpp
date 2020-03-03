#ifndef SVD_MATCHING_HPP_
#define SVD_MATCHING_HPP_

#include "Eigen/Eigen"
#include <vector>

class ScanMatcher{
public:
	ScanMatcher();

	void icp_ransac(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B);

	void icp(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B);

	Eigen::Matrix3d getT() { return mT; }

private:
	Eigen::Matrix3d fit_transform(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B);
	std::vector<int> random_select(int n, int min, int max);
	std::vector<int> sorted_random_select(int n, int min, int max);
	void closest(const Eigen::MatrixXd& src, const Eigen::MatrixXd& dst);
	void closest(const Eigen::MatrixXd& src, const Eigen::MatrixXd& dst, int k);
	double distance(const Eigen::Vector2d& p1, const Eigen::Vector2d& p2);

	Eigen::Matrix3d mT; // [R|T]:3*3
	std::vector<double> mPtPairsDistance;
	std::vector<int> mPtPairsIndex;
	int mPairNum;

	static const double NEIGHBOR_DISTANCE;
	static const bool NEAREST_FULL;
	static const bool INDEX_DEPEND;
	static const int NEAREST_K;
	static const double RANSAC_SAMPLE_RATE;
	static const int RANSAC_MAX_ITERATIONS;
	static const int MAX_ITERATIONS;
	static const double EPS;
};

#endif /* SVD_MATCHING_HPP_ */
