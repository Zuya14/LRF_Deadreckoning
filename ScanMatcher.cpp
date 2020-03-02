#include "ScanMatcher.hpp"

#include <stdio.h>
#include <iostream>

#include <numeric>
#include <chrono>

#include <algorithm>
#include <random>

#include "float.h"

#ifdef _OPENMP
#include <omp.h>
#endif

const double ScanMatcher::NEIGHBOR_DISTANCE = DBL_MAX; // [mm] これより近い点はご近所さん。DBL_MAXでOKなはず
const bool   ScanMatcher::NEAREST_FULL = true;        // NEAREST_FULL ? 全探索 : 近傍探索
const bool   ScanMatcher::INDEX_DEPEND = false;        // インデックスによる近傍探索をするか
const int    ScanMatcher::NEAREST_K = 25;              // 2k+1近傍探索

const double ScanMatcher::RANSAC_SAMPLE_RATE = 0.10;  // RANSACのサンプル率
const int    ScanMatcher::RANSAC_MAX_ITERATIONS = 10; // RANSACの試行回数 

const double ScanMatcher::EPS = 1e-6;          // マッチングの改善がEPS以下なら終了
const int    ScanMatcher::MAX_ITERATIONS = 10; // マッチングの最大試行回数

ScanMatcher::ScanMatcher() :
	mT(Eigen::MatrixXd::Identity(3, 3)),
	mPtPairsDistance(),
	mPtPairsIndex()
{}

void ScanMatcher::icp_ransac(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B) {
	int rowA = A.rows();
	int rowB = B.rows();

	Eigen::MatrixXd A_move = Eigen::MatrixXd::Ones(2, rowA);
	Eigen::MatrixXd A_move2 = Eigen::MatrixXd::Ones(2, rowA);
	Eigen::MatrixXd A_temp = Eigen::MatrixXd::Ones(2 + 1, rowA);
	Eigen::MatrixXd A_temp2 = Eigen::MatrixXd::Ones(2 + 1, rowA);
	//Eigen::MatrixXd choice = Eigen::MatrixXd::Ones(2, row);
	Eigen::Matrix3d T;
	double prev_error = 0;
	double error = 0;

	int sample_n = rowA * RANSAC_SAMPLE_RATE;
	Eigen::Matrix3d T_sample_best;
	Eigen::MatrixXd A_sample(sample_n, 2);
	Eigen::MatrixXd B_sample(sample_n, 2);
	Eigen::MatrixXd choice_sample = Eigen::MatrixXd::Ones(2, sample_n);
	Eigen::Matrix3d T_sample;
	double min_error = DBL_MAX;
	double error_temp = 0;

	#ifdef _OPENMP
	#pragma omp parallel for
	#endif
	for (int j = 0; j < rowA; j++) {
		A_move.block<2, 1>(0, j) = A.block<1, 2>(j, 0).transpose();
		A_temp.block<2, 1>(0, j) = A.block<1, 2>(j, 0).transpose();
	}

	int iter;
	for (iter = 0; iter < MAX_ITERATIONS; iter++) {

		// とりあえず、収束基準なしで定数回のRANSAC。
		for (int i = 0; i < RANSAC_MAX_ITERATIONS; i++) {

			// サンプリング
			if (INDEX_DEPEND) {
				int row = (rowA > rowB) ? rowA : rowB;
				std::vector<int> indices = sorted_random_select(sample_n, 0, row - 1);

				#ifdef _OPENMP
				#pragma omp parallel for
				#endif
				for (int i = 0; i < sample_n; i++) {
					int index = indices[i];
					A_sample.block<1, 2>(i, 0) = A_move.block<2, 1>(0, index);
					B_sample.block<1, 2>(i, 0) = B.block<1, 2>(index, 0);
				}
			}else {
				std::vector<int> indicesA = sorted_random_select(sample_n, 0, rowA - 1);
				std::vector<int> indicesB = sorted_random_select(sample_n, 0, rowB - 1);

				#ifdef _OPENMP
				#pragma omp parallel for
				#endif
				for (int i = 0; i < sample_n; i++) {
					A_sample.block<1, 2>(i, 0) = A_move.block<2, 1>(0, indicesA[i]);
					B_sample.block<1, 2>(i, 0) = B.block<1, 2>(indicesB[i], 0);
				}
			}

			// サンプル内での対応点探索
			if (NEAREST_FULL) {
				closest(A_sample, B_sample);
			}else {
				closest(A_sample, B_sample, NEAREST_K);
			}

			#ifdef _OPENMP
			#pragma omp parallel for
			#endif
			for (int j = 0; j < sample_n; j++) {
				choice_sample.block<2, 1>(0, j) = B_sample.block<1, 2>(mPtPairsIndex[j], 0);
			}

			T_sample = fit_transform(A_sample, choice_sample.transpose());
			A_temp2 = T_sample * A_temp;

			#ifdef _OPENMP
			#pragma omp parallel for
			#endif
			for (int j = 0; j < rowA; j++) {
				A_move2.block<2, 1>(0, j) = A_temp2.block<2, 1>(0, j);
			}

			// 全体での対応点探索
			if (NEAREST_FULL) {
				closest(A_move2.transpose(), B);
			}else {
				closest(A_move2.transpose(), B, NEAREST_K);
			}

			error_temp = std::accumulate(mPtPairsDistance.begin(), mPtPairsDistance.end(), 0.0) / mPtPairsDistance.size();

			if (error_temp < min_error) {
				min_error = error_temp;
				T_sample_best = T_sample;
			}
		}

		//std::cout << "best, iter:" << iter << std::endl;
		//std::cout << min_error << std::endl;
		//std::cout << T_sample_best << std::endl << std::endl;

		T = T_sample_best;
		A_temp = T * A_temp;

		#ifdef _OPENMP
		#pragma omp parallel for
		#endif
		for (int j = 0; j < rowA; j++) {
			A_move.block<2, 1>(0, j) = A_temp.block<2, 1>(0, j);
		}

		error = min_error;

		if (std::abs(prev_error - error) < EPS) {
			break;
		}

		prev_error = error;
	}
	printf("ITER: %d, EROOR: %.2f\n", iter, error);

	//mT = fit_transform(A_move.transpose(), A);
	mT = fit_transform(A, A_move.transpose());
}

void ScanMatcher::icp(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B) {

	int row = A.rows();
	
	Eigen::MatrixXd A_move = Eigen::MatrixXd::Ones(2, row);
	Eigen::MatrixXd A_temp = Eigen::MatrixXd::Ones(2 + 1, row);
	Eigen::MatrixXd choice = Eigen::MatrixXd::Ones(2, row);
	Eigen::Matrix3d T;

	double prev_error = 0;
	double error = 0;

	for (int j = 0; j < row; j++) {
		A_move.block<2, 1>(0, j) = A.block<1, 2>(j, 0).transpose();
		A_temp.block<2, 1>(0, j) = A.block<1, 2>(j, 0).transpose();
	}

	int iter;
	for (iter = 0; iter < MAX_ITERATIONS; iter++) {
		closest(A_move.transpose(), B, NEAREST_K);

		//std::cout << A_move.transpose().rows() << ", " << A_move.transpose().cols() << std::endl << std::endl;
		//std::cout << B.rows() << ", " << B.cols() << std::endl << std::endl;

		for (int j = 0; j < row; j++) {
			choice.block<2, 1>(0, j) = B.block<1, 2>(mPtPairsIndex[j], 0);
		}

		T = fit_transform(A_move.transpose(), choice.transpose());
		A_temp = T * A_temp;
		
		for (int j = 0; j < row; j++) {
			A_move.block<2, 1>(0, j) = A_temp.block<2, 1>(0, j);
		}

		error = std::accumulate(mPtPairsDistance.begin(), mPtPairsDistance.end(), 0.0) / mPtPairsDistance.size();
		
		if (std::abs(prev_error - error) < EPS) {
			break;
		}

		prev_error = error;
	}
	printf("ITER: %d\n", iter);

	mT = fit_transform(A, A_move.transpose());
}

// 時系列で点群がアフィン変換したものとして移動量予測
Eigen::Matrix3d ScanMatcher::fit_transform(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B) {

	Eigen::Matrix3d T = Eigen::MatrixXd::Identity(3, 3);

	Eigen::Vector2d centroid_A(0, 0), centroid_B(0, 0);
	Eigen::MatrixXd AA = A, BB = B;

	// 重心
	centroid_A = AA.colwise().mean();
	centroid_B = BB.colwise().mean();

	//// 正規化
	AA.rowwise() -= centroid_A.transpose();
	BB.rowwise() -= centroid_B.transpose();

	// 共分散
	Eigen::MatrixXd Sigma = AA.transpose() * BB;

	// 特異値分解
	Eigen::JacobiSVD<Eigen::MatrixXd> svd(Sigma, Eigen::ComputeFullU | Eigen::ComputeFullV);
	Eigen::MatrixXd U = svd.matrixU();
	Eigen::VectorXd S = svd.singularValues();
	Eigen::MatrixXd V = svd.matrixV();

	// 回転行列
	Eigen::Matrix2d R = V * U.transpose();

	// 並進行列
	Eigen::Vector2d t = centroid_B - R * centroid_A;

	T.block<2, 2>(0, 0) = R;
	T.block<2, 1>(0, 2) = t;

	return T;
}

// [min, max]の範囲でn個の乱数を重複なく見つける
std::vector<int> ScanMatcher::random_select(int n, int min, int max) {

	if (min > max)
		std::swap(min, max);
	const int max_min_diff = max - min + 1;

	if (max_min_diff < n) {
		printf("random_select: range error - %d, set range - %d", n, max_min_diff);
		n = max_min_diff;
	}

	// tmp := {min, min+1, ..., max-1, max}
	std::vector<int> tmp(max_min_diff);
	std::iota(tmp.begin(), tmp.end(), min);

	std::random_device rnd;
	std::mt19937_64 engine(rnd());

	// tmpのn番目までシャッフル
	#ifdef _OPENMP
	#pragma omp parallel for
	#endif
	for (int i = 0; i < n; ++i) {
		int pos = std::uniform_int_distribution<>(i, tmp.size() - 1)(engine);
		if (i != pos) std::swap(tmp[i], tmp[pos]);
	}
	// n番目まで残す
	tmp.erase(std::next(tmp.begin(), n), tmp.end());

	return std::move(tmp);
}

std::vector<int> ScanMatcher::sorted_random_select(int n, int min, int max) {
	std::vector<int> indices = random_select(n, min, max);
	std::sort(indices.begin(), indices.end());
	return std::move(indices);
}

// 距離が近い点のペアを探す
void ScanMatcher::closest(const Eigen::MatrixXd& src, const Eigen::MatrixXd& dst) {
	mPtPairsDistance.clear();
	mPtPairsIndex.clear();

	int row_src = src.rows();
	int row_dst = dst.rows();

	Eigen::Vector2d vec_src, vec_dst;

	double min = 0;
	double dist = 0;
	int index = 0;

	for (int i = 0; i < row_src; i++) {
		vec_src = src.block<1, 2>(i, 0).transpose();
		min = NEIGHBOR_DISTANCE;
		index = 0;
		dist = 0;
		#ifdef _OPENMP
		#pragma omp parallel for
		#endif
		for (int j = 0; j < row_dst; j++) {
			vec_dst = dst.block<1, 2>(j, 0).transpose();
			dist = distance(vec_src, vec_dst);
			#ifdef _OPENMP
			#pragma omp critical
			#endif
			{
				if (dist < min) {
					min = dist;
					index = j;
			}
		}
		}
		mPtPairsDistance.push_back(min);
		mPtPairsIndex.push_back(index);
	}
}


// インデックスで(2k+1)近傍。距離が近い点のペアを探す
void ScanMatcher::closest(const Eigen::MatrixXd& src, const Eigen::MatrixXd& dst, int k) {
	mPtPairsDistance.clear();
	mPtPairsIndex.clear();
	
	int row_src = src.rows();
	int row_dst = dst.rows();

	Eigen::Vector2d vec_src, vec_dst;

	double min = 0;
	double dist = 0;
	int index = 0;

	for (int i = 0; i < row_src; i++) {
		vec_src = src.block<1, 2>(i, 0).transpose();
		min = NEIGHBOR_DISTANCE;
		index = dist = 0;
		int ks = i - k;
		int kt = i + k;
		if (0 > ks)
			ks = 0;
		if (row_dst <= kt)
			kt = row_dst - 1;

		#ifdef _OPENMP
		#pragma omp parallel for
		#endif
		for (int j = ks; j <= kt; j++) {
			vec_dst = dst.block<1, 2>(j, 0).transpose();
			dist = distance(vec_src, vec_dst);
			#ifdef _OPENMP
			#pragma omp critical
			#endif
			{
				if (dist < min) {
					min = dist;
					index = j;
				}
			}
		}
		mPtPairsDistance.push_back(min);
		mPtPairsIndex.push_back(index);
	}
}

double ScanMatcher::distance(const Eigen::Vector2d& p1, const Eigen::Vector2d& p2) {
	double dx = p1[0] - p2[0];
	double dy = p1[1] - p2[1];
	return std::sqrt(dx * dx + dy * dy);
}
