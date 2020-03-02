#include <iostream>
#include <stdio.h>
#include <string>  
#include <fstream>  
#include <sstream>   

#include <vector>

#include <chrono>

#include "LRF_Deadreckoning.hpp"

const double PI = 3.14159265358979323846;

void test_pt() {
	int test_n = 1080;
	Eigen::MatrixXd A = Eigen::MatrixXd::Random(test_n, 2);

	Eigen::Vector2d t = Eigen::Vector2d(30, 10);

	Eigen::MatrixXd B;
	B = A;
	B.rowwise() += t.transpose();

	double thata = PI / 4.0;
	double cos = std::cos(thata);
	double sin = std::sin(thata);

	std::cout << t << std::endl << std::endl;
	std::cout << cos << ", " << sin << std::endl << std::endl;

	Eigen::MatrixXd C;
	C = B;

	for (int i = 0; i < A.rows(); i++) {
		C(i, 0) = B(i, 0) * cos - B(i, 1) * sin;
		C(i, 1) = B(i, 0) * sin + B(i, 1) * cos;
		//C(i, 0) = B(i, 0) * cos - B(i, 1) * sin + ((rand() % 100) - 50) / 100.0;
		//C(i, 1) = B(i, 0) * sin + B(i, 1) * cos + ((rand() % 100) - 50) / 100.0;
	}

	//std::cout << A << std::endl << std::endl;
	//std::cout << B << std::endl << std::endl;
	//std::cout << C << std::endl << std::endl;

	ScanMatcher matcher;

	matcher.icp(A, B);
	std::cout << matcher.getT() << std::endl << std::endl;
	matcher.icp_ransac(A, B);
	std::cout << matcher.getT() << std::endl << std::endl;

	matcher.icp(A, C);
	std::cout << matcher.getT() << std::endl << std::endl;
	matcher.icp_ransac(A, C);
	std::cout << matcher.getT() << std::endl << std::endl;
}

void test() {
	// std::string filename = "lidar_10_short.txt";
	//std::string filename = "lidar_10_long.txt";
	//std::string filename = "lidar_25_short.txt";
	std::string filename = "lidar_25_long.txt";

	std::ifstream read_file(filename, std::ios::in);
	std::string read_line;

	//printf("reading %s\n", filename);

	std::vector<double> points;
	points.reserve(1082);

	LRF_Deadreckoning lrfDR;

	while (!read_file.eof()) {
		points.clear();

		std::getline(read_file, read_line);
		std::string read_cell;
		std::istringstream iss(read_line);

		while (std::getline(iss, read_cell, '\t')) {
			points.push_back(std::stod(read_cell));
		}

		//printf("points.size:%u\n", points.size());

		if (points.size() == 1081) {
			std::chrono::system_clock::time_point start, end;
			start = std::chrono::system_clock::now();
			bool success = lrfDR.update(points);
			end = std::chrono::system_clock::now();
			double time = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0);
			if (success)
				printf("%f[ms]\n\n", time);
		}
	}
}

int main(){

	//test_pt();

	test();

	//int a[3] = { 1, 2, 3 };
	//std::vector<int> v(std::begin(a), std::end(a));

	//std::cout << typeid(a).name() << std::endl << std::endl;

	//for (auto& x : a)
	//	std::cout << x << " ";
	//std::cout << std::endl;

	//for (auto& x : v)
	//	std::cout << x << " ";
	//std::cout << std::endl;
}