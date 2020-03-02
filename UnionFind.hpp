#ifndef UNIONFIND_HPP_
#define UNIONFIND_HPP_

#include <vector>

#ifdef _OPENMP
#include <omp.h>
#else
#include <numeric>
#endif // _OPENMP

class UnionFind{
public:
	UnionFind(size_t n) :
		parent(n, 0.0),
		rank(n, 0.0)
	{
		#ifdef _OPENMP
			#pragma omp parallel for
		  	for(size_t i=0; i<n; i++) parent[i] = i;
		#else
			std::iota(parent.begin(), parent.end(), 0);
		#endif // _OPENMP

	}

	size_t find(size_t x) {
	#ifdef _PATH_COMPRESSION
		if(parent[x] == x){
			return x;
		}else{
			return parent[x] = find(parent[x]);
		}
	#else
		while (x != parent[x])
			x = parent[x];
		return x;
	#endif
	}

	bool unite(size_t x, size_t y) {
		x = find(x);
		y = find(y);
		if(x == y) return false;

		if(rank[x] < rank[y]){
			parent[x] = y;
		}else {
			parent[y] = x;
			if(rank[x] == rank[y]) rank[x]++;
		}
		return true;
	}

private:
	std::vector<size_t> parent;
	std::vector<size_t> rank;
};

#endif /* UNIONFIND_HPP_ */
