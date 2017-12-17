#ifndef _SYMMETRY_POINT_H
#define _SYMMETRY_POINT_H

//
// Author:  Christoph Dalitz
// Version: 2.0a from 2014/04/16
//

#include <opencv2/core/core.hpp>
#include <vector>

namespace sym
{
	typedef std::vector<cv::Point> PointVector;

	// classes for storing a list of symmetry points
	class SymmetryPoint {
	public:
	  cv::Point point;
	  double value;
	  SymmetryPoint(const SymmetryPoint& s) {
		point = s.point; value = s.value;
	  }
	  SymmetryPoint(const cv::Point& p, double v) {
		point = p; value = v;
	  }
	  SymmetryPoint& operator=(const SymmetryPoint& s) {
		point = s.point; value = s.value;
		return *this;
	  }
	  bool operator<(const SymmetryPoint& s) const {
		return (value < s.value);
	  }
	};
	typedef std::vector<SymmetryPoint> SymmetryPointVector;


	// helper function for descending sort
	inline bool sortinverse(const SymmetryPoint& p1, const SymmetryPoint& p2)
	{
	  return (p1.value>p2.value);
	}

	// computes the neighbors of a given point
	// warning: one pixel wide image border is ignored
	// return value: number of neighbors
	inline int get_neighbors(const cv::Point &p, PointVector* neighbors, int maxx, int maxy)
	{
	  neighbors->clear();
	  if (p.y>0 && p.y<maxy && p.x>0 && p.x<maxy) {
		neighbors->push_back(cv::Point(p.x-1,p.y-1));
		neighbors->push_back(cv::Point(p.x,p.y-1));
		neighbors->push_back(cv::Point(p.x+1,p.y-1));
		neighbors->push_back(cv::Point(p.x+1,p.y));
		neighbors->push_back(cv::Point(p.x+1,p.y+1));
		neighbors->push_back(cv::Point(p.x,p.y+1));
		neighbors->push_back(cv::Point(p.x-1,p.y+1));
		neighbors->push_back(cv::Point(p.x-1,p.y));
	  }
	  return neighbors->size();
	}

	#endif
}