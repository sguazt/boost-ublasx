#include <iterator>
#include <string>
#include <ctime>
#include <vector>
#include <iostream>
#include <fstream>
#include <boost/assign/std/vector.hpp> 
#include <boost/random.hpp>
#include <boost/assert.hpp>
#include <boost/numeric/ublas/io.hpp>
#include<iostream>
#include "expm.hpp"
using namespace boost::numeric::ublas;
using namespace std;
int main(void)
{
	matrix<complex<double> > mat(3,3);
	matrix<complex<double> > gen(3,3);  // Generator of rotaion around z aix in group theory
	complex<double> img = std::complex<double>(0,1);
	gen(0,0) = 0  ; gen(0,1) = -img; gen(0,2) = 0;
	gen(1,0) = img; gen(1,1) = 0   ; gen(1,2) = 0;
	gen(2,0) = 0  ; gen(2,1) = 0   ; gen(2,2) = 0;

	double theta = 1.5;
	mat = img * theta * gen;
	cout<< "Rotation Matrix : "<< expm_pad(gen) <<"\n\n";
	return 0;
}

