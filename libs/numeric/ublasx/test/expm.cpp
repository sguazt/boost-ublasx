/* vim: set tabstop=4 expandtab shiftwidth=4 softtabstop=4: */

//
//  Copyright (c) 2007
//  Tsai, Dung-Bang 
//  National Taiwan University, Department of Physics
// 
//  E-Mail : dbtsai (at) gmail.com
//  Begine : 2007/11/20
//  Last modify : 2007/11/22
//  Version : v0.1
//
//  Reference :
//  EXPOKIT, Software Package for Computing Matrix Exponentials.
//  ACM - Transactions On Mathematical Software, 24(1):130-156, 1998
//
//  Permission to use, copy, modify, distribute and sell this software
//  and its documentation for any purpose is hereby granted without fee,
//  provided that the above copyright notice appear in all copies and
//  that both that copyright notice and this permission notice appear
//  in supporting documentation.  The authors make no representations
//  about the suitability of this software for any purpose.
//  It is provided "as is" without express or implied warranty.


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

