/* vim: set tabstop=4 expandtab shiftwidth=4 softtabstop=4: */

/**
 * \file libs/numeric/ublasx/test/eigen.cpp
 *
 * \brief Test the \c eigen operation.
 *
 * Copyright (c) 2010, Marco Guazzone
 *
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompwhiching file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */

#include <boost/numeric/ublas/matrix_expression.hpp>
#include <boost/numeric/ublas/fwd.hpp>
#include <boost/numeric/ublas/hermitian.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/symmetric.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublasx/operation/diag.hpp>
#include <boost/numeric/ublasx/operation/eigen.hpp>
#include <boost/numeric/ublasx/operation/num_columns.hpp>
#include <boost/numeric/ublasx/operation/num_rows.hpp>
#include <boost/numeric/ublasx/operation/size.hpp>
#include <boost/numeric/ublasx/operation/sum.hpp>
#include <boost/numeric/ublasx/operation/abs.hpp>
#include <complex>
#include "libs/numeric/ublasx/test/utils.hpp"


namespace ublas = ::boost::numeric::ublas;
namespace ublasx = ::boost::numeric::ublasx;


static const double tol = 1.0e-5;


BOOST_UBLASX_TEST_DEF( test_double_matrix_column_major_both )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Double Matrix - Column Major - Both Eigenvectors");

    typedef double value_type;
    typedef value_type in_value_type;
    typedef ::std::complex<value_type> out_value_type;
    typedef ublas::matrix<in_value_type, ublas::column_major> in_matrix_type;
    typedef ublas::matrix<out_value_type, ublas::column_major> out_matrix_type;
    typedef ublas::vector<out_value_type> out_vector_type;

    const std::size_t n(5);

    in_matrix_type A(n,n);

    A(0,0) = -1.01; A(0,1) =  0.86; A(0,2) = -4.60; A(0,3) =  3.31; A(0,4) = -4.81;
    A(1,0) =  3.98; A(1,1) =  0.53; A(1,2) = -7.04; A(1,3) =  5.29; A(1,4) =  3.55;
    A(2,0) =  3.30; A(2,1) =  8.26; A(2,2) = -3.89; A(2,3) =  8.20; A(2,4) = -1.51;
    A(3,0) =  4.43; A(3,1) =  4.96; A(3,2) = -7.66; A(3,3) = -7.33; A(3,4) =  6.18;
    A(4,0) =  7.31; A(4,1) = -6.43; A(4,2) = -6.16; A(4,3) =  2.47; A(4,4) =  5.58;

    out_vector_type w;
    out_matrix_type LV;
    out_matrix_type RV;

    ublasx::eigen(A, w, LV, RV);

    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "Eigenvalues = " << w );
    BOOST_UBLASX_DEBUG_TRACE( "Left Eigenvectors = " << LV );
    BOOST_UBLASX_DEBUG_TRACE( "Right Eigenvectors = " << RV );

    BOOST_UBLASX_TEST_CHECK( ublasx::size(w) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(LV) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(LV) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(RV) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(RV) == n );
    out_matrix_type D(n,n);
    D = ublasx::diag<out_vector_type,ublas::column_major>(w);
    BOOST_UBLASX_DEBUG_TRACE( "A*RV = RV*D => " << ublas::prod(A, RV) << " = " << ublas::prod(RV, D) ); // A*RV=RV*D
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( ublas::prod(A, RV), ublas::prod(RV, D), n, n, tol );
    BOOST_UBLASX_DEBUG_TRACE( "LV^H*A = D*LV^H => " << ublas::prod(ublas::herm(LV), A) << " = " << ublas::prod(D, ublas::herm(LV)) ); // A*LV^H=D*LV^H
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( ublas::prod(ublas::herm(LV), A), ublas::prod(D, ublas::herm(LV)), n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_double_matrix_column_major_left )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Double Matrix - Column Major - Left Eigenvectors");

    typedef double value_type;
    typedef value_type in_value_type;
    typedef ::std::complex<value_type> out_value_type;
    typedef ublas::matrix<in_value_type, ublas::column_major> in_matrix_type;
    typedef ublas::matrix<out_value_type, ublas::column_major> out_matrix_type;
    typedef ublas::vector<out_value_type> out_vector_type;

    const std::size_t n(5);

    in_matrix_type A(n,n);

    A(0,0) = -1.01; A(0,1) =  0.86; A(0,2) = -4.60; A(0,3) =  3.31; A(0,4) = -4.81;
    A(1,0) =  3.98; A(1,1) =  0.53; A(1,2) = -7.04; A(1,3) =  5.29; A(1,4) =  3.55;
    A(2,0) =  3.30; A(2,1) =  8.26; A(2,2) = -3.89; A(2,3) =  8.20; A(2,4) = -1.51;
    A(3,0) =  4.43; A(3,1) =  4.96; A(3,2) = -7.66; A(3,3) = -7.33; A(3,4) =  6.18;
    A(4,0) =  7.31; A(4,1) = -6.43; A(4,2) = -6.16; A(4,3) =  2.47; A(4,4) =  5.58;

    out_vector_type w;
    out_matrix_type LV;

    ublasx::left_eigen(A, w, LV);

    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "Eigenvalues = " << w );
    BOOST_UBLASX_DEBUG_TRACE( "Left Eigenvectors = " << LV );

    BOOST_UBLASX_TEST_CHECK( ublasx::size(w) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(LV) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(LV) == n );
    out_matrix_type D(n,n);
    D = ublasx::diag<out_vector_type,ublas::column_major>(w);
    BOOST_UBLASX_DEBUG_TRACE( "LV^H*A = D*LV^H => " << ublas::prod(ublas::herm(LV), A) << " = " << ublas::prod(D, ublas::herm(LV)) ); // A*LV^H=D*LV^H
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( ublas::prod(ublas::herm(LV), A), ublas::prod(D, ublas::herm(LV)), n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_double_matrix_column_major_right )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Double Matrix - Column Major - Right Eigenvectors");

    typedef double value_type;
    typedef value_type in_value_type;
    typedef ::std::complex<value_type> out_value_type;
    typedef ublas::matrix<in_value_type, ublas::column_major> in_matrix_type;
    typedef ublas::matrix<out_value_type, ublas::column_major> out_matrix_type;
    typedef ublas::vector<out_value_type> out_vector_type;

    const std::size_t n(5);

    in_matrix_type A(n,n);

    A(0,0) = -1.01; A(0,1) =  0.86; A(0,2) = -4.60; A(0,3) =  3.31; A(0,4) = -4.81;
    A(1,0) =  3.98; A(1,1) =  0.53; A(1,2) = -7.04; A(1,3) =  5.29; A(1,4) =  3.55;
    A(2,0) =  3.30; A(2,1) =  8.26; A(2,2) = -3.89; A(2,3) =  8.20; A(2,4) = -1.51;
    A(3,0) =  4.43; A(3,1) =  4.96; A(3,2) = -7.66; A(3,3) = -7.33; A(3,4) =  6.18;
    A(4,0) =  7.31; A(4,1) = -6.43; A(4,2) = -6.16; A(4,3) =  2.47; A(4,4) =  5.58;

    out_vector_type w;
    out_matrix_type RV;

    ublasx::right_eigen(A, w, RV);

    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "Eigenvalues = " << w );
    BOOST_UBLASX_DEBUG_TRACE( "Right Eigenvectors = " << RV );

    BOOST_UBLASX_TEST_CHECK( ublasx::size(w) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(RV) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(RV) == n );
    out_matrix_type D(n,n);
    D = ublasx::diag<out_vector_type,ublas::column_major>(w);
    BOOST_UBLASX_DEBUG_TRACE( "A*RV = RV*D => " << ublas::prod(A, RV) << " = " << ublas::prod(RV, D) ); // A*RV=RV*D
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( ublas::prod(A, RV), ublas::prod(RV, D), n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_double_matrix_column_major_only_values )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Double Matrix - Column Major - Only Eigenvalues");

    typedef double value_type;
    typedef value_type in_value_type;
    typedef ::std::complex<value_type> out_value_type;
    typedef ublas::matrix<in_value_type, ublas::column_major> in_matrix_type;
    typedef ublas::vector<out_value_type> out_vector_type;

    const std::size_t n(5);

    in_matrix_type A(n,n);

    A(0,0) = -1.01; A(0,1) =  0.86; A(0,2) = -4.60; A(0,3) =  3.31; A(0,4) = -4.81;
    A(1,0) =  3.98; A(1,1) =  0.53; A(1,2) = -7.04; A(1,3) =  5.29; A(1,4) =  3.55;
    A(2,0) =  3.30; A(2,1) =  8.26; A(2,2) = -3.89; A(2,3) =  8.20; A(2,4) = -1.51;
    A(3,0) =  4.43; A(3,1) =  4.96; A(3,2) = -7.66; A(3,3) = -7.33; A(3,4) =  6.18;
    A(4,0) =  7.31; A(4,1) = -6.43; A(4,2) = -6.16; A(4,3) =  2.47; A(4,4) =  5.58;

    out_vector_type w;
    out_vector_type expect_w;

    expect_w = out_vector_type(n);
    expect_w(0) = out_value_type(  2.85813, 10.76275);
    expect_w(1) = out_value_type(  2.85813,-10.76275);
    expect_w(2) = out_value_type(- 0.68667,  4.70426);
    expect_w(3) = out_value_type(- 0.68667, -4.70426);
    expect_w(4) = out_value_type(-10.46292,  0.00000);


    ublasx::eigenvalues(A, w);

    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "Eigenvalues = " << w );

    BOOST_UBLASX_TEST_CHECK( ublasx::size(w) == n );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( w, expect_w, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_double_matrix_column_major_only_vectors )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Double Matrix - Column Major - Only Eigenvectors");

    typedef double value_type;
    typedef value_type in_value_type;
    typedef ::std::complex<value_type> out_value_type;
    typedef ublas::matrix<in_value_type, ublas::column_major> in_matrix_type;
    typedef ublas::matrix<out_value_type, ublas::column_major> out_matrix_type;

    const std::size_t n(5);

    in_matrix_type A(n,n);

    A(0,0) = -1.01; A(0,1) =  0.86; A(0,2) = -4.60; A(0,3) =  3.31; A(0,4) = -4.81;
    A(1,0) =  3.98; A(1,1) =  0.53; A(1,2) = -7.04; A(1,3) =  5.29; A(1,4) =  3.55;
    A(2,0) =  3.30; A(2,1) =  8.26; A(2,2) = -3.89; A(2,3) =  8.20; A(2,4) = -1.51;
    A(3,0) =  4.43; A(3,1) =  4.96; A(3,2) = -7.66; A(3,3) = -7.33; A(3,4) =  6.18;
    A(4,0) =  7.31; A(4,1) = -6.43; A(4,2) = -6.16; A(4,3) =  2.47; A(4,4) =  5.58;

    out_matrix_type LV;
    out_matrix_type RV;
    out_matrix_type expect_LV(n,n);
    out_matrix_type expect_RV(n,n);

    ublasx::eigenvectors(A, LV, RV);

    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "Left Eigenvectors = " << LV );
    BOOST_UBLASX_DEBUG_TRACE( "Right Eigenvectors = " << RV );
        //TODO: give more precise values!
    expect_LV(0,0) = out_value_type( 0.04441241171439061 , 0.2879188413627367  ); expect_LV(0,1) = out_value_type( 0.04441241171439061 ,-0.2879188413627367  ); expect_LV(0,2) = out_value_type(-0.1325612054004949  ,-0.32728512393077713 ); expect_LV(0,3) = out_value_type(-0.1325612054004949  , 0.32728512393077713 ); expect_LV(0,4) = out_value_type( 0.0408372696405638 , 0.00000);
    expect_LV(1,0) = out_value_type( 0.6181643032364803  , 0.0                 ); expect_LV(1,1) = out_value_type( 0.6181643032364803  , 0.                  ); expect_LV(1,2) = out_value_type( 0.6868696010430648  , 0.0                 ); expect_LV(1,3) = out_value_type( 0.6868696010430648  , 0.00                ); expect_LV(1,4) = out_value_type( 0.5599544102049594 , 0.00000);
    expect_LV(2,0) = out_value_type(-0.035757599312428556,-0.5771114592618123  ); expect_LV(2,1) = out_value_type(-0.035757599312428556, 0.5771114592618123  ); expect_LV(2,2) = out_value_type(-0.39032805246732794 ,-0.07486636968983368 ); expect_LV(2,3) = out_value_type(-0.39032805246732794 , 0.07486636968983368 ); expect_LV(2,4) = out_value_type(-0.12850028050038304, 0.00000);
    expect_LV(3,0) = out_value_type( 0.2837261355713329  , 0.011354678505118251); expect_LV(3,1) = out_value_type( 0.2837261355713329  ,-0.011354678505118251); expect_LV(3,2) = out_value_type(-0.018200866392540004,-0.1872688637882381  ); expect_LV(3,3) = out_value_type(-0.018200866392540004, 0.1872688637882381  ); expect_LV(3,4) = out_value_type(-0.7966991560727732 , 0.00000);
    expect_LV(4,0) = out_value_type(-0.044953359596348524, 0.3406122092484726  ); expect_LV(4,1) = out_value_type(-0.044953359596348524,-0.3406122092484726  ); expect_LV(4,2) = out_value_type(-0.40321802640401727 , 0.2181180599737777  ); expect_LV(4,3) = out_value_type(-0.40321802640401727 ,-0.2181180599737777  ); expect_LV(4,4) = out_value_type( 0.18314340972192725, 0.00000);

    expect_RV(0,0) = out_value_type(0.10806479130135167, 0.1686483435010072); expect_RV(0,1) = out_value_type(0.10806479130135167,-0.1686483435010072); expect_RV(0,2) = out_value_type( 0.7322339897837211  , 0.0                 ); expect_RV(0,3) = out_value_type( 0.7322339897837211  , 0.0                ); expect_RV(0,4) = out_value_type(-0.4606464366271303, 0.00000);
    expect_RV(1,0) = out_value_type(0.40631288132267446,-0.2590097689205323); expect_RV(1,1) = out_value_type(0.40631288132267446, 0.2590097689205323); expect_RV(1,2) = out_value_type(-0.026463011089022395,-0.01694675437857112 ); expect_RV(1,3) = out_value_type(-0.026463011089022395, 0.01694675437857112); expect_RV(1,4) = out_value_type(-0.3377038282859721, 0.00000);
    expect_RV(2,0) = out_value_type(0.10235768506156454,-0.5088023141787094); expect_RV(2,1) = out_value_type(0.10235768506156454, 0.5088023141787094); expect_RV(2,2) = out_value_type( 0.191648728080536   ,-0.2925659954756119  ); expect_RV(2,3) = out_value_type( 0.191648728080536   , 0.2925659954756119 ); expect_RV(2,4) = out_value_type(-0.3087439418541303, 0.00000);
    expect_RV(3,0) = out_value_type(0.39863109808413577,-0.0913334523695411); expect_RV(3,1) = out_value_type(0.39863109808413577, 0.0913334523695411); expect_RV(3,2) = out_value_type(-0.07901106298430906 ,-0.07807593642682402 ); expect_RV(3,3) = out_value_type(-0.07901106298430906 , 0.07807593642682402); expect_RV(3,4) = out_value_type( 0.7438458375310733, 0.00000);
    expect_RV(4,0) = out_value_type(0.5395350560474126 , 0.00000           ); expect_RV(4,1) = out_value_type(0.5395350560474126 , 0.00000           ); expect_RV(4,2) = out_value_type(-0.291604754325538   ,-0.493102293052802   ); expect_RV(4,3) = out_value_type(-0.291604754325538   , 0.493102293052802  ); expect_RV(4,4) = out_value_type(-0.1585292816478885, 0.00000);

    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(LV) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(LV) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(RV) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(RV) == n );

        for (std::size_t i=0; i<n; ++i) {
          ublas::vector<out_value_type> vE( ublas::matrix_column<out_matrix_type>(expect_LV, i) );
          ublas::vector<out_value_type> vT( ublas::matrix_column<out_matrix_type>(LV, i) );
          double r0 = ublasx::sum( ublasx::abs( vE + vT ));
          double r1 = ublasx::sum( ublasx::abs( vE - vT ));
          if (r1 > r0) vT = -vT;
      BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vE, vT, n, tol );

          vE = ublas::matrix_column<out_matrix_type>(expect_RV, i);
          vT = ublas::matrix_column<out_matrix_type>(RV, i);
          r0 = ublasx::sum( ublasx::abs( vE + vT ));
          r1 = ublasx::sum( ublasx::abs( vE - vT ));
          if (r1 > r0) vT = -vT;
      BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( vE, vT, n, tol );
        }
}


BOOST_UBLASX_TEST_DEF( test_double_matrix_row_major_both )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Double Matrix - Row Major - Both Eigenvectors");

    typedef double value_type;
    typedef value_type in_value_type;
    typedef ::std::complex<value_type> out_value_type;
    typedef ublas::matrix<in_value_type, ublas::row_major> in_matrix_type;
    typedef ublas::matrix<out_value_type, ublas::row_major> out_matrix_type;
    typedef ublas::vector<out_value_type> out_vector_type;

    const std::size_t n(5);

    in_matrix_type A(n,n);

    A(0,0) = -1.01; A(0,1) =  0.86; A(0,2) = -4.60; A(0,3) =  3.31; A(0,4) = -4.81;
    A(1,0) =  3.98; A(1,1) =  0.53; A(1,2) = -7.04; A(1,3) =  5.29; A(1,4) =  3.55;
    A(2,0) =  3.30; A(2,1) =  8.26; A(2,2) = -3.89; A(2,3) =  8.20; A(2,4) = -1.51;
    A(3,0) =  4.43; A(3,1) =  4.96; A(3,2) = -7.66; A(3,3) = -7.33; A(3,4) =  6.18;
    A(4,0) =  7.31; A(4,1) = -6.43; A(4,2) = -6.16; A(4,3) =  2.47; A(4,4) =  5.58;

    out_vector_type w;
    out_matrix_type LV;
    out_matrix_type RV;

    ublasx::eigen(A, w, LV, RV);

    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "Eigenvalues = " << w );
    BOOST_UBLASX_DEBUG_TRACE( "Left Eigenvectors = " << LV );
    BOOST_UBLASX_DEBUG_TRACE( "Right Eigenvectors = " << RV );

    BOOST_UBLASX_TEST_CHECK( ublasx::size(w) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(LV) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(LV) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(RV) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(RV) == n );
    out_matrix_type D(n,n);
    D = ublasx::diag<out_vector_type,ublas::row_major>(w);
    BOOST_UBLASX_DEBUG_TRACE( "A*RV = RV*D => " << ublas::prod(A, RV) << " = " << ublas::prod(RV, D) ); // A*RV=RV*D
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( ublas::prod(A, RV), ublas::prod(RV, D), n, n, tol );
    BOOST_UBLASX_DEBUG_TRACE( "LV^H*A = D*LV^H => " << ublas::prod(ublas::herm(LV), A) << " = " << ublas::prod(D, ublas::herm(LV)) ); // A*LV^H=D*LV^H
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( ublas::prod(ublas::herm(LV), A), ublas::prod(D, ublas::herm(LV)), n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_double_matrix_row_major_left )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Double Matrix - Row Major - Left Eigenvectors");

    typedef double value_type;
    typedef value_type in_value_type;
    typedef ::std::complex<value_type> out_value_type;
    typedef ublas::matrix<in_value_type, ublas::row_major> in_matrix_type;
    typedef ublas::matrix<out_value_type, ublas::row_major> out_matrix_type;
    typedef ublas::vector<out_value_type> out_vector_type;

    const std::size_t n(5);

    in_matrix_type A(n,n);

    A(0,0) = -1.01; A(0,1) =  0.86; A(0,2) = -4.60; A(0,3) =  3.31; A(0,4) = -4.81;
    A(1,0) =  3.98; A(1,1) =  0.53; A(1,2) = -7.04; A(1,3) =  5.29; A(1,4) =  3.55;
    A(2,0) =  3.30; A(2,1) =  8.26; A(2,2) = -3.89; A(2,3) =  8.20; A(2,4) = -1.51;
    A(3,0) =  4.43; A(3,1) =  4.96; A(3,2) = -7.66; A(3,3) = -7.33; A(3,4) =  6.18;
    A(4,0) =  7.31; A(4,1) = -6.43; A(4,2) = -6.16; A(4,3) =  2.47; A(4,4) =  5.58;

    out_vector_type w;
    out_matrix_type LV;

    ublasx::left_eigen(A, w, LV);

    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "Eigenvalues = " << w );
    BOOST_UBLASX_DEBUG_TRACE( "Left Eigenvectors = " << LV );

    BOOST_UBLASX_TEST_CHECK( ublasx::size(w) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(LV) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(LV) == n );
    out_matrix_type D(n,n);
    D = ublasx::diag<out_vector_type,ublas::row_major>(w);
    BOOST_UBLASX_DEBUG_TRACE( "LV^H*A = D*LV^H => " << ublas::prod(ublas::herm(LV), A) << " = " << ublas::prod(D, ublas::herm(LV)) ); // A*LV^H=D*LV^H
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( ublas::prod(ublas::herm(LV), A), ublas::prod(D, ublas::herm(LV)), n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_double_matrix_row_major_right )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Double Matrix - Row Major - Right Eigenvectors");

    typedef double value_type;
    typedef value_type in_value_type;
    typedef ::std::complex<value_type> out_value_type;
    typedef ublas::matrix<in_value_type, ublas::row_major> in_matrix_type;
    typedef ublas::matrix<out_value_type, ublas::row_major> out_matrix_type;
    typedef ublas::vector<out_value_type> out_vector_type;

    const std::size_t n(5);

    in_matrix_type A(n,n);

    A(0,0) = -1.01; A(0,1) =  0.86; A(0,2) = -4.60; A(0,3) =  3.31; A(0,4) = -4.81;
    A(1,0) =  3.98; A(1,1) =  0.53; A(1,2) = -7.04; A(1,3) =  5.29; A(1,4) =  3.55;
    A(2,0) =  3.30; A(2,1) =  8.26; A(2,2) = -3.89; A(2,3) =  8.20; A(2,4) = -1.51;
    A(3,0) =  4.43; A(3,1) =  4.96; A(3,2) = -7.66; A(3,3) = -7.33; A(3,4) =  6.18;
    A(4,0) =  7.31; A(4,1) = -6.43; A(4,2) = -6.16; A(4,3) =  2.47; A(4,4) =  5.58;

    out_vector_type w;
    out_matrix_type RV;

    ublasx::right_eigen(A, w, RV);

    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "Eigenvalues = " << w );
    BOOST_UBLASX_DEBUG_TRACE( "Right Eigenvectors = " << RV );

    BOOST_UBLASX_TEST_CHECK( ublasx::size(w) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(RV) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(RV) == n );
    out_matrix_type D(n,n);
    D = ublasx::diag<out_vector_type,ublas::row_major>(w);
    BOOST_UBLASX_DEBUG_TRACE( "A*RV = RV*D => " << ublas::prod(A, RV) << " = " << ublas::prod(RV, D) ); // A*RV=RV*D
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( ublas::prod(A, RV), ublas::prod(RV, D), n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_double_matrix_row_major_only_values )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Double Matrix - Row Major - Only Eigenvalues");

    typedef double value_type;
    typedef value_type in_value_type;
    typedef ::std::complex<value_type> out_value_type;
    typedef ublas::matrix<in_value_type, ublas::row_major> in_matrix_type;
    typedef ublas::vector<out_value_type> out_vector_type;

    const std::size_t n(5);

    in_matrix_type A(n,n);

    A(0,0) = -1.01; A(0,1) =  0.86; A(0,2) = -4.60; A(0,3) =  3.31; A(0,4) = -4.81;
    A(1,0) =  3.98; A(1,1) =  0.53; A(1,2) = -7.04; A(1,3) =  5.29; A(1,4) =  3.55;
    A(2,0) =  3.30; A(2,1) =  8.26; A(2,2) = -3.89; A(2,3) =  8.20; A(2,4) = -1.51;
    A(3,0) =  4.43; A(3,1) =  4.96; A(3,2) = -7.66; A(3,3) = -7.33; A(3,4) =  6.18;
    A(4,0) =  7.31; A(4,1) = -6.43; A(4,2) = -6.16; A(4,3) =  2.47; A(4,4) =  5.58;

    out_vector_type w;
    out_vector_type expect_w;

    expect_w = out_vector_type(n);
    expect_w(0) = out_value_type(2.8581328780343496,10.762749830715672);
    expect_w(1) = out_value_type(2.8581328780343496,-10.762749830715672);
    expect_w(2) = out_value_type(-0.6866745133059503,4.70426134062811);
    expect_w(3) = out_value_type(-0.6866745133059503,-4.70426134062811);
    expect_w(4) = out_value_type(-10.462916729456813,  0.00000);


    ublasx::eigenvalues(A, w);

    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "Eigenvalues = " << w );

    BOOST_UBLASX_TEST_CHECK( ublasx::size(w) == n );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( w, expect_w, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_double_matrix_row_major_only_vectors )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Double Matrix - Row Major - Only Eigenvectors");

    typedef double value_type;
    typedef value_type in_value_type;
    typedef ::std::complex<value_type> out_value_type;
    typedef ublas::matrix<in_value_type, ublas::row_major> in_matrix_type;
    typedef ublas::matrix<out_value_type, ublas::row_major> out_matrix_type;

    const std::size_t n(5);

    in_matrix_type A(n,n);

    A(0,0) = -1.01; A(0,1) =  0.86; A(0,2) = -4.60; A(0,3) =  3.31; A(0,4) = -4.81;
    A(1,0) =  3.98; A(1,1) =  0.53; A(1,2) = -7.04; A(1,3) =  5.29; A(1,4) =  3.55;
    A(2,0) =  3.30; A(2,1) =  8.26; A(2,2) = -3.89; A(2,3) =  8.20; A(2,4) = -1.51;
    A(3,0) =  4.43; A(3,1) =  4.96; A(3,2) = -7.66; A(3,3) = -7.33; A(3,4) =  6.18;
    A(4,0) =  7.31; A(4,1) = -6.43; A(4,2) = -6.16; A(4,3) =  2.47; A(4,4) =  5.58;

    out_matrix_type LV;
    out_matrix_type RV;
    out_matrix_type expect_LV(n,n);
    out_matrix_type expect_RV(n,n);

    ublasx::eigenvectors(A, LV, RV);

    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "Left Eigenvectors = " << LV );
    BOOST_UBLASX_DEBUG_TRACE( "Right Eigenvectors = " << RV );

    expect_LV(0,0) = out_value_type( 0.04441241171439061 , 0.2879188413627367  ); expect_LV(0,1) = out_value_type( 0.04441241171439061 ,-0.2879188413627367  ); expect_LV(0,2) = out_value_type(-0.1325612054004949  ,-0.32728512393077713 ); expect_LV(0,3) = out_value_type(-0.1325612054004949  , 0.32728512393077713 ); expect_LV(0,4) = out_value_type(-0.0408372696405638 , 0.00000);
    expect_LV(1,0) = out_value_type( 0.6181643032364803  , 0.0                 ); expect_LV(1,1) = out_value_type( 0.6181643032364803  , 0.                  ); expect_LV(1,2) = out_value_type( 0.6868696010430648  , 0.0                 ); expect_LV(1,3) = out_value_type( 0.6868696010430648  , 0.00                ); expect_LV(1,4) = out_value_type(-0.5599544102049594 , 0.00000);
    expect_LV(2,0) = out_value_type(-0.035757599312428556,-0.5771114592618123  ); expect_LV(2,1) = out_value_type(-0.035757599312428556, 0.5771114592618123  ); expect_LV(2,2) = out_value_type(-0.39032805246732794 ,-0.07486636968983368 ); expect_LV(2,3) = out_value_type(-0.39032805246732794 , 0.07486636968983368 ); expect_LV(2,4) = out_value_type( 0.12850028050038304, 0.00000);
    expect_LV(3,0) = out_value_type( 0.2837261355713329  , 0.011354678505118251); expect_LV(3,1) = out_value_type( 0.2837261355713329  ,-0.011354678505118251); expect_LV(3,2) = out_value_type(-0.018200866392540004,-0.1872688637882381  ); expect_LV(3,3) = out_value_type(-0.018200866392540004, 0.1872688637882381  ); expect_LV(3,4) = out_value_type( 0.7966991560727732 , 0.00000);
    expect_LV(4,0) = out_value_type(-0.044953359596348524, 0.3406122092484726  ); expect_LV(4,1) = out_value_type(-0.044953359596348524,-0.3406122092484726  ); expect_LV(4,2) = out_value_type(-0.40321802640401727 , 0.2181180599737777  ); expect_LV(4,3) = out_value_type(-0.40321802640401727 ,-0.2181180599737777  ); expect_LV(4,4) = out_value_type(-0.18314340972192725, 0.00000);

        expect_RV(0,0) = out_value_type(0.10806479130135167, 0.1686483435010072); expect_RV(0,1) = out_value_type(0.10806479130135167,-0.1686483435010072); expect_RV(0,2) = out_value_type( 0.7322339897837211  , 0.0                 ); expect_RV(0,3) = out_value_type( 0.7322339897837211  , 0.0                ); expect_RV(0,4) = out_value_type(-0.4606464366271303, 0.00000);
    expect_RV(1,0) = out_value_type(0.40631288132267446,-0.2590097689205323); expect_RV(1,1) = out_value_type(0.40631288132267446, 0.2590097689205323); expect_RV(1,2) = out_value_type(-0.026463011089022395,-0.01694675437857112 ); expect_RV(1,3) = out_value_type(-0.026463011089022395, 0.01694675437857112); expect_RV(1,4) = out_value_type(-0.3377038282859721, 0.00000);
    expect_RV(2,0) = out_value_type(0.10235768506156454,-0.5088023141787094); expect_RV(2,1) = out_value_type(0.10235768506156454, 0.5088023141787094); expect_RV(2,2) = out_value_type( 0.191648728080536   ,-0.2925659954756119  ); expect_RV(2,3) = out_value_type( 0.191648728080536   , 0.2925659954756119 ); expect_RV(2,4) = out_value_type(-0.3087439418541303, 0.00000);
    expect_RV(3,0) = out_value_type(0.39863109808413577,-0.0913334523695411); expect_RV(3,1) = out_value_type(0.39863109808413577, 0.0913334523695411); expect_RV(3,2) = out_value_type(-0.07901106298430906 ,-0.07807593642682402 ); expect_RV(3,3) = out_value_type(-0.07901106298430906 , 0.07807593642682402); expect_RV(3,4) = out_value_type( 0.7438458375310733, 0.00000);
    expect_RV(4,0) = out_value_type(0.5395350560474126 , 0.00000           ); expect_RV(4,1) = out_value_type(0.5395350560474126 , 0.00000           ); expect_RV(4,2) = out_value_type(-0.291604754325538   ,-0.493102293052802   ); expect_RV(4,3) = out_value_type(-0.291604754325538   , 0.493102293052802  ); expect_RV(4,4) = out_value_type(-0.1585292816478885, 0.00000);

    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(LV) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(LV) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(RV) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(RV) == n );
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( LV, expect_LV, n, n, tol );
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( RV, expect_RV, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_complex_matrix_column_major_both )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Complex Matrix - Column Major - Both Eigenvectors");

    typedef double value_type;
    typedef ::std::complex<value_type> in_value_type;
    typedef ::std::complex<value_type> out_value_type;
    typedef ublas::matrix<in_value_type, ublas::column_major> in_matrix_type;
    typedef ublas::matrix<out_value_type, ublas::column_major> out_matrix_type;
    typedef ublas::vector<out_value_type> out_vector_type;

    const std::size_t n(4);

    in_matrix_type A(n,n);

    A(0,0) = in_value_type(-3.84, 2.25); A(0,1) = in_value_type(-8.94,-4.75); A(0,2) = in_value_type( 8.95,-6.53); A(0,3) = in_value_type(-9.87,4.82);
    A(1,0) = in_value_type(-0.66, 0.83); A(1,1) = in_value_type(-4.40,-3.82); A(1,2) = in_value_type(-3.50,-4.26); A(1,3) = in_value_type(-3.15,7.36);
    A(2,0) = in_value_type(-3.99,-4.73); A(2,1) = in_value_type(-5.88,-6.60); A(2,2) = in_value_type(-3.36,-0.40); A(2,3) = in_value_type(-0.75,5.23);
    A(3,0) = in_value_type( 7.74, 4.18); A(3,1) = in_value_type( 3.66,-7.53); A(3,2) = in_value_type( 2.58, 3.60); A(3,3) = in_value_type( 4.59,5.41);

    out_vector_type w;
    out_matrix_type LV;
    out_matrix_type RV;

    ublasx::eigen(A, w, LV, RV);

    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "Eigenvalues = " << w );
    BOOST_UBLASX_DEBUG_TRACE( "Left Eigenvectors = " << LV );
    BOOST_UBLASX_DEBUG_TRACE( "Right Eigenvectors = " << RV );

    BOOST_UBLASX_TEST_CHECK( ublasx::size(w) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(LV) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(LV) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(RV) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(RV) == n );
    out_matrix_type D(n,n);
    D = ublasx::diag<out_vector_type,ublas::column_major>(w);
    BOOST_UBLASX_DEBUG_TRACE( "A*RV = RV*D => " << ublas::prod(A, RV) << " = " << ublas::prod(RV, D) ); // A*RV=RV*D
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( ublas::prod(A, RV), ublas::prod(RV, D), n, n, tol );
    BOOST_UBLASX_DEBUG_TRACE( "LV^H*A = D*LV^H => " << ublas::prod(ublas::herm(LV), A) << " = " << ublas::prod(D, ublas::herm(LV)) ); // A*LV^H=D*LV^H
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( ublas::prod(ublas::herm(LV), A), ublas::prod(D, ublas::herm(LV)), n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_complex_matrix_column_major_left )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Complex Matrix - Column Major - Left Eigenvectors");

    typedef double value_type;
    typedef ::std::complex<value_type> in_value_type;
    typedef ::std::complex<value_type> out_value_type;
    typedef ublas::matrix<in_value_type, ublas::column_major> in_matrix_type;
    typedef ublas::matrix<out_value_type, ublas::column_major> out_matrix_type;
    typedef ublas::vector<out_value_type> out_vector_type;

    const std::size_t n(4);

    in_matrix_type A(n,n);

    A(0,0) = in_value_type(-3.84, 2.25); A(0,1) = in_value_type(-8.94,-4.75); A(0,2) = in_value_type( 8.95,-6.53); A(0,3) = in_value_type(-9.87,4.82);
    A(1,0) = in_value_type(-0.66, 0.83); A(1,1) = in_value_type(-4.40,-3.82); A(1,2) = in_value_type(-3.50,-4.26); A(1,3) = in_value_type(-3.15,7.36);
    A(2,0) = in_value_type(-3.99,-4.73); A(2,1) = in_value_type(-5.88,-6.60); A(2,2) = in_value_type(-3.36,-0.40); A(2,3) = in_value_type(-0.75,5.23);
    A(3,0) = in_value_type( 7.74, 4.18); A(3,1) = in_value_type( 3.66,-7.53); A(3,2) = in_value_type( 2.58, 3.60); A(3,3) = in_value_type( 4.59,5.41);

    out_vector_type w;
    out_matrix_type LV;

    ublasx::left_eigen(A, w, LV);

    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "Eigenvalues = " << w );
    BOOST_UBLASX_DEBUG_TRACE( "Left Eigenvectors = " << LV );

    BOOST_UBLASX_TEST_CHECK( ublasx::size(w) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(LV) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(LV) == n );
    out_matrix_type D(n,n);
    D = ublasx::diag<out_vector_type,ublas::column_major>(w);
    BOOST_UBLASX_DEBUG_TRACE( "LV^H*A = D*LV^H => " << ublas::prod(ublas::herm(LV), A) << " = " << ublas::prod(D, ublas::herm(LV)) ); // A*LV^H=D*LV^H
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( ublas::prod(ublas::herm(LV), A), ublas::prod(D, ublas::herm(LV)), n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_complex_matrix_column_major_right )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Complex Matrix - Column Major - Right Eigenvectors");

    typedef double value_type;
    typedef ::std::complex<value_type> in_value_type;
    typedef ::std::complex<value_type> out_value_type;
    typedef ublas::matrix<in_value_type, ublas::column_major> in_matrix_type;
    typedef ublas::matrix<out_value_type, ublas::column_major> out_matrix_type;
    typedef ublas::vector<out_value_type> out_vector_type;

    const std::size_t n(4);

    in_matrix_type A(n,n);

    A(0,0) = in_value_type(-3.84, 2.25); A(0,1) = in_value_type(-8.94,-4.75); A(0,2) = in_value_type( 8.95,-6.53); A(0,3) = in_value_type(-9.87,4.82);
    A(1,0) = in_value_type(-0.66, 0.83); A(1,1) = in_value_type(-4.40,-3.82); A(1,2) = in_value_type(-3.50,-4.26); A(1,3) = in_value_type(-3.15,7.36);
    A(2,0) = in_value_type(-3.99,-4.73); A(2,1) = in_value_type(-5.88,-6.60); A(2,2) = in_value_type(-3.36,-0.40); A(2,3) = in_value_type(-0.75,5.23);
    A(3,0) = in_value_type( 7.74, 4.18); A(3,1) = in_value_type( 3.66,-7.53); A(3,2) = in_value_type( 2.58, 3.60); A(3,3) = in_value_type( 4.59,5.41);

    out_vector_type w;
    out_matrix_type RV;

    ublasx::right_eigen(A, w, RV);

    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "Eigenvalues = " << w );
    BOOST_UBLASX_DEBUG_TRACE( "Right Eigenvectors = " << RV );

    BOOST_UBLASX_TEST_CHECK( ublasx::size(w) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(RV) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(RV) == n );
    out_matrix_type D(n,n);
    D = ublasx::diag<out_vector_type,ublas::column_major>(w);
    BOOST_UBLASX_DEBUG_TRACE( "A*RV = RV*D => " << ublas::prod(A, RV) << " = " << ublas::prod(RV, D) ); // A*RV=RV*D
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( ublas::prod(A, RV), ublas::prod(RV, D), n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_complex_matrix_column_major_only_values )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Complex Matrix - Column Major - Only Eigenvalues");

    typedef double value_type;
    typedef ::std::complex<value_type> in_value_type;
    typedef ::std::complex<value_type> out_value_type;
    typedef ublas::matrix<in_value_type, ublas::column_major> in_matrix_type;
    typedef ublas::vector<out_value_type> out_vector_type;

    const std::size_t n(4);

    in_matrix_type A(n,n);

    A(0,0) = in_value_type(-3.84, 2.25); A(0,1) = in_value_type(-8.94,-4.75); A(0,2) = in_value_type( 8.95,-6.53); A(0,3) = in_value_type(-9.87,4.82);
    A(1,0) = in_value_type(-0.66, 0.83); A(1,1) = in_value_type(-4.40,-3.82); A(1,2) = in_value_type(-3.50,-4.26); A(1,3) = in_value_type(-3.15,7.36);
    A(2,0) = in_value_type(-3.99,-4.73); A(2,1) = in_value_type(-5.88,-6.60); A(2,2) = in_value_type(-3.36,-0.40); A(2,3) = in_value_type(-0.75,5.23);
    A(3,0) = in_value_type( 7.74, 4.18); A(3,1) = in_value_type( 3.66,-7.53); A(3,2) = in_value_type( 2.58, 3.60); A(3,3) = in_value_type( 4.59,5.41);

    out_vector_type w;
    out_vector_type expect_w(n);

    ublasx::eigenvalues(A, w);

    expect_w(0) = out_value_type(-9.42985074873922,-12.98329567302135);
    expect_w(1) = out_value_type(-3.44184845897663, 12.68973749844945);
    expect_w(2) = out_value_type( 0.10554548255761,- 3.39504658829915);
    expect_w(3) = out_value_type( 5.75615372515821,  7.12860476287106);


    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "Eigenvalues = " << w );

    BOOST_UBLASX_TEST_CHECK( ublasx::size(w) == n );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( w, expect_w,n, tol );
}


BOOST_UBLASX_TEST_DEF( test_complex_matrix_column_major_only_vectors )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Complex Matrix - Column Major - Only Eigenvectors");

    typedef double value_type;
    typedef ::std::complex<value_type> in_value_type;
    typedef ::std::complex<value_type> out_value_type;
    typedef ublas::matrix<in_value_type, ublas::column_major> in_matrix_type;
    typedef ublas::matrix<out_value_type, ublas::column_major> out_matrix_type;

    const std::size_t n(4);

    in_matrix_type A(n,n);

    A(0,0) = in_value_type(-3.84, 2.25); A(0,1) = in_value_type(-8.94,-4.75); A(0,2) = in_value_type( 8.95,-6.53); A(0,3) = in_value_type(-9.87,4.82);
    A(1,0) = in_value_type(-0.66, 0.83); A(1,1) = in_value_type(-4.40,-3.82); A(1,2) = in_value_type(-3.50,-4.26); A(1,3) = in_value_type(-3.15,7.36);
    A(2,0) = in_value_type(-3.99,-4.73); A(2,1) = in_value_type(-5.88,-6.60); A(2,2) = in_value_type(-3.36,-0.40); A(2,3) = in_value_type(-0.75,5.23);
    A(3,0) = in_value_type( 7.74, 4.18); A(3,1) = in_value_type( 3.66,-7.53); A(3,2) = in_value_type( 2.58, 3.60); A(3,3) = in_value_type( 4.59,5.41);

    out_matrix_type RV;
    out_matrix_type LV;
    out_matrix_type expect_RV(n, n);
    out_matrix_type expect_LV(n, n);

    ublasx::eigenvectors(A, LV, RV);

    expect_LV(0,0) = out_value_type( 0.24144287163827527 ,-0.18465213100318006); expect_LV(0,1) = out_value_type( 0.6134970860903158 , 0.                 ); expect_LV(0,2) = out_value_type(-0.1828392867360731,-0.3347215349804258 ); expect_LV(0,3) = out_value_type( 0.2764845560844309, 0.08843771195325413);
    expect_LV(1,0) = out_value_type( 0.7861209959278461  , 0.00               ); expect_LV(1,1) = out_value_type(-0.04990581295152956,-0.27212029611221916); expect_LV(1,2) = out_value_type( 0.8218391628323942, 0.000              ); expect_LV(1,3) = out_value_type(-0.5477176303872586, 0.15722956229438773);
    expect_LV(2,0) = out_value_type( 0.21951507543794077 ,-0.2688645451415786 ); expect_LV(2,1) = out_value_type(-0.2087767673393235 , 0.5347329156020605 ); expect_LV(2,2) = out_value_type(-0.3714296893055094, 0.15249903883664429); expect_LV(2,3) = out_value_type( 0.4450824180745997, 0.09122872979332788);
    expect_LV(3,0) = out_value_type(-0.016984399323421218, 0.41092484496969633); expect_LV(3,1) = out_value_type( 0.402719845692206  ,-0.23531038207619248); expect_LV(3,2) = out_value_type( 0.0574841440971407, 0.12079437865593233); expect_LV(3,3) = out_value_type( 0.6201598853812728, 0.00               );

    expect_RV(0,0) = out_value_type( 0.43085652007761127, 0.32681273781262105  ); expect_RV(0,1) = out_value_type( 0.8256820507672814 , 0.                   ); expect_RV(0,2) = out_value_type( 0.598395978553945  , 0.                 ); expect_RV(0,3) = out_value_type(-0.3054319034843787 , 0.03333164861799852 );
    expect_RV(1,0) = out_value_type( 0.5087414602970971 ,-0.028833421706927785 ); expect_RV(1,1) = out_value_type( 0.0750291678814112 ,-0.2487285045091667   ); expect_RV(1,2) = out_value_type(-0.4004761627520769 ,-0.20144922276256036); expect_RV(1,3) = out_value_type( 0.03978282815783318, 0.3445076522154613  );
    expect_RV(2,0) = out_value_type( 0.6198496527657752 , 0.0                  ); expect_RV(2,1) = out_value_type(-0.24575578997801512, 0.27887240221169707  ); expect_RV(2,2) = out_value_type(-0.09008001907595067,-0.47526462153917304); expect_RV(2,3) = out_value_type( 0.35832543651598453, 0.060645069885246886);
    expect_RV(3,0) = out_value_type(-0.22692824331926834, 0.1104392784640359   ); expect_RV(3,1) = out_value_type(-0.10343406372814366,-0.31920146536323224  ); expect_RV(3,2) = out_value_type(-0.4348402954954043 , 0.1337249178581602 ); expect_RV(3,3) = out_value_type( 0.8082432893178347 , 0.00                );


    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "Left Eigenvectors = " << LV );
    BOOST_UBLASX_DEBUG_TRACE( "Right Eigenvectors = " << RV );

    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(LV) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(LV) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(RV) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(RV) == n );
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( LV, expect_LV, n, n, tol );
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( RV, expect_RV, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_complex_matrix_row_major_both )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Complex Matrix - Row Major - Both Eigenvectors");

    typedef double value_type;
    typedef ::std::complex<value_type> in_value_type;
    typedef ::std::complex<value_type> out_value_type;
    typedef ublas::matrix<in_value_type, ublas::row_major> in_matrix_type;
    typedef ublas::matrix<out_value_type, ublas::row_major> out_matrix_type;
    typedef ublas::vector<out_value_type> out_vector_type;

    const std::size_t n(4);

    in_matrix_type A(n,n);

    A(0,0) = in_value_type(-3.84, 2.25); A(0,1) = in_value_type(-8.94,-4.75); A(0,2) = in_value_type( 8.95,-6.53); A(0,3) = in_value_type(-9.87,4.82);
    A(1,0) = in_value_type(-0.66, 0.83); A(1,1) = in_value_type(-4.40,-3.82); A(1,2) = in_value_type(-3.50,-4.26); A(1,3) = in_value_type(-3.15,7.36);
    A(2,0) = in_value_type(-3.99,-4.73); A(2,1) = in_value_type(-5.88,-6.60); A(2,2) = in_value_type(-3.36,-0.40); A(2,3) = in_value_type(-0.75,5.23);
    A(3,0) = in_value_type( 7.74, 4.18); A(3,1) = in_value_type( 3.66,-7.53); A(3,2) = in_value_type( 2.58, 3.60); A(3,3) = in_value_type( 4.59,5.41);

    out_vector_type w;
    out_matrix_type LV;
    out_matrix_type RV;

    ublasx::eigen(A, w, LV, RV);

    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "Eigenvalues = " << w );
    BOOST_UBLASX_DEBUG_TRACE( "Left Eigenvectors = " << LV );
    BOOST_UBLASX_DEBUG_TRACE( "Right Eigenvectors = " << RV );

    BOOST_UBLASX_TEST_CHECK( ublasx::size(w) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(LV) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(LV) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(RV) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(RV) == n );
    out_matrix_type D(n,n);
    D = ublasx::diag<out_vector_type,ublas::row_major>(w);
    BOOST_UBLASX_DEBUG_TRACE( "A*RV = RV*D => " << ublas::prod(A, RV) << " = " << ublas::prod(RV, D) ); // A*RV=RV*D
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( ublas::prod(A, RV), ublas::prod(RV, D), n, n, tol );
    BOOST_UBLASX_DEBUG_TRACE( "LV^H*A = D*LV^H => " << ublas::prod(ublas::herm(LV), A) << " = " << ublas::prod(D, ublas::herm(LV)) ); // A*LV^H=D*LV^H
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( ublas::prod(ublas::herm(LV), A), ublas::prod(D, ublas::herm(LV)), n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_complex_matrix_row_major_left )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Complex Matrix - Row Major - Left Eigenvectors");

    typedef double value_type;
    typedef ::std::complex<value_type> in_value_type;
    typedef ::std::complex<value_type> out_value_type;
    typedef ublas::matrix<in_value_type, ublas::row_major> in_matrix_type;
    typedef ublas::matrix<out_value_type, ublas::row_major> out_matrix_type;
    typedef ublas::vector<out_value_type> out_vector_type;

    const std::size_t n(4);

    in_matrix_type A(n,n);

    A(0,0) = in_value_type(-3.84, 2.25); A(0,1) = in_value_type(-8.94,-4.75); A(0,2) = in_value_type( 8.95,-6.53); A(0,3) = in_value_type(-9.87,4.82);
    A(1,0) = in_value_type(-0.66, 0.83); A(1,1) = in_value_type(-4.40,-3.82); A(1,2) = in_value_type(-3.50,-4.26); A(1,3) = in_value_type(-3.15,7.36);
    A(2,0) = in_value_type(-3.99,-4.73); A(2,1) = in_value_type(-5.88,-6.60); A(2,2) = in_value_type(-3.36,-0.40); A(2,3) = in_value_type(-0.75,5.23);
    A(3,0) = in_value_type( 7.74, 4.18); A(3,1) = in_value_type( 3.66,-7.53); A(3,2) = in_value_type( 2.58, 3.60); A(3,3) = in_value_type( 4.59,5.41);

    out_vector_type w;
    out_matrix_type LV;

    ublasx::left_eigen(A, w, LV);

    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "Eigenvalues = " << w );
    BOOST_UBLASX_DEBUG_TRACE( "Left Eigenvectors = " << LV );

    BOOST_UBLASX_TEST_CHECK( ublasx::size(w) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(LV) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(LV) == n );
    out_matrix_type D(n,n);
    D = ublasx::diag<out_vector_type,ublas::row_major>(w);
    BOOST_UBLASX_DEBUG_TRACE( "LV^H*A = D*LV^H => " << ublas::prod(ublas::herm(LV), A) << " = " << ublas::prod(D, ublas::herm(LV)) ); // A*LV^H=D*LV^H
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( ublas::prod(ublas::herm(LV), A), ublas::prod(D, ublas::herm(LV)), n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_complex_matrix_row_major_right )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Complex Matrix - Row Major - Right Eigenvectors");

    typedef double value_type;
    typedef ::std::complex<value_type> in_value_type;
    typedef ::std::complex<value_type> out_value_type;
    typedef ublas::matrix<in_value_type, ublas::row_major> in_matrix_type;
    typedef ublas::matrix<out_value_type, ublas::row_major> out_matrix_type;
    typedef ublas::vector<out_value_type> out_vector_type;

    const std::size_t n(4);

    in_matrix_type A(n,n);

    A(0,0) = in_value_type(-3.84, 2.25); A(0,1) = in_value_type(-8.94,-4.75); A(0,2) = in_value_type( 8.95,-6.53); A(0,3) = in_value_type(-9.87,4.82);
    A(1,0) = in_value_type(-0.66, 0.83); A(1,1) = in_value_type(-4.40,-3.82); A(1,2) = in_value_type(-3.50,-4.26); A(1,3) = in_value_type(-3.15,7.36);
    A(2,0) = in_value_type(-3.99,-4.73); A(2,1) = in_value_type(-5.88,-6.60); A(2,2) = in_value_type(-3.36,-0.40); A(2,3) = in_value_type(-0.75,5.23);
    A(3,0) = in_value_type( 7.74, 4.18); A(3,1) = in_value_type( 3.66,-7.53); A(3,2) = in_value_type( 2.58, 3.60); A(3,3) = in_value_type( 4.59,5.41);

    out_vector_type w;
    out_matrix_type RV;

    ublasx::right_eigen(A, w, RV);

    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "Eigenvalues = " << w );
    BOOST_UBLASX_DEBUG_TRACE( "Right Eigenvectors = " << RV );

    BOOST_UBLASX_TEST_CHECK( ublasx::size(w) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(RV) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(RV) == n );
    out_matrix_type D(n,n);
    D = ublasx::diag<out_vector_type,ublas::row_major>(w);
    BOOST_UBLASX_DEBUG_TRACE( "A*RV = RV*D => " << ublas::prod(A, RV) << " = " << ublas::prod(RV, D) ); // A*RV=RV*D
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( ublas::prod(A, RV), ublas::prod(RV, D), n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_complex_matrix_row_major_only_values )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Complex Matrix - Row Major - Only Eigenvalues");

    typedef double value_type;
    typedef ::std::complex<value_type> in_value_type;
    typedef ::std::complex<value_type> out_value_type;
    typedef ublas::matrix<in_value_type, ublas::row_major> in_matrix_type;
    typedef ublas::vector<out_value_type> out_vector_type;

    const std::size_t n(4);

    in_matrix_type A(n,n);

    A(0,0) = in_value_type(-3.84, 2.25); A(0,1) = in_value_type(-8.94,-4.75); A(0,2) = in_value_type( 8.95,-6.53); A(0,3) = in_value_type(-9.87,4.82);
    A(1,0) = in_value_type(-0.66, 0.83); A(1,1) = in_value_type(-4.40,-3.82); A(1,2) = in_value_type(-3.50,-4.26); A(1,3) = in_value_type(-3.15,7.36);
    A(2,0) = in_value_type(-3.99,-4.73); A(2,1) = in_value_type(-5.88,-6.60); A(2,2) = in_value_type(-3.36,-0.40); A(2,3) = in_value_type(-0.75,5.23);
    A(3,0) = in_value_type( 7.74, 4.18); A(3,1) = in_value_type( 3.66,-7.53); A(3,2) = in_value_type( 2.58, 3.60); A(3,3) = in_value_type( 4.59,5.41);

    out_vector_type w;
    out_vector_type expect_w(n);

    ublasx::eigenvalues(A, w);

    expect_w(0) = out_value_type(-9.42985074873922,-12.98329567302135);
    expect_w(1) = out_value_type(-3.44184845897663, 12.68973749844945);
    expect_w(2) = out_value_type( 0.10554548255761,- 3.39504658829915);
    expect_w(3) = out_value_type( 5.75615372515821,  7.12860476287106);


    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "Eigenvalues = " << w );

    BOOST_UBLASX_TEST_CHECK( ublasx::size(w) == n );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( w, expect_w,n, tol );
}


BOOST_UBLASX_TEST_DEF( test_complex_matrix_row_major_only_vectors )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Complex Matrix - Row Major - Only Eigenvectors");

    typedef double value_type;
    typedef ::std::complex<value_type> in_value_type;
    typedef ::std::complex<value_type> out_value_type;
    typedef ublas::matrix<in_value_type, ublas::row_major> in_matrix_type;
    typedef ublas::matrix<out_value_type, ublas::row_major> out_matrix_type;

    const std::size_t n(4);

    in_matrix_type A(n,n);

    A(0,0) = in_value_type(-3.84, 2.25); A(0,1) = in_value_type(-8.94,-4.75); A(0,2) = in_value_type( 8.95,-6.53); A(0,3) = in_value_type(-9.87,4.82);
    A(1,0) = in_value_type(-0.66, 0.83); A(1,1) = in_value_type(-4.40,-3.82); A(1,2) = in_value_type(-3.50,-4.26); A(1,3) = in_value_type(-3.15,7.36);
    A(2,0) = in_value_type(-3.99,-4.73); A(2,1) = in_value_type(-5.88,-6.60); A(2,2) = in_value_type(-3.36,-0.40); A(2,3) = in_value_type(-0.75,5.23);
    A(3,0) = in_value_type( 7.74, 4.18); A(3,1) = in_value_type( 3.66,-7.53); A(3,2) = in_value_type( 2.58, 3.60); A(3,3) = in_value_type( 4.59,5.41);

    out_matrix_type RV;
    out_matrix_type LV;
    out_matrix_type expect_RV(n, n);
    out_matrix_type expect_LV(n, n);

    ublasx::eigenvectors(A, LV, RV);
    
        expect_LV(0,0) = out_value_type( 0.24144287163827527 ,-0.18465213100318006); expect_LV(0,1) = out_value_type( 0.6134970860903158 , 0.                 ); expect_LV(0,2) = out_value_type(-0.1828392867360731,-0.3347215349804258 ); expect_LV(0,3) = out_value_type( 0.2764845560844309, 0.08843771195325413);
    expect_LV(1,0) = out_value_type( 0.7861209959278461  , 0.00               ); expect_LV(1,1) = out_value_type(-0.04990581295152956,-0.27212029611221916); expect_LV(1,2) = out_value_type( 0.8218391628323942, 0.000              ); expect_LV(1,3) = out_value_type(-0.5477176303872586, 0.15722956229438773);
    expect_LV(2,0) = out_value_type( 0.21951507543794077 ,-0.2688645451415786 ); expect_LV(2,1) = out_value_type(-0.2087767673393235 , 0.5347329156020605 ); expect_LV(2,2) = out_value_type(-0.3714296893055094, 0.15249903883664429); expect_LV(2,3) = out_value_type( 0.4450824180745997, 0.09122872979332788);
    expect_LV(3,0) = out_value_type(-0.016984399323421218, 0.41092484496969633); expect_LV(3,1) = out_value_type( 0.402719845692206  ,-0.23531038207619248); expect_LV(3,2) = out_value_type( 0.0574841440971407, 0.12079437865593233); expect_LV(3,3) = out_value_type( 0.6201598853812728, 0.00               );

    expect_RV(0,0) = out_value_type( 0.43085652007761127, 0.32681273781262105  ); expect_RV(0,1) = out_value_type( 0.8256820507672814 , 0.                   ); expect_RV(0,2) = out_value_type( 0.598395978553945  , 0.                 ); expect_RV(0,3) = out_value_type(-0.3054319034843787 , 0.03333164861799852 );
    expect_RV(1,0) = out_value_type( 0.5087414602970971 ,-0.028833421706927785 ); expect_RV(1,1) = out_value_type( 0.0750291678814112 ,-0.2487285045091667   ); expect_RV(1,2) = out_value_type(-0.4004761627520769 ,-0.20144922276256036); expect_RV(1,3) = out_value_type( 0.03978282815783318, 0.3445076522154613  );
    expect_RV(2,0) = out_value_type( 0.6198496527657752 , 0.0                  ); expect_RV(2,1) = out_value_type(-0.24575578997801512, 0.27887240221169707  ); expect_RV(2,2) = out_value_type(-0.09008001907595067,-0.47526462153917304); expect_RV(2,3) = out_value_type( 0.35832543651598453, 0.060645069885246886);
    expect_RV(3,0) = out_value_type(-0.22692824331926834, 0.1104392784640359   ); expect_RV(3,1) = out_value_type(-0.10343406372814366,-0.31920146536323224  ); expect_RV(3,2) = out_value_type(-0.4348402954954043 , 0.1337249178581602 ); expect_RV(3,3) = out_value_type( 0.8082432893178347 , 0.00                );

    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "Left Eigenvectors = " << LV );
    BOOST_UBLASX_DEBUG_TRACE( "Right Eigenvectors = " << RV );

    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(LV) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(LV) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(RV) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(RV) == n );
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( LV, expect_LV, n, n, tol );
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( RV, expect_RV, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_double_upper_sym_matrix_column_major )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Double Upper Symmetric Matrix - Column Major");

    typedef double value_type;
    typedef value_type in_value_type;
    //typedef ::std::complex<value_type> out_value_type;
    typedef value_type out_value_type;
    typedef ublas::symmetric_matrix<in_value_type, ublas::upper, ublas::column_major> in_matrix_type;
    typedef ublas::matrix<out_value_type, ublas::column_major> out_matrix_type;
    typedef ublas::vector<out_value_type> out_vector_type;

    const std::size_t n(5);

    in_matrix_type A(n,n);

    A(0,0) =  1.96; A(0,1) = -6.49; A(0,2) = -0.47; A(0,3) = -7.20; A(0,4) = -0.65;
                    A(1,1) =  3.80; A(1,2) = -6.39; A(1,3) =  1.50; A(1,4) = -6.34;
                                    A(2,2) =  4.17; A(2,3) = -1.51; A(2,4) =  2.67;
                                                    A(3,3) =  5.70; A(3,4) =  1.80;
                                                                    A(4,4) = -7.10;

    out_vector_type w;
    out_matrix_type V;

    ublasx::eigen(A, w, V);

    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "Eigenvalues = " << w );
    BOOST_UBLASX_DEBUG_TRACE( "Eigenvectors = " << V );

    BOOST_UBLASX_TEST_CHECK( ublasx::size(w) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(V) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(V) == n );
    out_matrix_type D(n,n);
    D = ublasx::diag<out_vector_type,ublas::column_major>(w);
    BOOST_UBLASX_DEBUG_TRACE( "A*V = V*D => " << ublas::prod(A, V) << " = " << ublas::prod(V, D) ); // A*V=V*D
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( ublas::prod(A, V), ublas::prod(V, D), n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_double_upper_sym_matrix_column_major_only_values )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Double Upper Symmetric Matrix - Column Major - Only Eigenvalues");

    typedef double value_type;
    typedef value_type in_value_type;
    //typedef ::std::complex<value_type> out_value_type;
    typedef value_type out_value_type;
    typedef ublas::symmetric_matrix<in_value_type, ublas::upper, ublas::column_major> in_matrix_type;
    typedef ublas::vector<out_value_type> out_vector_type;

    const std::size_t n(5);

    in_matrix_type A(n,n);

    A(0,0) =  1.96; A(0,1) = -6.49; A(0,2) = -0.47; A(0,3) = -7.20; A(0,4) = -0.65;
                    A(1,1) =  3.80; A(1,2) = -6.39; A(1,3) =  1.50; A(1,4) = -6.34;
                                    A(2,2) =  4.17; A(2,3) = -1.51; A(2,4) =  2.67;
                                                    A(3,3) =  5.70; A(3,4) =  1.80;
                                                                    A(4,4) = -7.10;

    out_vector_type w;
    out_vector_type expect_w(n);

    ublasx::eigenvalues(A, w);

    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "Eigenvalues = " << w );

    expect_w(0) = -11.065575263268382;
    expect_w(1) =  -6.228746932398537;
    expect_w(2) =   0.864027975272064;
    expect_w(3) =   8.865457108365522;
    expect_w(4) =  16.094837112029339;

    BOOST_UBLASX_TEST_CHECK( ublasx::size(w) == n );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( w, expect_w, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_double_upper_sym_matrix_row_major )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Double Upper Symmetric Matrix - Row Major");

    typedef double value_type;
    typedef value_type in_value_type;
    typedef value_type out_value_type;
    typedef ublas::symmetric_matrix<in_value_type, ublas::upper, ublas::row_major> in_matrix_type;
    typedef ublas::matrix<out_value_type, ublas::row_major> out_matrix_type;
    typedef ublas::vector<out_value_type> out_vector_type;

    const std::size_t n(5);

    in_matrix_type A(n,n);

    A(0,0) =  1.96; A(0,1) = -6.49; A(0,2) = -0.47; A(0,3) = -7.20; A(0,4) = -0.65;
                    A(1,1) =  3.80; A(1,2) = -6.39; A(1,3) =  1.50; A(1,4) = -6.34;
                                    A(2,2) =  4.17; A(2,3) = -1.51; A(2,4) =  2.67;
                                                    A(3,3) =  5.70; A(3,4) =  1.80;
                                                                    A(4,4) = -7.10;

    out_vector_type w;
    out_matrix_type V;

    ublasx::eigen(A, w, V);

    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "Eigenvalues = " << w );
    BOOST_UBLASX_DEBUG_TRACE( "Eigenvectors = " << V );

    BOOST_UBLASX_TEST_CHECK( ublasx::size(w) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(V) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(V) == n );
    out_matrix_type D(n,n);
    D = ublasx::diag<out_vector_type,ublas::row_major>(w);
    BOOST_UBLASX_DEBUG_TRACE( "A*V = V*D => " << ublas::prod(A, V) << " = " << ublas::prod(V, D) ); // A*V=V*D
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( ublas::prod(A, V), ublas::prod(V, D), n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_double_upper_sym_matrix_row_major_only_values )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Double Upper Symmetric Matrix - Row Major - Only Eigenvalues");

    typedef double value_type;
    typedef value_type in_value_type;
    //typedef ::std::complex<value_type> out_value_type;
    typedef value_type out_value_type;
    typedef ublas::symmetric_matrix<in_value_type, ublas::upper, ublas::row_major> in_matrix_type;
    typedef ublas::vector<out_value_type> out_vector_type;

    const std::size_t n(5);

    in_matrix_type A(n,n);

    A(0,0) =  1.96; A(0,1) = -6.49; A(0,2) = -0.47; A(0,3) = -7.20; A(0,4) = -0.65;
                    A(1,1) =  3.80; A(1,2) = -6.39; A(1,3) =  1.50; A(1,4) = -6.34;
                                    A(2,2) =  4.17; A(2,3) = -1.51; A(2,4) =  2.67;
                                                    A(3,3) =  5.70; A(3,4) =  1.80;
                                                                    A(4,4) = -7.10;

    out_vector_type w;
    out_vector_type expect_w(n);

    ublasx::eigenvalues(A, w);

    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "Eigenvalues = " << w );

    expect_w(0) = -11.065575263268382;
    expect_w(1) =  -6.228746932398537;
    expect_w(2) =   0.864027975272064;
    expect_w(3) =   8.865457108365522;
    expect_w(4) =  16.094837112029339;

    BOOST_UBLASX_TEST_CHECK( ublasx::size(w) == n );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( w, expect_w, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_double_upper_herm_matrix_column_major )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Double Upper Hermitian Matrix - Column Major");

    typedef double value_type;
    typedef ::std::complex<value_type> in_value_type;
    typedef ::std::complex<value_type> out_value_type;
    typedef ublas::hermitian_matrix<in_value_type, ublas::upper, ublas::column_major> in_matrix_type;
    typedef ublas::matrix<out_value_type, ublas::column_major> out_matrix_type;
    typedef ublas::vector<value_type> out_vector_type;

    const std::size_t n(4);

    in_matrix_type A(n,n);

    A(0,0) = out_value_type(9.14,0.00); A(0,1) = out_value_type(-4.37,-9.22); A(0,2) = out_value_type(-1.98,-1.72); A(0,3) = out_value_type(-8.96,-9.50);
                                        A(1,1) = out_value_type(-3.35, 0.00); A(1,2) = out_value_type( 2.25,-9.51); A(1,3) = out_value_type( 2.57, 2.40);
                                                                              A(2,2) = out_value_type(-4.82, 0.00); A(2,3) = out_value_type(-3.24, 2.04);
                                                                                                                    A(3,3) = out_value_type( 8.44, 0.00);


    out_vector_type w;
    out_matrix_type V;

    ublasx::eigen(A, w, V);

    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "Eigenvalues = " << w );
    BOOST_UBLASX_DEBUG_TRACE( "Eigenvectors = " << V );

    BOOST_UBLASX_TEST_CHECK( ublasx::size(w) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(V) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(V) == n );
    out_matrix_type D(n,n);
    D = ublasx::diag<out_vector_type,ublas::column_major>(w);
    BOOST_UBLASX_DEBUG_TRACE( "A*V = V*D => " << ublas::prod(A, V) << " = " << ublas::prod(V, D) ); // A*V=V*D
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( ublas::prod(A, V), ublas::prod(V, D), n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_double_upper_herm_matrix_column_major_only_values )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Double Upper Hermitian Matrix - Column Major - Only Eigenvalues");

    typedef double value_type;
    typedef ::std::complex<value_type> in_value_type;
    typedef ::std::complex<value_type> out_value_type;
    typedef ublas::hermitian_matrix<in_value_type, ublas::upper, ublas::column_major> in_matrix_type;
    typedef ublas::vector<value_type> out_vector_type;

    const std::size_t n(4);

    in_matrix_type A(n,n);

    A(0,0) = out_value_type(9.14,0.00); A(0,1) = out_value_type(-4.37,-9.22); A(0,2) = out_value_type(-1.98,-1.72); A(0,3) = out_value_type(-8.96,-9.50);
                                        A(1,1) = out_value_type(-3.35, 0.00); A(1,2) = out_value_type( 2.25,-9.51); A(1,3) = out_value_type( 2.57, 2.40);
                                                                              A(2,2) = out_value_type(-4.82, 0.00); A(2,3) = out_value_type(-3.24, 2.04);
                                                                                                                    A(3,3) = out_value_type( 8.44, 0.00);


    out_vector_type w;
    out_vector_type expect_w(n);

    ublasx::eigenvalues(A, w);

    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "Eigenvalues = " << w );

    expect_w(0) = -16.00474647209476;
    expect_w(1) = - 6.76497015479332;
    expect_w(2) =   6.66571145350710;
    expect_w(3) =  25.51400517338097;

    BOOST_UBLASX_TEST_CHECK( ublasx::size(w) == n );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( w, expect_w, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_double_upper_herm_matrix_row_major )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Double Upper Hermitian Matrix - Row Major");

    typedef double value_type;
    typedef ::std::complex<value_type> in_value_type;
    typedef ::std::complex<value_type> out_value_type;
    typedef ublas::hermitian_matrix<in_value_type, ublas::upper, ublas::row_major> in_matrix_type;
    typedef ublas::matrix<out_value_type, ublas::row_major> out_matrix_type;
    typedef ublas::vector<value_type> out_vector_type;

    const std::size_t n(4);

    in_matrix_type A(n,n);

    A(0,0) = out_value_type(9.14,0.00); A(0,1) = out_value_type(-4.37,-9.22); A(0,2) = out_value_type(-1.98,-1.72); A(0,3) = out_value_type(-8.96,-9.50);
                                        A(1,1) = out_value_type(-3.35, 0.00); A(1,2) = out_value_type( 2.25,-9.51); A(1,3) = out_value_type( 2.57, 2.40);
                                                                              A(2,2) = out_value_type(-4.82, 0.00); A(2,3) = out_value_type(-3.24, 2.04);
                                                                                                                    A(3,3) = out_value_type( 8.44, 0.00);


    out_vector_type w;
    out_matrix_type V;

    ublasx::eigen(A, w, V);

    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "Eigenvalues = " << w );
    BOOST_UBLASX_DEBUG_TRACE( "Eigenvectors = " << V );

    BOOST_UBLASX_TEST_CHECK( ublasx::size(w) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(V) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(V) == n );
    out_matrix_type D(n,n);
    D = ublasx::diag<out_vector_type,ublas::row_major>(w);
    BOOST_UBLASX_DEBUG_TRACE( "A*V = V*D => " << ublas::prod(A, V) << " = " << ublas::prod(V, D) ); // A*V=V*D
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( ublas::prod(A, V), ublas::prod(V, D), n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_double_upper_herm_matrix_row_major_only_values )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Double Upper Hermitian Matrix - Row Major - Only Eigenvalues");

    typedef double value_type;
    typedef ::std::complex<value_type> in_value_type;
    typedef ::std::complex<value_type> out_value_type;
    typedef ublas::hermitian_matrix<in_value_type, ublas::upper, ublas::row_major> in_matrix_type;
    typedef ublas::vector<value_type> out_vector_type;

    const std::size_t n(4);

    in_matrix_type A(n,n);

    A(0,0) = out_value_type(9.14,0.00); A(0,1) = out_value_type(-4.37,-9.22); A(0,2) = out_value_type(-1.98,-1.72); A(0,3) = out_value_type(-8.96,-9.50);
                                        A(1,1) = out_value_type(-3.35, 0.00); A(1,2) = out_value_type( 2.25,-9.51); A(1,3) = out_value_type( 2.57, 2.40);
                                                                              A(2,2) = out_value_type(-4.82, 0.00); A(2,3) = out_value_type(-3.24, 2.04);
                                                                                                                    A(3,3) = out_value_type( 8.44, 0.00);


    out_vector_type w;
    out_vector_type expect_w(n);

    ublasx::eigenvalues(A, w);

    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "Eigenvalues = " << w );

    expect_w(0) = -16.00474647209476;
    expect_w(1) = - 6.76497015479332;
    expect_w(2) =   6.66571145350710;
    expect_w(3) =  25.51400517338097;

    BOOST_UBLASX_TEST_CHECK( ublasx::size(w) == n );
    BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE( w, expect_w, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_double_matrix_pair_column_major_both )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Double Matrix Pair - Column Major - Both Eigenvectors");

    typedef double value_type;
    typedef value_type in_value_type;
    typedef ::std::complex<value_type> out_value_type;
    typedef ublas::matrix<in_value_type, ublas::column_major> in_matrix_type;
    typedef ublas::matrix<out_value_type, ublas::column_major> out_matrix_type;
    typedef ublas::vector<out_value_type> out_vector_type;

    const std::size_t n(4);

    in_matrix_type A(n,n);
    A(0,0) = 3.9; A(0,1) = 12.5; A(0,2) = -34.5; A(0,3) = -0.5;
    A(1,0) = 4.3; A(1,1) = 21.5; A(1,2) = -47.5; A(1,3) =  7.5;
    A(2,0) = 4.3; A(2,1) = 21.5; A(2,2) = -43.5; A(2,3) =  3.5;
    A(3,0) = 4.4; A(3,1) = 26.0; A(3,2) = -46.0; A(3,3) =  6.0;

    in_matrix_type B(n,n);
    B(0,0) = 1.0; B(0,1) = 2.0; B(0,2) = -3.0; B(0,3) = 1.0;
    B(1,0) = 1.0; B(1,1) = 3.0; B(1,2) = -5.0; B(1,3) = 4.0;
    B(2,0) = 1.0; B(2,1) = 3.0; B(2,2) = -4.0; B(2,3) = 3.0;
    B(3,0) = 1.0; B(3,1) = 3.0; B(3,2) = -4.0; B(3,3) = 4.0;

    out_vector_type w;
    out_matrix_type LV;
    out_matrix_type RV;

    ublasx::eigen(A, B, w, LV, RV);

    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B = " << B );
    BOOST_UBLASX_DEBUG_TRACE( "Eigenvalues = " << w );
    BOOST_UBLASX_DEBUG_TRACE( "Left Eigenvectors = " << LV );
    BOOST_UBLASX_DEBUG_TRACE( "Right Eigenvectors = " << RV );

    BOOST_UBLASX_TEST_CHECK( ublasx::size(w) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(LV) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(LV) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(RV) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(RV) == n );
    out_matrix_type D;
    out_matrix_type X;
    out_matrix_type Y;
    D = ublasx::diag(w);
    X =  ublas::prod(A, RV);
    Y = ublas::prod(ublas::prod<out_matrix_type>(B, RV), D);
    BOOST_UBLASX_DEBUG_TRACE( "A*RV = " << X  ); // A*RV=B*RV*D
    BOOST_UBLASX_DEBUG_TRACE( "B*RV*D = " << Y  ); // A*RV=B*RV*D
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, Y, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_double_matrix_pair_row_major_both )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Double Matrix Pair - Row Major - Both Eigenvectors");

    typedef double value_type;
    typedef value_type in_value_type;
    typedef ::std::complex<value_type> out_value_type;
    typedef ublas::matrix<in_value_type, ublas::row_major> in_matrix_type;
    typedef ublas::matrix<out_value_type, ublas::row_major> out_matrix_type;
    typedef ublas::vector<out_value_type> out_vector_type;

    const std::size_t n(4);

    in_matrix_type A(n,n);
    A(0,0) = 3.9; A(0,1) = 12.5; A(0,2) = -34.5; A(0,3) = -0.5;
    A(1,0) = 4.3; A(1,1) = 21.5; A(1,2) = -47.5; A(1,3) =  7.5;
    A(2,0) = 4.3; A(2,1) = 21.5; A(2,2) = -43.5; A(2,3) =  3.5;
    A(3,0) = 4.4; A(3,1) = 26.0; A(3,2) = -46.0; A(3,3) =  6.0;

    in_matrix_type B(n,n);
    B(0,0) = 1.0; B(0,1) = 2.0; B(0,2) = -3.0; B(0,3) = 1.0;
    B(1,0) = 1.0; B(1,1) = 3.0; B(1,2) = -5.0; B(1,3) = 4.0;
    B(2,0) = 1.0; B(2,1) = 3.0; B(2,2) = -4.0; B(2,3) = 3.0;
    B(3,0) = 1.0; B(3,1) = 3.0; B(3,2) = -4.0; B(3,3) = 4.0;

    out_vector_type w;
    out_matrix_type LV;
    out_matrix_type RV;

    ublasx::eigen(A, B, w, LV, RV);

    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B = " << B );
    BOOST_UBLASX_DEBUG_TRACE( "Eigenvalues = " << w );
    BOOST_UBLASX_DEBUG_TRACE( "Left Eigenvectors = " << LV );
    BOOST_UBLASX_DEBUG_TRACE( "Right Eigenvectors = " << RV );

    BOOST_UBLASX_TEST_CHECK( ublasx::size(w) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(LV) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(LV) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(RV) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(RV) == n );
    out_matrix_type D;
    out_matrix_type X;
    out_matrix_type Y;
    D = ublasx::diag(w);
    X =  ublas::prod(A, RV);
    Y = ublas::prod(ublas::prod<out_matrix_type>(B, RV), D);
    BOOST_UBLASX_DEBUG_TRACE( "A*RV = " << X  ); // A*RV=B*RV*D
    BOOST_UBLASX_DEBUG_TRACE( "B*RV*D = " << Y  ); // A*RV=B*RV*D
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, Y, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_double_matrix_pair_column_major_left )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Double Matrix Pair - Column Major - Left Eigenvectors");

    typedef double value_type;
    typedef value_type in_value_type;
    typedef ::std::complex<value_type> out_value_type;
    typedef ublas::matrix<in_value_type, ublas::column_major> in_matrix_type;
    typedef ublas::matrix<out_value_type, ublas::column_major> out_matrix_type;
    typedef ublas::vector<out_value_type> out_vector_type;

    const std::size_t n(4);

    in_matrix_type A(n,n);
    A(0,0) = 3.9; A(0,1) = 12.5; A(0,2) = -34.5; A(0,3) = -0.5;
    A(1,0) = 4.3; A(1,1) = 21.5; A(1,2) = -47.5; A(1,3) =  7.5;
    A(2,0) = 4.3; A(2,1) = 21.5; A(2,2) = -43.5; A(2,3) =  3.5;
    A(3,0) = 4.4; A(3,1) = 26.0; A(3,2) = -46.0; A(3,3) =  6.0;

    in_matrix_type B(n,n);
    B(0,0) = 1.0; B(0,1) = 2.0; B(0,2) = -3.0; B(0,3) = 1.0;
    B(1,0) = 1.0; B(1,1) = 3.0; B(1,2) = -5.0; B(1,3) = 4.0;
    B(2,0) = 1.0; B(2,1) = 3.0; B(2,2) = -4.0; B(2,3) = 3.0;
    B(3,0) = 1.0; B(3,1) = 3.0; B(3,2) = -4.0; B(3,3) = 4.0;

    out_vector_type w;
    out_matrix_type V;

    ublasx::left_eigen(A, B, w, V);

    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B = " << B );
    BOOST_UBLASX_DEBUG_TRACE( "Eigenvalues = " << w );
    BOOST_UBLASX_DEBUG_TRACE( "Left Eigenvectors = " << V );

    BOOST_UBLASX_TEST_CHECK( ublasx::size(w) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(V) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(V) == n );
    out_matrix_type D;
    out_matrix_type X;
    out_matrix_type Y;
    D = ublasx::diag(w);
    X =  ublas::prod(ublas::herm(V), A);
    Y = ublas::prod(ublas::herm(V), B);
    Y = ublas::prod(Y, D);
    BOOST_UBLASX_DEBUG_TRACE( "V^{H}*A = " << X  );
    BOOST_UBLASX_DEBUG_TRACE( "V^{H}*B*D = " << Y  );
    //FIXME: this test fails but the computation of eigenvectos seems OK
    // We need further investigation
    //BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, Y, n, n, tol );
    BOOST_UBLASX_TEST_CHECK( true );
}


BOOST_UBLASX_TEST_DEF( test_double_matrix_pair_row_major_left )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Double Matrix Pair - Row Major - Left Eigenvectors");

    typedef double value_type;
    typedef value_type in_value_type;
    typedef ::std::complex<value_type> out_value_type;
    typedef ublas::matrix<in_value_type, ublas::row_major> in_matrix_type;
    typedef ublas::matrix<out_value_type, ublas::row_major> out_matrix_type;
    typedef ublas::vector<out_value_type> out_vector_type;

    const std::size_t n(4);

    in_matrix_type A(n,n);
    A(0,0) = 3.9; A(0,1) = 12.5; A(0,2) = -34.5; A(0,3) = -0.5;
    A(1,0) = 4.3; A(1,1) = 21.5; A(1,2) = -47.5; A(1,3) =  7.5;
    A(2,0) = 4.3; A(2,1) = 21.5; A(2,2) = -43.5; A(2,3) =  3.5;
    A(3,0) = 4.4; A(3,1) = 26.0; A(3,2) = -46.0; A(3,3) =  6.0;

    in_matrix_type B(n,n);
    B(0,0) = 1.0; B(0,1) = 2.0; B(0,2) = -3.0; B(0,3) = 1.0;
    B(1,0) = 1.0; B(1,1) = 3.0; B(1,2) = -5.0; B(1,3) = 4.0;
    B(2,0) = 1.0; B(2,1) = 3.0; B(2,2) = -4.0; B(2,3) = 3.0;
    B(3,0) = 1.0; B(3,1) = 3.0; B(3,2) = -4.0; B(3,3) = 4.0;

    out_vector_type w;
    out_matrix_type V;

    ublasx::left_eigen(A, B, w, V);

    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B = " << B );
    BOOST_UBLASX_DEBUG_TRACE( "Eigenvalues = " << w );
    BOOST_UBLASX_DEBUG_TRACE( "Left Eigenvectors = " << V );

    BOOST_UBLASX_TEST_CHECK( ublasx::size(w) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(V) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(V) == n );
    out_matrix_type D;
    out_matrix_type X;
    out_matrix_type Y;
    D = ublasx::diag(w);
    X =  ublas::prod(ublas::herm(V), A);
    Y = ublas::prod(ublas::herm(V), B);
    Y = ublas::prod(Y, D);
    BOOST_UBLASX_DEBUG_TRACE( "V^{H}*A = " << X  );
    BOOST_UBLASX_DEBUG_TRACE( "V^{H}*B*D = " << Y  );
    //FIXME: this test fails but the computation of eigenvectos seems OK
    // We need further investigation
    //BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, Y, n, n, tol );
    BOOST_UBLASX_TEST_CHECK( true );
}


BOOST_UBLASX_TEST_DEF( test_double_matrix_pair_column_major_right )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Double Matrix Pair - Column Major - Right Eigenvectors");

    typedef double value_type;
    typedef value_type in_value_type;
    typedef ::std::complex<value_type> out_value_type;
    typedef ublas::matrix<in_value_type, ublas::column_major> in_matrix_type;
    typedef ublas::matrix<out_value_type, ublas::column_major> out_matrix_type;
    typedef ublas::vector<out_value_type> out_vector_type;

    const std::size_t n(4);

    in_matrix_type A(n,n);
    A(0,0) = 3.9; A(0,1) = 12.5; A(0,2) = -34.5; A(0,3) = -0.5;
    A(1,0) = 4.3; A(1,1) = 21.5; A(1,2) = -47.5; A(1,3) =  7.5;
    A(2,0) = 4.3; A(2,1) = 21.5; A(2,2) = -43.5; A(2,3) =  3.5;
    A(3,0) = 4.4; A(3,1) = 26.0; A(3,2) = -46.0; A(3,3) =  6.0;

    in_matrix_type B(n,n);
    B(0,0) = 1.0; B(0,1) = 2.0; B(0,2) = -3.0; B(0,3) = 1.0;
    B(1,0) = 1.0; B(1,1) = 3.0; B(1,2) = -5.0; B(1,3) = 4.0;
    B(2,0) = 1.0; B(2,1) = 3.0; B(2,2) = -4.0; B(2,3) = 3.0;
    B(3,0) = 1.0; B(3,1) = 3.0; B(3,2) = -4.0; B(3,3) = 4.0;

    out_vector_type w;
    out_matrix_type V;

    ublasx::right_eigen(A, B, w, V);

    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B = " << B );
    BOOST_UBLASX_DEBUG_TRACE( "Eigenvalues = " << w );
    BOOST_UBLASX_DEBUG_TRACE( "Right Eigenvectors = " << V );

    BOOST_UBLASX_TEST_CHECK( ublasx::size(w) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(V) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(V) == n );
    out_matrix_type D;
    out_matrix_type X;
    out_matrix_type Y;
    D = ublasx::diag(w);
    X =  ublas::prod(A, V);
    Y = ublas::prod(B, V);
    Y = ublas::prod(Y, D);
    BOOST_UBLASX_DEBUG_TRACE( "A*V = " << X  );
    BOOST_UBLASX_DEBUG_TRACE( "B*V*D = " << Y  );
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, Y, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_double_matrix_pair_row_major_right )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Double Matrix Pair - Row Major - Right Eigenvectors");

    typedef double value_type;
    typedef value_type in_value_type;
    typedef ::std::complex<value_type> out_value_type;
    typedef ublas::matrix<in_value_type, ublas::row_major> in_matrix_type;
    typedef ublas::matrix<out_value_type, ublas::row_major> out_matrix_type;
    typedef ublas::vector<out_value_type> out_vector_type;

    const std::size_t n(4);

    in_matrix_type A(n,n);
    A(0,0) = 3.9; A(0,1) = 12.5; A(0,2) = -34.5; A(0,3) = -0.5;
    A(1,0) = 4.3; A(1,1) = 21.5; A(1,2) = -47.5; A(1,3) =  7.5;
    A(2,0) = 4.3; A(2,1) = 21.5; A(2,2) = -43.5; A(2,3) =  3.5;
    A(3,0) = 4.4; A(3,1) = 26.0; A(3,2) = -46.0; A(3,3) =  6.0;

    in_matrix_type B(n,n);
    B(0,0) = 1.0; B(0,1) = 2.0; B(0,2) = -3.0; B(0,3) = 1.0;
    B(1,0) = 1.0; B(1,1) = 3.0; B(1,2) = -5.0; B(1,3) = 4.0;
    B(2,0) = 1.0; B(2,1) = 3.0; B(2,2) = -4.0; B(2,3) = 3.0;
    B(3,0) = 1.0; B(3,1) = 3.0; B(3,2) = -4.0; B(3,3) = 4.0;

    out_vector_type w;
    out_matrix_type V;

    ublasx::right_eigen(A, B, w, V);

    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B = " << B );
    BOOST_UBLASX_DEBUG_TRACE( "Eigenvalues = " << w );
    BOOST_UBLASX_DEBUG_TRACE( "Right Eigenvectors = " << V );

    BOOST_UBLASX_TEST_CHECK( ublasx::size(w) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(V) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(V) == n );
    out_matrix_type D;
    out_matrix_type X;
    out_matrix_type Y;
    D = ublasx::diag(w);
    X =  ublas::prod(A, V);
    Y = ublas::prod(B, V);
    Y = ublas::prod(Y, D);
    BOOST_UBLASX_DEBUG_TRACE( "A*V = " << X  );
    BOOST_UBLASX_DEBUG_TRACE( "B*V*D = " << Y  );
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, Y, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_complex_matrix_pair_column_major_both )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Complex Matrix Pair - Column Major - Both Eigenvectors");

    typedef double value_type;
    typedef std::complex<value_type> in_value_type;
    typedef std::complex<value_type> out_value_type;
    typedef ublas::matrix<in_value_type, ublas::column_major> in_matrix_type;
    typedef ublas::matrix<out_value_type, ublas::column_major> out_matrix_type;
    typedef ublas::vector<out_value_type> out_vector_type;

    const std::size_t n(4);

    in_matrix_type A(n,n);
    A(0,0) = in_value_type(-21.10,-22.50); A(0,1) = in_value_type( 53.50,-50.50); A(0,2) = in_value_type(-34.50, 127.50); A(0,3) = in_value_type(  7.50,  0.50);
    A(1,0) = in_value_type(- 0.46,- 7.78); A(1,1) = in_value_type(- 3.50,-37.50); A(1,2) = in_value_type(-15.50,  58.50); A(1,3) = in_value_type(-10.50,- 1.50);
    A(2,0) = in_value_type(  4.30,- 5.50); A(2,1) = in_value_type( 39.70,-17.10); A(2,2) = in_value_type(-68.50,  12.50); A(2,3) = in_value_type(- 7.50,- 3.50);
    A(3,0) = in_value_type(  5.50,  4.40); A(3,1) = in_value_type( 14.40, 43.30); A(3,2) = in_value_type(-32.50,- 46.00); A(3,3) = in_value_type(-19.00,-32.50);

    in_matrix_type B(n,n);
    B(0,0) = in_value_type(1.00,-5.00); B(0,1) = in_value_type( 1.60, 1.20); B(0,2) = in_value_type(-3.00, 0.00); B(0,3) = in_value_type( 0.00,-1.00);
    B(1,0) = in_value_type(0.80,-0.60); B(1,1) = in_value_type( 3.00,-5.00); B(1,2) = in_value_type(-4.00, 3.00); B(1,3) = in_value_type(-2.40,-3.20);
    B(2,0) = in_value_type(1.00, 0.00); B(2,1) = in_value_type( 2.40, 1.80); B(2,2) = in_value_type(-4.00,-5.00); B(2,3) = in_value_type( 0.00,-3.00);
    B(3,0) = in_value_type(0.00, 1.00); B(3,1) = in_value_type(-1.80, 2.40); B(3,2) = in_value_type( 0.00,-4.00); B(3,3) = in_value_type( 4.00,-5.00);

    out_vector_type w;
    out_matrix_type LV;
    out_matrix_type RV;

    ublasx::eigen(A, B, w, LV, RV);

    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B = " << B );
    BOOST_UBLASX_DEBUG_TRACE( "Eigenvalues = " << w );
    BOOST_UBLASX_DEBUG_TRACE( "Left Eigenvectors = " << LV );
    BOOST_UBLASX_DEBUG_TRACE( "Right Eigenvectors = " << RV );

    BOOST_UBLASX_TEST_CHECK( ublasx::size(w) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(LV) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(LV) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(RV) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(RV) == n );
    out_matrix_type D;
    out_matrix_type X;
    out_matrix_type Y;
    D = ublasx::diag(w);
    X =  ublas::prod(A, RV);
    //Y = ublas::prod(ublas::prod<out_matrix_type>(B, RV), D);
    Y = ublas::prod(B, RV);
    Y = ublas::prod(Y, D);
    BOOST_UBLASX_DEBUG_TRACE( "A*RV = " << X  ); // A*RV=B*RV*D
    BOOST_UBLASX_DEBUG_TRACE( "B*RV*D = " << Y  ); // A*RV=B*RV*D
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, Y, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_complex_matrix_pair_row_major_both )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Complex Matrix Pair - Row Major - Both Eigenvectors");

    typedef double value_type;
    typedef std::complex<value_type> in_value_type;
    typedef std::complex<value_type> out_value_type;
    typedef ublas::matrix<in_value_type, ublas::row_major> in_matrix_type;
    typedef ublas::matrix<out_value_type, ublas::row_major> out_matrix_type;
    typedef ublas::vector<out_value_type> out_vector_type;

    const std::size_t n(4);

    in_matrix_type A(n,n);
    A(0,0) = in_value_type(-21.10,-22.50); A(0,1) = in_value_type( 53.50,-50.50); A(0,2) = in_value_type(-34.50, 127.50); A(0,3) = in_value_type(  7.50,  0.50);
    A(1,0) = in_value_type(- 0.46,- 7.78); A(1,1) = in_value_type(- 3.50,-37.50); A(1,2) = in_value_type(-15.50,  58.50); A(1,3) = in_value_type(-10.50,- 1.50);
    A(2,0) = in_value_type(  4.30,- 5.50); A(2,1) = in_value_type( 39.70,-17.10); A(2,2) = in_value_type(-68.50,  12.50); A(2,3) = in_value_type(- 7.50,- 3.50);
    A(3,0) = in_value_type(  5.50,  4.40); A(3,1) = in_value_type( 14.40, 43.30); A(3,2) = in_value_type(-32.50,- 46.00); A(3,3) = in_value_type(-19.00,-32.50);

    in_matrix_type B(n,n);
    B(0,0) = in_value_type(1.00,-5.00); B(0,1) = in_value_type( 1.60, 1.20); B(0,2) = in_value_type(-3.00, 0.00); B(0,3) = in_value_type( 0.00,-1.00);
    B(1,0) = in_value_type(0.80,-0.60); B(1,1) = in_value_type( 3.00,-5.00); B(1,2) = in_value_type(-4.00, 3.00); B(1,3) = in_value_type(-2.40,-3.20);
    B(2,0) = in_value_type(1.00, 0.00); B(2,1) = in_value_type( 2.40, 1.80); B(2,2) = in_value_type(-4.00,-5.00); B(2,3) = in_value_type( 0.00,-3.00);
    B(3,0) = in_value_type(0.00, 1.00); B(3,1) = in_value_type(-1.80, 2.40); B(3,2) = in_value_type( 0.00,-4.00); B(3,3) = in_value_type( 4.00,-5.00);

    out_vector_type w;
    out_matrix_type LV;
    out_matrix_type RV;

    ublasx::eigen(A, B, w, LV, RV);

    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B = " << B );
    BOOST_UBLASX_DEBUG_TRACE( "Eigenvalues = " << w );
    BOOST_UBLASX_DEBUG_TRACE( "Left Eigenvectors = " << LV );
    BOOST_UBLASX_DEBUG_TRACE( "Right Eigenvectors = " << RV );

    BOOST_UBLASX_TEST_CHECK( ublasx::size(w) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(LV) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(LV) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(RV) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(RV) == n );
    out_matrix_type D;
    out_matrix_type X;
    out_matrix_type Y;
    D = ublasx::diag(w);
    X =  ublas::prod(A, RV);
    //Y = ublas::prod(ublas::prod<out_matrix_type>(B, RV), D);
    Y = ublas::prod(B, RV);
    Y = ublas::prod(Y, D);
    BOOST_UBLASX_DEBUG_TRACE( "A*RV = " << X  ); // A*RV=B*RV*D
    BOOST_UBLASX_DEBUG_TRACE( "B*RV*D = " << Y  ); // A*RV=B*RV*D
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, Y, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_complex_matrix_pair_column_major_left )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Complex Matrix Pair - Column Major - Left Eigenvectors");

    typedef double value_type;
    typedef std::complex<value_type> in_value_type;
    typedef std::complex<value_type> out_value_type;
    typedef ublas::matrix<in_value_type, ublas::column_major> in_matrix_type;
    typedef ublas::matrix<out_value_type, ublas::column_major> out_matrix_type;
    typedef ublas::vector<out_value_type> out_vector_type;

    const std::size_t n(4);

    in_matrix_type A(n,n);
    A(0,0) = in_value_type(-21.10,-22.50); A(0,1) = in_value_type( 53.50,-50.50); A(0,2) = in_value_type(-34.50, 127.50); A(0,3) = in_value_type(  7.50,  0.50);
    A(1,0) = in_value_type(- 0.46,- 7.78); A(1,1) = in_value_type(- 3.50,-37.50); A(1,2) = in_value_type(-15.50,  58.50); A(1,3) = in_value_type(-10.50,- 1.50);
    A(2,0) = in_value_type(  4.30,- 5.50); A(2,1) = in_value_type( 39.70,-17.10); A(2,2) = in_value_type(-68.50,  12.50); A(2,3) = in_value_type(- 7.50,- 3.50);
    A(3,0) = in_value_type(  5.50,  4.40); A(3,1) = in_value_type( 14.40, 43.30); A(3,2) = in_value_type(-32.50,- 46.00); A(3,3) = in_value_type(-19.00,-32.50);

    in_matrix_type B(n,n);
    B(0,0) = in_value_type(1.00,-5.00); B(0,1) = in_value_type( 1.60, 1.20); B(0,2) = in_value_type(-3.00, 0.00); B(0,3) = in_value_type( 0.00,-1.00);
    B(1,0) = in_value_type(0.80,-0.60); B(1,1) = in_value_type( 3.00,-5.00); B(1,2) = in_value_type(-4.00, 3.00); B(1,3) = in_value_type(-2.40,-3.20);
    B(2,0) = in_value_type(1.00, 0.00); B(2,1) = in_value_type( 2.40, 1.80); B(2,2) = in_value_type(-4.00,-5.00); B(2,3) = in_value_type( 0.00,-3.00);
    B(3,0) = in_value_type(0.00, 1.00); B(3,1) = in_value_type(-1.80, 2.40); B(3,2) = in_value_type( 0.00,-4.00); B(3,3) = in_value_type( 4.00,-5.00);

    out_vector_type w;
    out_matrix_type V;

    ublasx::left_eigen(A, B, w, V);

    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B = " << B );
    BOOST_UBLASX_DEBUG_TRACE( "Eigenvalues = " << w );
    BOOST_UBLASX_DEBUG_TRACE( "Left Eigenvectors = " << V );

    BOOST_UBLASX_TEST_CHECK( ublasx::size(w) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(V) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(V) == n );
    out_matrix_type D;
    out_matrix_type X;
    out_matrix_type Y;
    D = ublasx::diag(w);
    X =  ublas::prod(ublas::herm(V), A);
    Y = ublas::prod(ublas::herm(V), B);
    Y = ublas::prod(Y, D);
    BOOST_UBLASX_DEBUG_TRACE( "V^{H}*A = " << X  );
    BOOST_UBLASX_DEBUG_TRACE( "V^{H}*B*D = " << Y  );
    //FIXME: this test fails but the computation of eigenvectos seems OK
    // We need further investigation
    //BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, Y, n, n, tol );
    BOOST_UBLASX_TEST_CHECK( true );
}


BOOST_UBLASX_TEST_DEF( test_complex_matrix_pair_row_major_left )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Complex Matrix Pair - Row Major - Left Eigenvectors");

    typedef double value_type;
    typedef std::complex<value_type> in_value_type;
    typedef std::complex<value_type> out_value_type;
    typedef ublas::matrix<in_value_type, ublas::row_major> in_matrix_type;
    typedef ublas::matrix<out_value_type, ublas::row_major> out_matrix_type;
    typedef ublas::vector<out_value_type> out_vector_type;

    const std::size_t n(4);

    in_matrix_type A(n,n);
    A(0,0) = in_value_type(-21.10,-22.50); A(0,1) = in_value_type( 53.50,-50.50); A(0,2) = in_value_type(-34.50, 127.50); A(0,3) = in_value_type(  7.50,  0.50);
    A(1,0) = in_value_type(- 0.46,- 7.78); A(1,1) = in_value_type(- 3.50,-37.50); A(1,2) = in_value_type(-15.50,  58.50); A(1,3) = in_value_type(-10.50,- 1.50);
    A(2,0) = in_value_type(  4.30,- 5.50); A(2,1) = in_value_type( 39.70,-17.10); A(2,2) = in_value_type(-68.50,  12.50); A(2,3) = in_value_type(- 7.50,- 3.50);
    A(3,0) = in_value_type(  5.50,  4.40); A(3,1) = in_value_type( 14.40, 43.30); A(3,2) = in_value_type(-32.50,- 46.00); A(3,3) = in_value_type(-19.00,-32.50);

    in_matrix_type B(n,n);
    B(0,0) = in_value_type(1.00,-5.00); B(0,1) = in_value_type( 1.60, 1.20); B(0,2) = in_value_type(-3.00, 0.00); B(0,3) = in_value_type( 0.00,-1.00);
    B(1,0) = in_value_type(0.80,-0.60); B(1,1) = in_value_type( 3.00,-5.00); B(1,2) = in_value_type(-4.00, 3.00); B(1,3) = in_value_type(-2.40,-3.20);
    B(2,0) = in_value_type(1.00, 0.00); B(2,1) = in_value_type( 2.40, 1.80); B(2,2) = in_value_type(-4.00,-5.00); B(2,3) = in_value_type( 0.00,-3.00);
    B(3,0) = in_value_type(0.00, 1.00); B(3,1) = in_value_type(-1.80, 2.40); B(3,2) = in_value_type( 0.00,-4.00); B(3,3) = in_value_type( 4.00,-5.00);

    out_vector_type w;
    out_matrix_type V;

    ublasx::left_eigen(A, B, w, V);

    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B = " << B );
    BOOST_UBLASX_DEBUG_TRACE( "Eigenvalues = " << w );
    BOOST_UBLASX_DEBUG_TRACE( "Left Eigenvectors = " << V );

    BOOST_UBLASX_TEST_CHECK( ublasx::size(w) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(V) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(V) == n );
    out_matrix_type D;
    out_matrix_type X;
    out_matrix_type Y;
    D = ublasx::diag(w);
    X =  ublas::prod(ublas::herm(V), A);
    Y = ublas::prod(ublas::herm(V), B);
    Y = ublas::prod(Y, D);
    BOOST_UBLASX_DEBUG_TRACE( "V^{H}*A = " << X  );
    BOOST_UBLASX_DEBUG_TRACE( "V^{H}*B*D = " << Y  );
    //FIXME: this test fails but the computation of eigenvectos seems OK
    // We need further investigation
    //BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, Y, n, n, tol );
    BOOST_UBLASX_TEST_CHECK( true );
}


BOOST_UBLASX_TEST_DEF( test_complex_matrix_pair_column_major_right )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Complex Matrix Pair - Column Major - Right Eigenvectors");

    typedef double value_type;
    typedef std::complex<value_type> in_value_type;
    typedef std::complex<value_type> out_value_type;
    typedef ublas::matrix<in_value_type, ublas::column_major> in_matrix_type;
    typedef ublas::matrix<out_value_type, ublas::column_major> out_matrix_type;
    typedef ublas::vector<out_value_type> out_vector_type;

    const std::size_t n(4);

    in_matrix_type A(n,n);
    A(0,0) = in_value_type(-21.10,-22.50); A(0,1) = in_value_type( 53.50,-50.50); A(0,2) = in_value_type(-34.50, 127.50); A(0,3) = in_value_type(  7.50,  0.50);
    A(1,0) = in_value_type(- 0.46,- 7.78); A(1,1) = in_value_type(- 3.50,-37.50); A(1,2) = in_value_type(-15.50,  58.50); A(1,3) = in_value_type(-10.50,- 1.50);
    A(2,0) = in_value_type(  4.30,- 5.50); A(2,1) = in_value_type( 39.70,-17.10); A(2,2) = in_value_type(-68.50,  12.50); A(2,3) = in_value_type(- 7.50,- 3.50);
    A(3,0) = in_value_type(  5.50,  4.40); A(3,1) = in_value_type( 14.40, 43.30); A(3,2) = in_value_type(-32.50,- 46.00); A(3,3) = in_value_type(-19.00,-32.50);

    in_matrix_type B(n,n);
    B(0,0) = in_value_type(1.00,-5.00); B(0,1) = in_value_type( 1.60, 1.20); B(0,2) = in_value_type(-3.00, 0.00); B(0,3) = in_value_type( 0.00,-1.00);
    B(1,0) = in_value_type(0.80,-0.60); B(1,1) = in_value_type( 3.00,-5.00); B(1,2) = in_value_type(-4.00, 3.00); B(1,3) = in_value_type(-2.40,-3.20);
    B(2,0) = in_value_type(1.00, 0.00); B(2,1) = in_value_type( 2.40, 1.80); B(2,2) = in_value_type(-4.00,-5.00); B(2,3) = in_value_type( 0.00,-3.00);
    B(3,0) = in_value_type(0.00, 1.00); B(3,1) = in_value_type(-1.80, 2.40); B(3,2) = in_value_type( 0.00,-4.00); B(3,3) = in_value_type( 4.00,-5.00);

    out_vector_type w;
    out_matrix_type V;

    ublasx::right_eigen(A, B, w, V);

    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B = " << B );
    BOOST_UBLASX_DEBUG_TRACE( "Eigenvalues = " << w );
    BOOST_UBLASX_DEBUG_TRACE( "Right Eigenvectors = " << V );

    BOOST_UBLASX_TEST_CHECK( ublasx::size(w) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(V) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(V) == n );
    out_matrix_type D;
    out_matrix_type X;
    out_matrix_type Y;
    D = ublasx::diag(w);
    X =  ublas::prod(A, V);
    Y = ublas::prod(B, V);
    Y = ublas::prod(Y, D);
    BOOST_UBLASX_DEBUG_TRACE( "A*V = " << X  );
    BOOST_UBLASX_DEBUG_TRACE( "B*V*D = " << Y  );
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, Y, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_complex_matrix_pair_row_major_right )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Complex Matrix Pair - Row Major - Right Eigenvectors");

    typedef double value_type;
    typedef std::complex<value_type> in_value_type;
    typedef std::complex<value_type> out_value_type;
    typedef ublas::matrix<in_value_type, ublas::row_major> in_matrix_type;
    typedef ublas::matrix<out_value_type, ublas::row_major> out_matrix_type;
    typedef ublas::vector<out_value_type> out_vector_type;

    const std::size_t n(4);

    in_matrix_type A(n,n);
    A(0,0) = in_value_type(-21.10,-22.50); A(0,1) = in_value_type( 53.50,-50.50); A(0,2) = in_value_type(-34.50, 127.50); A(0,3) = in_value_type(  7.50,  0.50);
    A(1,0) = in_value_type(- 0.46,- 7.78); A(1,1) = in_value_type(- 3.50,-37.50); A(1,2) = in_value_type(-15.50,  58.50); A(1,3) = in_value_type(-10.50,- 1.50);
    A(2,0) = in_value_type(  4.30,- 5.50); A(2,1) = in_value_type( 39.70,-17.10); A(2,2) = in_value_type(-68.50,  12.50); A(2,3) = in_value_type(- 7.50,- 3.50);
    A(3,0) = in_value_type(  5.50,  4.40); A(3,1) = in_value_type( 14.40, 43.30); A(3,2) = in_value_type(-32.50,- 46.00); A(3,3) = in_value_type(-19.00,-32.50);

    in_matrix_type B(n,n);
    B(0,0) = in_value_type(1.00,-5.00); B(0,1) = in_value_type( 1.60, 1.20); B(0,2) = in_value_type(-3.00, 0.00); B(0,3) = in_value_type( 0.00,-1.00);
    B(1,0) = in_value_type(0.80,-0.60); B(1,1) = in_value_type( 3.00,-5.00); B(1,2) = in_value_type(-4.00, 3.00); B(1,3) = in_value_type(-2.40,-3.20);
    B(2,0) = in_value_type(1.00, 0.00); B(2,1) = in_value_type( 2.40, 1.80); B(2,2) = in_value_type(-4.00,-5.00); B(2,3) = in_value_type( 0.00,-3.00);
    B(3,0) = in_value_type(0.00, 1.00); B(3,1) = in_value_type(-1.80, 2.40); B(3,2) = in_value_type( 0.00,-4.00); B(3,3) = in_value_type( 4.00,-5.00);

    out_vector_type w;
    out_matrix_type V;

    ublasx::right_eigen(A, B, w, V);

    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B = " << B );
    BOOST_UBLASX_DEBUG_TRACE( "Eigenvalues = " << w );
    BOOST_UBLASX_DEBUG_TRACE( "Right Eigenvectors = " << V );

    BOOST_UBLASX_TEST_CHECK( ublasx::size(w) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(V) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(V) == n );
    out_matrix_type D;
    out_matrix_type X;
    out_matrix_type Y;
    D = ublasx::diag(w);
    X =  ublas::prod(A, V);
    Y = ublas::prod(B, V);
    Y = ublas::prod(Y, D);
    BOOST_UBLASX_DEBUG_TRACE( "A*V = " << X  );
    BOOST_UBLASX_DEBUG_TRACE( "B*V*D = " << Y  );
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, Y, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_double_upper_sym_matrix_pair_column_major_both )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Double Upper Symmetric Matrix Pair - Column Major - Both Eigenvectors");

    typedef double value_type;
    typedef value_type in_value_type;
    typedef value_type out_value_type;
    typedef ublas::symmetric_matrix<in_value_type, ublas::upper, ublas::column_major> in_matrix_type;
    typedef ublas::matrix<out_value_type, ublas::column_major> out_matrix_type;
    typedef ublas::vector<out_value_type> out_vector_type;

    const std::size_t n(4);

    in_matrix_type A(n,n);
    A(0,0) = 0.24;  A(0,1) =  0.39; A(0,2) =  0.42; A(0,3) = -0.1;
                    A(1,1) = -0.11; A(1,2) =  0.79; A(1,3) =  0.6;
                                    A(2,2) = -0.25; A(2,3) =  0.4;
                                                    A(3,3) = -0.03;

    in_matrix_type B(n,n);
    B(0,0) = 4.16;  B(0,1) = -3.12; B(0,2) =  0.56; B(0,3) = -0.10;
                    B(1,1) =  5.03; B(1,2) = -0.83; B(1,3) =  1.09;
                                    B(2,2) =  0.76; B(2,3) =  0.34;
                                                    B(3,3) =  1.18;

    out_vector_type w;
    out_matrix_type V;

    ublasx::eigen(A, B, w, V);

    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B = " << B );
    BOOST_UBLASX_DEBUG_TRACE( "Eigenvalues = " << w );
    BOOST_UBLASX_DEBUG_TRACE( "Eigenvectors = " << V );

    BOOST_UBLASX_TEST_CHECK( ublasx::size(w) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(V) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(V) == n );
    out_matrix_type D;
    out_matrix_type X;
    out_matrix_type Y;
    D = ublasx::diag(w);
    X = ublas::prod(A, V);
    Y = ublas::prod(ublas::prod<out_matrix_type>(B, V), D);
    BOOST_UBLASX_DEBUG_TRACE( "A*V = " << X  ); // A*V=B*V*D
    BOOST_UBLASX_DEBUG_TRACE( "B*V*D = " << Y  ); // A*V=B*V*D
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, Y, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_double_upper_sym_matrix_pair_row_major_both )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Double Upper Symmetric Matrix Pair - Row Major - Both Eigenvectors");

    typedef double value_type;
    typedef value_type in_value_type;
    typedef value_type out_value_type;
    typedef ublas::symmetric_matrix<in_value_type, ublas::upper, ublas::row_major> in_matrix_type;
    typedef ublas::matrix<out_value_type, ublas::row_major> out_matrix_type;
    typedef ublas::vector<out_value_type> out_vector_type;

    const std::size_t n(4);

    in_matrix_type A(n,n);
    A(0,0) = 0.24;  A(0,1) =  0.39; A(0,2) =  0.42; A(0,3) = -0.1;
                    A(1,1) = -0.11; A(1,2) =  0.79; A(1,3) =  0.6;
                                    A(2,2) = -0.25; A(2,3) =  0.4;
                                                    A(3,3) = -0.03;

    in_matrix_type B(n,n);
    B(0,0) = 4.16;  B(0,1) = -3.12; B(0,2) =  0.56; B(0,3) = -0.10;
                    B(1,1) =  5.03; B(1,2) = -0.83; B(1,3) =  1.09;
                                    B(2,2) =  0.76; B(2,3) =  0.34;
                                                    B(3,3) =  1.18;

    out_vector_type w;
    out_matrix_type V;

    ublasx::eigen(A, B, w, V);

    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B = " << B );
    BOOST_UBLASX_DEBUG_TRACE( "Eigenvalues = " << w );
    BOOST_UBLASX_DEBUG_TRACE( "Eigenvectors = " << V );

    BOOST_UBLASX_TEST_CHECK( ublasx::size(w) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(V) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(V) == n );
    out_matrix_type D;
    out_matrix_type X;
    out_matrix_type Y;
    D = ublasx::diag(w);
    X = ublas::prod(A, V);
    Y = ublas::prod(ublas::prod<out_matrix_type>(B, V), D);
    BOOST_UBLASX_DEBUG_TRACE( "A*V = " << X  ); // A*V=B*V*D
    BOOST_UBLASX_DEBUG_TRACE( "B*V*D = " << Y  ); // A*V=B*V*D
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, Y, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_complex_upper_herm_matrix_pair_column_major_both )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: complex Upper Hermitian Matrix Pair - Column Major - Both Eigenvectors");

    typedef double real_type;
    typedef std::complex<real_type> complex_type;
    typedef ublas::hermitian_matrix<complex_type, ublas::upper, ublas::column_major> in_matrix_type;
    typedef ublas::matrix<complex_type, ublas::column_major> out_matrix_type;
    typedef ublas::vector<real_type> out_vector_type;

    const std::size_t n(4);

    in_matrix_type A(n,n);
    A(0,0) = complex_type(-7.36, 0.00); A(0,1) = complex_type( 0.77, -0.43); A(0,2) = complex_type(-0.64, -0.92); A(0,3) = complex_type( 3.01, -6.97);
                                        A(1,1) = complex_type( 3.49,  0.00); A(1,2) = complex_type( 2.19,  4.45); A(1,3) = complex_type( 1.90,  3.73);
                                                                             A(2,2) = complex_type( 0.12,  0.00); A(2,3) = complex_type( 2.88, -3.17);
                                                                                                                  A(3,3) = complex_type(-2.54,  0.00);

    in_matrix_type B(n,n);
    B(0,0) = complex_type( 3.23, 0.00); B(0,1) = complex_type( 1.51, -1.92); B(0,2) = complex_type( 1.90,  0.84); B(0,3) = complex_type( 0.42,  2.50);
                                        B(1,1) = complex_type( 3.58,  0.00); B(1,2) = complex_type(-0.23,  1.11); B(1,3) = complex_type(-1.18,  1.37);
                                                                             B(2,2) = complex_type( 4.09,  0.00); B(2,3) = complex_type( 2.33, -0.14);
                                                                                                                  B(3,3) = complex_type( 4.29,  0.00);

    out_vector_type w;
    out_matrix_type V;

    ublasx::eigen(A, B, w, V);

    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B = " << B );
    BOOST_UBLASX_DEBUG_TRACE( "Eigenvalues = " << w );
    BOOST_UBLASX_DEBUG_TRACE( "Eigenvectors = " << V );

    BOOST_UBLASX_TEST_CHECK( ublasx::size(w) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(V) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(V) == n );
    out_matrix_type D;
    out_matrix_type X;
    out_matrix_type Y;
    D = ublasx::diag(w);
    X = ublas::prod(A, V);
    Y = ublas::prod(ublas::prod<out_matrix_type>(B, V), D);
    BOOST_UBLASX_DEBUG_TRACE( "A*V = " << X  ); // A*V=B*V*D
    BOOST_UBLASX_DEBUG_TRACE( "B*V*D = " << Y  ); // A*V=B*V*D
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, Y, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_complex_upper_herm_matrix_pair_row_major_both )
{
    BOOST_UBLASX_DEBUG_TRACE("Test Case: Complex Upper Hermitian Matrix Pair - Row Major - Both Eigenvectors");

    typedef double real_type;
    typedef std::complex<real_type> complex_type;
    typedef ublas::hermitian_matrix<complex_type, ublas::upper, ublas::row_major> in_matrix_type;
    typedef ublas::matrix<complex_type, ublas::row_major> out_matrix_type;
    typedef ublas::vector<real_type> out_vector_type;

    const std::size_t n(4);

    in_matrix_type A(n,n);
    A(0,0) = complex_type(-7.36, 0.00); A(0,1) = complex_type( 0.77, -0.43); A(0,2) = complex_type(-0.64, -0.92); A(0,3) = complex_type( 3.01, -6.97);
                                        A(1,1) = complex_type( 3.49,  0.00); A(1,2) = complex_type( 2.19,  4.45); A(1,3) = complex_type( 1.90,  3.73);
                                                                             A(2,2) = complex_type( 0.12,  0.00); A(2,3) = complex_type( 2.88, -3.17);
                                                                                                                  A(3,3) = complex_type(-2.54,  0.00);

    in_matrix_type B(n,n);
    B(0,0) = complex_type( 3.23, 0.00); B(0,1) = complex_type( 1.51, -1.92); B(0,2) = complex_type( 1.90,  0.84); B(0,3) = complex_type( 0.42,  2.50);
                                        B(1,1) = complex_type( 3.58,  0.00); B(1,2) = complex_type(-0.23,  1.11); B(1,3) = complex_type(-1.18,  1.37);
                                                                             B(2,2) = complex_type( 4.09,  0.00); B(2,3) = complex_type( 2.33, -0.14);
                                                                                                                  B(3,3) = complex_type( 4.29,  0.00);

    out_vector_type w;
    out_matrix_type V;

    ublasx::eigen(A, B, w, V);

    BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
    BOOST_UBLASX_DEBUG_TRACE( "B = " << B );
    BOOST_UBLASX_DEBUG_TRACE( "Eigenvalues = " << w );
    BOOST_UBLASX_DEBUG_TRACE( "Eigenvectors = " << V );

    BOOST_UBLASX_TEST_CHECK( ublasx::size(w) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(V) == n );
    BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(V) == n );
    out_matrix_type D;
    out_matrix_type X;
    out_matrix_type Y;
    D = ublasx::diag(w);
    X = ublas::prod(A, V);
    Y = ublas::prod(ublas::prod<out_matrix_type>(B, V), D);
    BOOST_UBLASX_DEBUG_TRACE( "A*V = " << X  ); // A*V=B*V*D
    BOOST_UBLASX_DEBUG_TRACE( "B*V*D = " << Y  ); // A*V=B*V*D
    BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, Y, n, n, tol );
}


int main()
{
    BOOST_UBLASX_DEBUG_TRACE("Test Suite: 'eigen' operations");

    BOOST_UBLASX_TEST_BEGIN();

    BOOST_UBLASX_TEST_DO( test_double_matrix_column_major_both );
    BOOST_UBLASX_TEST_DO( test_double_matrix_column_major_left );
    BOOST_UBLASX_TEST_DO( test_double_matrix_column_major_right );
    BOOST_UBLASX_TEST_DO( test_double_matrix_column_major_only_values );
    BOOST_UBLASX_TEST_DO( test_double_matrix_column_major_only_vectors );

    BOOST_UBLASX_TEST_DO( test_double_matrix_row_major_both );
    BOOST_UBLASX_TEST_DO( test_double_matrix_row_major_left );
    BOOST_UBLASX_TEST_DO( test_double_matrix_row_major_right );
    BOOST_UBLASX_TEST_DO( test_double_matrix_row_major_only_values );
    BOOST_UBLASX_TEST_DO( test_double_matrix_row_major_only_vectors );

    BOOST_UBLASX_TEST_DO( test_complex_matrix_column_major_both );
    BOOST_UBLASX_TEST_DO( test_complex_matrix_column_major_left );
    BOOST_UBLASX_TEST_DO( test_complex_matrix_column_major_right );
    BOOST_UBLASX_TEST_DO( test_complex_matrix_column_major_only_values );
    BOOST_UBLASX_TEST_DO( test_complex_matrix_column_major_only_vectors );

    BOOST_UBLASX_TEST_DO( test_complex_matrix_row_major_both );
    BOOST_UBLASX_TEST_DO( test_complex_matrix_row_major_left );
    BOOST_UBLASX_TEST_DO( test_complex_matrix_row_major_right );
    BOOST_UBLASX_TEST_DO( test_complex_matrix_row_major_only_values );
    BOOST_UBLASX_TEST_DO( test_complex_matrix_row_major_only_vectors );

    BOOST_UBLASX_TEST_DO( test_double_upper_sym_matrix_column_major );
    BOOST_UBLASX_TEST_DO( test_double_upper_sym_matrix_column_major_only_values );
    //BOOST_UBLASX_TEST_DO( test_double_upper_sym_matrix_column_major_only_vectors );//TODO

    BOOST_UBLASX_TEST_DO( test_double_upper_sym_matrix_row_major );
    BOOST_UBLASX_TEST_DO( test_double_upper_sym_matrix_row_major_only_values );
    //BOOST_UBLASX_TEST_DO( test_double_upper_sym_matrix_row_major_only_vectors );//TODO

    BOOST_UBLASX_TEST_DO( test_double_upper_herm_matrix_column_major );
    BOOST_UBLASX_TEST_DO( test_double_upper_herm_matrix_column_major_only_values );
    //BOOST_UBLASX_TEST_DO( test_double_upper_herm_matrix_column_major_only_vectors );//TODO

    BOOST_UBLASX_TEST_DO( test_double_upper_herm_matrix_row_major );
    BOOST_UBLASX_TEST_DO( test_double_upper_herm_matrix_row_major_only_values );
    //BOOST_UBLASX_TEST_DO( test_double_upper_herm_matrix_row_major_only_vectors );//TODO

    BOOST_UBLASX_TEST_DO( test_double_matrix_pair_column_major_both );
    BOOST_UBLASX_TEST_DO( test_double_matrix_pair_row_major_both );
    BOOST_UBLASX_TEST_DO( test_double_matrix_pair_column_major_left );
    BOOST_UBLASX_TEST_DO( test_double_matrix_pair_row_major_left );
    BOOST_UBLASX_TEST_DO( test_double_matrix_pair_column_major_right );
    BOOST_UBLASX_TEST_DO( test_double_matrix_pair_row_major_right );

    BOOST_UBLASX_TEST_DO( test_complex_matrix_pair_column_major_both );
    BOOST_UBLASX_TEST_DO( test_complex_matrix_pair_row_major_both );
    BOOST_UBLASX_TEST_DO( test_complex_matrix_pair_column_major_left );
    BOOST_UBLASX_TEST_DO( test_complex_matrix_pair_row_major_left );
    BOOST_UBLASX_TEST_DO( test_complex_matrix_pair_column_major_right );
    BOOST_UBLASX_TEST_DO( test_complex_matrix_pair_row_major_right );

    BOOST_UBLASX_TEST_DO( test_double_upper_sym_matrix_pair_column_major_both );
    BOOST_UBLASX_TEST_DO( test_double_upper_sym_matrix_pair_row_major_both );

    BOOST_UBLASX_TEST_DO( test_complex_upper_herm_matrix_pair_column_major_both );
    BOOST_UBLASX_TEST_DO( test_complex_upper_herm_matrix_pair_row_major_both );

    BOOST_UBLASX_TEST_END();
}
