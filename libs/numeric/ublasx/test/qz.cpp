/**
 * \file libs/numeric/ublasx/test/qz.cpp
 *
 * \brief Test suite for the QZ decomposition.
 *
 * Copyright (c) 2010, Marco Guazzone
 *
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompwhiching file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */

#include <boost/numeric/ublas/expression_types.hpp>
#include <boost/numeric/ublas/fwd.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_expression.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublasx/operation/diag.hpp>
#include <boost/numeric/ublasx/operation/num_columns.hpp>
#include <boost/numeric/ublasx/operation/num_rows.hpp>
#include <boost/numeric/ublasx/operation/qz.hpp>
#include <boost/numeric/ublasx/operation/size.hpp>
#include "libs/numeric/ublasx/test/utils.hpp"


static const double tol = 1.0e-5;


namespace ublas = boost::numeric::ublas;
namespace ublasx = boost::numeric::ublasx;


BOOST_UBLASX_TEST_DEF( test_real_column_major_decomp )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Real Matrix - Column Major - Decomposition");

	typedef double value_type;
	typedef ublas::matrix<value_type, ublas::column_major> matrix_type;

	const std::size_t n(5);

	matrix_type A(n,n);
	A(0,0) = -0.180557; A(0,1) =  0.322289; A(0,2) = -0.651789; A(0,3) =  0.793637; A(0,4) = -0.141086;
	A(1,0) =  0.729781; A(1,1) =  1.665989; A(1,2) =  0.620091; A(1,3) = -1.541503; A(1,4) =  0.146673;
	A(2,0) = -0.594370; A(2,1) =  0.494804; A(2,2) =  1.004784; A(2,3) = -0.221373; A(2,4) = -2.196082;
	A(3,0) = -1.106269; A(3,1) =  0.026697; A(3,2) =  2.687083; A(3,3) =  0.763162; A(3,4) =  1.203514;
	A(4,0) = -0.021184; A(4,1) = -0.882220; A(4,2) = -1.618234; A(4,3) =  1.119524; A(4,4) =  2.588165;

	matrix_type B(n,n);
	B(0,0) = -1.592710; B(0,1) =  0.057283; B(0,2) = -1.862275; B(0,3) =  0.712471; B(0,4) =  0.463207;
	B(1,0) =  1.072859; B(1,1) = -1.384371; B(1,2) =  0.777754; B(1,3) =  1.914787; B(1,4) =  0.082774;
	B(2,0) = -0.451744; B(2,1) = -0.131528; B(2,2) = -0.636187; B(2,3) =  0.984480; B(2,4) =  0.011728;
	B(3,0) = -0.876629; B(3,1) = -0.083787; B(3,2) =  0.474227; B(3,3) = -0.042328; B(3,4) = -0.529845;
	B(4,0) = -0.812610; B(4,1) =  0.142456; B(4,2) =  0.033739; B(4,3) = -2.000422; B(4,4) = -0.765401;


	matrix_type S;
	matrix_type T;
	matrix_type Q;
	matrix_type Z;

	ublasx::qz_decompose(A, B, S, T, Q, Z);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "B = " << B );
	BOOST_UBLASX_DEBUG_TRACE( "S = " << S );
	BOOST_UBLASX_DEBUG_TRACE( "T = " << T );
	BOOST_UBLASX_DEBUG_TRACE( "Q = " << Q );
	BOOST_UBLASX_DEBUG_TRACE( "Z = " << Z );
	matrix_type X;
	X = ublas::prod(Q, S);
	X = ublas::prod(X, ublas::trans(Z));
	BOOST_UBLASX_DEBUG_TRACE( "Q*S*Z' = " << X );
	X = ublas::prod(Q, T);
	X = ublas::prod(X, ublas::trans(Z));
	BOOST_UBLASX_DEBUG_TRACE( "Q*T*Z' = " << X );

	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(S) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(S) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(T) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(T) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(Q) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(Q) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(Z) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(Z) == n );
	X = ublas::prod(Q, S);
	X = ublas::prod(X, ublas::trans(Z));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, A, n, n, tol );
	X = ublas::prod(Q, T);
	X = ublas::prod(X, ublas::trans(Z));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, B, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_real_row_major_decomp )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Real Matrix - Row Major - Decomposition");

	typedef double value_type;
	typedef ublas::matrix<value_type, ublas::row_major> matrix_type;

	const std::size_t n(5);

	matrix_type A(n,n);
	A(0,0) = -0.180557; A(0,1) =  0.322289; A(0,2) = -0.651789; A(0,3) =  0.793637; A(0,4) = -0.141086;
	A(1,0) =  0.729781; A(1,1) =  1.665989; A(1,2) =  0.620091; A(1,3) = -1.541503; A(1,4) =  0.146673;
	A(2,0) = -0.594370; A(2,1) =  0.494804; A(2,2) =  1.004784; A(2,3) = -0.221373; A(2,4) = -2.196082;
	A(3,0) = -1.106269; A(3,1) =  0.026697; A(3,2) =  2.687083; A(3,3) =  0.763162; A(3,4) =  1.203514;
	A(4,0) = -0.021184; A(4,1) = -0.882220; A(4,2) = -1.618234; A(4,3) =  1.119524; A(4,4) =  2.588165;

	matrix_type B(n,n);
	B(0,0) = -1.592710; B(0,1) =  0.057283; B(0,2) = -1.862275; B(0,3) =  0.712471; B(0,4) =  0.463207;
	B(1,0) =  1.072859; B(1,1) = -1.384371; B(1,2) =  0.777754; B(1,3) =  1.914787; B(1,4) =  0.082774;
	B(2,0) = -0.451744; B(2,1) = -0.131528; B(2,2) = -0.636187; B(2,3) =  0.984480; B(2,4) =  0.011728;
	B(3,0) = -0.876629; B(3,1) = -0.083787; B(3,2) =  0.474227; B(3,3) = -0.042328; B(3,4) = -0.529845;
	B(4,0) = -0.812610; B(4,1) =  0.142456; B(4,2) =  0.033739; B(4,3) = -2.000422; B(4,4) = -0.765401;


	matrix_type S;
	matrix_type T;
	matrix_type Q;
	matrix_type Z;

	ublasx::qz_decompose(A, B, S, T, Q, Z);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "B = " << B );
	BOOST_UBLASX_DEBUG_TRACE( "S = " << S );
	BOOST_UBLASX_DEBUG_TRACE( "T = " << T );
	BOOST_UBLASX_DEBUG_TRACE( "Q = " << Q );
	BOOST_UBLASX_DEBUG_TRACE( "Z = " << Z );
	matrix_type X;
	X = ublas::prod(Q, S);
	X = ublas::prod(X, ublas::trans(Z));
	BOOST_UBLASX_DEBUG_TRACE( "Q*S*Z' = " << X );
	X = ublas::prod(Q, T);
	X = ublas::prod(X, ublas::trans(Z));
	BOOST_UBLASX_DEBUG_TRACE( "Q*T*Z' = " << X );

	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(S) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(S) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(T) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(T) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(Q) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(Q) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(Z) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(Z) == n );
	X = ublas::prod(Q, S);
	X = ublas::prod(X, ublas::trans(Z));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, A, n, n, tol );
	X = ublas::prod(Q, T);
	X = ublas::prod(X, ublas::trans(Z));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, B, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_real_column_major_decomp_lhp_reorder )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Real Matrix - Column Major - Decomposition and LHP Reordering");

	typedef double value_type;
	typedef ublas::matrix<value_type, ublas::column_major> matrix_type;

	const std::size_t n(5);

	matrix_type A(n,n);
	A(0,0) = -0.180557; A(0,1) =  0.322289; A(0,2) = -0.651789; A(0,3) =  0.793637; A(0,4) = -0.141086;
	A(1,0) =  0.729781; A(1,1) =  1.665989; A(1,2) =  0.620091; A(1,3) = -1.541503; A(1,4) =  0.146673;
	A(2,0) = -0.594370; A(2,1) =  0.494804; A(2,2) =  1.004784; A(2,3) = -0.221373; A(2,4) = -2.196082;
	A(3,0) = -1.106269; A(3,1) =  0.026697; A(3,2) =  2.687083; A(3,3) =  0.763162; A(3,4) =  1.203514;
	A(4,0) = -0.021184; A(4,1) = -0.882220; A(4,2) = -1.618234; A(4,3) =  1.119524; A(4,4) =  2.588165;

	matrix_type B(n,n);
	B(0,0) = -1.592710; B(0,1) =  0.057283; B(0,2) = -1.862275; B(0,3) =  0.712471; B(0,4) =  0.463207;
	B(1,0) =  1.072859; B(1,1) = -1.384371; B(1,2) =  0.777754; B(1,3) =  1.914787; B(1,4) =  0.082774;
	B(2,0) = -0.451744; B(2,1) = -0.131528; B(2,2) = -0.636187; B(2,3) =  0.984480; B(2,4) =  0.011728;
	B(3,0) = -0.876629; B(3,1) = -0.083787; B(3,2) =  0.474227; B(3,3) = -0.042328; B(3,4) = -0.529845;
	B(4,0) = -0.812610; B(4,1) =  0.142456; B(4,2) =  0.033739; B(4,3) = -2.000422; B(4,4) = -0.765401;


	matrix_type S;
	matrix_type T;
	matrix_type Q;
	matrix_type Z;

	ublasx::qz_decompose(A, B, S, T, Q, Z, ublasx::lhp_qz_eigenvalues);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "B = " << B );
	BOOST_UBLASX_DEBUG_TRACE( "S = " << S );
	BOOST_UBLASX_DEBUG_TRACE( "T = " << T );
	BOOST_UBLASX_DEBUG_TRACE( "Q = " << Q );
	BOOST_UBLASX_DEBUG_TRACE( "Z = " << Z );
	matrix_type X;
	X = ublas::prod(Q, S);
	X = ublas::prod(X, ublas::trans(Z));
	BOOST_UBLASX_DEBUG_TRACE( "Q*S*Z' = " << X );
	X = ublas::prod(Q, T);
	X = ublas::prod(X, ublas::trans(Z));
	BOOST_UBLASX_DEBUG_TRACE( "Q*T*Z' = " << X );

	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(S) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(S) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(T) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(T) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(Q) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(Q) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(Z) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(Z) == n );
	X = ublas::prod(Q, S);
	X = ublas::prod(X, ublas::trans(Z));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, A, n, n, tol );
	X = ublas::prod(Q, T);
	X = ublas::prod(X, ublas::trans(Z));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, B, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_real_row_major_decomp_lhp_reorder )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Real Matrix - Row Major - Decomposition and LHP Reordering");

	typedef double value_type;
	typedef ublas::matrix<value_type, ublas::row_major> matrix_type;

	const std::size_t n(5);

	matrix_type A(n,n);
	A(0,0) = -0.180557; A(0,1) =  0.322289; A(0,2) = -0.651789; A(0,3) =  0.793637; A(0,4) = -0.141086;
	A(1,0) =  0.729781; A(1,1) =  1.665989; A(1,2) =  0.620091; A(1,3) = -1.541503; A(1,4) =  0.146673;
	A(2,0) = -0.594370; A(2,1) =  0.494804; A(2,2) =  1.004784; A(2,3) = -0.221373; A(2,4) = -2.196082;
	A(3,0) = -1.106269; A(3,1) =  0.026697; A(3,2) =  2.687083; A(3,3) =  0.763162; A(3,4) =  1.203514;
	A(4,0) = -0.021184; A(4,1) = -0.882220; A(4,2) = -1.618234; A(4,3) =  1.119524; A(4,4) =  2.588165;

	matrix_type B(n,n);
	B(0,0) = -1.592710; B(0,1) =  0.057283; B(0,2) = -1.862275; B(0,3) =  0.712471; B(0,4) =  0.463207;
	B(1,0) =  1.072859; B(1,1) = -1.384371; B(1,2) =  0.777754; B(1,3) =  1.914787; B(1,4) =  0.082774;
	B(2,0) = -0.451744; B(2,1) = -0.131528; B(2,2) = -0.636187; B(2,3) =  0.984480; B(2,4) =  0.011728;
	B(3,0) = -0.876629; B(3,1) = -0.083787; B(3,2) =  0.474227; B(3,3) = -0.042328; B(3,4) = -0.529845;
	B(4,0) = -0.812610; B(4,1) =  0.142456; B(4,2) =  0.033739; B(4,3) = -2.000422; B(4,4) = -0.765401;


	matrix_type S;
	matrix_type T;
	matrix_type Q;
	matrix_type Z;

	ublasx::qz_decompose(A, B, S, T, Q, Z, ublasx::lhp_qz_eigenvalues);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "B = " << B );
	BOOST_UBLASX_DEBUG_TRACE( "S = " << S );
	BOOST_UBLASX_DEBUG_TRACE( "T = " << T );
	BOOST_UBLASX_DEBUG_TRACE( "Q = " << Q );
	BOOST_UBLASX_DEBUG_TRACE( "Z = " << Z );
	matrix_type X;
	X = ublas::prod(Q, S);
	X = ublas::prod(X, ublas::trans(Z));
	BOOST_UBLASX_DEBUG_TRACE( "Q*S*Z' = " << X );
	X = ublas::prod(Q, T);
	X = ublas::prod(X, ublas::trans(Z));
	BOOST_UBLASX_DEBUG_TRACE( "Q*T*Z' = " << X );

	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(S) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(S) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(T) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(T) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(Q) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(Q) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(Z) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(Z) == n );
	X = ublas::prod(Q, S);
	X = ublas::prod(X, ublas::trans(Z));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, A, n, n, tol );
	X = ublas::prod(Q, T);
	X = ublas::prod(X, ublas::trans(Z));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, B, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_real_column_major_decomp_rhp_reorder )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Real Matrix - Column Major - Decomposition and RHP Reordering");

	typedef double value_type;
	typedef ublas::matrix<value_type, ublas::column_major> matrix_type;

	const std::size_t n(5);

	matrix_type A(n,n);
	A(0,0) = -0.180557; A(0,1) =  0.322289; A(0,2) = -0.651789; A(0,3) =  0.793637; A(0,4) = -0.141086;
	A(1,0) =  0.729781; A(1,1) =  1.665989; A(1,2) =  0.620091; A(1,3) = -1.541503; A(1,4) =  0.146673;
	A(2,0) = -0.594370; A(2,1) =  0.494804; A(2,2) =  1.004784; A(2,3) = -0.221373; A(2,4) = -2.196082;
	A(3,0) = -1.106269; A(3,1) =  0.026697; A(3,2) =  2.687083; A(3,3) =  0.763162; A(3,4) =  1.203514;
	A(4,0) = -0.021184; A(4,1) = -0.882220; A(4,2) = -1.618234; A(4,3) =  1.119524; A(4,4) =  2.588165;

	matrix_type B(n,n);
	B(0,0) = -1.592710; B(0,1) =  0.057283; B(0,2) = -1.862275; B(0,3) =  0.712471; B(0,4) =  0.463207;
	B(1,0) =  1.072859; B(1,1) = -1.384371; B(1,2) =  0.777754; B(1,3) =  1.914787; B(1,4) =  0.082774;
	B(2,0) = -0.451744; B(2,1) = -0.131528; B(2,2) = -0.636187; B(2,3) =  0.984480; B(2,4) =  0.011728;
	B(3,0) = -0.876629; B(3,1) = -0.083787; B(3,2) =  0.474227; B(3,3) = -0.042328; B(3,4) = -0.529845;
	B(4,0) = -0.812610; B(4,1) =  0.142456; B(4,2) =  0.033739; B(4,3) = -2.000422; B(4,4) = -0.765401;


	matrix_type S;
	matrix_type T;
	matrix_type Q;
	matrix_type Z;

	ublasx::qz_decompose(A, B, S, T, Q, Z, ublasx::rhp_qz_eigenvalues);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "B = " << B );
	BOOST_UBLASX_DEBUG_TRACE( "S = " << S );
	BOOST_UBLASX_DEBUG_TRACE( "T = " << T );
	BOOST_UBLASX_DEBUG_TRACE( "Q = " << Q );
	BOOST_UBLASX_DEBUG_TRACE( "Z = " << Z );
	matrix_type X;
	X = ublas::prod(Q, S);
	X = ublas::prod(X, ublas::trans(Z));
	BOOST_UBLASX_DEBUG_TRACE( "Q*S*Z' = " << X );
	X = ublas::prod(Q, T);
	X = ublas::prod(X, ublas::trans(Z));
	BOOST_UBLASX_DEBUG_TRACE( "Q*T*Z' = " << X );

	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(S) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(S) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(T) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(T) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(Q) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(Q) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(Z) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(Z) == n );
	X = ublas::prod(Q, S);
	X = ublas::prod(X, ublas::trans(Z));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, A, n, n, tol );
	X = ublas::prod(Q, T);
	X = ublas::prod(X, ublas::trans(Z));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, B, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_real_row_major_decomp_rhp_reorder )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Real Matrix - Row Major - Decomposition and RHP Reordering");

	typedef double value_type;
	typedef ublas::matrix<value_type, ublas::row_major> matrix_type;

	const std::size_t n(5);

	matrix_type A(n,n);
	A(0,0) = -0.180557; A(0,1) =  0.322289; A(0,2) = -0.651789; A(0,3) =  0.793637; A(0,4) = -0.141086;
	A(1,0) =  0.729781; A(1,1) =  1.665989; A(1,2) =  0.620091; A(1,3) = -1.541503; A(1,4) =  0.146673;
	A(2,0) = -0.594370; A(2,1) =  0.494804; A(2,2) =  1.004784; A(2,3) = -0.221373; A(2,4) = -2.196082;
	A(3,0) = -1.106269; A(3,1) =  0.026697; A(3,2) =  2.687083; A(3,3) =  0.763162; A(3,4) =  1.203514;
	A(4,0) = -0.021184; A(4,1) = -0.882220; A(4,2) = -1.618234; A(4,3) =  1.119524; A(4,4) =  2.588165;

	matrix_type B(n,n);
	B(0,0) = -1.592710; B(0,1) =  0.057283; B(0,2) = -1.862275; B(0,3) =  0.712471; B(0,4) =  0.463207;
	B(1,0) =  1.072859; B(1,1) = -1.384371; B(1,2) =  0.777754; B(1,3) =  1.914787; B(1,4) =  0.082774;
	B(2,0) = -0.451744; B(2,1) = -0.131528; B(2,2) = -0.636187; B(2,3) =  0.984480; B(2,4) =  0.011728;
	B(3,0) = -0.876629; B(3,1) = -0.083787; B(3,2) =  0.474227; B(3,3) = -0.042328; B(3,4) = -0.529845;
	B(4,0) = -0.812610; B(4,1) =  0.142456; B(4,2) =  0.033739; B(4,3) = -2.000422; B(4,4) = -0.765401;


	matrix_type S;
	matrix_type T;
	matrix_type Q;
	matrix_type Z;

	ublasx::qz_decompose(A, B, S, T, Q, Z, ublasx::rhp_qz_eigenvalues);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "B = " << B );
	BOOST_UBLASX_DEBUG_TRACE( "S = " << S );
	BOOST_UBLASX_DEBUG_TRACE( "T = " << T );
	BOOST_UBLASX_DEBUG_TRACE( "Q = " << Q );
	BOOST_UBLASX_DEBUG_TRACE( "Z = " << Z );
	matrix_type X;
	X = ublas::prod(Q, S);
	X = ublas::prod(X, ublas::trans(Z));
	BOOST_UBLASX_DEBUG_TRACE( "Q*S*Z' = " << X );
	X = ublas::prod(Q, T);
	X = ublas::prod(X, ublas::trans(Z));
	BOOST_UBLASX_DEBUG_TRACE( "Q*T*Z' = " << X );

	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(S) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(S) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(T) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(T) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(Q) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(Q) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(Z) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(Z) == n );
	X = ublas::prod(Q, S);
	X = ublas::prod(X, ublas::trans(Z));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, A, n, n, tol );
	X = ublas::prod(Q, T);
	X = ublas::prod(X, ublas::trans(Z));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, B, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_real_column_major_decomp_udi_reorder )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Real Matrix - Column Major - Decomposition and UDI Reordering");

	typedef double value_type;
	typedef ublas::matrix<value_type, ublas::column_major> matrix_type;

	const std::size_t n(5);

	matrix_type A(n,n);
	A(0,0) = -0.180557; A(0,1) =  0.322289; A(0,2) = -0.651789; A(0,3) =  0.793637; A(0,4) = -0.141086;
	A(1,0) =  0.729781; A(1,1) =  1.665989; A(1,2) =  0.620091; A(1,3) = -1.541503; A(1,4) =  0.146673;
	A(2,0) = -0.594370; A(2,1) =  0.494804; A(2,2) =  1.004784; A(2,3) = -0.221373; A(2,4) = -2.196082;
	A(3,0) = -1.106269; A(3,1) =  0.026697; A(3,2) =  2.687083; A(3,3) =  0.763162; A(3,4) =  1.203514;
	A(4,0) = -0.021184; A(4,1) = -0.882220; A(4,2) = -1.618234; A(4,3) =  1.119524; A(4,4) =  2.588165;

	matrix_type B(n,n);
	B(0,0) = -1.592710; B(0,1) =  0.057283; B(0,2) = -1.862275; B(0,3) =  0.712471; B(0,4) =  0.463207;
	B(1,0) =  1.072859; B(1,1) = -1.384371; B(1,2) =  0.777754; B(1,3) =  1.914787; B(1,4) =  0.082774;
	B(2,0) = -0.451744; B(2,1) = -0.131528; B(2,2) = -0.636187; B(2,3) =  0.984480; B(2,4) =  0.011728;
	B(3,0) = -0.876629; B(3,1) = -0.083787; B(3,2) =  0.474227; B(3,3) = -0.042328; B(3,4) = -0.529845;
	B(4,0) = -0.812610; B(4,1) =  0.142456; B(4,2) =  0.033739; B(4,3) = -2.000422; B(4,4) = -0.765401;


	matrix_type S;
	matrix_type T;
	matrix_type Q;
	matrix_type Z;

	ublasx::qz_decompose(A, B, S, T, Q, Z, ublasx::udi_qz_eigenvalues);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "B = " << B );
	BOOST_UBLASX_DEBUG_TRACE( "S = " << S );
	BOOST_UBLASX_DEBUG_TRACE( "T = " << T );
	BOOST_UBLASX_DEBUG_TRACE( "Q = " << Q );
	BOOST_UBLASX_DEBUG_TRACE( "Z = " << Z );
	matrix_type X;
	X = ublas::prod(Q, S);
	X = ublas::prod(X, ublas::trans(Z));
	BOOST_UBLASX_DEBUG_TRACE( "Q*S*Z' = " << X );
	X = ublas::prod(Q, T);
	X = ublas::prod(X, ublas::trans(Z));
	BOOST_UBLASX_DEBUG_TRACE( "Q*T*Z' = " << X );

	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(S) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(S) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(T) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(T) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(Q) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(Q) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(Z) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(Z) == n );
	X = ublas::prod(Q, S);
	X = ublas::prod(X, ublas::trans(Z));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, A, n, n, tol );
	X = ublas::prod(Q, T);
	X = ublas::prod(X, ublas::trans(Z));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, B, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_real_row_major_decomp_udi_reorder )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Real Matrix - Row Major - Decomposition and UDI Reordering");

	typedef double value_type;
	typedef ublas::matrix<value_type, ublas::row_major> matrix_type;

	const std::size_t n(5);

	matrix_type A(n,n);
	A(0,0) = -0.180557; A(0,1) =  0.322289; A(0,2) = -0.651789; A(0,3) =  0.793637; A(0,4) = -0.141086;
	A(1,0) =  0.729781; A(1,1) =  1.665989; A(1,2) =  0.620091; A(1,3) = -1.541503; A(1,4) =  0.146673;
	A(2,0) = -0.594370; A(2,1) =  0.494804; A(2,2) =  1.004784; A(2,3) = -0.221373; A(2,4) = -2.196082;
	A(3,0) = -1.106269; A(3,1) =  0.026697; A(3,2) =  2.687083; A(3,3) =  0.763162; A(3,4) =  1.203514;
	A(4,0) = -0.021184; A(4,1) = -0.882220; A(4,2) = -1.618234; A(4,3) =  1.119524; A(4,4) =  2.588165;

	matrix_type B(n,n);
	B(0,0) = -1.592710; B(0,1) =  0.057283; B(0,2) = -1.862275; B(0,3) =  0.712471; B(0,4) =  0.463207;
	B(1,0) =  1.072859; B(1,1) = -1.384371; B(1,2) =  0.777754; B(1,3) =  1.914787; B(1,4) =  0.082774;
	B(2,0) = -0.451744; B(2,1) = -0.131528; B(2,2) = -0.636187; B(2,3) =  0.984480; B(2,4) =  0.011728;
	B(3,0) = -0.876629; B(3,1) = -0.083787; B(3,2) =  0.474227; B(3,3) = -0.042328; B(3,4) = -0.529845;
	B(4,0) = -0.812610; B(4,1) =  0.142456; B(4,2) =  0.033739; B(4,3) = -2.000422; B(4,4) = -0.765401;


	matrix_type S;
	matrix_type T;
	matrix_type Q;
	matrix_type Z;

	ublasx::qz_decompose(A, B, S, T, Q, Z, ublasx::udi_qz_eigenvalues);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "B = " << B );
	BOOST_UBLASX_DEBUG_TRACE( "S = " << S );
	BOOST_UBLASX_DEBUG_TRACE( "T = " << T );
	BOOST_UBLASX_DEBUG_TRACE( "Q = " << Q );
	BOOST_UBLASX_DEBUG_TRACE( "Z = " << Z );
	matrix_type X;
	X = ublas::prod(Q, S);
	X = ublas::prod(X, ublas::trans(Z));
	BOOST_UBLASX_DEBUG_TRACE( "Q*S*Z' = " << X );
	X = ublas::prod(Q, T);
	X = ublas::prod(X, ublas::trans(Z));
	BOOST_UBLASX_DEBUG_TRACE( "Q*T*Z' = " << X );

	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(S) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(S) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(T) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(T) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(Q) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(Q) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(Z) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(Z) == n );
	X = ublas::prod(Q, S);
	X = ublas::prod(X, ublas::trans(Z));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, A, n, n, tol );
	X = ublas::prod(Q, T);
	X = ublas::prod(X, ublas::trans(Z));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, B, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_real_column_major_decomp_udo_reorder )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Real Matrix - Column Major - Decomposition and UDO Reordering");

	typedef double value_type;
	typedef ublas::matrix<value_type, ublas::column_major> matrix_type;

	const std::size_t n(5);

	matrix_type A(n,n);
	A(0,0) = -0.180557; A(0,1) =  0.322289; A(0,2) = -0.651789; A(0,3) =  0.793637; A(0,4) = -0.141086;
	A(1,0) =  0.729781; A(1,1) =  1.665989; A(1,2) =  0.620091; A(1,3) = -1.541503; A(1,4) =  0.146673;
	A(2,0) = -0.594370; A(2,1) =  0.494804; A(2,2) =  1.004784; A(2,3) = -0.221373; A(2,4) = -2.196082;
	A(3,0) = -1.106269; A(3,1) =  0.026697; A(3,2) =  2.687083; A(3,3) =  0.763162; A(3,4) =  1.203514;
	A(4,0) = -0.021184; A(4,1) = -0.882220; A(4,2) = -1.618234; A(4,3) =  1.119524; A(4,4) =  2.588165;

	matrix_type B(n,n);
	B(0,0) = -1.592710; B(0,1) =  0.057283; B(0,2) = -1.862275; B(0,3) =  0.712471; B(0,4) =  0.463207;
	B(1,0) =  1.072859; B(1,1) = -1.384371; B(1,2) =  0.777754; B(1,3) =  1.914787; B(1,4) =  0.082774;
	B(2,0) = -0.451744; B(2,1) = -0.131528; B(2,2) = -0.636187; B(2,3) =  0.984480; B(2,4) =  0.011728;
	B(3,0) = -0.876629; B(3,1) = -0.083787; B(3,2) =  0.474227; B(3,3) = -0.042328; B(3,4) = -0.529845;
	B(4,0) = -0.812610; B(4,1) =  0.142456; B(4,2) =  0.033739; B(4,3) = -2.000422; B(4,4) = -0.765401;


	matrix_type S;
	matrix_type T;
	matrix_type Q;
	matrix_type Z;

	ublasx::qz_decompose(A, B, S, T, Q, Z, ublasx::udo_qz_eigenvalues);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "B = " << B );
	BOOST_UBLASX_DEBUG_TRACE( "S = " << S );
	BOOST_UBLASX_DEBUG_TRACE( "T = " << T );
	BOOST_UBLASX_DEBUG_TRACE( "Q = " << Q );
	BOOST_UBLASX_DEBUG_TRACE( "Z = " << Z );
	matrix_type X;
	X = ublas::prod(Q, S);
	X = ublas::prod(X, ublas::trans(Z));
	BOOST_UBLASX_DEBUG_TRACE( "Q*S*Z' = " << X );
	X = ublas::prod(Q, T);
	X = ublas::prod(X, ublas::trans(Z));
	BOOST_UBLASX_DEBUG_TRACE( "Q*T*Z' = " << X );

	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(S) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(S) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(T) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(T) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(Q) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(Q) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(Z) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(Z) == n );
	X = ublas::prod(Q, S);
	X = ublas::prod(X, ublas::trans(Z));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, A, n, n, tol );
	X = ublas::prod(Q, T);
	X = ublas::prod(X, ublas::trans(Z));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, B, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_real_row_major_decomp_udo_reorder )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Real Matrix - Row Major - Decomposition and UDO Reordering");

	typedef double value_type;
	typedef ublas::matrix<value_type, ublas::row_major> matrix_type;

	const std::size_t n(5);

	matrix_type A(n,n);
	A(0,0) = -0.180557; A(0,1) =  0.322289; A(0,2) = -0.651789; A(0,3) =  0.793637; A(0,4) = -0.141086;
	A(1,0) =  0.729781; A(1,1) =  1.665989; A(1,2) =  0.620091; A(1,3) = -1.541503; A(1,4) =  0.146673;
	A(2,0) = -0.594370; A(2,1) =  0.494804; A(2,2) =  1.004784; A(2,3) = -0.221373; A(2,4) = -2.196082;
	A(3,0) = -1.106269; A(3,1) =  0.026697; A(3,2) =  2.687083; A(3,3) =  0.763162; A(3,4) =  1.203514;
	A(4,0) = -0.021184; A(4,1) = -0.882220; A(4,2) = -1.618234; A(4,3) =  1.119524; A(4,4) =  2.588165;

	matrix_type B(n,n);
	B(0,0) = -1.592710; B(0,1) =  0.057283; B(0,2) = -1.862275; B(0,3) =  0.712471; B(0,4) =  0.463207;
	B(1,0) =  1.072859; B(1,1) = -1.384371; B(1,2) =  0.777754; B(1,3) =  1.914787; B(1,4) =  0.082774;
	B(2,0) = -0.451744; B(2,1) = -0.131528; B(2,2) = -0.636187; B(2,3) =  0.984480; B(2,4) =  0.011728;
	B(3,0) = -0.876629; B(3,1) = -0.083787; B(3,2) =  0.474227; B(3,3) = -0.042328; B(3,4) = -0.529845;
	B(4,0) = -0.812610; B(4,1) =  0.142456; B(4,2) =  0.033739; B(4,3) = -2.000422; B(4,4) = -0.765401;


	matrix_type S;
	matrix_type T;
	matrix_type Q;
	matrix_type Z;

	ublasx::qz_decompose(A, B, S, T, Q, Z, ublasx::udo_qz_eigenvalues);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "B = " << B );
	BOOST_UBLASX_DEBUG_TRACE( "S = " << S );
	BOOST_UBLASX_DEBUG_TRACE( "T = " << T );
	BOOST_UBLASX_DEBUG_TRACE( "Q = " << Q );
	BOOST_UBLASX_DEBUG_TRACE( "Z = " << Z );
	matrix_type X;
	X = ublas::prod(Q, S);
	X = ublas::prod(X, ublas::trans(Z));
	BOOST_UBLASX_DEBUG_TRACE( "Q*S*Z' = " << X );
	X = ublas::prod(Q, T);
	X = ublas::prod(X, ublas::trans(Z));
	BOOST_UBLASX_DEBUG_TRACE( "Q*T*Z' = " << X );

	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(S) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(S) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(T) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(T) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(Q) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(Q) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(Z) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(Z) == n );
	X = ublas::prod(Q, S);
	X = ublas::prod(X, ublas::trans(Z));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, A, n, n, tol );
	X = ublas::prod(Q, T);
	X = ublas::prod(X, ublas::trans(Z));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, B, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_real_column_major_decomp_custom_reorder )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Real Matrix - Column Major - Decomposition and Custom Reordering");

	typedef double value_type;
	typedef ublas::matrix<value_type, ublas::column_major> matrix_type;

	const std::size_t n(5);

	matrix_type A(n,n);
	A(0,0) = -0.180557; A(0,1) =  0.322289; A(0,2) = -0.651789; A(0,3) =  0.793637; A(0,4) = -0.141086;
	A(1,0) =  0.729781; A(1,1) =  1.665989; A(1,2) =  0.620091; A(1,3) = -1.541503; A(1,4) =  0.146673;
	A(2,0) = -0.594370; A(2,1) =  0.494804; A(2,2) =  1.004784; A(2,3) = -0.221373; A(2,4) = -2.196082;
	A(3,0) = -1.106269; A(3,1) =  0.026697; A(3,2) =  2.687083; A(3,3) =  0.763162; A(3,4) =  1.203514;
	A(4,0) = -0.021184; A(4,1) = -0.882220; A(4,2) = -1.618234; A(4,3) =  1.119524; A(4,4) =  2.588165;

	matrix_type B(n,n);
	B(0,0) = -1.592710; B(0,1) =  0.057283; B(0,2) = -1.862275; B(0,3) =  0.712471; B(0,4) =  0.463207;
	B(1,0) =  1.072859; B(1,1) = -1.384371; B(1,2) =  0.777754; B(1,3) =  1.914787; B(1,4) =  0.082774;
	B(2,0) = -0.451744; B(2,1) = -0.131528; B(2,2) = -0.636187; B(2,3) =  0.984480; B(2,4) =  0.011728;
	B(3,0) = -0.876629; B(3,1) = -0.083787; B(3,2) =  0.474227; B(3,3) = -0.042328; B(3,4) = -0.529845;
	B(4,0) = -0.812610; B(4,1) =  0.142456; B(4,2) =  0.033739; B(4,3) = -2.000422; B(4,4) = -0.765401;


	matrix_type S;
	matrix_type T;
	matrix_type Q;
	matrix_type Z;

	ublasx::qz_decompose(A, B, S, T, Q, Z);

	// Selection: select eigenvalues at even position (no sense ... just a test)
	ublas::vector<bool> selection(n);
	for (std::size_t i = 0; i < n; ++i)
	{
		selection(i) = ((i+1) % 2) ? true : false;
	}

	matrix_type SS;
	matrix_type TS;
	matrix_type QS;
	matrix_type ZS;

	ublasx::qz_reorder(S, T, Q, Z, selection, SS, TS, QS, ZS);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "B = " << B );
	BOOST_UBLASX_DEBUG_TRACE( "S = " << S );
	BOOST_UBLASX_DEBUG_TRACE( "T = " << T );
	BOOST_UBLASX_DEBUG_TRACE( "Q = " << Q );
	BOOST_UBLASX_DEBUG_TRACE( "Z = " << Z );
	BOOST_UBLASX_DEBUG_TRACE( "SS = " << SS );
	BOOST_UBLASX_DEBUG_TRACE( "TS = " << TS );
	BOOST_UBLASX_DEBUG_TRACE( "QS = " << QS );
	BOOST_UBLASX_DEBUG_TRACE( "ZS = " << ZS );
	matrix_type X;
	X = ublas::prod(QS, SS);
	X = ublas::prod(X, ublas::trans(ZS));
	BOOST_UBLASX_DEBUG_TRACE( "QS*SS*ZS' = " << X );
	X = ublas::prod(QS, TS);
	X = ublas::prod(X, ublas::trans(ZS));
	BOOST_UBLASX_DEBUG_TRACE( "QS*TS*ZS' = " << X );

	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(SS) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(SS) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(TS) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(TS) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(QS) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(QS) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(ZS) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(ZS) == n );
	X = ublas::prod(QS, SS);
	X = ublas::prod(X, ublas::trans(ZS));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, A, n, n, tol );
	X = ublas::prod(QS, TS);
	X = ublas::prod(X, ublas::trans(ZS));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, B, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_real_row_major_decomp_custom_reorder )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Real Matrix - Row Major - Decomposition and Custom Reordering");

	typedef double value_type;
	typedef ublas::matrix<value_type, ublas::row_major> matrix_type;

	const std::size_t n(5);

	matrix_type A(n,n);
	A(0,0) = -0.180557; A(0,1) =  0.322289; A(0,2) = -0.651789; A(0,3) =  0.793637; A(0,4) = -0.141086;
	A(1,0) =  0.729781; A(1,1) =  1.665989; A(1,2) =  0.620091; A(1,3) = -1.541503; A(1,4) =  0.146673;
	A(2,0) = -0.594370; A(2,1) =  0.494804; A(2,2) =  1.004784; A(2,3) = -0.221373; A(2,4) = -2.196082;
	A(3,0) = -1.106269; A(3,1) =  0.026697; A(3,2) =  2.687083; A(3,3) =  0.763162; A(3,4) =  1.203514;
	A(4,0) = -0.021184; A(4,1) = -0.882220; A(4,2) = -1.618234; A(4,3) =  1.119524; A(4,4) =  2.588165;

	matrix_type B(n,n);
	B(0,0) = -1.592710; B(0,1) =  0.057283; B(0,2) = -1.862275; B(0,3) =  0.712471; B(0,4) =  0.463207;
	B(1,0) =  1.072859; B(1,1) = -1.384371; B(1,2) =  0.777754; B(1,3) =  1.914787; B(1,4) =  0.082774;
	B(2,0) = -0.451744; B(2,1) = -0.131528; B(2,2) = -0.636187; B(2,3) =  0.984480; B(2,4) =  0.011728;
	B(3,0) = -0.876629; B(3,1) = -0.083787; B(3,2) =  0.474227; B(3,3) = -0.042328; B(3,4) = -0.529845;
	B(4,0) = -0.812610; B(4,1) =  0.142456; B(4,2) =  0.033739; B(4,3) = -2.000422; B(4,4) = -0.765401;


	matrix_type S;
	matrix_type T;
	matrix_type Q;
	matrix_type Z;

	ublasx::qz_decompose(A, B, S, T, Q, Z);

	// Selection: select eigenvalues at even position (no sense ... just a test)
	ublas::vector<bool> selection(n);
	for (std::size_t i = 0; i < n; ++i)
	{
		selection(i) = ((i+1) % 2) ? true : false;
	}

	matrix_type SS;
	matrix_type TS;
	matrix_type QS;
	matrix_type ZS;

	ublasx::qz_reorder(S, T, Q, Z, selection, SS, TS, QS, ZS);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "B = " << B );
	BOOST_UBLASX_DEBUG_TRACE( "S = " << S );
	BOOST_UBLASX_DEBUG_TRACE( "T = " << T );
	BOOST_UBLASX_DEBUG_TRACE( "Q = " << Q );
	BOOST_UBLASX_DEBUG_TRACE( "Z = " << Z );
	BOOST_UBLASX_DEBUG_TRACE( "SS = " << SS );
	BOOST_UBLASX_DEBUG_TRACE( "TS = " << TS );
	BOOST_UBLASX_DEBUG_TRACE( "QS = " << QS );
	BOOST_UBLASX_DEBUG_TRACE( "ZS = " << ZS );
	matrix_type X;
	X = ublas::prod(QS, SS);
	X = ublas::prod(X, ublas::trans(ZS));
	BOOST_UBLASX_DEBUG_TRACE( "QS*SS*ZS' = " << X );
	X = ublas::prod(QS, TS);
	X = ublas::prod(X, ublas::trans(ZS));
	BOOST_UBLASX_DEBUG_TRACE( "QS*TS*ZS' = " << X );

	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(SS) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(SS) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(TS) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(TS) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(QS) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(QS) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(ZS) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(ZS) == n );
	X = ublas::prod(QS, SS);
	X = ublas::prod(X, ublas::trans(ZS));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, A, n, n, tol );
	X = ublas::prod(QS, TS);
	X = ublas::prod(X, ublas::trans(ZS));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, B, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_complex_column_major_decomp )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Complex Matrix - Column Major - Decomposition");

	typedef std::complex<double> value_type;
	typedef ublas::matrix<value_type, ublas::column_major> matrix_type;

	const std::size_t n(5);

	matrix_type A(n,n);
	A(0,0) = value_type(-0.180557,-0.947835); A(0,1) = value_type( 0.322289, 1.603339); A(0,2) = value_type(-0.651789,-0.902809); A(0,3) = value_type( 0.793637, 0.031147); A(0,4) = value_type(-0.141086,-0.769742);
	A(1,0) = value_type( 0.729781, 0.693097); A(1,1) = value_type( 1.665989, 1.258667); A(1,2) = value_type( 0.620091,-0.192144); A(1,3) = value_type(-1.541503, 0.545104); A(1,4) = value_type( 0.146673,-1.553689);
	A(2,0) = value_type(-0.594370, 0.071316); A(2,1) = value_type( 0.494804,-2.381227); A(2,2) = value_type( 1.004784, 1.097852); A(2,3) = value_type(-0.221373,-0.585458); A(2,4) = value_type(-2.196082,-0.518945);
	A(3,0) = value_type(-1.106269,-1.015812); A(3,1) = value_type( 0.026697, 2.866180); A(3,2) = value_type( 2.687083,-0.115506); A(3,3) = value_type( 0.763162, 0.382183); A(3,4) = value_type( 1.203514, 0.203552);
	A(4,0) = value_type(-0.021184,-1.006383); A(4,1) = value_type(-0.882220, 0.232555); A(4,2) = value_type(-1.618234, 0.615528); A(4,3) = value_type( 1.119524,-2.178697); A(4,4) = value_type( 2.588165,-0.054936);

	matrix_type B(n,n);
	B(0,0) = value_type(-1.592710, 0.804795); B(0,1) = value_type( 0.057283, 0.470640); B(0,2) = value_type(-1.862275,-0.454486); B(0,3) = value_type( 0.712471, 0.887654); B(0,4) = value_type( 0.463207,-0.045117);
	B(1,0) = value_type( 1.072859,-2.009749); B(1,1) = value_type(-1.384371,-0.778200); B(1,2) = value_type( 0.777754, 1.245456); B(1,3) = value_type( 1.914787,-1.246236); B(1,4) = value_type( 0.082774, 0.358980);
	B(2,0) = value_type(-0.451744,-0.353845); B(2,1) = value_type(-0.131528,-2.236258); B(2,2) = value_type(-0.636187,-0.957378); B(2,3) = value_type( 0.984480,-1.840536); B(2,4) = value_type( 0.011728, 1.682497);
	B(3,0) = value_type(-0.876629, 0.764240); B(3,1) = value_type(-0.083787, 2.238476); B(3,2) = value_type( 0.474227,-2.426134); B(3,3) = value_type(-0.042328, 1.129135); B(3,4) = value_type(-0.529845, 0.653758);
	B(4,0) = value_type(-0.812610, 0.996806); B(4,1) = value_type( 0.142456, 0.101454); B(4,2) = value_type( 0.033739,-0.501220); B(4,3) = value_type(-2.000422, 0.181931); B(4,4) = value_type(-0.765401,-1.004076);


	matrix_type S;
	matrix_type T;
	matrix_type Q;
	matrix_type Z;

	ublasx::qz_decompose(A, B, S, T, Q, Z);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "B = " << B );
	BOOST_UBLASX_DEBUG_TRACE( "S = " << S );
	BOOST_UBLASX_DEBUG_TRACE( "T = " << T );
	BOOST_UBLASX_DEBUG_TRACE( "Q = " << Q );
	BOOST_UBLASX_DEBUG_TRACE( "Z = " << Z );
//	BOOST_UBLASX_DEBUG_TRACE( "Q*S*Z' = " << ublas::prod(ublas::prod(Q, S), ublas::herm(Z)) );
//	BOOST_UBLASX_DEBUG_TRACE( "Q*T*Z' = " << ublas::prod(ublas::prod(Q, T), ublas::herm(Z)) );
	matrix_type X;
	X = ublas::prod(Q, S);
	X = ublas::prod(X, ublas::herm(Z));
	BOOST_UBLASX_DEBUG_TRACE( "Q*S*Z' = " << X );
	X = ublas::prod(Q, T);
	X = ublas::prod(X, ublas::herm(Z));
	BOOST_UBLASX_DEBUG_TRACE( "Q*T*Z' = " << X );

	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(S) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(S) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(T) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(T) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(Q) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(Q) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(Z) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(Z) == n );
	X = ublas::prod(Q, S);
	X = ublas::prod(X, ublas::herm(Z));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, A, n, n, tol );
	X = ublas::prod(Q, T);
	X = ublas::prod(X, ublas::herm(Z));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, B, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_complex_row_major_decomp )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Complex Matrix - Row Major - Decomposition");

	typedef std::complex<double> value_type;
	typedef ublas::matrix<value_type, ublas::row_major> matrix_type;

	const std::size_t n(5);

	matrix_type A(n,n);
	A(0,0) = value_type(-0.180557,-0.947835); A(0,1) = value_type( 0.322289, 1.603339); A(0,2) = value_type(-0.651789,-0.902809); A(0,3) = value_type( 0.793637, 0.031147); A(0,4) = value_type(-0.141086,-0.769742);
	A(1,0) = value_type( 0.729781, 0.693097); A(1,1) = value_type( 1.665989, 1.258667); A(1,2) = value_type( 0.620091,-0.192144); A(1,3) = value_type(-1.541503, 0.545104); A(1,4) = value_type( 0.146673,-1.553689);
	A(2,0) = value_type(-0.594370, 0.071316); A(2,1) = value_type( 0.494804,-2.381227); A(2,2) = value_type( 1.004784, 1.097852); A(2,3) = value_type(-0.221373,-0.585458); A(2,4) = value_type(-2.196082,-0.518945);
	A(3,0) = value_type(-1.106269,-1.015812); A(3,1) = value_type( 0.026697, 2.866180); A(3,2) = value_type( 2.687083,-0.115506); A(3,3) = value_type( 0.763162, 0.382183); A(3,4) = value_type( 1.203514, 0.203552);
	A(4,0) = value_type(-0.021184,-1.006383); A(4,1) = value_type(-0.882220, 0.232555); A(4,2) = value_type(-1.618234, 0.615528); A(4,3) = value_type( 1.119524,-2.178697); A(4,4) = value_type( 2.588165,-0.054936);

	matrix_type B(n,n);
	B(0,0) = value_type(-1.592710, 0.804795); B(0,1) = value_type( 0.057283, 0.470640); B(0,2) = value_type(-1.862275,-0.454486); B(0,3) = value_type( 0.712471, 0.887654); B(0,4) = value_type( 0.463207,-0.045117);
	B(1,0) = value_type( 1.072859,-2.009749); B(1,1) = value_type(-1.384371,-0.778200); B(1,2) = value_type( 0.777754, 1.245456); B(1,3) = value_type( 1.914787,-1.246236); B(1,4) = value_type( 0.082774, 0.358980);
	B(2,0) = value_type(-0.451744,-0.353845); B(2,1) = value_type(-0.131528,-2.236258); B(2,2) = value_type(-0.636187,-0.957378); B(2,3) = value_type( 0.984480,-1.840536); B(2,4) = value_type( 0.011728, 1.682497);
	B(3,0) = value_type(-0.876629, 0.764240); B(3,1) = value_type(-0.083787, 2.238476); B(3,2) = value_type( 0.474227,-2.426134); B(3,3) = value_type(-0.042328, 1.129135); B(3,4) = value_type(-0.529845, 0.653758);
	B(4,0) = value_type(-0.812610, 0.996806); B(4,1) = value_type( 0.142456, 0.101454); B(4,2) = value_type( 0.033739,-0.501220); B(4,3) = value_type(-2.000422, 0.181931); B(4,4) = value_type(-0.765401,-1.004076);


	matrix_type S;
	matrix_type T;
	matrix_type Q;
	matrix_type Z;

	ublasx::qz_decompose(A, B, S, T, Q, Z);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "B = " << B );
	BOOST_UBLASX_DEBUG_TRACE( "S = " << S );
	BOOST_UBLASX_DEBUG_TRACE( "T = " << T );
	BOOST_UBLASX_DEBUG_TRACE( "Q = " << Q );
	BOOST_UBLASX_DEBUG_TRACE( "Z = " << Z );
//	BOOST_UBLASX_DEBUG_TRACE( "Q*S*Z' = " << ublas::prod(ublas::prod(Q, S), ublas::herm(Z)) );
//	BOOST_UBLASX_DEBUG_TRACE( "Q*T*Z' = " << ublas::prod(ublas::prod(Q, T), ublas::herm(Z)) );
	matrix_type X;
	X = ublas::prod(Q, S);
	X = ublas::prod(X, ublas::herm(Z));
	BOOST_UBLASX_DEBUG_TRACE( "Q*S*Z' = " << X );
	X = ublas::prod(Q, T);
	X = ublas::prod(X, ublas::herm(Z));
	BOOST_UBLASX_DEBUG_TRACE( "Q*T*Z' = " << X );

	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(S) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(S) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(T) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(T) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(Q) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(Q) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(Z) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(Z) == n );
	X = ublas::prod(Q, S);
	X = ublas::prod(X, ublas::herm(Z));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, A, n, n, tol );
	X = ublas::prod(Q, T);
	X = ublas::prod(X, ublas::herm(Z));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, B, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_complex_column_major_decomp_lhp_reorder )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Complex Matrix - Column Major - Decomposition and LHP Reordering");

	typedef std::complex<double> value_type;
	typedef ublas::matrix<value_type, ublas::column_major> matrix_type;

	const std::size_t n(5);

	matrix_type A(n,n);
	A(0,0) = value_type(-0.180557,-0.947835); A(0,1) = value_type( 0.322289, 1.603339); A(0,2) = value_type(-0.651789,-0.902809); A(0,3) = value_type( 0.793637, 0.031147); A(0,4) = value_type(-0.141086,-0.769742);
	A(1,0) = value_type( 0.729781, 0.693097); A(1,1) = value_type( 1.665989, 1.258667); A(1,2) = value_type( 0.620091,-0.192144); A(1,3) = value_type(-1.541503, 0.545104); A(1,4) = value_type( 0.146673,-1.553689);
	A(2,0) = value_type(-0.594370, 0.071316); A(2,1) = value_type( 0.494804,-2.381227); A(2,2) = value_type( 1.004784, 1.097852); A(2,3) = value_type(-0.221373,-0.585458); A(2,4) = value_type(-2.196082,-0.518945);
	A(3,0) = value_type(-1.106269,-1.015812); A(3,1) = value_type( 0.026697, 2.866180); A(3,2) = value_type( 2.687083,-0.115506); A(3,3) = value_type( 0.763162, 0.382183); A(3,4) = value_type( 1.203514, 0.203552);
	A(4,0) = value_type(-0.021184,-1.006383); A(4,1) = value_type(-0.882220, 0.232555); A(4,2) = value_type(-1.618234, 0.615528); A(4,3) = value_type( 1.119524,-2.178697); A(4,4) = value_type( 2.588165,-0.054936);

	matrix_type B(n,n);
	B(0,0) = value_type(-1.592710, 0.804795); B(0,1) = value_type( 0.057283, 0.470640); B(0,2) = value_type(-1.862275,-0.454486); B(0,3) = value_type( 0.712471, 0.887654); B(0,4) = value_type( 0.463207,-0.045117);
	B(1,0) = value_type( 1.072859,-2.009749); B(1,1) = value_type(-1.384371,-0.778200); B(1,2) = value_type( 0.777754, 1.245456); B(1,3) = value_type( 1.914787,-1.246236); B(1,4) = value_type( 0.082774, 0.358980);
	B(2,0) = value_type(-0.451744,-0.353845); B(2,1) = value_type(-0.131528,-2.236258); B(2,2) = value_type(-0.636187,-0.957378); B(2,3) = value_type( 0.984480,-1.840536); B(2,4) = value_type( 0.011728, 1.682497);
	B(3,0) = value_type(-0.876629, 0.764240); B(3,1) = value_type(-0.083787, 2.238476); B(3,2) = value_type( 0.474227,-2.426134); B(3,3) = value_type(-0.042328, 1.129135); B(3,4) = value_type(-0.529845, 0.653758);
	B(4,0) = value_type(-0.812610, 0.996806); B(4,1) = value_type( 0.142456, 0.101454); B(4,2) = value_type( 0.033739,-0.501220); B(4,3) = value_type(-2.000422, 0.181931); B(4,4) = value_type(-0.765401,-1.004076);


	matrix_type S;
	matrix_type T;
	matrix_type Q;
	matrix_type Z;

	ublasx::qz_decompose(A, B, S, T, Q, Z, ublasx::lhp_qz_eigenvalues);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "B = " << B );
	BOOST_UBLASX_DEBUG_TRACE( "S = " << S );
	BOOST_UBLASX_DEBUG_TRACE( "T = " << T );
	BOOST_UBLASX_DEBUG_TRACE( "Q = " << Q );
	BOOST_UBLASX_DEBUG_TRACE( "Z = " << Z );
//	BOOST_UBLASX_DEBUG_TRACE( "Q*S*Z' = " << ublas::prod(ublas::prod(Q, S), ublas::herm(Z)) );
//	BOOST_UBLASX_DEBUG_TRACE( "Q*T*Z' = " << ublas::prod(ublas::prod(Q, T), ublas::herm(Z)) );
	matrix_type X;
	X = ublas::prod(Q, S);
	X = ublas::prod(X, ublas::herm(Z));
	BOOST_UBLASX_DEBUG_TRACE( "Q*S*Z' = " << X );
	X = ublas::prod(Q, T);
	X = ublas::prod(X, ublas::herm(Z));
	BOOST_UBLASX_DEBUG_TRACE( "Q*T*Z' = " << X );

	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(S) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(S) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(T) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(T) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(Q) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(Q) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(Z) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(Z) == n );
	X = ublas::prod(Q, S);
	X = ublas::prod(X, ublas::herm(Z));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, A, n, n, tol );
	X = ublas::prod(Q, T);
	X = ublas::prod(X, ublas::herm(Z));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, B, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_complex_row_major_decomp_lhp_reorder )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Complex Matrix - Row Major - Decomposition and LHP Reordering");

	typedef std::complex<double> value_type;
	typedef ublas::matrix<value_type, ublas::row_major> matrix_type;

	const std::size_t n(5);

	matrix_type A(n,n);
	A(0,0) = value_type(-0.180557,-0.947835); A(0,1) = value_type( 0.322289, 1.603339); A(0,2) = value_type(-0.651789,-0.902809); A(0,3) = value_type( 0.793637, 0.031147); A(0,4) = value_type(-0.141086,-0.769742);
	A(1,0) = value_type( 0.729781, 0.693097); A(1,1) = value_type( 1.665989, 1.258667); A(1,2) = value_type( 0.620091,-0.192144); A(1,3) = value_type(-1.541503, 0.545104); A(1,4) = value_type( 0.146673,-1.553689);
	A(2,0) = value_type(-0.594370, 0.071316); A(2,1) = value_type( 0.494804,-2.381227); A(2,2) = value_type( 1.004784, 1.097852); A(2,3) = value_type(-0.221373,-0.585458); A(2,4) = value_type(-2.196082,-0.518945);
	A(3,0) = value_type(-1.106269,-1.015812); A(3,1) = value_type( 0.026697, 2.866180); A(3,2) = value_type( 2.687083,-0.115506); A(3,3) = value_type( 0.763162, 0.382183); A(3,4) = value_type( 1.203514, 0.203552);
	A(4,0) = value_type(-0.021184,-1.006383); A(4,1) = value_type(-0.882220, 0.232555); A(4,2) = value_type(-1.618234, 0.615528); A(4,3) = value_type( 1.119524,-2.178697); A(4,4) = value_type( 2.588165,-0.054936);

	matrix_type B(n,n);
	B(0,0) = value_type(-1.592710, 0.804795); B(0,1) = value_type( 0.057283, 0.470640); B(0,2) = value_type(-1.862275,-0.454486); B(0,3) = value_type( 0.712471, 0.887654); B(0,4) = value_type( 0.463207,-0.045117);
	B(1,0) = value_type( 1.072859,-2.009749); B(1,1) = value_type(-1.384371,-0.778200); B(1,2) = value_type( 0.777754, 1.245456); B(1,3) = value_type( 1.914787,-1.246236); B(1,4) = value_type( 0.082774, 0.358980);
	B(2,0) = value_type(-0.451744,-0.353845); B(2,1) = value_type(-0.131528,-2.236258); B(2,2) = value_type(-0.636187,-0.957378); B(2,3) = value_type( 0.984480,-1.840536); B(2,4) = value_type( 0.011728, 1.682497);
	B(3,0) = value_type(-0.876629, 0.764240); B(3,1) = value_type(-0.083787, 2.238476); B(3,2) = value_type( 0.474227,-2.426134); B(3,3) = value_type(-0.042328, 1.129135); B(3,4) = value_type(-0.529845, 0.653758);
	B(4,0) = value_type(-0.812610, 0.996806); B(4,1) = value_type( 0.142456, 0.101454); B(4,2) = value_type( 0.033739,-0.501220); B(4,3) = value_type(-2.000422, 0.181931); B(4,4) = value_type(-0.765401,-1.004076);


	matrix_type S;
	matrix_type T;
	matrix_type Q;
	matrix_type Z;

	ublasx::qz_decompose(A, B, S, T, Q, Z, ublasx::lhp_qz_eigenvalues);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "B = " << B );
	BOOST_UBLASX_DEBUG_TRACE( "S = " << S );
	BOOST_UBLASX_DEBUG_TRACE( "T = " << T );
	BOOST_UBLASX_DEBUG_TRACE( "Q = " << Q );
	BOOST_UBLASX_DEBUG_TRACE( "Z = " << Z );
//	BOOST_UBLASX_DEBUG_TRACE( "Q*S*Z' = " << ublas::prod(ublas::prod(Q, S), ublas::herm(Z)) );
//	BOOST_UBLASX_DEBUG_TRACE( "Q*T*Z' = " << ublas::prod(ublas::prod(Q, T), ublas::herm(Z)) );
	matrix_type X;
	X = ublas::prod(Q, S);
	X = ublas::prod(X, ublas::herm(Z));
	BOOST_UBLASX_DEBUG_TRACE( "Q*S*Z' = " << X );
	X = ublas::prod(Q, T);
	X = ublas::prod(X, ublas::herm(Z));
	BOOST_UBLASX_DEBUG_TRACE( "Q*T*Z' = " << X );

	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(S) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(S) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(T) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(T) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(Q) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(Q) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(Z) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(Z) == n );
	X = ublas::prod(Q, S);
	X = ublas::prod(X, ublas::herm(Z));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, A, n, n, tol );
	X = ublas::prod(Q, T);
	X = ublas::prod(X, ublas::herm(Z));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, B, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_complex_column_major_decomp_rhp_reorder )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Complex Matrix - Column Major - Decomposition and RHP Reordering");

	typedef std::complex<double> value_type;
	typedef ublas::matrix<value_type, ublas::column_major> matrix_type;

	const std::size_t n(5);

	matrix_type A(n,n);
	A(0,0) = value_type(-0.180557,-0.947835); A(0,1) = value_type( 0.322289, 1.603339); A(0,2) = value_type(-0.651789,-0.902809); A(0,3) = value_type( 0.793637, 0.031147); A(0,4) = value_type(-0.141086,-0.769742);
	A(1,0) = value_type( 0.729781, 0.693097); A(1,1) = value_type( 1.665989, 1.258667); A(1,2) = value_type( 0.620091,-0.192144); A(1,3) = value_type(-1.541503, 0.545104); A(1,4) = value_type( 0.146673,-1.553689);
	A(2,0) = value_type(-0.594370, 0.071316); A(2,1) = value_type( 0.494804,-2.381227); A(2,2) = value_type( 1.004784, 1.097852); A(2,3) = value_type(-0.221373,-0.585458); A(2,4) = value_type(-2.196082,-0.518945);
	A(3,0) = value_type(-1.106269,-1.015812); A(3,1) = value_type( 0.026697, 2.866180); A(3,2) = value_type( 2.687083,-0.115506); A(3,3) = value_type( 0.763162, 0.382183); A(3,4) = value_type( 1.203514, 0.203552);
	A(4,0) = value_type(-0.021184,-1.006383); A(4,1) = value_type(-0.882220, 0.232555); A(4,2) = value_type(-1.618234, 0.615528); A(4,3) = value_type( 1.119524,-2.178697); A(4,4) = value_type( 2.588165,-0.054936);

	matrix_type B(n,n);
	B(0,0) = value_type(-1.592710, 0.804795); B(0,1) = value_type( 0.057283, 0.470640); B(0,2) = value_type(-1.862275,-0.454486); B(0,3) = value_type( 0.712471, 0.887654); B(0,4) = value_type( 0.463207,-0.045117);
	B(1,0) = value_type( 1.072859,-2.009749); B(1,1) = value_type(-1.384371,-0.778200); B(1,2) = value_type( 0.777754, 1.245456); B(1,3) = value_type( 1.914787,-1.246236); B(1,4) = value_type( 0.082774, 0.358980);
	B(2,0) = value_type(-0.451744,-0.353845); B(2,1) = value_type(-0.131528,-2.236258); B(2,2) = value_type(-0.636187,-0.957378); B(2,3) = value_type( 0.984480,-1.840536); B(2,4) = value_type( 0.011728, 1.682497);
	B(3,0) = value_type(-0.876629, 0.764240); B(3,1) = value_type(-0.083787, 2.238476); B(3,2) = value_type( 0.474227,-2.426134); B(3,3) = value_type(-0.042328, 1.129135); B(3,4) = value_type(-0.529845, 0.653758);
	B(4,0) = value_type(-0.812610, 0.996806); B(4,1) = value_type( 0.142456, 0.101454); B(4,2) = value_type( 0.033739,-0.501220); B(4,3) = value_type(-2.000422, 0.181931); B(4,4) = value_type(-0.765401,-1.004076);


	matrix_type S;
	matrix_type T;
	matrix_type Q;
	matrix_type Z;

	ublasx::qz_decompose(A, B, S, T, Q, Z, ublasx::rhp_qz_eigenvalues);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "B = " << B );
	BOOST_UBLASX_DEBUG_TRACE( "S = " << S );
	BOOST_UBLASX_DEBUG_TRACE( "T = " << T );
	BOOST_UBLASX_DEBUG_TRACE( "Q = " << Q );
	BOOST_UBLASX_DEBUG_TRACE( "Z = " << Z );
//	BOOST_UBLASX_DEBUG_TRACE( "Q*S*Z' = " << ublas::prod(ublas::prod(Q, S), ublas::herm(Z)) );
//	BOOST_UBLASX_DEBUG_TRACE( "Q*T*Z' = " << ublas::prod(ublas::prod(Q, T), ublas::herm(Z)) );
	matrix_type X;
	X = ublas::prod(Q, S);
	X = ublas::prod(X, ublas::herm(Z));
	BOOST_UBLASX_DEBUG_TRACE( "Q*S*Z' = " << X );
	X = ublas::prod(Q, T);
	X = ublas::prod(X, ublas::herm(Z));
	BOOST_UBLASX_DEBUG_TRACE( "Q*T*Z' = " << X );

	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(S) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(S) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(T) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(T) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(Q) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(Q) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(Z) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(Z) == n );
	X = ublas::prod(Q, S);
	X = ublas::prod(X, ublas::herm(Z));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, A, n, n, tol );
	X = ublas::prod(Q, T);
	X = ublas::prod(X, ublas::herm(Z));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, B, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_complex_row_major_decomp_rhp_reorder )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Complex Matrix - Row Major - Decomposition and RHP Reordering");

	typedef std::complex<double> value_type;
	typedef ublas::matrix<value_type, ublas::row_major> matrix_type;

	const std::size_t n(5);

	matrix_type A(n,n);
	A(0,0) = value_type(-0.180557,-0.947835); A(0,1) = value_type( 0.322289, 1.603339); A(0,2) = value_type(-0.651789,-0.902809); A(0,3) = value_type( 0.793637, 0.031147); A(0,4) = value_type(-0.141086,-0.769742);
	A(1,0) = value_type( 0.729781, 0.693097); A(1,1) = value_type( 1.665989, 1.258667); A(1,2) = value_type( 0.620091,-0.192144); A(1,3) = value_type(-1.541503, 0.545104); A(1,4) = value_type( 0.146673,-1.553689);
	A(2,0) = value_type(-0.594370, 0.071316); A(2,1) = value_type( 0.494804,-2.381227); A(2,2) = value_type( 1.004784, 1.097852); A(2,3) = value_type(-0.221373,-0.585458); A(2,4) = value_type(-2.196082,-0.518945);
	A(3,0) = value_type(-1.106269,-1.015812); A(3,1) = value_type( 0.026697, 2.866180); A(3,2) = value_type( 2.687083,-0.115506); A(3,3) = value_type( 0.763162, 0.382183); A(3,4) = value_type( 1.203514, 0.203552);
	A(4,0) = value_type(-0.021184,-1.006383); A(4,1) = value_type(-0.882220, 0.232555); A(4,2) = value_type(-1.618234, 0.615528); A(4,3) = value_type( 1.119524,-2.178697); A(4,4) = value_type( 2.588165,-0.054936);

	matrix_type B(n,n);
	B(0,0) = value_type(-1.592710, 0.804795); B(0,1) = value_type( 0.057283, 0.470640); B(0,2) = value_type(-1.862275,-0.454486); B(0,3) = value_type( 0.712471, 0.887654); B(0,4) = value_type( 0.463207,-0.045117);
	B(1,0) = value_type( 1.072859,-2.009749); B(1,1) = value_type(-1.384371,-0.778200); B(1,2) = value_type( 0.777754, 1.245456); B(1,3) = value_type( 1.914787,-1.246236); B(1,4) = value_type( 0.082774, 0.358980);
	B(2,0) = value_type(-0.451744,-0.353845); B(2,1) = value_type(-0.131528,-2.236258); B(2,2) = value_type(-0.636187,-0.957378); B(2,3) = value_type( 0.984480,-1.840536); B(2,4) = value_type( 0.011728, 1.682497);
	B(3,0) = value_type(-0.876629, 0.764240); B(3,1) = value_type(-0.083787, 2.238476); B(3,2) = value_type( 0.474227,-2.426134); B(3,3) = value_type(-0.042328, 1.129135); B(3,4) = value_type(-0.529845, 0.653758);
	B(4,0) = value_type(-0.812610, 0.996806); B(4,1) = value_type( 0.142456, 0.101454); B(4,2) = value_type( 0.033739,-0.501220); B(4,3) = value_type(-2.000422, 0.181931); B(4,4) = value_type(-0.765401,-1.004076);


	matrix_type S;
	matrix_type T;
	matrix_type Q;
	matrix_type Z;

	ublasx::qz_decompose(A, B, S, T, Q, Z, ublasx::rhp_qz_eigenvalues);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "B = " << B );
	BOOST_UBLASX_DEBUG_TRACE( "S = " << S );
	BOOST_UBLASX_DEBUG_TRACE( "T = " << T );
	BOOST_UBLASX_DEBUG_TRACE( "Q = " << Q );
	BOOST_UBLASX_DEBUG_TRACE( "Z = " << Z );
//	BOOST_UBLASX_DEBUG_TRACE( "Q*S*Z' = " << ublas::prod(ublas::prod(Q, S), ublas::herm(Z)) );
//	BOOST_UBLASX_DEBUG_TRACE( "Q*T*Z' = " << ublas::prod(ublas::prod(Q, T), ublas::herm(Z)) );
	matrix_type X;
	X = ublas::prod(Q, S);
	X = ublas::prod(X, ublas::herm(Z));
	BOOST_UBLASX_DEBUG_TRACE( "Q*S*Z' = " << X );
	X = ublas::prod(Q, T);
	X = ublas::prod(X, ublas::herm(Z));
	BOOST_UBLASX_DEBUG_TRACE( "Q*T*Z' = " << X );

	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(S) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(S) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(T) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(T) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(Q) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(Q) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(Z) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(Z) == n );
	X = ublas::prod(Q, S);
	X = ublas::prod(X, ublas::herm(Z));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, A, n, n, tol );
	X = ublas::prod(Q, T);
	X = ublas::prod(X, ublas::herm(Z));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, B, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_complex_column_major_decomp_udi_reorder )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Complex Matrix - Column Major - Decomposition and UDI Reordering");

	typedef std::complex<double> value_type;
	typedef ublas::matrix<value_type, ublas::column_major> matrix_type;

	const std::size_t n(5);

	matrix_type A(n,n);
	A(0,0) = value_type(-0.180557,-0.947835); A(0,1) = value_type( 0.322289, 1.603339); A(0,2) = value_type(-0.651789,-0.902809); A(0,3) = value_type( 0.793637, 0.031147); A(0,4) = value_type(-0.141086,-0.769742);
	A(1,0) = value_type( 0.729781, 0.693097); A(1,1) = value_type( 1.665989, 1.258667); A(1,2) = value_type( 0.620091,-0.192144); A(1,3) = value_type(-1.541503, 0.545104); A(1,4) = value_type( 0.146673,-1.553689);
	A(2,0) = value_type(-0.594370, 0.071316); A(2,1) = value_type( 0.494804,-2.381227); A(2,2) = value_type( 1.004784, 1.097852); A(2,3) = value_type(-0.221373,-0.585458); A(2,4) = value_type(-2.196082,-0.518945);
	A(3,0) = value_type(-1.106269,-1.015812); A(3,1) = value_type( 0.026697, 2.866180); A(3,2) = value_type( 2.687083,-0.115506); A(3,3) = value_type( 0.763162, 0.382183); A(3,4) = value_type( 1.203514, 0.203552);
	A(4,0) = value_type(-0.021184,-1.006383); A(4,1) = value_type(-0.882220, 0.232555); A(4,2) = value_type(-1.618234, 0.615528); A(4,3) = value_type( 1.119524,-2.178697); A(4,4) = value_type( 2.588165,-0.054936);

	matrix_type B(n,n);
	B(0,0) = value_type(-1.592710, 0.804795); B(0,1) = value_type( 0.057283, 0.470640); B(0,2) = value_type(-1.862275,-0.454486); B(0,3) = value_type( 0.712471, 0.887654); B(0,4) = value_type( 0.463207,-0.045117);
	B(1,0) = value_type( 1.072859,-2.009749); B(1,1) = value_type(-1.384371,-0.778200); B(1,2) = value_type( 0.777754, 1.245456); B(1,3) = value_type( 1.914787,-1.246236); B(1,4) = value_type( 0.082774, 0.358980);
	B(2,0) = value_type(-0.451744,-0.353845); B(2,1) = value_type(-0.131528,-2.236258); B(2,2) = value_type(-0.636187,-0.957378); B(2,3) = value_type( 0.984480,-1.840536); B(2,4) = value_type( 0.011728, 1.682497);
	B(3,0) = value_type(-0.876629, 0.764240); B(3,1) = value_type(-0.083787, 2.238476); B(3,2) = value_type( 0.474227,-2.426134); B(3,3) = value_type(-0.042328, 1.129135); B(3,4) = value_type(-0.529845, 0.653758);
	B(4,0) = value_type(-0.812610, 0.996806); B(4,1) = value_type( 0.142456, 0.101454); B(4,2) = value_type( 0.033739,-0.501220); B(4,3) = value_type(-2.000422, 0.181931); B(4,4) = value_type(-0.765401,-1.004076);


	matrix_type S;
	matrix_type T;
	matrix_type Q;
	matrix_type Z;

	ublasx::qz_decompose(A, B, S, T, Q, Z, ublasx::udi_qz_eigenvalues);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "B = " << B );
	BOOST_UBLASX_DEBUG_TRACE( "S = " << S );
	BOOST_UBLASX_DEBUG_TRACE( "T = " << T );
	BOOST_UBLASX_DEBUG_TRACE( "Q = " << Q );
	BOOST_UBLASX_DEBUG_TRACE( "Z = " << Z );
//	BOOST_UBLASX_DEBUG_TRACE( "Q*S*Z' = " << ublas::prod(ublas::prod(Q, S), ublas::herm(Z)) );
//	BOOST_UBLASX_DEBUG_TRACE( "Q*T*Z' = " << ublas::prod(ublas::prod(Q, T), ublas::herm(Z)) );
	matrix_type X;
	X = ublas::prod(Q, S);
	X = ublas::prod(X, ublas::herm(Z));
	BOOST_UBLASX_DEBUG_TRACE( "Q*S*Z' = " << X );
	X = ublas::prod(Q, T);
	X = ublas::prod(X, ublas::herm(Z));
	BOOST_UBLASX_DEBUG_TRACE( "Q*T*Z' = " << X );

	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(S) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(S) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(T) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(T) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(Q) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(Q) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(Z) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(Z) == n );
	X = ublas::prod(Q, S);
	X = ublas::prod(X, ublas::herm(Z));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, A, n, n, tol );
	X = ublas::prod(Q, T);
	X = ublas::prod(X, ublas::herm(Z));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, B, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_complex_row_major_decomp_udi_reorder )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Complex Matrix - Row Major - Decomposition and UDI Reordering");

	typedef std::complex<double> value_type;
	typedef ublas::matrix<value_type, ublas::row_major> matrix_type;

	const std::size_t n(5);

	matrix_type A(n,n);
	A(0,0) = value_type(-0.180557,-0.947835); A(0,1) = value_type( 0.322289, 1.603339); A(0,2) = value_type(-0.651789,-0.902809); A(0,3) = value_type( 0.793637, 0.031147); A(0,4) = value_type(-0.141086,-0.769742);
	A(1,0) = value_type( 0.729781, 0.693097); A(1,1) = value_type( 1.665989, 1.258667); A(1,2) = value_type( 0.620091,-0.192144); A(1,3) = value_type(-1.541503, 0.545104); A(1,4) = value_type( 0.146673,-1.553689);
	A(2,0) = value_type(-0.594370, 0.071316); A(2,1) = value_type( 0.494804,-2.381227); A(2,2) = value_type( 1.004784, 1.097852); A(2,3) = value_type(-0.221373,-0.585458); A(2,4) = value_type(-2.196082,-0.518945);
	A(3,0) = value_type(-1.106269,-1.015812); A(3,1) = value_type( 0.026697, 2.866180); A(3,2) = value_type( 2.687083,-0.115506); A(3,3) = value_type( 0.763162, 0.382183); A(3,4) = value_type( 1.203514, 0.203552);
	A(4,0) = value_type(-0.021184,-1.006383); A(4,1) = value_type(-0.882220, 0.232555); A(4,2) = value_type(-1.618234, 0.615528); A(4,3) = value_type( 1.119524,-2.178697); A(4,4) = value_type( 2.588165,-0.054936);

	matrix_type B(n,n);
	B(0,0) = value_type(-1.592710, 0.804795); B(0,1) = value_type( 0.057283, 0.470640); B(0,2) = value_type(-1.862275,-0.454486); B(0,3) = value_type( 0.712471, 0.887654); B(0,4) = value_type( 0.463207,-0.045117);
	B(1,0) = value_type( 1.072859,-2.009749); B(1,1) = value_type(-1.384371,-0.778200); B(1,2) = value_type( 0.777754, 1.245456); B(1,3) = value_type( 1.914787,-1.246236); B(1,4) = value_type( 0.082774, 0.358980);
	B(2,0) = value_type(-0.451744,-0.353845); B(2,1) = value_type(-0.131528,-2.236258); B(2,2) = value_type(-0.636187,-0.957378); B(2,3) = value_type( 0.984480,-1.840536); B(2,4) = value_type( 0.011728, 1.682497);
	B(3,0) = value_type(-0.876629, 0.764240); B(3,1) = value_type(-0.083787, 2.238476); B(3,2) = value_type( 0.474227,-2.426134); B(3,3) = value_type(-0.042328, 1.129135); B(3,4) = value_type(-0.529845, 0.653758);
	B(4,0) = value_type(-0.812610, 0.996806); B(4,1) = value_type( 0.142456, 0.101454); B(4,2) = value_type( 0.033739,-0.501220); B(4,3) = value_type(-2.000422, 0.181931); B(4,4) = value_type(-0.765401,-1.004076);


	matrix_type S;
	matrix_type T;
	matrix_type Q;
	matrix_type Z;

	ublasx::qz_decompose(A, B, S, T, Q, Z, ublasx::udi_qz_eigenvalues);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "B = " << B );
	BOOST_UBLASX_DEBUG_TRACE( "S = " << S );
	BOOST_UBLASX_DEBUG_TRACE( "T = " << T );
	BOOST_UBLASX_DEBUG_TRACE( "Q = " << Q );
	BOOST_UBLASX_DEBUG_TRACE( "Z = " << Z );
//	BOOST_UBLASX_DEBUG_TRACE( "Q*S*Z' = " << ublas::prod(ublas::prod(Q, S), ublas::herm(Z)) );
//	BOOST_UBLASX_DEBUG_TRACE( "Q*T*Z' = " << ublas::prod(ublas::prod(Q, T), ublas::herm(Z)) );
	matrix_type X;
	X = ublas::prod(Q, S);
	X = ublas::prod(X, ublas::herm(Z));
	BOOST_UBLASX_DEBUG_TRACE( "Q*S*Z' = " << X );
	X = ublas::prod(Q, T);
	X = ublas::prod(X, ublas::herm(Z));
	BOOST_UBLASX_DEBUG_TRACE( "Q*T*Z' = " << X );

	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(S) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(S) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(T) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(T) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(Q) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(Q) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(Z) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(Z) == n );
	X = ublas::prod(Q, S);
	X = ublas::prod(X, ublas::herm(Z));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, A, n, n, tol );
	X = ublas::prod(Q, T);
	X = ublas::prod(X, ublas::herm(Z));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, B, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_complex_column_major_decomp_udo_reorder )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Complex Matrix - Column Major - Decomposition and UDO Reordering");

	typedef std::complex<double> value_type;
	typedef ublas::matrix<value_type, ublas::column_major> matrix_type;

	const std::size_t n(5);

	matrix_type A(n,n);
	A(0,0) = value_type(-0.180557,-0.947835); A(0,1) = value_type( 0.322289, 1.603339); A(0,2) = value_type(-0.651789,-0.902809); A(0,3) = value_type( 0.793637, 0.031147); A(0,4) = value_type(-0.141086,-0.769742);
	A(1,0) = value_type( 0.729781, 0.693097); A(1,1) = value_type( 1.665989, 1.258667); A(1,2) = value_type( 0.620091,-0.192144); A(1,3) = value_type(-1.541503, 0.545104); A(1,4) = value_type( 0.146673,-1.553689);
	A(2,0) = value_type(-0.594370, 0.071316); A(2,1) = value_type( 0.494804,-2.381227); A(2,2) = value_type( 1.004784, 1.097852); A(2,3) = value_type(-0.221373,-0.585458); A(2,4) = value_type(-2.196082,-0.518945);
	A(3,0) = value_type(-1.106269,-1.015812); A(3,1) = value_type( 0.026697, 2.866180); A(3,2) = value_type( 2.687083,-0.115506); A(3,3) = value_type( 0.763162, 0.382183); A(3,4) = value_type( 1.203514, 0.203552);
	A(4,0) = value_type(-0.021184,-1.006383); A(4,1) = value_type(-0.882220, 0.232555); A(4,2) = value_type(-1.618234, 0.615528); A(4,3) = value_type( 1.119524,-2.178697); A(4,4) = value_type( 2.588165,-0.054936);

	matrix_type B(n,n);
	B(0,0) = value_type(-1.592710, 0.804795); B(0,1) = value_type( 0.057283, 0.470640); B(0,2) = value_type(-1.862275,-0.454486); B(0,3) = value_type( 0.712471, 0.887654); B(0,4) = value_type( 0.463207,-0.045117);
	B(1,0) = value_type( 1.072859,-2.009749); B(1,1) = value_type(-1.384371,-0.778200); B(1,2) = value_type( 0.777754, 1.245456); B(1,3) = value_type( 1.914787,-1.246236); B(1,4) = value_type( 0.082774, 0.358980);
	B(2,0) = value_type(-0.451744,-0.353845); B(2,1) = value_type(-0.131528,-2.236258); B(2,2) = value_type(-0.636187,-0.957378); B(2,3) = value_type( 0.984480,-1.840536); B(2,4) = value_type( 0.011728, 1.682497);
	B(3,0) = value_type(-0.876629, 0.764240); B(3,1) = value_type(-0.083787, 2.238476); B(3,2) = value_type( 0.474227,-2.426134); B(3,3) = value_type(-0.042328, 1.129135); B(3,4) = value_type(-0.529845, 0.653758);
	B(4,0) = value_type(-0.812610, 0.996806); B(4,1) = value_type( 0.142456, 0.101454); B(4,2) = value_type( 0.033739,-0.501220); B(4,3) = value_type(-2.000422, 0.181931); B(4,4) = value_type(-0.765401,-1.004076);


	matrix_type S;
	matrix_type T;
	matrix_type Q;
	matrix_type Z;

	ublasx::qz_decompose(A, B, S, T, Q, Z, ublasx::udo_qz_eigenvalues);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "B = " << B );
	BOOST_UBLASX_DEBUG_TRACE( "S = " << S );
	BOOST_UBLASX_DEBUG_TRACE( "T = " << T );
	BOOST_UBLASX_DEBUG_TRACE( "Q = " << Q );
	BOOST_UBLASX_DEBUG_TRACE( "Z = " << Z );
//	BOOST_UBLASX_DEBUG_TRACE( "Q*S*Z' = " << ublas::prod(ublas::prod(Q, S), ublas::herm(Z)) );
//	BOOST_UBLASX_DEBUG_TRACE( "Q*T*Z' = " << ublas::prod(ublas::prod(Q, T), ublas::herm(Z)) );
	matrix_type X;
	X = ublas::prod(Q, S);
	X = ublas::prod(X, ublas::herm(Z));
	BOOST_UBLASX_DEBUG_TRACE( "Q*S*Z' = " << X );
	X = ublas::prod(Q, T);
	X = ublas::prod(X, ublas::herm(Z));
	BOOST_UBLASX_DEBUG_TRACE( "Q*T*Z' = " << X );

	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(S) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(S) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(T) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(T) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(Q) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(Q) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(Z) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(Z) == n );
	X = ublas::prod(Q, S);
	X = ublas::prod(X, ublas::herm(Z));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, A, n, n, tol );
	X = ublas::prod(Q, T);
	X = ublas::prod(X, ublas::herm(Z));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, B, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_complex_row_major_decomp_udo_reorder )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Complex Matrix - Row Major - Decomposition and UDO Reordering");

	typedef std::complex<double> value_type;
	typedef ublas::matrix<value_type, ublas::row_major> matrix_type;

	const std::size_t n(5);

	matrix_type A(n,n);
	A(0,0) = value_type(-0.180557,-0.947835); A(0,1) = value_type( 0.322289, 1.603339); A(0,2) = value_type(-0.651789,-0.902809); A(0,3) = value_type( 0.793637, 0.031147); A(0,4) = value_type(-0.141086,-0.769742);
	A(1,0) = value_type( 0.729781, 0.693097); A(1,1) = value_type( 1.665989, 1.258667); A(1,2) = value_type( 0.620091,-0.192144); A(1,3) = value_type(-1.541503, 0.545104); A(1,4) = value_type( 0.146673,-1.553689);
	A(2,0) = value_type(-0.594370, 0.071316); A(2,1) = value_type( 0.494804,-2.381227); A(2,2) = value_type( 1.004784, 1.097852); A(2,3) = value_type(-0.221373,-0.585458); A(2,4) = value_type(-2.196082,-0.518945);
	A(3,0) = value_type(-1.106269,-1.015812); A(3,1) = value_type( 0.026697, 2.866180); A(3,2) = value_type( 2.687083,-0.115506); A(3,3) = value_type( 0.763162, 0.382183); A(3,4) = value_type( 1.203514, 0.203552);
	A(4,0) = value_type(-0.021184,-1.006383); A(4,1) = value_type(-0.882220, 0.232555); A(4,2) = value_type(-1.618234, 0.615528); A(4,3) = value_type( 1.119524,-2.178697); A(4,4) = value_type( 2.588165,-0.054936);

	matrix_type B(n,n);
	B(0,0) = value_type(-1.592710, 0.804795); B(0,1) = value_type( 0.057283, 0.470640); B(0,2) = value_type(-1.862275,-0.454486); B(0,3) = value_type( 0.712471, 0.887654); B(0,4) = value_type( 0.463207,-0.045117);
	B(1,0) = value_type( 1.072859,-2.009749); B(1,1) = value_type(-1.384371,-0.778200); B(1,2) = value_type( 0.777754, 1.245456); B(1,3) = value_type( 1.914787,-1.246236); B(1,4) = value_type( 0.082774, 0.358980);
	B(2,0) = value_type(-0.451744,-0.353845); B(2,1) = value_type(-0.131528,-2.236258); B(2,2) = value_type(-0.636187,-0.957378); B(2,3) = value_type( 0.984480,-1.840536); B(2,4) = value_type( 0.011728, 1.682497);
	B(3,0) = value_type(-0.876629, 0.764240); B(3,1) = value_type(-0.083787, 2.238476); B(3,2) = value_type( 0.474227,-2.426134); B(3,3) = value_type(-0.042328, 1.129135); B(3,4) = value_type(-0.529845, 0.653758);
	B(4,0) = value_type(-0.812610, 0.996806); B(4,1) = value_type( 0.142456, 0.101454); B(4,2) = value_type( 0.033739,-0.501220); B(4,3) = value_type(-2.000422, 0.181931); B(4,4) = value_type(-0.765401,-1.004076);


	matrix_type S;
	matrix_type T;
	matrix_type Q;
	matrix_type Z;

	ublasx::qz_decompose(A, B, S, T, Q, Z, ublasx::udo_qz_eigenvalues);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "B = " << B );
	BOOST_UBLASX_DEBUG_TRACE( "S = " << S );
	BOOST_UBLASX_DEBUG_TRACE( "T = " << T );
	BOOST_UBLASX_DEBUG_TRACE( "Q = " << Q );
	BOOST_UBLASX_DEBUG_TRACE( "Z = " << Z );
//	BOOST_UBLASX_DEBUG_TRACE( "Q*S*Z' = " << ublas::prod(ublas::prod(Q, S), ublas::herm(Z)) );
//	BOOST_UBLASX_DEBUG_TRACE( "Q*T*Z' = " << ublas::prod(ublas::prod(Q, T), ublas::herm(Z)) );
	matrix_type X;
	X = ublas::prod(Q, S);
	X = ublas::prod(X, ublas::herm(Z));
	BOOST_UBLASX_DEBUG_TRACE( "Q*S*Z' = " << X );
	X = ublas::prod(Q, T);
	X = ublas::prod(X, ublas::herm(Z));
	BOOST_UBLASX_DEBUG_TRACE( "Q*T*Z' = " << X );

	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(S) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(S) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(T) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(T) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(Q) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(Q) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(Z) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(Z) == n );
	X = ublas::prod(Q, S);
	X = ublas::prod(X, ublas::herm(Z));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, A, n, n, tol );
	X = ublas::prod(Q, T);
	X = ublas::prod(X, ublas::herm(Z));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, B, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_complex_column_major_decomp_custom_reorder )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Complex Matrix - Column Major - Decomposition and Custom Reordering");

	typedef std::complex<double> value_type;
	typedef ublas::matrix<value_type, ublas::column_major> matrix_type;

	const std::size_t n(5);

	matrix_type A(n,n);
	A(0,0) = value_type(-0.180557,-0.947835); A(0,1) = value_type( 0.322289, 1.603339); A(0,2) = value_type(-0.651789,-0.902809); A(0,3) = value_type( 0.793637, 0.031147); A(0,4) = value_type(-0.141086,-0.769742);
	A(1,0) = value_type( 0.729781, 0.693097); A(1,1) = value_type( 1.665989, 1.258667); A(1,2) = value_type( 0.620091,-0.192144); A(1,3) = value_type(-1.541503, 0.545104); A(1,4) = value_type( 0.146673,-1.553689);
	A(2,0) = value_type(-0.594370, 0.071316); A(2,1) = value_type( 0.494804,-2.381227); A(2,2) = value_type( 1.004784, 1.097852); A(2,3) = value_type(-0.221373,-0.585458); A(2,4) = value_type(-2.196082,-0.518945);
	A(3,0) = value_type(-1.106269,-1.015812); A(3,1) = value_type( 0.026697, 2.866180); A(3,2) = value_type( 2.687083,-0.115506); A(3,3) = value_type( 0.763162, 0.382183); A(3,4) = value_type( 1.203514, 0.203552);
	A(4,0) = value_type(-0.021184,-1.006383); A(4,1) = value_type(-0.882220, 0.232555); A(4,2) = value_type(-1.618234, 0.615528); A(4,3) = value_type( 1.119524,-2.178697); A(4,4) = value_type( 2.588165,-0.054936);

	matrix_type B(n,n);
	B(0,0) = value_type(-1.592710, 0.804795); B(0,1) = value_type( 0.057283, 0.470640); B(0,2) = value_type(-1.862275,-0.454486); B(0,3) = value_type( 0.712471, 0.887654); B(0,4) = value_type( 0.463207,-0.045117);
	B(1,0) = value_type( 1.072859,-2.009749); B(1,1) = value_type(-1.384371,-0.778200); B(1,2) = value_type( 0.777754, 1.245456); B(1,3) = value_type( 1.914787,-1.246236); B(1,4) = value_type( 0.082774, 0.358980);
	B(2,0) = value_type(-0.451744,-0.353845); B(2,1) = value_type(-0.131528,-2.236258); B(2,2) = value_type(-0.636187,-0.957378); B(2,3) = value_type( 0.984480,-1.840536); B(2,4) = value_type( 0.011728, 1.682497);
	B(3,0) = value_type(-0.876629, 0.764240); B(3,1) = value_type(-0.083787, 2.238476); B(3,2) = value_type( 0.474227,-2.426134); B(3,3) = value_type(-0.042328, 1.129135); B(3,4) = value_type(-0.529845, 0.653758);
	B(4,0) = value_type(-0.812610, 0.996806); B(4,1) = value_type( 0.142456, 0.101454); B(4,2) = value_type( 0.033739,-0.501220); B(4,3) = value_type(-2.000422, 0.181931); B(4,4) = value_type(-0.765401,-1.004076);


	matrix_type S;
	matrix_type T;
	matrix_type Q;
	matrix_type Z;

	ublasx::qz_decompose(A, B, S, T, Q, Z);

	// Selection: select eigenvalues at even position (no sense ... just a test)
	ublas::vector<bool> selection(n);
	for (std::size_t i = 0; i < n; ++i)
	{
		selection(i) = ((i+1) % 2) ? true : false;
	}

	matrix_type SS;
	matrix_type TS;
	matrix_type QS;
	matrix_type ZS;

	ublasx::qz_reorder(S, T, Q, Z, selection, SS, TS, QS, ZS);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "B = " << B );
	BOOST_UBLASX_DEBUG_TRACE( "S = " << S );
	BOOST_UBLASX_DEBUG_TRACE( "T = " << T );
	BOOST_UBLASX_DEBUG_TRACE( "Q = " << Q );
	BOOST_UBLASX_DEBUG_TRACE( "Z = " << Z );
//	BOOST_UBLASX_DEBUG_TRACE( "Q*S*Z' = " << ublas::prod(ublas::prod(Q, S), ublas::herm(Z)) );
//	BOOST_UBLASX_DEBUG_TRACE( "Q*T*Z' = " << ublas::prod(ublas::prod(Q, T), ublas::herm(Z)) );
	matrix_type X;
	X = ublas::prod(Q, S);
	X = ublas::prod(X, ublas::herm(Z));
	BOOST_UBLASX_DEBUG_TRACE( "Q*S*Z' = " << X );
	X = ublas::prod(Q, T);
	X = ublas::prod(X, ublas::herm(Z));
	BOOST_UBLASX_DEBUG_TRACE( "Q*T*Z' = " << X );

	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(S) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(S) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(T) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(T) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(Q) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(Q) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(Z) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(Z) == n );
	X = ublas::prod(Q, S);
	X = ublas::prod(X, ublas::herm(Z));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, A, n, n, tol );
	X = ublas::prod(Q, T);
	X = ublas::prod(X, ublas::herm(Z));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, B, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_complex_row_major_decomp_custom_reorder )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Complex Matrix - Row Major - Decomposition and Custom Reordering");

	typedef std::complex<double> value_type;
	typedef ublas::matrix<value_type, ublas::row_major> matrix_type;

	const std::size_t n(5);

	matrix_type A(n,n);
	A(0,0) = value_type(-0.180557,-0.947835); A(0,1) = value_type( 0.322289, 1.603339); A(0,2) = value_type(-0.651789,-0.902809); A(0,3) = value_type( 0.793637, 0.031147); A(0,4) = value_type(-0.141086,-0.769742);
	A(1,0) = value_type( 0.729781, 0.693097); A(1,1) = value_type( 1.665989, 1.258667); A(1,2) = value_type( 0.620091,-0.192144); A(1,3) = value_type(-1.541503, 0.545104); A(1,4) = value_type( 0.146673,-1.553689);
	A(2,0) = value_type(-0.594370, 0.071316); A(2,1) = value_type( 0.494804,-2.381227); A(2,2) = value_type( 1.004784, 1.097852); A(2,3) = value_type(-0.221373,-0.585458); A(2,4) = value_type(-2.196082,-0.518945);
	A(3,0) = value_type(-1.106269,-1.015812); A(3,1) = value_type( 0.026697, 2.866180); A(3,2) = value_type( 2.687083,-0.115506); A(3,3) = value_type( 0.763162, 0.382183); A(3,4) = value_type( 1.203514, 0.203552);
	A(4,0) = value_type(-0.021184,-1.006383); A(4,1) = value_type(-0.882220, 0.232555); A(4,2) = value_type(-1.618234, 0.615528); A(4,3) = value_type( 1.119524,-2.178697); A(4,4) = value_type( 2.588165,-0.054936);

	matrix_type B(n,n);
	B(0,0) = value_type(-1.592710, 0.804795); B(0,1) = value_type( 0.057283, 0.470640); B(0,2) = value_type(-1.862275,-0.454486); B(0,3) = value_type( 0.712471, 0.887654); B(0,4) = value_type( 0.463207,-0.045117);
	B(1,0) = value_type( 1.072859,-2.009749); B(1,1) = value_type(-1.384371,-0.778200); B(1,2) = value_type( 0.777754, 1.245456); B(1,3) = value_type( 1.914787,-1.246236); B(1,4) = value_type( 0.082774, 0.358980);
	B(2,0) = value_type(-0.451744,-0.353845); B(2,1) = value_type(-0.131528,-2.236258); B(2,2) = value_type(-0.636187,-0.957378); B(2,3) = value_type( 0.984480,-1.840536); B(2,4) = value_type( 0.011728, 1.682497);
	B(3,0) = value_type(-0.876629, 0.764240); B(3,1) = value_type(-0.083787, 2.238476); B(3,2) = value_type( 0.474227,-2.426134); B(3,3) = value_type(-0.042328, 1.129135); B(3,4) = value_type(-0.529845, 0.653758);
	B(4,0) = value_type(-0.812610, 0.996806); B(4,1) = value_type( 0.142456, 0.101454); B(4,2) = value_type( 0.033739,-0.501220); B(4,3) = value_type(-2.000422, 0.181931); B(4,4) = value_type(-0.765401,-1.004076);


	matrix_type S;
	matrix_type T;
	matrix_type Q;
	matrix_type Z;

	ublasx::qz_decompose(A, B, S, T, Q, Z);

	// Selection: select eigenvalues at even position (no sense ... just a test)
	ublas::vector<bool> selection(n);
	for (std::size_t i = 0; i < n; ++i)
	{
		selection(i) = ((i+1) % 2) ? true : false;
	}

	matrix_type SS;
	matrix_type TS;
	matrix_type QS;
	matrix_type ZS;

	ublasx::qz_reorder(S, T, Q, Z, selection, SS, TS, QS, ZS);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "B = " << B );
	BOOST_UBLASX_DEBUG_TRACE( "S = " << S );
	BOOST_UBLASX_DEBUG_TRACE( "T = " << T );
	BOOST_UBLASX_DEBUG_TRACE( "Q = " << Q );
	BOOST_UBLASX_DEBUG_TRACE( "Z = " << Z );
//	BOOST_UBLASX_DEBUG_TRACE( "Q*S*Z' = " << ublas::prod(ublas::prod(Q, S), ublas::herm(Z)) );
//	BOOST_UBLASX_DEBUG_TRACE( "Q*T*Z' = " << ublas::prod(ublas::prod(Q, T), ublas::herm(Z)) );
	matrix_type X;
	X = ublas::prod(Q, S);
	X = ublas::prod(X, ublas::herm(Z));
	BOOST_UBLASX_DEBUG_TRACE( "Q*S*Z' = " << X );
	X = ublas::prod(Q, T);
	X = ublas::prod(X, ublas::herm(Z));
	BOOST_UBLASX_DEBUG_TRACE( "Q*T*Z' = " << X );

	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(S) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(S) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(T) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(T) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(Q) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(Q) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(Z) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(Z) == n );
	X = ublas::prod(Q, S);
	X = ublas::prod(X, ublas::herm(Z));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, A, n, n, tol );
	X = ublas::prod(Q, T);
	X = ublas::prod(X, ublas::herm(Z));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, B, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_real_column_major_oo_decomp )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Real Matrix - Column Major - QZ Object - Decomposition");

	typedef double value_type;
	typedef ublas::matrix<value_type, ublas::column_major> matrix_type;

	const std::size_t n(5);

	matrix_type A(n,n);
	A(0,0) = -0.180557; A(0,1) =  0.322289; A(0,2) = -0.651789; A(0,3) =  0.793637; A(0,4) = -0.141086;
	A(1,0) =  0.729781; A(1,1) =  1.665989; A(1,2) =  0.620091; A(1,3) = -1.541503; A(1,4) =  0.146673;
	A(2,0) = -0.594370; A(2,1) =  0.494804; A(2,2) =  1.004784; A(2,3) = -0.221373; A(2,4) = -2.196082;
	A(3,0) = -1.106269; A(3,1) =  0.026697; A(3,2) =  2.687083; A(3,3) =  0.763162; A(3,4) =  1.203514;
	A(4,0) = -0.021184; A(4,1) = -0.882220; A(4,2) = -1.618234; A(4,3) =  1.119524; A(4,4) =  2.588165;

	matrix_type B(n,n);
	B(0,0) = -1.592710; B(0,1) =  0.057283; B(0,2) = -1.862275; B(0,3) =  0.712471; B(0,4) =  0.463207;
	B(1,0) =  1.072859; B(1,1) = -1.384371; B(1,2) =  0.777754; B(1,3) =  1.914787; B(1,4) =  0.082774;
	B(2,0) = -0.451744; B(2,1) = -0.131528; B(2,2) = -0.636187; B(2,3) =  0.984480; B(2,4) =  0.011728;
	B(3,0) = -0.876629; B(3,1) = -0.083787; B(3,2) =  0.474227; B(3,3) = -0.042328; B(3,4) = -0.529845;
	B(4,0) = -0.812610; B(4,1) =  0.142456; B(4,2) =  0.033739; B(4,3) = -2.000422; B(4,4) = -0.765401;


	ublasx::qz_decomposition<value_type> qz;

	qz.decompose(A, B);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "B = " << B );
	BOOST_UBLASX_DEBUG_TRACE( "S = " << qz.S() );
	BOOST_UBLASX_DEBUG_TRACE( "T = " << qz.T() );
	BOOST_UBLASX_DEBUG_TRACE( "Q = " << qz.Q() );
	BOOST_UBLASX_DEBUG_TRACE( "Z = " << qz.Z() );
	BOOST_UBLASX_DEBUG_TRACE( "alpha = " << qz.alpha() );
	BOOST_UBLASX_DEBUG_TRACE( "beta = " << qz.beta() );
	BOOST_UBLASX_DEBUG_TRACE( "eigenvalues lambda = " << qz.eigenvalues() );
	BOOST_UBLASX_DEBUG_TRACE( "right eigenvectors V = " << qz.right_eigenvectors() );
	BOOST_UBLASX_DEBUG_TRACE( "left eigenvectors W = " << qz.left_eigenvectors() );
	matrix_type X;
//	vector_type x;
//	vector_type y;
	X = ublas::prod(qz.Q(), qz.S());
	X = ublas::prod(X, ublas::trans(qz.Z()));
	BOOST_UBLASX_DEBUG_TRACE( "Q*S*Z' = " << X );
	X = ublas::prod(qz.Q(), qz.T());
	X = ublas::prod(X, ublas::trans(qz.Z()));
	BOOST_UBLASX_DEBUG_TRACE( "Q*T*Z' = " << X );
//	x = ublas::prod(ublas::prod(A, qz.right_eigenvectors()), ublasx::diag(qz.T()));
//	y = ublas::prod(ublas::prod(B, qz.right_eigenvectors()), ublasx::diag(qz.S()));
//	BOOST_UBLASX_DEBUG_TRACE( "A*V*alpha = " << x );
//	BOOST_UBLASX_DEBUG_TRACE( "B*V*beta = " << y );
//	x = ublas::prod(ublas::prod(ublasx::diag(qz.T()), ublas::trans(qz.left_eigenvectors())), A);
//	y = ublas::prod(ublas::prod(ublasx::diag(qz.S()), ublas::trans(qz.left_eigenvectors())), B);
//	BOOST_UBLASX_DEBUG_TRACE( "beta'*W'*A = " << x );
//	BOOST_UBLASX_DEBUG_TRACE( "alpha'*W'*B = " << y );

	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.S()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.S()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.T()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.T()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.Q()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.Q()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.Z()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.Z()) == n );
	X = ublas::prod(qz.Q(), qz.S());
	X = ublas::prod(X, ublas::trans(qz.Z()));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, A, n, n, tol );
	X = ublas::prod(qz.Q(), qz.T());
	X = ublas::prod(X, ublas::trans(qz.Z()));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, B, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_real_row_major_oo_decomp )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Real Matrix - Row Major - QZ Object - Decomposition");

	typedef double value_type;
	typedef ublas::matrix<value_type, ublas::row_major> matrix_type;

	const std::size_t n(5);

	matrix_type A(n,n);
	A(0,0) = -0.180557; A(0,1) =  0.322289; A(0,2) = -0.651789; A(0,3) =  0.793637; A(0,4) = -0.141086;
	A(1,0) =  0.729781; A(1,1) =  1.665989; A(1,2) =  0.620091; A(1,3) = -1.541503; A(1,4) =  0.146673;
	A(2,0) = -0.594370; A(2,1) =  0.494804; A(2,2) =  1.004784; A(2,3) = -0.221373; A(2,4) = -2.196082;
	A(3,0) = -1.106269; A(3,1) =  0.026697; A(3,2) =  2.687083; A(3,3) =  0.763162; A(3,4) =  1.203514;
	A(4,0) = -0.021184; A(4,1) = -0.882220; A(4,2) = -1.618234; A(4,3) =  1.119524; A(4,4) =  2.588165;

	matrix_type B(n,n);
	B(0,0) = -1.592710; B(0,1) =  0.057283; B(0,2) = -1.862275; B(0,3) =  0.712471; B(0,4) =  0.463207;
	B(1,0) =  1.072859; B(1,1) = -1.384371; B(1,2) =  0.777754; B(1,3) =  1.914787; B(1,4) =  0.082774;
	B(2,0) = -0.451744; B(2,1) = -0.131528; B(2,2) = -0.636187; B(2,3) =  0.984480; B(2,4) =  0.011728;
	B(3,0) = -0.876629; B(3,1) = -0.083787; B(3,2) =  0.474227; B(3,3) = -0.042328; B(3,4) = -0.529845;
	B(4,0) = -0.812610; B(4,1) =  0.142456; B(4,2) =  0.033739; B(4,3) = -2.000422; B(4,4) = -0.765401;


	ublasx::qz_decomposition<value_type> qz;

	qz.decompose(A, B);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "B = " << B );
	BOOST_UBLASX_DEBUG_TRACE( "S = " << qz.S() );
	BOOST_UBLASX_DEBUG_TRACE( "T = " << qz.T() );
	BOOST_UBLASX_DEBUG_TRACE( "Q = " << qz.Q() );
	BOOST_UBLASX_DEBUG_TRACE( "Z = " << qz.Z() );
	BOOST_UBLASX_DEBUG_TRACE( "alpha = " << qz.alpha() );
	BOOST_UBLASX_DEBUG_TRACE( "beta = " << qz.beta() );
	BOOST_UBLASX_DEBUG_TRACE( "eigenvalues lambda = " << qz.eigenvalues() );
	BOOST_UBLASX_DEBUG_TRACE( "right eigenvectors V = " << qz.right_eigenvectors() );
	BOOST_UBLASX_DEBUG_TRACE( "left eigenvectors W = " << qz.left_eigenvectors() );
	matrix_type X;
	X = ublas::prod(qz.Q(), qz.S());
	X = ublas::prod(X, ublas::trans(qz.Z()));
	BOOST_UBLASX_DEBUG_TRACE( "Q*S*Z' = " << X );
	X = ublas::prod(qz.Q(), qz.T());
	X = ublas::prod(X, ublas::trans(qz.Z()));
	BOOST_UBLASX_DEBUG_TRACE( "Q*T*Z' = " << X );

	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.S()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.S()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.T()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.T()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.Q()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.Q()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.Z()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.Z()) == n );
	X = ublas::prod(qz.Q(), qz.S());
	X = ublas::prod(X, ublas::trans(qz.Z()));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, A, n, n, tol );
	X = ublas::prod(qz.Q(), qz.T());
	X = ublas::prod(X, ublas::trans(qz.Z()));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, B, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_real_column_major_oo_lhp_reorder )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Real Matrix - Column Major - QZ Object - LHP Reordering");

	typedef double value_type;
	typedef ublas::matrix<value_type, ublas::column_major> matrix_type;

	const std::size_t n(5);

	matrix_type A(n,n);
	A(0,0) = -0.180557; A(0,1) =  0.322289; A(0,2) = -0.651789; A(0,3) =  0.793637; A(0,4) = -0.141086;
	A(1,0) =  0.729781; A(1,1) =  1.665989; A(1,2) =  0.620091; A(1,3) = -1.541503; A(1,4) =  0.146673;
	A(2,0) = -0.594370; A(2,1) =  0.494804; A(2,2) =  1.004784; A(2,3) = -0.221373; A(2,4) = -2.196082;
	A(3,0) = -1.106269; A(3,1) =  0.026697; A(3,2) =  2.687083; A(3,3) =  0.763162; A(3,4) =  1.203514;
	A(4,0) = -0.021184; A(4,1) = -0.882220; A(4,2) = -1.618234; A(4,3) =  1.119524; A(4,4) =  2.588165;

	matrix_type B(n,n);
	B(0,0) = -1.592710; B(0,1) =  0.057283; B(0,2) = -1.862275; B(0,3) =  0.712471; B(0,4) =  0.463207;
	B(1,0) =  1.072859; B(1,1) = -1.384371; B(1,2) =  0.777754; B(1,3) =  1.914787; B(1,4) =  0.082774;
	B(2,0) = -0.451744; B(2,1) = -0.131528; B(2,2) = -0.636187; B(2,3) =  0.984480; B(2,4) =  0.011728;
	B(3,0) = -0.876629; B(3,1) = -0.083787; B(3,2) =  0.474227; B(3,3) = -0.042328; B(3,4) = -0.529845;
	B(4,0) = -0.812610; B(4,1) =  0.142456; B(4,2) =  0.033739; B(4,3) = -2.000422; B(4,4) = -0.765401;


	ublasx::qz_decomposition<value_type> qz;

	qz.decompose(A, B);
	qz.reorder(ublasx::lhp_qz_eigenvalues);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "B = " << B );
	BOOST_UBLASX_DEBUG_TRACE( "S = " << qz.S() );
	BOOST_UBLASX_DEBUG_TRACE( "T = " << qz.T() );
	BOOST_UBLASX_DEBUG_TRACE( "Q = " << qz.Q() );
	BOOST_UBLASX_DEBUG_TRACE( "Z = " << qz.Z() );
	BOOST_UBLASX_DEBUG_TRACE( "alpha = " << qz.alpha() );
	BOOST_UBLASX_DEBUG_TRACE( "beta = " << qz.beta() );
	BOOST_UBLASX_DEBUG_TRACE( "eigenvalues lambda = " << qz.eigenvalues() );
	BOOST_UBLASX_DEBUG_TRACE( "right eigenvectors V = " << qz.right_eigenvectors() );
	BOOST_UBLASX_DEBUG_TRACE( "left eigenvectors W = " << qz.left_eigenvectors() );
	matrix_type X;
	X = ublas::prod(qz.Q(), qz.S());
	X = ublas::prod(X, ublas::trans(qz.Z()));
	BOOST_UBLASX_DEBUG_TRACE( "Q*S*Z' = " << X );
	X = ublas::prod(qz.Q(), qz.T());
	X = ublas::prod(X, ublas::trans(qz.Z()));
	BOOST_UBLASX_DEBUG_TRACE( "Q*T*Z' = " << X );

	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.S()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.S()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.T()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.T()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.Q()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.Q()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.Z()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.Z()) == n );
	X = ublas::prod(qz.Q(), qz.S());
	X = ublas::prod(X, ublas::trans(qz.Z()));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, A, n, n, tol );
	X = ublas::prod(qz.Q(), qz.T());
	X = ublas::prod(X, ublas::trans(qz.Z()));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, B, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_real_row_major_oo_lhp_reorder )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Real Matrix - Row Major - QZ Object - LHP Reordering");

	typedef double value_type;
	typedef ublas::matrix<value_type, ublas::row_major> matrix_type;

	const std::size_t n(5);

	matrix_type A(n,n);
	A(0,0) = -0.180557; A(0,1) =  0.322289; A(0,2) = -0.651789; A(0,3) =  0.793637; A(0,4) = -0.141086;
	A(1,0) =  0.729781; A(1,1) =  1.665989; A(1,2) =  0.620091; A(1,3) = -1.541503; A(1,4) =  0.146673;
	A(2,0) = -0.594370; A(2,1) =  0.494804; A(2,2) =  1.004784; A(2,3) = -0.221373; A(2,4) = -2.196082;
	A(3,0) = -1.106269; A(3,1) =  0.026697; A(3,2) =  2.687083; A(3,3) =  0.763162; A(3,4) =  1.203514;
	A(4,0) = -0.021184; A(4,1) = -0.882220; A(4,2) = -1.618234; A(4,3) =  1.119524; A(4,4) =  2.588165;

	matrix_type B(n,n);
	B(0,0) = -1.592710; B(0,1) =  0.057283; B(0,2) = -1.862275; B(0,3) =  0.712471; B(0,4) =  0.463207;
	B(1,0) =  1.072859; B(1,1) = -1.384371; B(1,2) =  0.777754; B(1,3) =  1.914787; B(1,4) =  0.082774;
	B(2,0) = -0.451744; B(2,1) = -0.131528; B(2,2) = -0.636187; B(2,3) =  0.984480; B(2,4) =  0.011728;
	B(3,0) = -0.876629; B(3,1) = -0.083787; B(3,2) =  0.474227; B(3,3) = -0.042328; B(3,4) = -0.529845;
	B(4,0) = -0.812610; B(4,1) =  0.142456; B(4,2) =  0.033739; B(4,3) = -2.000422; B(4,4) = -0.765401;


	ublasx::qz_decomposition<value_type> qz;

	qz.decompose(A, B);
	qz.reorder(ublasx::lhp_qz_eigenvalues);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "B = " << B );
	BOOST_UBLASX_DEBUG_TRACE( "S = " << qz.S() );
	BOOST_UBLASX_DEBUG_TRACE( "T = " << qz.T() );
	BOOST_UBLASX_DEBUG_TRACE( "Q = " << qz.Q() );
	BOOST_UBLASX_DEBUG_TRACE( "Z = " << qz.Z() );
	BOOST_UBLASX_DEBUG_TRACE( "alpha = " << qz.alpha() );
	BOOST_UBLASX_DEBUG_TRACE( "beta = " << qz.beta() );
	BOOST_UBLASX_DEBUG_TRACE( "eigenvalues lambda = " << qz.eigenvalues() );
	BOOST_UBLASX_DEBUG_TRACE( "right eigenvectors V = " << qz.right_eigenvectors() );
	BOOST_UBLASX_DEBUG_TRACE( "left eigenvectors W = " << qz.left_eigenvectors() );
	matrix_type X;
	X = ublas::prod(qz.Q(), qz.S());
	X = ublas::prod(X, ublas::trans(qz.Z()));
	BOOST_UBLASX_DEBUG_TRACE( "Q*S*Z' = " << X );
	X = ublas::prod(qz.Q(), qz.T());
	X = ublas::prod(X, ublas::trans(qz.Z()));
	BOOST_UBLASX_DEBUG_TRACE( "Q*T*Z' = " << X );

	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.S()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.S()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.T()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.T()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.Q()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.Q()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.Z()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.Z()) == n );
	X = ublas::prod(qz.Q(), qz.S());
	X = ublas::prod(X, ublas::trans(qz.Z()));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, A, n, n, tol );
	X = ublas::prod(qz.Q(), qz.T());
	X = ublas::prod(X, ublas::trans(qz.Z()));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, B, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_real_column_major_oo_rhp_reorder )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Real Matrix - Column Major - QZ Object - RHP Reordering");

	typedef double value_type;
	typedef ublas::matrix<value_type, ublas::column_major> matrix_type;

	const std::size_t n(5);

	matrix_type A(n,n);
	A(0,0) = -0.180557; A(0,1) =  0.322289; A(0,2) = -0.651789; A(0,3) =  0.793637; A(0,4) = -0.141086;
	A(1,0) =  0.729781; A(1,1) =  1.665989; A(1,2) =  0.620091; A(1,3) = -1.541503; A(1,4) =  0.146673;
	A(2,0) = -0.594370; A(2,1) =  0.494804; A(2,2) =  1.004784; A(2,3) = -0.221373; A(2,4) = -2.196082;
	A(3,0) = -1.106269; A(3,1) =  0.026697; A(3,2) =  2.687083; A(3,3) =  0.763162; A(3,4) =  1.203514;
	A(4,0) = -0.021184; A(4,1) = -0.882220; A(4,2) = -1.618234; A(4,3) =  1.119524; A(4,4) =  2.588165;

	matrix_type B(n,n);
	B(0,0) = -1.592710; B(0,1) =  0.057283; B(0,2) = -1.862275; B(0,3) =  0.712471; B(0,4) =  0.463207;
	B(1,0) =  1.072859; B(1,1) = -1.384371; B(1,2) =  0.777754; B(1,3) =  1.914787; B(1,4) =  0.082774;
	B(2,0) = -0.451744; B(2,1) = -0.131528; B(2,2) = -0.636187; B(2,3) =  0.984480; B(2,4) =  0.011728;
	B(3,0) = -0.876629; B(3,1) = -0.083787; B(3,2) =  0.474227; B(3,3) = -0.042328; B(3,4) = -0.529845;
	B(4,0) = -0.812610; B(4,1) =  0.142456; B(4,2) =  0.033739; B(4,3) = -2.000422; B(4,4) = -0.765401;


	ublasx::qz_decomposition<value_type> qz;

	qz.decompose(A, B);
	qz.reorder(ublasx::rhp_qz_eigenvalues);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "B = " << B );
	BOOST_UBLASX_DEBUG_TRACE( "S = " << qz.S() );
	BOOST_UBLASX_DEBUG_TRACE( "T = " << qz.T() );
	BOOST_UBLASX_DEBUG_TRACE( "Q = " << qz.Q() );
	BOOST_UBLASX_DEBUG_TRACE( "Z = " << qz.Z() );
	BOOST_UBLASX_DEBUG_TRACE( "alpha = " << qz.alpha() );
	BOOST_UBLASX_DEBUG_TRACE( "beta = " << qz.beta() );
	BOOST_UBLASX_DEBUG_TRACE( "eigenvalues lambda = " << qz.eigenvalues() );
	BOOST_UBLASX_DEBUG_TRACE( "right eigenvectors V = " << qz.right_eigenvectors() );
	BOOST_UBLASX_DEBUG_TRACE( "left eigenvectors W = " << qz.left_eigenvectors() );
	matrix_type X;
	X = ublas::prod(qz.Q(), qz.S());
	X = ublas::prod(X, ublas::trans(qz.Z()));
	BOOST_UBLASX_DEBUG_TRACE( "Q*S*Z' = " << X );
	X = ublas::prod(qz.Q(), qz.T());
	X = ublas::prod(X, ublas::trans(qz.Z()));
	BOOST_UBLASX_DEBUG_TRACE( "Q*T*Z' = " << X );

	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.S()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.S()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.T()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.T()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.Q()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.Q()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.Z()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.Z()) == n );
	X = ublas::prod(qz.Q(), qz.S());
	X = ublas::prod(X, ublas::trans(qz.Z()));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, A, n, n, tol );
	X = ublas::prod(qz.Q(), qz.T());
	X = ublas::prod(X, ublas::trans(qz.Z()));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, B, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_real_row_major_oo_rhp_reorder )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Real Matrix - Row Major - QZ Object - RHP Reordering");

	typedef double value_type;
	typedef ublas::matrix<value_type, ublas::row_major> matrix_type;

	const std::size_t n(5);

	matrix_type A(n,n);
	A(0,0) = -0.180557; A(0,1) =  0.322289; A(0,2) = -0.651789; A(0,3) =  0.793637; A(0,4) = -0.141086;
	A(1,0) =  0.729781; A(1,1) =  1.665989; A(1,2) =  0.620091; A(1,3) = -1.541503; A(1,4) =  0.146673;
	A(2,0) = -0.594370; A(2,1) =  0.494804; A(2,2) =  1.004784; A(2,3) = -0.221373; A(2,4) = -2.196082;
	A(3,0) = -1.106269; A(3,1) =  0.026697; A(3,2) =  2.687083; A(3,3) =  0.763162; A(3,4) =  1.203514;
	A(4,0) = -0.021184; A(4,1) = -0.882220; A(4,2) = -1.618234; A(4,3) =  1.119524; A(4,4) =  2.588165;

	matrix_type B(n,n);
	B(0,0) = -1.592710; B(0,1) =  0.057283; B(0,2) = -1.862275; B(0,3) =  0.712471; B(0,4) =  0.463207;
	B(1,0) =  1.072859; B(1,1) = -1.384371; B(1,2) =  0.777754; B(1,3) =  1.914787; B(1,4) =  0.082774;
	B(2,0) = -0.451744; B(2,1) = -0.131528; B(2,2) = -0.636187; B(2,3) =  0.984480; B(2,4) =  0.011728;
	B(3,0) = -0.876629; B(3,1) = -0.083787; B(3,2) =  0.474227; B(3,3) = -0.042328; B(3,4) = -0.529845;
	B(4,0) = -0.812610; B(4,1) =  0.142456; B(4,2) =  0.033739; B(4,3) = -2.000422; B(4,4) = -0.765401;


	ublasx::qz_decomposition<value_type> qz;

	qz.decompose(A, B);
	qz.reorder(ublasx::rhp_qz_eigenvalues);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "B = " << B );
	BOOST_UBLASX_DEBUG_TRACE( "S = " << qz.S() );
	BOOST_UBLASX_DEBUG_TRACE( "T = " << qz.T() );
	BOOST_UBLASX_DEBUG_TRACE( "Q = " << qz.Q() );
	BOOST_UBLASX_DEBUG_TRACE( "Z = " << qz.Z() );
	BOOST_UBLASX_DEBUG_TRACE( "alpha = " << qz.alpha() );
	BOOST_UBLASX_DEBUG_TRACE( "beta = " << qz.beta() );
	BOOST_UBLASX_DEBUG_TRACE( "eigenvalues lambda = " << qz.eigenvalues() );
	BOOST_UBLASX_DEBUG_TRACE( "right eigenvectors V = " << qz.right_eigenvectors() );
	BOOST_UBLASX_DEBUG_TRACE( "left eigenvectors W = " << qz.left_eigenvectors() );
	matrix_type X;
	X = ublas::prod(qz.Q(), qz.S());
	X = ublas::prod(X, ublas::trans(qz.Z()));
	BOOST_UBLASX_DEBUG_TRACE( "Q*S*Z' = " << X );
	X = ublas::prod(qz.Q(), qz.T());
	X = ublas::prod(X, ublas::trans(qz.Z()));
	BOOST_UBLASX_DEBUG_TRACE( "Q*T*Z' = " << X );

	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.S()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.S()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.T()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.T()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.Q()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.Q()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.Z()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.Z()) == n );
	X = ublas::prod(qz.Q(), qz.S());
	X = ublas::prod(X, ublas::trans(qz.Z()));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, A, n, n, tol );
	X = ublas::prod(qz.Q(), qz.T());
	X = ublas::prod(X, ublas::trans(qz.Z()));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, B, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_real_column_major_oo_udi_reorder )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Real Matrix - Column Major - QZ Object - UDI Reordering");

	typedef double value_type;
	typedef ublas::matrix<value_type, ublas::column_major> matrix_type;

	const std::size_t n(5);

	matrix_type A(n,n);
	A(0,0) = -0.180557; A(0,1) =  0.322289; A(0,2) = -0.651789; A(0,3) =  0.793637; A(0,4) = -0.141086;
	A(1,0) =  0.729781; A(1,1) =  1.665989; A(1,2) =  0.620091; A(1,3) = -1.541503; A(1,4) =  0.146673;
	A(2,0) = -0.594370; A(2,1) =  0.494804; A(2,2) =  1.004784; A(2,3) = -0.221373; A(2,4) = -2.196082;
	A(3,0) = -1.106269; A(3,1) =  0.026697; A(3,2) =  2.687083; A(3,3) =  0.763162; A(3,4) =  1.203514;
	A(4,0) = -0.021184; A(4,1) = -0.882220; A(4,2) = -1.618234; A(4,3) =  1.119524; A(4,4) =  2.588165;

	matrix_type B(n,n);
	B(0,0) = -1.592710; B(0,1) =  0.057283; B(0,2) = -1.862275; B(0,3) =  0.712471; B(0,4) =  0.463207;
	B(1,0) =  1.072859; B(1,1) = -1.384371; B(1,2) =  0.777754; B(1,3) =  1.914787; B(1,4) =  0.082774;
	B(2,0) = -0.451744; B(2,1) = -0.131528; B(2,2) = -0.636187; B(2,3) =  0.984480; B(2,4) =  0.011728;
	B(3,0) = -0.876629; B(3,1) = -0.083787; B(3,2) =  0.474227; B(3,3) = -0.042328; B(3,4) = -0.529845;
	B(4,0) = -0.812610; B(4,1) =  0.142456; B(4,2) =  0.033739; B(4,3) = -2.000422; B(4,4) = -0.765401;


	ublasx::qz_decomposition<value_type> qz;

	qz.decompose(A, B);
	qz.reorder(ublasx::udi_qz_eigenvalues);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "B = " << B );
	BOOST_UBLASX_DEBUG_TRACE( "S = " << qz.S() );
	BOOST_UBLASX_DEBUG_TRACE( "T = " << qz.T() );
	BOOST_UBLASX_DEBUG_TRACE( "Q = " << qz.Q() );
	BOOST_UBLASX_DEBUG_TRACE( "Z = " << qz.Z() );
	BOOST_UBLASX_DEBUG_TRACE( "alpha = " << qz.alpha() );
	BOOST_UBLASX_DEBUG_TRACE( "beta = " << qz.beta() );
	BOOST_UBLASX_DEBUG_TRACE( "eigenvalues lambda = " << qz.eigenvalues() );
	BOOST_UBLASX_DEBUG_TRACE( "right eigenvectors V = " << qz.right_eigenvectors() );
	BOOST_UBLASX_DEBUG_TRACE( "left eigenvectors W = " << qz.left_eigenvectors() );
	matrix_type X;
	X = ublas::prod(qz.Q(), qz.S());
	X = ublas::prod(X, ublas::trans(qz.Z()));
	BOOST_UBLASX_DEBUG_TRACE( "Q*S*Z' = " << X );
	X = ublas::prod(qz.Q(), qz.T());
	X = ublas::prod(X, ublas::trans(qz.Z()));
	BOOST_UBLASX_DEBUG_TRACE( "Q*T*Z' = " << X );

	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.S()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.S()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.T()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.T()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.Q()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.Q()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.Z()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.Z()) == n );
	X = ublas::prod(qz.Q(), qz.S());
	X = ublas::prod(X, ublas::trans(qz.Z()));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, A, n, n, tol );
	X = ublas::prod(qz.Q(), qz.T());
	X = ublas::prod(X, ublas::trans(qz.Z()));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, B, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_real_row_major_oo_udi_reorder )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Real Matrix - Row Major - QZ Object - UDI Reordering");

	typedef double value_type;
	typedef ublas::matrix<value_type, ublas::row_major> matrix_type;

	const std::size_t n(5);

	matrix_type A(n,n);
	A(0,0) = -0.180557; A(0,1) =  0.322289; A(0,2) = -0.651789; A(0,3) =  0.793637; A(0,4) = -0.141086;
	A(1,0) =  0.729781; A(1,1) =  1.665989; A(1,2) =  0.620091; A(1,3) = -1.541503; A(1,4) =  0.146673;
	A(2,0) = -0.594370; A(2,1) =  0.494804; A(2,2) =  1.004784; A(2,3) = -0.221373; A(2,4) = -2.196082;
	A(3,0) = -1.106269; A(3,1) =  0.026697; A(3,2) =  2.687083; A(3,3) =  0.763162; A(3,4) =  1.203514;
	A(4,0) = -0.021184; A(4,1) = -0.882220; A(4,2) = -1.618234; A(4,3) =  1.119524; A(4,4) =  2.588165;

	matrix_type B(n,n);
	B(0,0) = -1.592710; B(0,1) =  0.057283; B(0,2) = -1.862275; B(0,3) =  0.712471; B(0,4) =  0.463207;
	B(1,0) =  1.072859; B(1,1) = -1.384371; B(1,2) =  0.777754; B(1,3) =  1.914787; B(1,4) =  0.082774;
	B(2,0) = -0.451744; B(2,1) = -0.131528; B(2,2) = -0.636187; B(2,3) =  0.984480; B(2,4) =  0.011728;
	B(3,0) = -0.876629; B(3,1) = -0.083787; B(3,2) =  0.474227; B(3,3) = -0.042328; B(3,4) = -0.529845;
	B(4,0) = -0.812610; B(4,1) =  0.142456; B(4,2) =  0.033739; B(4,3) = -2.000422; B(4,4) = -0.765401;


	ublasx::qz_decomposition<value_type> qz;

	qz.decompose(A, B);
	qz.reorder(ublasx::udi_qz_eigenvalues);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "B = " << B );
	BOOST_UBLASX_DEBUG_TRACE( "S = " << qz.S() );
	BOOST_UBLASX_DEBUG_TRACE( "T = " << qz.T() );
	BOOST_UBLASX_DEBUG_TRACE( "Q = " << qz.Q() );
	BOOST_UBLASX_DEBUG_TRACE( "Z = " << qz.Z() );
	BOOST_UBLASX_DEBUG_TRACE( "alpha = " << qz.alpha() );
	BOOST_UBLASX_DEBUG_TRACE( "beta = " << qz.beta() );
	BOOST_UBLASX_DEBUG_TRACE( "eigenvalues lambda = " << qz.eigenvalues() );
	BOOST_UBLASX_DEBUG_TRACE( "right eigenvectors V = " << qz.right_eigenvectors() );
	BOOST_UBLASX_DEBUG_TRACE( "left eigenvectors W = " << qz.left_eigenvectors() );
	matrix_type X;
	X = ublas::prod(qz.Q(), qz.S());
	X = ublas::prod(X, ublas::trans(qz.Z()));
	BOOST_UBLASX_DEBUG_TRACE( "Q*S*Z' = " << X );
	X = ublas::prod(qz.Q(), qz.T());
	X = ublas::prod(X, ublas::trans(qz.Z()));
	BOOST_UBLASX_DEBUG_TRACE( "Q*T*Z' = " << X );

	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.S()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.S()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.T()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.T()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.Q()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.Q()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.Z()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.Z()) == n );
	X = ublas::prod(qz.Q(), qz.S());
	X = ublas::prod(X, ublas::trans(qz.Z()));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, A, n, n, tol );
	X = ublas::prod(qz.Q(), qz.T());
	X = ublas::prod(X, ublas::trans(qz.Z()));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, B, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_real_column_major_oo_udo_reorder )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Real Matrix - Column Major - QZ Object - UDO Reordering");

	typedef double value_type;
	typedef ublas::matrix<value_type, ublas::column_major> matrix_type;

	const std::size_t n(5);

	matrix_type A(n,n);
	A(0,0) = -0.180557; A(0,1) =  0.322289; A(0,2) = -0.651789; A(0,3) =  0.793637; A(0,4) = -0.141086;
	A(1,0) =  0.729781; A(1,1) =  1.665989; A(1,2) =  0.620091; A(1,3) = -1.541503; A(1,4) =  0.146673;
	A(2,0) = -0.594370; A(2,1) =  0.494804; A(2,2) =  1.004784; A(2,3) = -0.221373; A(2,4) = -2.196082;
	A(3,0) = -1.106269; A(3,1) =  0.026697; A(3,2) =  2.687083; A(3,3) =  0.763162; A(3,4) =  1.203514;
	A(4,0) = -0.021184; A(4,1) = -0.882220; A(4,2) = -1.618234; A(4,3) =  1.119524; A(4,4) =  2.588165;

	matrix_type B(n,n);
	B(0,0) = -1.592710; B(0,1) =  0.057283; B(0,2) = -1.862275; B(0,3) =  0.712471; B(0,4) =  0.463207;
	B(1,0) =  1.072859; B(1,1) = -1.384371; B(1,2) =  0.777754; B(1,3) =  1.914787; B(1,4) =  0.082774;
	B(2,0) = -0.451744; B(2,1) = -0.131528; B(2,2) = -0.636187; B(2,3) =  0.984480; B(2,4) =  0.011728;
	B(3,0) = -0.876629; B(3,1) = -0.083787; B(3,2) =  0.474227; B(3,3) = -0.042328; B(3,4) = -0.529845;
	B(4,0) = -0.812610; B(4,1) =  0.142456; B(4,2) =  0.033739; B(4,3) = -2.000422; B(4,4) = -0.765401;


	ublasx::qz_decomposition<value_type> qz;

	qz.decompose(A, B);
	qz.reorder(ublasx::udo_qz_eigenvalues);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "B = " << B );
	BOOST_UBLASX_DEBUG_TRACE( "S = " << qz.S() );
	BOOST_UBLASX_DEBUG_TRACE( "T = " << qz.T() );
	BOOST_UBLASX_DEBUG_TRACE( "Q = " << qz.Q() );
	BOOST_UBLASX_DEBUG_TRACE( "Z = " << qz.Z() );
	BOOST_UBLASX_DEBUG_TRACE( "alpha = " << qz.alpha() );
	BOOST_UBLASX_DEBUG_TRACE( "beta = " << qz.beta() );
	BOOST_UBLASX_DEBUG_TRACE( "eigenvalues lambda = " << qz.eigenvalues() );
	BOOST_UBLASX_DEBUG_TRACE( "right eigenvectors V = " << qz.right_eigenvectors() );
	BOOST_UBLASX_DEBUG_TRACE( "left eigenvectors W = " << qz.left_eigenvectors() );
	matrix_type X;
	X = ublas::prod(qz.Q(), qz.S());
	X = ublas::prod(X, ublas::trans(qz.Z()));
	BOOST_UBLASX_DEBUG_TRACE( "Q*S*Z' = " << X );
	X = ublas::prod(qz.Q(), qz.T());
	X = ublas::prod(X, ublas::trans(qz.Z()));
	BOOST_UBLASX_DEBUG_TRACE( "Q*T*Z' = " << X );

	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.S()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.S()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.T()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.T()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.Q()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.Q()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.Z()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.Z()) == n );
	X = ublas::prod(qz.Q(), qz.S());
	X = ublas::prod(X, ublas::trans(qz.Z()));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, A, n, n, tol );
	X = ublas::prod(qz.Q(), qz.T());
	X = ublas::prod(X, ublas::trans(qz.Z()));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, B, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_real_row_major_oo_udo_reorder )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Real Matrix - Row Major - QZ Object - UDO Reordering");

	typedef double value_type;
	typedef ublas::matrix<value_type, ublas::row_major> matrix_type;

	const std::size_t n(5);

	matrix_type A(n,n);
	A(0,0) = -0.180557; A(0,1) =  0.322289; A(0,2) = -0.651789; A(0,3) =  0.793637; A(0,4) = -0.141086;
	A(1,0) =  0.729781; A(1,1) =  1.665989; A(1,2) =  0.620091; A(1,3) = -1.541503; A(1,4) =  0.146673;
	A(2,0) = -0.594370; A(2,1) =  0.494804; A(2,2) =  1.004784; A(2,3) = -0.221373; A(2,4) = -2.196082;
	A(3,0) = -1.106269; A(3,1) =  0.026697; A(3,2) =  2.687083; A(3,3) =  0.763162; A(3,4) =  1.203514;
	A(4,0) = -0.021184; A(4,1) = -0.882220; A(4,2) = -1.618234; A(4,3) =  1.119524; A(4,4) =  2.588165;

	matrix_type B(n,n);
	B(0,0) = -1.592710; B(0,1) =  0.057283; B(0,2) = -1.862275; B(0,3) =  0.712471; B(0,4) =  0.463207;
	B(1,0) =  1.072859; B(1,1) = -1.384371; B(1,2) =  0.777754; B(1,3) =  1.914787; B(1,4) =  0.082774;
	B(2,0) = -0.451744; B(2,1) = -0.131528; B(2,2) = -0.636187; B(2,3) =  0.984480; B(2,4) =  0.011728;
	B(3,0) = -0.876629; B(3,1) = -0.083787; B(3,2) =  0.474227; B(3,3) = -0.042328; B(3,4) = -0.529845;
	B(4,0) = -0.812610; B(4,1) =  0.142456; B(4,2) =  0.033739; B(4,3) = -2.000422; B(4,4) = -0.765401;


	ublasx::qz_decomposition<value_type> qz;

	qz.decompose(A, B);
	qz.reorder(ublasx::udo_qz_eigenvalues);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "B = " << B );
	BOOST_UBLASX_DEBUG_TRACE( "S = " << qz.S() );
	BOOST_UBLASX_DEBUG_TRACE( "T = " << qz.T() );
	BOOST_UBLASX_DEBUG_TRACE( "Q = " << qz.Q() );
	BOOST_UBLASX_DEBUG_TRACE( "Z = " << qz.Z() );
	BOOST_UBLASX_DEBUG_TRACE( "alpha = " << qz.alpha() );
	BOOST_UBLASX_DEBUG_TRACE( "beta = " << qz.beta() );
	BOOST_UBLASX_DEBUG_TRACE( "eigenvalues lambda = " << qz.eigenvalues() );
	BOOST_UBLASX_DEBUG_TRACE( "right eigenvectors V = " << qz.right_eigenvectors() );
	BOOST_UBLASX_DEBUG_TRACE( "left eigenvectors W = " << qz.left_eigenvectors() );
	matrix_type X;
	X = ublas::prod(qz.Q(), qz.S());
	X = ublas::prod(X, ublas::trans(qz.Z()));
	BOOST_UBLASX_DEBUG_TRACE( "Q*S*Z' = " << X );
	X = ublas::prod(qz.Q(), qz.T());
	X = ublas::prod(X, ublas::trans(qz.Z()));
	BOOST_UBLASX_DEBUG_TRACE( "Q*T*Z' = " << X );

	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.S()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.S()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.T()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.T()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.Q()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.Q()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.Z()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.Z()) == n );
	X = ublas::prod(qz.Q(), qz.S());
	X = ublas::prod(X, ublas::trans(qz.Z()));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, A, n, n, tol );
	X = ublas::prod(qz.Q(), qz.T());
	X = ublas::prod(X, ublas::trans(qz.Z()));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, B, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_real_column_major_oo_custom_reorder )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Real Matrix - Column Major - QZ Object - Custom Reordering");

	typedef double value_type;
	typedef ublas::matrix<value_type, ublas::column_major> matrix_type;

	const std::size_t n(5);

	matrix_type A(n,n);
	A(0,0) = -0.180557; A(0,1) =  0.322289; A(0,2) = -0.651789; A(0,3) =  0.793637; A(0,4) = -0.141086;
	A(1,0) =  0.729781; A(1,1) =  1.665989; A(1,2) =  0.620091; A(1,3) = -1.541503; A(1,4) =  0.146673;
	A(2,0) = -0.594370; A(2,1) =  0.494804; A(2,2) =  1.004784; A(2,3) = -0.221373; A(2,4) = -2.196082;
	A(3,0) = -1.106269; A(3,1) =  0.026697; A(3,2) =  2.687083; A(3,3) =  0.763162; A(3,4) =  1.203514;
	A(4,0) = -0.021184; A(4,1) = -0.882220; A(4,2) = -1.618234; A(4,3) =  1.119524; A(4,4) =  2.588165;

	matrix_type B(n,n);
	B(0,0) = -1.592710; B(0,1) =  0.057283; B(0,2) = -1.862275; B(0,3) =  0.712471; B(0,4) =  0.463207;
	B(1,0) =  1.072859; B(1,1) = -1.384371; B(1,2) =  0.777754; B(1,3) =  1.914787; B(1,4) =  0.082774;
	B(2,0) = -0.451744; B(2,1) = -0.131528; B(2,2) = -0.636187; B(2,3) =  0.984480; B(2,4) =  0.011728;
	B(3,0) = -0.876629; B(3,1) = -0.083787; B(3,2) =  0.474227; B(3,3) = -0.042328; B(3,4) = -0.529845;
	B(4,0) = -0.812610; B(4,1) =  0.142456; B(4,2) =  0.033739; B(4,3) = -2.000422; B(4,4) = -0.765401;


	ublasx::qz_decomposition<value_type> qz;

	qz.decompose(A, B);

	// Selection: select eigenvalues at even position (no sense ... just a test)
	std::size_t n_e = ublasx::size(qz.eigenvalues());
	ublas::vector<bool> selection(n_e);
	for (std::size_t i = 0; i < n_e; ++i)
	{
		selection(i) = ((i+1) % 2) ? true : false;
	}

	qz.reorder(selection);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "B = " << B );
	BOOST_UBLASX_DEBUG_TRACE( "S = " << qz.S() );
	BOOST_UBLASX_DEBUG_TRACE( "T = " << qz.T() );
	BOOST_UBLASX_DEBUG_TRACE( "Q = " << qz.Q() );
	BOOST_UBLASX_DEBUG_TRACE( "Z = " << qz.Z() );
	BOOST_UBLASX_DEBUG_TRACE( "alpha = " << qz.alpha() );
	BOOST_UBLASX_DEBUG_TRACE( "beta = " << qz.beta() );
	BOOST_UBLASX_DEBUG_TRACE( "eigenvalues lambda = " << qz.eigenvalues() );
	BOOST_UBLASX_DEBUG_TRACE( "right eigenvectors V = " << qz.right_eigenvectors() );
	BOOST_UBLASX_DEBUG_TRACE( "left eigenvectors W = " << qz.left_eigenvectors() );
	matrix_type X;
	X = ublas::prod(qz.Q(), qz.S());
	X = ublas::prod(X, ublas::trans(qz.Z()));
	BOOST_UBLASX_DEBUG_TRACE( "Q*S*Z' = " << X );
	X = ublas::prod(qz.Q(), qz.T());
	X = ublas::prod(X, ublas::trans(qz.Z()));
	BOOST_UBLASX_DEBUG_TRACE( "Q*T*Z' = " << X );

	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.S()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.S()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.T()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.T()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.Q()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.Q()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.Z()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.Z()) == n );
	X = ublas::prod(qz.Q(), qz.S());
	X = ublas::prod(X, ublas::trans(qz.Z()));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, A, n, n, tol );
	X = ublas::prod(qz.Q(), qz.T());
	X = ublas::prod(X, ublas::trans(qz.Z()));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, B, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_real_row_major_oo_custom_reorder )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Real Matrix - Row Major - QZ Object - Custom Reordering");

	typedef double value_type;
	typedef ublas::matrix<value_type, ublas::row_major> matrix_type;

	const std::size_t n(5);

	matrix_type A(n,n);
	A(0,0) = -0.180557; A(0,1) =  0.322289; A(0,2) = -0.651789; A(0,3) =  0.793637; A(0,4) = -0.141086;
	A(1,0) =  0.729781; A(1,1) =  1.665989; A(1,2) =  0.620091; A(1,3) = -1.541503; A(1,4) =  0.146673;
	A(2,0) = -0.594370; A(2,1) =  0.494804; A(2,2) =  1.004784; A(2,3) = -0.221373; A(2,4) = -2.196082;
	A(3,0) = -1.106269; A(3,1) =  0.026697; A(3,2) =  2.687083; A(3,3) =  0.763162; A(3,4) =  1.203514;
	A(4,0) = -0.021184; A(4,1) = -0.882220; A(4,2) = -1.618234; A(4,3) =  1.119524; A(4,4) =  2.588165;

	matrix_type B(n,n);
	B(0,0) = -1.592710; B(0,1) =  0.057283; B(0,2) = -1.862275; B(0,3) =  0.712471; B(0,4) =  0.463207;
	B(1,0) =  1.072859; B(1,1) = -1.384371; B(1,2) =  0.777754; B(1,3) =  1.914787; B(1,4) =  0.082774;
	B(2,0) = -0.451744; B(2,1) = -0.131528; B(2,2) = -0.636187; B(2,3) =  0.984480; B(2,4) =  0.011728;
	B(3,0) = -0.876629; B(3,1) = -0.083787; B(3,2) =  0.474227; B(3,3) = -0.042328; B(3,4) = -0.529845;
	B(4,0) = -0.812610; B(4,1) =  0.142456; B(4,2) =  0.033739; B(4,3) = -2.000422; B(4,4) = -0.765401;


	ublasx::qz_decomposition<value_type> qz;

	qz.decompose(A, B);

	// Selection: select eigenvalues at even position (no sense ... just a test)
	std::size_t n_e = ublasx::size(qz.eigenvalues());
	ublas::vector<bool> selection(n_e);
	for (std::size_t i = 0; i < n_e; ++i)
	{
		selection(i) = ((i+1) % 2) ? true : false;
	}

	qz.reorder(selection);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "B = " << B );
	BOOST_UBLASX_DEBUG_TRACE( "S = " << qz.S() );
	BOOST_UBLASX_DEBUG_TRACE( "T = " << qz.T() );
	BOOST_UBLASX_DEBUG_TRACE( "Q = " << qz.Q() );
	BOOST_UBLASX_DEBUG_TRACE( "Z = " << qz.Z() );
	BOOST_UBLASX_DEBUG_TRACE( "alpha = " << qz.alpha() );
	BOOST_UBLASX_DEBUG_TRACE( "beta = " << qz.beta() );
	BOOST_UBLASX_DEBUG_TRACE( "eigenvalues lambda = " << qz.eigenvalues() );
	BOOST_UBLASX_DEBUG_TRACE( "right eigenvectors V = " << qz.right_eigenvectors() );
	BOOST_UBLASX_DEBUG_TRACE( "left eigenvectors W = " << qz.left_eigenvectors() );
	matrix_type X;
	X = ublas::prod(qz.Q(), qz.S());
	X = ublas::prod(X, ublas::trans(qz.Z()));
	BOOST_UBLASX_DEBUG_TRACE( "Q*S*Z' = " << X );
	X = ublas::prod(qz.Q(), qz.T());
	X = ublas::prod(X, ublas::trans(qz.Z()));
	BOOST_UBLASX_DEBUG_TRACE( "Q*T*Z' = " << X );

	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.S()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.S()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.T()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.T()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.Q()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.Q()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.Z()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.Z()) == n );
	X = ublas::prod(qz.Q(), qz.S());
	X = ublas::prod(X, ublas::trans(qz.Z()));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, A, n, n, tol );
	X = ublas::prod(qz.Q(), qz.T());
	X = ublas::prod(X, ublas::trans(qz.Z()));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, B, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_complex_column_major_oo_decomp )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Complex Matrix - Column Major - QZ Object - Decomposition");

	typedef std::complex<double> value_type;
	typedef ublas::matrix<value_type, ublas::column_major> matrix_type;

	const std::size_t n(5);

	matrix_type A(n,n);
	A(0,0) = value_type(-0.180557,-0.947835); A(0,1) = value_type( 0.322289, 1.603339); A(0,2) = value_type(-0.651789,-0.902809); A(0,3) = value_type( 0.793637, 0.031147); A(0,4) = value_type(-0.141086,-0.769742);
	A(1,0) = value_type( 0.729781, 0.693097); A(1,1) = value_type( 1.665989, 1.258667); A(1,2) = value_type( 0.620091,-0.192144); A(1,3) = value_type(-1.541503, 0.545104); A(1,4) = value_type( 0.146673,-1.553689);
	A(2,0) = value_type(-0.594370, 0.071316); A(2,1) = value_type( 0.494804,-2.381227); A(2,2) = value_type( 1.004784, 1.097852); A(2,3) = value_type(-0.221373,-0.585458); A(2,4) = value_type(-2.196082,-0.518945);
	A(3,0) = value_type(-1.106269,-1.015812); A(3,1) = value_type( 0.026697, 2.866180); A(3,2) = value_type( 2.687083,-0.115506); A(3,3) = value_type( 0.763162, 0.382183); A(3,4) = value_type( 1.203514, 0.203552);
	A(4,0) = value_type(-0.021184,-1.006383); A(4,1) = value_type(-0.882220, 0.232555); A(4,2) = value_type(-1.618234, 0.615528); A(4,3) = value_type( 1.119524,-2.178697); A(4,4) = value_type( 2.588165,-0.054936);

	matrix_type B(n,n);
	B(0,0) = value_type(-1.592710, 0.804795); B(0,1) = value_type( 0.057283, 0.470640); B(0,2) = value_type(-1.862275,-0.454486); B(0,3) = value_type( 0.712471, 0.887654); B(0,4) = value_type( 0.463207,-0.045117);
	B(1,0) = value_type( 1.072859,-2.009749); B(1,1) = value_type(-1.384371,-0.778200); B(1,2) = value_type( 0.777754, 1.245456); B(1,3) = value_type( 1.914787,-1.246236); B(1,4) = value_type( 0.082774, 0.358980);
	B(2,0) = value_type(-0.451744,-0.353845); B(2,1) = value_type(-0.131528,-2.236258); B(2,2) = value_type(-0.636187,-0.957378); B(2,3) = value_type( 0.984480,-1.840536); B(2,4) = value_type( 0.011728, 1.682497);
	B(3,0) = value_type(-0.876629, 0.764240); B(3,1) = value_type(-0.083787, 2.238476); B(3,2) = value_type( 0.474227,-2.426134); B(3,3) = value_type(-0.042328, 1.129135); B(3,4) = value_type(-0.529845, 0.653758);
	B(4,0) = value_type(-0.812610, 0.996806); B(4,1) = value_type( 0.142456, 0.101454); B(4,2) = value_type( 0.033739,-0.501220); B(4,3) = value_type(-2.000422, 0.181931); B(4,4) = value_type(-0.765401,-1.004076);


	ublasx::qz_decomposition<value_type> qz;

	qz.decompose(A, B);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "B = " << B );
	BOOST_UBLASX_DEBUG_TRACE( "S = " << qz.S() );
	BOOST_UBLASX_DEBUG_TRACE( "T = " << qz.T() );
	BOOST_UBLASX_DEBUG_TRACE( "Q = " << qz.Q() );
	BOOST_UBLASX_DEBUG_TRACE( "Z = " << qz.Z() );
	BOOST_UBLASX_DEBUG_TRACE( "alpha = " << qz.alpha() );
	BOOST_UBLASX_DEBUG_TRACE( "beta = " << qz.beta() );
	BOOST_UBLASX_DEBUG_TRACE( "eigenvalues lambda = " << qz.eigenvalues() );
	BOOST_UBLASX_DEBUG_TRACE( "right eigenvectors V = " << qz.right_eigenvectors() );
	BOOST_UBLASX_DEBUG_TRACE( "left eigenvectors W = " << qz.left_eigenvectors() );
	matrix_type X;
	X = ublas::prod(qz.Q(), qz.S());
	X = ublas::prod(X, ublas::trans(qz.Z()));
	BOOST_UBLASX_DEBUG_TRACE( "Q*S*Z' = " << X );
	X = ublas::prod(qz.Q(), qz.T());
	X = ublas::prod(X, ublas::trans(qz.Z()));
	BOOST_UBLASX_DEBUG_TRACE( "Q*T*Z' = " << X );

	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.S()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.S()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.T()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.T()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.Q()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.Q()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.Z()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.Z()) == n );
	X = ublas::prod(qz.Q(), qz.S());
	X = ublas::prod(X, ublas::herm(qz.Z()));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, A, n, n, tol );
	X = ublas::prod(qz.Q(), qz.T());
	X = ublas::prod(X, ublas::herm(qz.Z()));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, B, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_complex_row_major_oo_decomp )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Complex Matrix - Row Major - QZ Object - Decomposition");

	typedef std::complex<double> value_type;
	typedef ublas::matrix<value_type, ublas::row_major> matrix_type;

	const std::size_t n(5);

	matrix_type A(n,n);
	A(0,0) = value_type(-0.180557,-0.947835); A(0,1) = value_type( 0.322289, 1.603339); A(0,2) = value_type(-0.651789,-0.902809); A(0,3) = value_type( 0.793637, 0.031147); A(0,4) = value_type(-0.141086,-0.769742);
	A(1,0) = value_type( 0.729781, 0.693097); A(1,1) = value_type( 1.665989, 1.258667); A(1,2) = value_type( 0.620091,-0.192144); A(1,3) = value_type(-1.541503, 0.545104); A(1,4) = value_type( 0.146673,-1.553689);
	A(2,0) = value_type(-0.594370, 0.071316); A(2,1) = value_type( 0.494804,-2.381227); A(2,2) = value_type( 1.004784, 1.097852); A(2,3) = value_type(-0.221373,-0.585458); A(2,4) = value_type(-2.196082,-0.518945);
	A(3,0) = value_type(-1.106269,-1.015812); A(3,1) = value_type( 0.026697, 2.866180); A(3,2) = value_type( 2.687083,-0.115506); A(3,3) = value_type( 0.763162, 0.382183); A(3,4) = value_type( 1.203514, 0.203552);
	A(4,0) = value_type(-0.021184,-1.006383); A(4,1) = value_type(-0.882220, 0.232555); A(4,2) = value_type(-1.618234, 0.615528); A(4,3) = value_type( 1.119524,-2.178697); A(4,4) = value_type( 2.588165,-0.054936);

	matrix_type B(n,n);
	B(0,0) = value_type(-1.592710, 0.804795); B(0,1) = value_type( 0.057283, 0.470640); B(0,2) = value_type(-1.862275,-0.454486); B(0,3) = value_type( 0.712471, 0.887654); B(0,4) = value_type( 0.463207,-0.045117);
	B(1,0) = value_type( 1.072859,-2.009749); B(1,1) = value_type(-1.384371,-0.778200); B(1,2) = value_type( 0.777754, 1.245456); B(1,3) = value_type( 1.914787,-1.246236); B(1,4) = value_type( 0.082774, 0.358980);
	B(2,0) = value_type(-0.451744,-0.353845); B(2,1) = value_type(-0.131528,-2.236258); B(2,2) = value_type(-0.636187,-0.957378); B(2,3) = value_type( 0.984480,-1.840536); B(2,4) = value_type( 0.011728, 1.682497);
	B(3,0) = value_type(-0.876629, 0.764240); B(3,1) = value_type(-0.083787, 2.238476); B(3,2) = value_type( 0.474227,-2.426134); B(3,3) = value_type(-0.042328, 1.129135); B(3,4) = value_type(-0.529845, 0.653758);
	B(4,0) = value_type(-0.812610, 0.996806); B(4,1) = value_type( 0.142456, 0.101454); B(4,2) = value_type( 0.033739,-0.501220); B(4,3) = value_type(-2.000422, 0.181931); B(4,4) = value_type(-0.765401,-1.004076);


	ublasx::qz_decomposition<value_type> qz;

	qz.decompose(A, B);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "B = " << B );
	BOOST_UBLASX_DEBUG_TRACE( "S = " << qz.S() );
	BOOST_UBLASX_DEBUG_TRACE( "T = " << qz.T() );
	BOOST_UBLASX_DEBUG_TRACE( "Q = " << qz.Q() );
	BOOST_UBLASX_DEBUG_TRACE( "Z = " << qz.Z() );
	BOOST_UBLASX_DEBUG_TRACE( "alpha = " << qz.alpha() );
	BOOST_UBLASX_DEBUG_TRACE( "beta = " << qz.beta() );
	BOOST_UBLASX_DEBUG_TRACE( "eigenvalues lambda = " << qz.eigenvalues() );
	BOOST_UBLASX_DEBUG_TRACE( "right eigenvectors V = " << qz.right_eigenvectors() );
	BOOST_UBLASX_DEBUG_TRACE( "left eigenvectors W = " << qz.left_eigenvectors() );
	matrix_type X;
	X = ublas::prod(qz.Q(), qz.S());
	X = ublas::prod(X, ublas::trans(qz.Z()));
	BOOST_UBLASX_DEBUG_TRACE( "Q*S*Z' = " << X );
	X = ublas::prod(qz.Q(), qz.T());
	X = ublas::prod(X, ublas::trans(qz.Z()));
	BOOST_UBLASX_DEBUG_TRACE( "Q*T*Z' = " << X );

	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.S()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.S()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.T()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.T()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.Q()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.Q()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.Z()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.Z()) == n );
	X = ublas::prod(qz.Q(), qz.S());
	X = ublas::prod(X, ublas::herm(qz.Z()));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, A, n, n, tol );
	X = ublas::prod(qz.Q(), qz.T());
	X = ublas::prod(X, ublas::herm(qz.Z()));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, B, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_complex_column_major_oo_lhp_reorder )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Complex Matrix - Column Major - QZ Object - LHP Reordering");

	typedef std::complex<double> value_type;
	typedef ublas::matrix<value_type, ublas::column_major> matrix_type;

	const std::size_t n(5);

	matrix_type A(n,n);
	A(0,0) = value_type(-0.180557,-0.947835); A(0,1) = value_type( 0.322289, 1.603339); A(0,2) = value_type(-0.651789,-0.902809); A(0,3) = value_type( 0.793637, 0.031147); A(0,4) = value_type(-0.141086,-0.769742);
	A(1,0) = value_type( 0.729781, 0.693097); A(1,1) = value_type( 1.665989, 1.258667); A(1,2) = value_type( 0.620091,-0.192144); A(1,3) = value_type(-1.541503, 0.545104); A(1,4) = value_type( 0.146673,-1.553689);
	A(2,0) = value_type(-0.594370, 0.071316); A(2,1) = value_type( 0.494804,-2.381227); A(2,2) = value_type( 1.004784, 1.097852); A(2,3) = value_type(-0.221373,-0.585458); A(2,4) = value_type(-2.196082,-0.518945);
	A(3,0) = value_type(-1.106269,-1.015812); A(3,1) = value_type( 0.026697, 2.866180); A(3,2) = value_type( 2.687083,-0.115506); A(3,3) = value_type( 0.763162, 0.382183); A(3,4) = value_type( 1.203514, 0.203552);
	A(4,0) = value_type(-0.021184,-1.006383); A(4,1) = value_type(-0.882220, 0.232555); A(4,2) = value_type(-1.618234, 0.615528); A(4,3) = value_type( 1.119524,-2.178697); A(4,4) = value_type( 2.588165,-0.054936);

	matrix_type B(n,n);
	B(0,0) = value_type(-1.592710, 0.804795); B(0,1) = value_type( 0.057283, 0.470640); B(0,2) = value_type(-1.862275,-0.454486); B(0,3) = value_type( 0.712471, 0.887654); B(0,4) = value_type( 0.463207,-0.045117);
	B(1,0) = value_type( 1.072859,-2.009749); B(1,1) = value_type(-1.384371,-0.778200); B(1,2) = value_type( 0.777754, 1.245456); B(1,3) = value_type( 1.914787,-1.246236); B(1,4) = value_type( 0.082774, 0.358980);
	B(2,0) = value_type(-0.451744,-0.353845); B(2,1) = value_type(-0.131528,-2.236258); B(2,2) = value_type(-0.636187,-0.957378); B(2,3) = value_type( 0.984480,-1.840536); B(2,4) = value_type( 0.011728, 1.682497);
	B(3,0) = value_type(-0.876629, 0.764240); B(3,1) = value_type(-0.083787, 2.238476); B(3,2) = value_type( 0.474227,-2.426134); B(3,3) = value_type(-0.042328, 1.129135); B(3,4) = value_type(-0.529845, 0.653758);
	B(4,0) = value_type(-0.812610, 0.996806); B(4,1) = value_type( 0.142456, 0.101454); B(4,2) = value_type( 0.033739,-0.501220); B(4,3) = value_type(-2.000422, 0.181931); B(4,4) = value_type(-0.765401,-1.004076);


	ublasx::qz_decomposition<value_type> qz;

	qz.decompose(A, B);
	qz.reorder(ublasx::lhp_qz_eigenvalues);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "B = " << B );
	BOOST_UBLASX_DEBUG_TRACE( "S = " << qz.S() );
	BOOST_UBLASX_DEBUG_TRACE( "T = " << qz.T() );
	BOOST_UBLASX_DEBUG_TRACE( "Q = " << qz.Q() );
	BOOST_UBLASX_DEBUG_TRACE( "Z = " << qz.Z() );
	BOOST_UBLASX_DEBUG_TRACE( "alpha = " << qz.alpha() );
	BOOST_UBLASX_DEBUG_TRACE( "beta = " << qz.beta() );
	BOOST_UBLASX_DEBUG_TRACE( "eigenvalues lambda = " << qz.eigenvalues() );
	BOOST_UBLASX_DEBUG_TRACE( "right eigenvectors V = " << qz.right_eigenvectors() );
	BOOST_UBLASX_DEBUG_TRACE( "left eigenvectors W = " << qz.left_eigenvectors() );
	matrix_type X;
	X = ublas::prod(qz.Q(), qz.S());
	X = ublas::prod(X, ublas::trans(qz.Z()));
	BOOST_UBLASX_DEBUG_TRACE( "Q*S*Z' = " << X );
	X = ublas::prod(qz.Q(), qz.T());
	X = ublas::prod(X, ublas::trans(qz.Z()));
	BOOST_UBLASX_DEBUG_TRACE( "Q*T*Z' = " << X );

	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.S()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.S()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.T()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.T()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.Q()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.Q()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.Z()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.Z()) == n );
	X = ublas::prod(qz.Q(), qz.S());
	X = ublas::prod(X, ublas::herm(qz.Z()));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, A, n, n, tol );
	X = ublas::prod(qz.Q(), qz.T());
	X = ublas::prod(X, ublas::herm(qz.Z()));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, B, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_complex_row_major_oo_lhp_reorder )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Complex Matrix - Row Major - QZ Object - LHP Reordering");

	typedef std::complex<double> value_type;
	typedef ublas::matrix<value_type, ublas::row_major> matrix_type;

	const std::size_t n(5);

	matrix_type A(n,n);
	A(0,0) = value_type(-0.180557,-0.947835); A(0,1) = value_type( 0.322289, 1.603339); A(0,2) = value_type(-0.651789,-0.902809); A(0,3) = value_type( 0.793637, 0.031147); A(0,4) = value_type(-0.141086,-0.769742);
	A(1,0) = value_type( 0.729781, 0.693097); A(1,1) = value_type( 1.665989, 1.258667); A(1,2) = value_type( 0.620091,-0.192144); A(1,3) = value_type(-1.541503, 0.545104); A(1,4) = value_type( 0.146673,-1.553689);
	A(2,0) = value_type(-0.594370, 0.071316); A(2,1) = value_type( 0.494804,-2.381227); A(2,2) = value_type( 1.004784, 1.097852); A(2,3) = value_type(-0.221373,-0.585458); A(2,4) = value_type(-2.196082,-0.518945);
	A(3,0) = value_type(-1.106269,-1.015812); A(3,1) = value_type( 0.026697, 2.866180); A(3,2) = value_type( 2.687083,-0.115506); A(3,3) = value_type( 0.763162, 0.382183); A(3,4) = value_type( 1.203514, 0.203552);
	A(4,0) = value_type(-0.021184,-1.006383); A(4,1) = value_type(-0.882220, 0.232555); A(4,2) = value_type(-1.618234, 0.615528); A(4,3) = value_type( 1.119524,-2.178697); A(4,4) = value_type( 2.588165,-0.054936);

	matrix_type B(n,n);
	B(0,0) = value_type(-1.592710, 0.804795); B(0,1) = value_type( 0.057283, 0.470640); B(0,2) = value_type(-1.862275,-0.454486); B(0,3) = value_type( 0.712471, 0.887654); B(0,4) = value_type( 0.463207,-0.045117);
	B(1,0) = value_type( 1.072859,-2.009749); B(1,1) = value_type(-1.384371,-0.778200); B(1,2) = value_type( 0.777754, 1.245456); B(1,3) = value_type( 1.914787,-1.246236); B(1,4) = value_type( 0.082774, 0.358980);
	B(2,0) = value_type(-0.451744,-0.353845); B(2,1) = value_type(-0.131528,-2.236258); B(2,2) = value_type(-0.636187,-0.957378); B(2,3) = value_type( 0.984480,-1.840536); B(2,4) = value_type( 0.011728, 1.682497);
	B(3,0) = value_type(-0.876629, 0.764240); B(3,1) = value_type(-0.083787, 2.238476); B(3,2) = value_type( 0.474227,-2.426134); B(3,3) = value_type(-0.042328, 1.129135); B(3,4) = value_type(-0.529845, 0.653758);
	B(4,0) = value_type(-0.812610, 0.996806); B(4,1) = value_type( 0.142456, 0.101454); B(4,2) = value_type( 0.033739,-0.501220); B(4,3) = value_type(-2.000422, 0.181931); B(4,4) = value_type(-0.765401,-1.004076);


	ublasx::qz_decomposition<value_type> qz;

	qz.decompose(A, B);
	qz.reorder(ublasx::lhp_qz_eigenvalues);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "B = " << B );
	BOOST_UBLASX_DEBUG_TRACE( "S = " << qz.S() );
	BOOST_UBLASX_DEBUG_TRACE( "T = " << qz.T() );
	BOOST_UBLASX_DEBUG_TRACE( "Q = " << qz.Q() );
	BOOST_UBLASX_DEBUG_TRACE( "Z = " << qz.Z() );
	BOOST_UBLASX_DEBUG_TRACE( "alpha = " << qz.alpha() );
	BOOST_UBLASX_DEBUG_TRACE( "beta = " << qz.beta() );
	BOOST_UBLASX_DEBUG_TRACE( "eigenvalues lambda = " << qz.eigenvalues() );
	BOOST_UBLASX_DEBUG_TRACE( "right eigenvectors V = " << qz.right_eigenvectors() );
	BOOST_UBLASX_DEBUG_TRACE( "left eigenvectors W = " << qz.left_eigenvectors() );
	matrix_type X;
	X = ublas::prod(qz.Q(), qz.S());
	X = ublas::prod(X, ublas::trans(qz.Z()));
	BOOST_UBLASX_DEBUG_TRACE( "Q*S*Z' = " << X );
	X = ublas::prod(qz.Q(), qz.T());
	X = ublas::prod(X, ublas::trans(qz.Z()));
	BOOST_UBLASX_DEBUG_TRACE( "Q*T*Z' = " << X );

	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.S()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.S()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.T()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.T()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.Q()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.Q()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.Z()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.Z()) == n );
	X = ublas::prod(qz.Q(), qz.S());
	X = ublas::prod(X, ublas::herm(qz.Z()));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, A, n, n, tol );
	X = ublas::prod(qz.Q(), qz.T());
	X = ublas::prod(X, ublas::herm(qz.Z()));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, B, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_complex_column_major_oo_rhp_reorder )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Complex Matrix - Column Major - QZ Object - RHP Reordering");

	typedef std::complex<double> value_type;
	typedef ublas::matrix<value_type, ublas::column_major> matrix_type;

	const std::size_t n(5);

	matrix_type A(n,n);
	A(0,0) = value_type(-0.180557,-0.947835); A(0,1) = value_type( 0.322289, 1.603339); A(0,2) = value_type(-0.651789,-0.902809); A(0,3) = value_type( 0.793637, 0.031147); A(0,4) = value_type(-0.141086,-0.769742);
	A(1,0) = value_type( 0.729781, 0.693097); A(1,1) = value_type( 1.665989, 1.258667); A(1,2) = value_type( 0.620091,-0.192144); A(1,3) = value_type(-1.541503, 0.545104); A(1,4) = value_type( 0.146673,-1.553689);
	A(2,0) = value_type(-0.594370, 0.071316); A(2,1) = value_type( 0.494804,-2.381227); A(2,2) = value_type( 1.004784, 1.097852); A(2,3) = value_type(-0.221373,-0.585458); A(2,4) = value_type(-2.196082,-0.518945);
	A(3,0) = value_type(-1.106269,-1.015812); A(3,1) = value_type( 0.026697, 2.866180); A(3,2) = value_type( 2.687083,-0.115506); A(3,3) = value_type( 0.763162, 0.382183); A(3,4) = value_type( 1.203514, 0.203552);
	A(4,0) = value_type(-0.021184,-1.006383); A(4,1) = value_type(-0.882220, 0.232555); A(4,2) = value_type(-1.618234, 0.615528); A(4,3) = value_type( 1.119524,-2.178697); A(4,4) = value_type( 2.588165,-0.054936);

	matrix_type B(n,n);
	B(0,0) = value_type(-1.592710, 0.804795); B(0,1) = value_type( 0.057283, 0.470640); B(0,2) = value_type(-1.862275,-0.454486); B(0,3) = value_type( 0.712471, 0.887654); B(0,4) = value_type( 0.463207,-0.045117);
	B(1,0) = value_type( 1.072859,-2.009749); B(1,1) = value_type(-1.384371,-0.778200); B(1,2) = value_type( 0.777754, 1.245456); B(1,3) = value_type( 1.914787,-1.246236); B(1,4) = value_type( 0.082774, 0.358980);
	B(2,0) = value_type(-0.451744,-0.353845); B(2,1) = value_type(-0.131528,-2.236258); B(2,2) = value_type(-0.636187,-0.957378); B(2,3) = value_type( 0.984480,-1.840536); B(2,4) = value_type( 0.011728, 1.682497);
	B(3,0) = value_type(-0.876629, 0.764240); B(3,1) = value_type(-0.083787, 2.238476); B(3,2) = value_type( 0.474227,-2.426134); B(3,3) = value_type(-0.042328, 1.129135); B(3,4) = value_type(-0.529845, 0.653758);
	B(4,0) = value_type(-0.812610, 0.996806); B(4,1) = value_type( 0.142456, 0.101454); B(4,2) = value_type( 0.033739,-0.501220); B(4,3) = value_type(-2.000422, 0.181931); B(4,4) = value_type(-0.765401,-1.004076);


	ublasx::qz_decomposition<value_type> qz;

	qz.decompose(A, B);
	qz.reorder(ublasx::rhp_qz_eigenvalues);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "B = " << B );
	BOOST_UBLASX_DEBUG_TRACE( "S = " << qz.S() );
	BOOST_UBLASX_DEBUG_TRACE( "T = " << qz.T() );
	BOOST_UBLASX_DEBUG_TRACE( "Q = " << qz.Q() );
	BOOST_UBLASX_DEBUG_TRACE( "Z = " << qz.Z() );
	BOOST_UBLASX_DEBUG_TRACE( "alpha = " << qz.alpha() );
	BOOST_UBLASX_DEBUG_TRACE( "beta = " << qz.beta() );
	BOOST_UBLASX_DEBUG_TRACE( "eigenvalues lambda = " << qz.eigenvalues() );
	BOOST_UBLASX_DEBUG_TRACE( "right eigenvectors V = " << qz.right_eigenvectors() );
	BOOST_UBLASX_DEBUG_TRACE( "left eigenvectors W = " << qz.left_eigenvectors() );
	matrix_type X;
	X = ublas::prod(qz.Q(), qz.S());
	X = ublas::prod(X, ublas::trans(qz.Z()));
	BOOST_UBLASX_DEBUG_TRACE( "Q*S*Z' = " << X );
	X = ublas::prod(qz.Q(), qz.T());
	X = ublas::prod(X, ublas::trans(qz.Z()));
	BOOST_UBLASX_DEBUG_TRACE( "Q*T*Z' = " << X );

	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.S()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.S()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.T()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.T()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.Q()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.Q()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.Z()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.Z()) == n );
	X = ublas::prod(qz.Q(), qz.S());
	X = ublas::prod(X, ublas::herm(qz.Z()));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, A, n, n, tol );
	X = ublas::prod(qz.Q(), qz.T());
	X = ublas::prod(X, ublas::herm(qz.Z()));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, B, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_complex_row_major_oo_rhp_reorder )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Complex Matrix - Row Major - QZ Object - RHP Reordering");

	typedef std::complex<double> value_type;
	typedef ublas::matrix<value_type, ublas::row_major> matrix_type;

	const std::size_t n(5);

	matrix_type A(n,n);
	A(0,0) = value_type(-0.180557,-0.947835); A(0,1) = value_type( 0.322289, 1.603339); A(0,2) = value_type(-0.651789,-0.902809); A(0,3) = value_type( 0.793637, 0.031147); A(0,4) = value_type(-0.141086,-0.769742);
	A(1,0) = value_type( 0.729781, 0.693097); A(1,1) = value_type( 1.665989, 1.258667); A(1,2) = value_type( 0.620091,-0.192144); A(1,3) = value_type(-1.541503, 0.545104); A(1,4) = value_type( 0.146673,-1.553689);
	A(2,0) = value_type(-0.594370, 0.071316); A(2,1) = value_type( 0.494804,-2.381227); A(2,2) = value_type( 1.004784, 1.097852); A(2,3) = value_type(-0.221373,-0.585458); A(2,4) = value_type(-2.196082,-0.518945);
	A(3,0) = value_type(-1.106269,-1.015812); A(3,1) = value_type( 0.026697, 2.866180); A(3,2) = value_type( 2.687083,-0.115506); A(3,3) = value_type( 0.763162, 0.382183); A(3,4) = value_type( 1.203514, 0.203552);
	A(4,0) = value_type(-0.021184,-1.006383); A(4,1) = value_type(-0.882220, 0.232555); A(4,2) = value_type(-1.618234, 0.615528); A(4,3) = value_type( 1.119524,-2.178697); A(4,4) = value_type( 2.588165,-0.054936);

	matrix_type B(n,n);
	B(0,0) = value_type(-1.592710, 0.804795); B(0,1) = value_type( 0.057283, 0.470640); B(0,2) = value_type(-1.862275,-0.454486); B(0,3) = value_type( 0.712471, 0.887654); B(0,4) = value_type( 0.463207,-0.045117);
	B(1,0) = value_type( 1.072859,-2.009749); B(1,1) = value_type(-1.384371,-0.778200); B(1,2) = value_type( 0.777754, 1.245456); B(1,3) = value_type( 1.914787,-1.246236); B(1,4) = value_type( 0.082774, 0.358980);
	B(2,0) = value_type(-0.451744,-0.353845); B(2,1) = value_type(-0.131528,-2.236258); B(2,2) = value_type(-0.636187,-0.957378); B(2,3) = value_type( 0.984480,-1.840536); B(2,4) = value_type( 0.011728, 1.682497);
	B(3,0) = value_type(-0.876629, 0.764240); B(3,1) = value_type(-0.083787, 2.238476); B(3,2) = value_type( 0.474227,-2.426134); B(3,3) = value_type(-0.042328, 1.129135); B(3,4) = value_type(-0.529845, 0.653758);
	B(4,0) = value_type(-0.812610, 0.996806); B(4,1) = value_type( 0.142456, 0.101454); B(4,2) = value_type( 0.033739,-0.501220); B(4,3) = value_type(-2.000422, 0.181931); B(4,4) = value_type(-0.765401,-1.004076);


	ublasx::qz_decomposition<value_type> qz;

	qz.decompose(A, B);
	qz.reorder(ublasx::rhp_qz_eigenvalues);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "B = " << B );
	BOOST_UBLASX_DEBUG_TRACE( "S = " << qz.S() );
	BOOST_UBLASX_DEBUG_TRACE( "T = " << qz.T() );
	BOOST_UBLASX_DEBUG_TRACE( "Q = " << qz.Q() );
	BOOST_UBLASX_DEBUG_TRACE( "Z = " << qz.Z() );
	BOOST_UBLASX_DEBUG_TRACE( "alpha = " << qz.alpha() );
	BOOST_UBLASX_DEBUG_TRACE( "beta = " << qz.beta() );
	BOOST_UBLASX_DEBUG_TRACE( "eigenvalues lambda = " << qz.eigenvalues() );
	BOOST_UBLASX_DEBUG_TRACE( "right eigenvectors V = " << qz.right_eigenvectors() );
	BOOST_UBLASX_DEBUG_TRACE( "left eigenvectors W = " << qz.left_eigenvectors() );
	matrix_type X;
	X = ublas::prod(qz.Q(), qz.S());
	X = ublas::prod(X, ublas::trans(qz.Z()));
	BOOST_UBLASX_DEBUG_TRACE( "Q*S*Z' = " << X );
	X = ublas::prod(qz.Q(), qz.T());
	X = ublas::prod(X, ublas::trans(qz.Z()));
	BOOST_UBLASX_DEBUG_TRACE( "Q*T*Z' = " << X );

	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.S()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.S()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.T()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.T()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.Q()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.Q()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.Z()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.Z()) == n );
	X = ublas::prod(qz.Q(), qz.S());
	X = ublas::prod(X, ublas::herm(qz.Z()));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, A, n, n, tol );
	X = ublas::prod(qz.Q(), qz.T());
	X = ublas::prod(X, ublas::herm(qz.Z()));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, B, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_complex_column_major_oo_udi_reorder )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Complex Matrix - Column Major - QZ Object - UDI Reordering");

	typedef std::complex<double> value_type;
	typedef ublas::matrix<value_type, ublas::column_major> matrix_type;

	const std::size_t n(5);

	matrix_type A(n,n);
	A(0,0) = value_type(-0.180557,-0.947835); A(0,1) = value_type( 0.322289, 1.603339); A(0,2) = value_type(-0.651789,-0.902809); A(0,3) = value_type( 0.793637, 0.031147); A(0,4) = value_type(-0.141086,-0.769742);
	A(1,0) = value_type( 0.729781, 0.693097); A(1,1) = value_type( 1.665989, 1.258667); A(1,2) = value_type( 0.620091,-0.192144); A(1,3) = value_type(-1.541503, 0.545104); A(1,4) = value_type( 0.146673,-1.553689);
	A(2,0) = value_type(-0.594370, 0.071316); A(2,1) = value_type( 0.494804,-2.381227); A(2,2) = value_type( 1.004784, 1.097852); A(2,3) = value_type(-0.221373,-0.585458); A(2,4) = value_type(-2.196082,-0.518945);
	A(3,0) = value_type(-1.106269,-1.015812); A(3,1) = value_type( 0.026697, 2.866180); A(3,2) = value_type( 2.687083,-0.115506); A(3,3) = value_type( 0.763162, 0.382183); A(3,4) = value_type( 1.203514, 0.203552);
	A(4,0) = value_type(-0.021184,-1.006383); A(4,1) = value_type(-0.882220, 0.232555); A(4,2) = value_type(-1.618234, 0.615528); A(4,3) = value_type( 1.119524,-2.178697); A(4,4) = value_type( 2.588165,-0.054936);

	matrix_type B(n,n);
	B(0,0) = value_type(-1.592710, 0.804795); B(0,1) = value_type( 0.057283, 0.470640); B(0,2) = value_type(-1.862275,-0.454486); B(0,3) = value_type( 0.712471, 0.887654); B(0,4) = value_type( 0.463207,-0.045117);
	B(1,0) = value_type( 1.072859,-2.009749); B(1,1) = value_type(-1.384371,-0.778200); B(1,2) = value_type( 0.777754, 1.245456); B(1,3) = value_type( 1.914787,-1.246236); B(1,4) = value_type( 0.082774, 0.358980);
	B(2,0) = value_type(-0.451744,-0.353845); B(2,1) = value_type(-0.131528,-2.236258); B(2,2) = value_type(-0.636187,-0.957378); B(2,3) = value_type( 0.984480,-1.840536); B(2,4) = value_type( 0.011728, 1.682497);
	B(3,0) = value_type(-0.876629, 0.764240); B(3,1) = value_type(-0.083787, 2.238476); B(3,2) = value_type( 0.474227,-2.426134); B(3,3) = value_type(-0.042328, 1.129135); B(3,4) = value_type(-0.529845, 0.653758);
	B(4,0) = value_type(-0.812610, 0.996806); B(4,1) = value_type( 0.142456, 0.101454); B(4,2) = value_type( 0.033739,-0.501220); B(4,3) = value_type(-2.000422, 0.181931); B(4,4) = value_type(-0.765401,-1.004076);


	ublasx::qz_decomposition<value_type> qz;

	qz.decompose(A, B);
	qz.reorder(ublasx::udi_qz_eigenvalues);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "B = " << B );
	BOOST_UBLASX_DEBUG_TRACE( "S = " << qz.S() );
	BOOST_UBLASX_DEBUG_TRACE( "T = " << qz.T() );
	BOOST_UBLASX_DEBUG_TRACE( "Q = " << qz.Q() );
	BOOST_UBLASX_DEBUG_TRACE( "Z = " << qz.Z() );
	BOOST_UBLASX_DEBUG_TRACE( "alpha = " << qz.alpha() );
	BOOST_UBLASX_DEBUG_TRACE( "beta = " << qz.beta() );
	BOOST_UBLASX_DEBUG_TRACE( "eigenvalues lambda = " << qz.eigenvalues() );
	BOOST_UBLASX_DEBUG_TRACE( "right eigenvectors V = " << qz.right_eigenvectors() );
	BOOST_UBLASX_DEBUG_TRACE( "left eigenvectors W = " << qz.left_eigenvectors() );
	matrix_type X;
	X = ublas::prod(qz.Q(), qz.S());
	X = ublas::prod(X, ublas::trans(qz.Z()));
	BOOST_UBLASX_DEBUG_TRACE( "Q*S*Z' = " << X );
	X = ublas::prod(qz.Q(), qz.T());
	X = ublas::prod(X, ublas::trans(qz.Z()));
	BOOST_UBLASX_DEBUG_TRACE( "Q*T*Z' = " << X );

	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.S()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.S()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.T()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.T()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.Q()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.Q()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.Z()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.Z()) == n );
	X = ublas::prod(qz.Q(), qz.S());
	X = ublas::prod(X, ublas::herm(qz.Z()));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, A, n, n, tol );
	X = ublas::prod(qz.Q(), qz.T());
	X = ublas::prod(X, ublas::herm(qz.Z()));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, B, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_complex_row_major_oo_udi_reorder )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Complex Matrix - Row Major - QZ Object - UDI Reordering");

	typedef std::complex<double> value_type;
	typedef ublas::matrix<value_type, ublas::row_major> matrix_type;

	const std::size_t n(5);

	matrix_type A(n,n);
	A(0,0) = value_type(-0.180557,-0.947835); A(0,1) = value_type( 0.322289, 1.603339); A(0,2) = value_type(-0.651789,-0.902809); A(0,3) = value_type( 0.793637, 0.031147); A(0,4) = value_type(-0.141086,-0.769742);
	A(1,0) = value_type( 0.729781, 0.693097); A(1,1) = value_type( 1.665989, 1.258667); A(1,2) = value_type( 0.620091,-0.192144); A(1,3) = value_type(-1.541503, 0.545104); A(1,4) = value_type( 0.146673,-1.553689);
	A(2,0) = value_type(-0.594370, 0.071316); A(2,1) = value_type( 0.494804,-2.381227); A(2,2) = value_type( 1.004784, 1.097852); A(2,3) = value_type(-0.221373,-0.585458); A(2,4) = value_type(-2.196082,-0.518945);
	A(3,0) = value_type(-1.106269,-1.015812); A(3,1) = value_type( 0.026697, 2.866180); A(3,2) = value_type( 2.687083,-0.115506); A(3,3) = value_type( 0.763162, 0.382183); A(3,4) = value_type( 1.203514, 0.203552);
	A(4,0) = value_type(-0.021184,-1.006383); A(4,1) = value_type(-0.882220, 0.232555); A(4,2) = value_type(-1.618234, 0.615528); A(4,3) = value_type( 1.119524,-2.178697); A(4,4) = value_type( 2.588165,-0.054936);

	matrix_type B(n,n);
	B(0,0) = value_type(-1.592710, 0.804795); B(0,1) = value_type( 0.057283, 0.470640); B(0,2) = value_type(-1.862275,-0.454486); B(0,3) = value_type( 0.712471, 0.887654); B(0,4) = value_type( 0.463207,-0.045117);
	B(1,0) = value_type( 1.072859,-2.009749); B(1,1) = value_type(-1.384371,-0.778200); B(1,2) = value_type( 0.777754, 1.245456); B(1,3) = value_type( 1.914787,-1.246236); B(1,4) = value_type( 0.082774, 0.358980);
	B(2,0) = value_type(-0.451744,-0.353845); B(2,1) = value_type(-0.131528,-2.236258); B(2,2) = value_type(-0.636187,-0.957378); B(2,3) = value_type( 0.984480,-1.840536); B(2,4) = value_type( 0.011728, 1.682497);
	B(3,0) = value_type(-0.876629, 0.764240); B(3,1) = value_type(-0.083787, 2.238476); B(3,2) = value_type( 0.474227,-2.426134); B(3,3) = value_type(-0.042328, 1.129135); B(3,4) = value_type(-0.529845, 0.653758);
	B(4,0) = value_type(-0.812610, 0.996806); B(4,1) = value_type( 0.142456, 0.101454); B(4,2) = value_type( 0.033739,-0.501220); B(4,3) = value_type(-2.000422, 0.181931); B(4,4) = value_type(-0.765401,-1.004076);


	ublasx::qz_decomposition<value_type> qz;

	qz.decompose(A, B);
	qz.reorder(ublasx::udi_qz_eigenvalues);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "B = " << B );
	BOOST_UBLASX_DEBUG_TRACE( "S = " << qz.S() );
	BOOST_UBLASX_DEBUG_TRACE( "T = " << qz.T() );
	BOOST_UBLASX_DEBUG_TRACE( "Q = " << qz.Q() );
	BOOST_UBLASX_DEBUG_TRACE( "Z = " << qz.Z() );
	BOOST_UBLASX_DEBUG_TRACE( "alpha = " << qz.alpha() );
	BOOST_UBLASX_DEBUG_TRACE( "beta = " << qz.beta() );
	BOOST_UBLASX_DEBUG_TRACE( "eigenvalues lambda = " << qz.eigenvalues() );
	BOOST_UBLASX_DEBUG_TRACE( "right eigenvectors V = " << qz.right_eigenvectors() );
	BOOST_UBLASX_DEBUG_TRACE( "left eigenvectors W = " << qz.left_eigenvectors() );
	matrix_type X;
	X = ublas::prod(qz.Q(), qz.S());
	X = ublas::prod(X, ublas::trans(qz.Z()));
	BOOST_UBLASX_DEBUG_TRACE( "Q*S*Z' = " << X );
	X = ublas::prod(qz.Q(), qz.T());
	X = ublas::prod(X, ublas::trans(qz.Z()));
	BOOST_UBLASX_DEBUG_TRACE( "Q*T*Z' = " << X );

	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.S()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.S()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.T()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.T()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.Q()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.Q()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.Z()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.Z()) == n );
	X = ublas::prod(qz.Q(), qz.S());
	X = ublas::prod(X, ublas::herm(qz.Z()));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, A, n, n, tol );
	X = ublas::prod(qz.Q(), qz.T());
	X = ublas::prod(X, ublas::herm(qz.Z()));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, B, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_complex_column_major_oo_udo_reorder )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Complex Matrix - Column Major - QZ Object - UDO Reordering");

	typedef std::complex<double> value_type;
	typedef ublas::matrix<value_type, ublas::column_major> matrix_type;

	const std::size_t n(5);

	matrix_type A(n,n);
	A(0,0) = value_type(-0.180557,-0.947835); A(0,1) = value_type( 0.322289, 1.603339); A(0,2) = value_type(-0.651789,-0.902809); A(0,3) = value_type( 0.793637, 0.031147); A(0,4) = value_type(-0.141086,-0.769742);
	A(1,0) = value_type( 0.729781, 0.693097); A(1,1) = value_type( 1.665989, 1.258667); A(1,2) = value_type( 0.620091,-0.192144); A(1,3) = value_type(-1.541503, 0.545104); A(1,4) = value_type( 0.146673,-1.553689);
	A(2,0) = value_type(-0.594370, 0.071316); A(2,1) = value_type( 0.494804,-2.381227); A(2,2) = value_type( 1.004784, 1.097852); A(2,3) = value_type(-0.221373,-0.585458); A(2,4) = value_type(-2.196082,-0.518945);
	A(3,0) = value_type(-1.106269,-1.015812); A(3,1) = value_type( 0.026697, 2.866180); A(3,2) = value_type( 2.687083,-0.115506); A(3,3) = value_type( 0.763162, 0.382183); A(3,4) = value_type( 1.203514, 0.203552);
	A(4,0) = value_type(-0.021184,-1.006383); A(4,1) = value_type(-0.882220, 0.232555); A(4,2) = value_type(-1.618234, 0.615528); A(4,3) = value_type( 1.119524,-2.178697); A(4,4) = value_type( 2.588165,-0.054936);

	matrix_type B(n,n);
	B(0,0) = value_type(-1.592710, 0.804795); B(0,1) = value_type( 0.057283, 0.470640); B(0,2) = value_type(-1.862275,-0.454486); B(0,3) = value_type( 0.712471, 0.887654); B(0,4) = value_type( 0.463207,-0.045117);
	B(1,0) = value_type( 1.072859,-2.009749); B(1,1) = value_type(-1.384371,-0.778200); B(1,2) = value_type( 0.777754, 1.245456); B(1,3) = value_type( 1.914787,-1.246236); B(1,4) = value_type( 0.082774, 0.358980);
	B(2,0) = value_type(-0.451744,-0.353845); B(2,1) = value_type(-0.131528,-2.236258); B(2,2) = value_type(-0.636187,-0.957378); B(2,3) = value_type( 0.984480,-1.840536); B(2,4) = value_type( 0.011728, 1.682497);
	B(3,0) = value_type(-0.876629, 0.764240); B(3,1) = value_type(-0.083787, 2.238476); B(3,2) = value_type( 0.474227,-2.426134); B(3,3) = value_type(-0.042328, 1.129135); B(3,4) = value_type(-0.529845, 0.653758);
	B(4,0) = value_type(-0.812610, 0.996806); B(4,1) = value_type( 0.142456, 0.101454); B(4,2) = value_type( 0.033739,-0.501220); B(4,3) = value_type(-2.000422, 0.181931); B(4,4) = value_type(-0.765401,-1.004076);


	ublasx::qz_decomposition<value_type> qz;

	qz.decompose(A, B);
	qz.reorder(ublasx::udo_qz_eigenvalues);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "B = " << B );
	BOOST_UBLASX_DEBUG_TRACE( "S = " << qz.S() );
	BOOST_UBLASX_DEBUG_TRACE( "T = " << qz.T() );
	BOOST_UBLASX_DEBUG_TRACE( "Q = " << qz.Q() );
	BOOST_UBLASX_DEBUG_TRACE( "Z = " << qz.Z() );
	BOOST_UBLASX_DEBUG_TRACE( "alpha = " << qz.alpha() );
	BOOST_UBLASX_DEBUG_TRACE( "beta = " << qz.beta() );
	BOOST_UBLASX_DEBUG_TRACE( "eigenvalues lambda = " << qz.eigenvalues() );
	BOOST_UBLASX_DEBUG_TRACE( "right eigenvectors V = " << qz.right_eigenvectors() );
	BOOST_UBLASX_DEBUG_TRACE( "left eigenvectors W = " << qz.left_eigenvectors() );
	matrix_type X;
	X = ublas::prod(qz.Q(), qz.S());
	X = ublas::prod(X, ublas::trans(qz.Z()));
	BOOST_UBLASX_DEBUG_TRACE( "Q*S*Z' = " << X );
	X = ublas::prod(qz.Q(), qz.T());
	X = ublas::prod(X, ublas::trans(qz.Z()));
	BOOST_UBLASX_DEBUG_TRACE( "Q*T*Z' = " << X );

	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.S()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.S()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.T()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.T()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.Q()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.Q()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.Z()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.Z()) == n );
	X = ublas::prod(qz.Q(), qz.S());
	X = ublas::prod(X, ublas::herm(qz.Z()));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, A, n, n, tol );
	X = ublas::prod(qz.Q(), qz.T());
	X = ublas::prod(X, ublas::herm(qz.Z()));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, B, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_complex_row_major_oo_udo_reorder )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Complex Matrix - Row Major - QZ Object - UDO Reordering");

	typedef std::complex<double> value_type;
	typedef ublas::matrix<value_type, ublas::row_major> matrix_type;

	const std::size_t n(5);

	matrix_type A(n,n);
	A(0,0) = value_type(-0.180557,-0.947835); A(0,1) = value_type( 0.322289, 1.603339); A(0,2) = value_type(-0.651789,-0.902809); A(0,3) = value_type( 0.793637, 0.031147); A(0,4) = value_type(-0.141086,-0.769742);
	A(1,0) = value_type( 0.729781, 0.693097); A(1,1) = value_type( 1.665989, 1.258667); A(1,2) = value_type( 0.620091,-0.192144); A(1,3) = value_type(-1.541503, 0.545104); A(1,4) = value_type( 0.146673,-1.553689);
	A(2,0) = value_type(-0.594370, 0.071316); A(2,1) = value_type( 0.494804,-2.381227); A(2,2) = value_type( 1.004784, 1.097852); A(2,3) = value_type(-0.221373,-0.585458); A(2,4) = value_type(-2.196082,-0.518945);
	A(3,0) = value_type(-1.106269,-1.015812); A(3,1) = value_type( 0.026697, 2.866180); A(3,2) = value_type( 2.687083,-0.115506); A(3,3) = value_type( 0.763162, 0.382183); A(3,4) = value_type( 1.203514, 0.203552);
	A(4,0) = value_type(-0.021184,-1.006383); A(4,1) = value_type(-0.882220, 0.232555); A(4,2) = value_type(-1.618234, 0.615528); A(4,3) = value_type( 1.119524,-2.178697); A(4,4) = value_type( 2.588165,-0.054936);

	matrix_type B(n,n);
	B(0,0) = value_type(-1.592710, 0.804795); B(0,1) = value_type( 0.057283, 0.470640); B(0,2) = value_type(-1.862275,-0.454486); B(0,3) = value_type( 0.712471, 0.887654); B(0,4) = value_type( 0.463207,-0.045117);
	B(1,0) = value_type( 1.072859,-2.009749); B(1,1) = value_type(-1.384371,-0.778200); B(1,2) = value_type( 0.777754, 1.245456); B(1,3) = value_type( 1.914787,-1.246236); B(1,4) = value_type( 0.082774, 0.358980);
	B(2,0) = value_type(-0.451744,-0.353845); B(2,1) = value_type(-0.131528,-2.236258); B(2,2) = value_type(-0.636187,-0.957378); B(2,3) = value_type( 0.984480,-1.840536); B(2,4) = value_type( 0.011728, 1.682497);
	B(3,0) = value_type(-0.876629, 0.764240); B(3,1) = value_type(-0.083787, 2.238476); B(3,2) = value_type( 0.474227,-2.426134); B(3,3) = value_type(-0.042328, 1.129135); B(3,4) = value_type(-0.529845, 0.653758);
	B(4,0) = value_type(-0.812610, 0.996806); B(4,1) = value_type( 0.142456, 0.101454); B(4,2) = value_type( 0.033739,-0.501220); B(4,3) = value_type(-2.000422, 0.181931); B(4,4) = value_type(-0.765401,-1.004076);


	ublasx::qz_decomposition<value_type> qz;

	qz.decompose(A, B);
	qz.reorder(ublasx::udo_qz_eigenvalues);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "B = " << B );
	BOOST_UBLASX_DEBUG_TRACE( "S = " << qz.S() );
	BOOST_UBLASX_DEBUG_TRACE( "T = " << qz.T() );
	BOOST_UBLASX_DEBUG_TRACE( "Q = " << qz.Q() );
	BOOST_UBLASX_DEBUG_TRACE( "Z = " << qz.Z() );
	BOOST_UBLASX_DEBUG_TRACE( "alpha = " << qz.alpha() );
	BOOST_UBLASX_DEBUG_TRACE( "beta = " << qz.beta() );
	BOOST_UBLASX_DEBUG_TRACE( "eigenvalues lambda = " << qz.eigenvalues() );
	BOOST_UBLASX_DEBUG_TRACE( "right eigenvectors V = " << qz.right_eigenvectors() );
	BOOST_UBLASX_DEBUG_TRACE( "left eigenvectors W = " << qz.left_eigenvectors() );
	matrix_type X;
	X = ublas::prod(qz.Q(), qz.S());
	X = ublas::prod(X, ublas::trans(qz.Z()));
	BOOST_UBLASX_DEBUG_TRACE( "Q*S*Z' = " << X );
	X = ublas::prod(qz.Q(), qz.T());
	X = ublas::prod(X, ublas::trans(qz.Z()));
	BOOST_UBLASX_DEBUG_TRACE( "Q*T*Z' = " << X );

	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.S()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.S()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.T()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.T()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.Q()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.Q()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.Z()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.Z()) == n );
	X = ublas::prod(qz.Q(), qz.S());
	X = ublas::prod(X, ublas::herm(qz.Z()));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, A, n, n, tol );
	X = ublas::prod(qz.Q(), qz.T());
	X = ublas::prod(X, ublas::herm(qz.Z()));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, B, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_complex_column_major_oo_custom_reorder )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Complex Matrix - Column Major - QZ Object - Custom Reordering");

	typedef std::complex<double> value_type;
	typedef ublas::matrix<value_type, ublas::column_major> matrix_type;

	const std::size_t n(5);

	matrix_type A(n,n);
	A(0,0) = value_type(-0.180557,-0.947835); A(0,1) = value_type( 0.322289, 1.603339); A(0,2) = value_type(-0.651789,-0.902809); A(0,3) = value_type( 0.793637, 0.031147); A(0,4) = value_type(-0.141086,-0.769742);
	A(1,0) = value_type( 0.729781, 0.693097); A(1,1) = value_type( 1.665989, 1.258667); A(1,2) = value_type( 0.620091,-0.192144); A(1,3) = value_type(-1.541503, 0.545104); A(1,4) = value_type( 0.146673,-1.553689);
	A(2,0) = value_type(-0.594370, 0.071316); A(2,1) = value_type( 0.494804,-2.381227); A(2,2) = value_type( 1.004784, 1.097852); A(2,3) = value_type(-0.221373,-0.585458); A(2,4) = value_type(-2.196082,-0.518945);
	A(3,0) = value_type(-1.106269,-1.015812); A(3,1) = value_type( 0.026697, 2.866180); A(3,2) = value_type( 2.687083,-0.115506); A(3,3) = value_type( 0.763162, 0.382183); A(3,4) = value_type( 1.203514, 0.203552);
	A(4,0) = value_type(-0.021184,-1.006383); A(4,1) = value_type(-0.882220, 0.232555); A(4,2) = value_type(-1.618234, 0.615528); A(4,3) = value_type( 1.119524,-2.178697); A(4,4) = value_type( 2.588165,-0.054936);

	matrix_type B(n,n);
	B(0,0) = value_type(-1.592710, 0.804795); B(0,1) = value_type( 0.057283, 0.470640); B(0,2) = value_type(-1.862275,-0.454486); B(0,3) = value_type( 0.712471, 0.887654); B(0,4) = value_type( 0.463207,-0.045117);
	B(1,0) = value_type( 1.072859,-2.009749); B(1,1) = value_type(-1.384371,-0.778200); B(1,2) = value_type( 0.777754, 1.245456); B(1,3) = value_type( 1.914787,-1.246236); B(1,4) = value_type( 0.082774, 0.358980);
	B(2,0) = value_type(-0.451744,-0.353845); B(2,1) = value_type(-0.131528,-2.236258); B(2,2) = value_type(-0.636187,-0.957378); B(2,3) = value_type( 0.984480,-1.840536); B(2,4) = value_type( 0.011728, 1.682497);
	B(3,0) = value_type(-0.876629, 0.764240); B(3,1) = value_type(-0.083787, 2.238476); B(3,2) = value_type( 0.474227,-2.426134); B(3,3) = value_type(-0.042328, 1.129135); B(3,4) = value_type(-0.529845, 0.653758);
	B(4,0) = value_type(-0.812610, 0.996806); B(4,1) = value_type( 0.142456, 0.101454); B(4,2) = value_type( 0.033739,-0.501220); B(4,3) = value_type(-2.000422, 0.181931); B(4,4) = value_type(-0.765401,-1.004076);


	ublasx::qz_decomposition<value_type> qz;

	qz.decompose(A, B);

	// Selection: select eigenvalues at even position (no sense ... just a test)
	std::size_t n_e = ublasx::size(qz.eigenvalues());
	ublas::vector<bool> selection(n_e);
	for (std::size_t i = 0; i < n_e; ++i)
	{
		selection(i) = ((i+1) % 2) ? true : false;
	}

	qz.reorder(selection);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "B = " << B );
	BOOST_UBLASX_DEBUG_TRACE( "S = " << qz.S() );
	BOOST_UBLASX_DEBUG_TRACE( "T = " << qz.T() );
	BOOST_UBLASX_DEBUG_TRACE( "Q = " << qz.Q() );
	BOOST_UBLASX_DEBUG_TRACE( "Z = " << qz.Z() );
	BOOST_UBLASX_DEBUG_TRACE( "alpha = " << qz.alpha() );
	BOOST_UBLASX_DEBUG_TRACE( "beta = " << qz.beta() );
	BOOST_UBLASX_DEBUG_TRACE( "eigenvalues lambda = " << qz.eigenvalues() );
	BOOST_UBLASX_DEBUG_TRACE( "right eigenvectors V = " << qz.right_eigenvectors() );
	BOOST_UBLASX_DEBUG_TRACE( "left eigenvectors W = " << qz.left_eigenvectors() );
	matrix_type X;
	X = ublas::prod(qz.Q(), qz.S());
	X = ublas::prod(X, ublas::trans(qz.Z()));
	BOOST_UBLASX_DEBUG_TRACE( "Q*S*Z' = " << X );
	X = ublas::prod(qz.Q(), qz.T());
	X = ublas::prod(X, ublas::trans(qz.Z()));
	BOOST_UBLASX_DEBUG_TRACE( "Q*T*Z' = " << X );

	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.S()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.S()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.T()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.T()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.Q()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.Q()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.Z()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.Z()) == n );
	X = ublas::prod(qz.Q(), qz.S());
	X = ublas::prod(X, ublas::herm(qz.Z()));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, A, n, n, tol );
	X = ublas::prod(qz.Q(), qz.T());
	X = ublas::prod(X, ublas::herm(qz.Z()));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, B, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( test_complex_row_major_oo_custom_reorder )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Complex Matrix - Row Major - QZ Object - Custom Reordering");

	typedef std::complex<double> value_type;
	typedef ublas::matrix<value_type, ublas::row_major> matrix_type;

	const std::size_t n(5);

	matrix_type A(n,n);
	A(0,0) = value_type(-0.180557,-0.947835); A(0,1) = value_type( 0.322289, 1.603339); A(0,2) = value_type(-0.651789,-0.902809); A(0,3) = value_type( 0.793637, 0.031147); A(0,4) = value_type(-0.141086,-0.769742);
	A(1,0) = value_type( 0.729781, 0.693097); A(1,1) = value_type( 1.665989, 1.258667); A(1,2) = value_type( 0.620091,-0.192144); A(1,3) = value_type(-1.541503, 0.545104); A(1,4) = value_type( 0.146673,-1.553689);
	A(2,0) = value_type(-0.594370, 0.071316); A(2,1) = value_type( 0.494804,-2.381227); A(2,2) = value_type( 1.004784, 1.097852); A(2,3) = value_type(-0.221373,-0.585458); A(2,4) = value_type(-2.196082,-0.518945);
	A(3,0) = value_type(-1.106269,-1.015812); A(3,1) = value_type( 0.026697, 2.866180); A(3,2) = value_type( 2.687083,-0.115506); A(3,3) = value_type( 0.763162, 0.382183); A(3,4) = value_type( 1.203514, 0.203552);
	A(4,0) = value_type(-0.021184,-1.006383); A(4,1) = value_type(-0.882220, 0.232555); A(4,2) = value_type(-1.618234, 0.615528); A(4,3) = value_type( 1.119524,-2.178697); A(4,4) = value_type( 2.588165,-0.054936);

	matrix_type B(n,n);
	B(0,0) = value_type(-1.592710, 0.804795); B(0,1) = value_type( 0.057283, 0.470640); B(0,2) = value_type(-1.862275,-0.454486); B(0,3) = value_type( 0.712471, 0.887654); B(0,4) = value_type( 0.463207,-0.045117);
	B(1,0) = value_type( 1.072859,-2.009749); B(1,1) = value_type(-1.384371,-0.778200); B(1,2) = value_type( 0.777754, 1.245456); B(1,3) = value_type( 1.914787,-1.246236); B(1,4) = value_type( 0.082774, 0.358980);
	B(2,0) = value_type(-0.451744,-0.353845); B(2,1) = value_type(-0.131528,-2.236258); B(2,2) = value_type(-0.636187,-0.957378); B(2,3) = value_type( 0.984480,-1.840536); B(2,4) = value_type( 0.011728, 1.682497);
	B(3,0) = value_type(-0.876629, 0.764240); B(3,1) = value_type(-0.083787, 2.238476); B(3,2) = value_type( 0.474227,-2.426134); B(3,3) = value_type(-0.042328, 1.129135); B(3,4) = value_type(-0.529845, 0.653758);
	B(4,0) = value_type(-0.812610, 0.996806); B(4,1) = value_type( 0.142456, 0.101454); B(4,2) = value_type( 0.033739,-0.501220); B(4,3) = value_type(-2.000422, 0.181931); B(4,4) = value_type(-0.765401,-1.004076);


	ublasx::qz_decomposition<value_type> qz;

	qz.decompose(A, B);

	// Selection: select eigenvalues at even position (no sense ... just a test)
	std::size_t n_e = ublasx::size(qz.eigenvalues());
	ublas::vector<bool> selection(n_e);
	for (std::size_t i = 0; i < n_e; ++i)
	{
		selection(i) = ((i+1) % 2) ? true : false;
	}

	qz.reorder(selection);

	BOOST_UBLASX_DEBUG_TRACE( "A = " << A );
	BOOST_UBLASX_DEBUG_TRACE( "B = " << B );
	BOOST_UBLASX_DEBUG_TRACE( "S = " << qz.S() );
	BOOST_UBLASX_DEBUG_TRACE( "T = " << qz.T() );
	BOOST_UBLASX_DEBUG_TRACE( "Q = " << qz.Q() );
	BOOST_UBLASX_DEBUG_TRACE( "Z = " << qz.Z() );
	BOOST_UBLASX_DEBUG_TRACE( "alpha = " << qz.alpha() );
	BOOST_UBLASX_DEBUG_TRACE( "beta = " << qz.beta() );
	BOOST_UBLASX_DEBUG_TRACE( "eigenvalues lambda = " << qz.eigenvalues() );
	BOOST_UBLASX_DEBUG_TRACE( "right eigenvectors V = " << qz.right_eigenvectors() );
	BOOST_UBLASX_DEBUG_TRACE( "left eigenvectors W = " << qz.left_eigenvectors() );
	matrix_type X;
	X = ublas::prod(qz.Q(), qz.S());
	X = ublas::prod(X, ublas::trans(qz.Z()));
	BOOST_UBLASX_DEBUG_TRACE( "Q*S*Z' = " << X );
	X = ublas::prod(qz.Q(), qz.T());
	X = ublas::prod(X, ublas::trans(qz.Z()));
	BOOST_UBLASX_DEBUG_TRACE( "Q*T*Z' = " << X );

	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.S()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.S()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.T()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.T()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.Q()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.Q()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_rows(qz.Z()) == n );
	BOOST_UBLASX_TEST_CHECK( ublasx::num_columns(qz.Z()) == n );
	X = ublas::prod(qz.Q(), qz.S());
	X = ublas::prod(X, ublas::herm(qz.Z()));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, A, n, n, tol );
	X = ublas::prod(qz.Q(), qz.T());
	X = ublas::prod(X, ublas::herm(qz.Z()));
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, B, n, n, tol );
}


int main()
{
	BOOST_UBLASX_DEBUG_TRACE("Test Suite: QZ factorization");

	BOOST_UBLASX_TEST_BEGIN();

	BOOST_UBLASX_TEST_DO( test_real_column_major_decomp );
	BOOST_UBLASX_TEST_DO( test_real_row_major_decomp );
	BOOST_UBLASX_TEST_DO( test_real_column_major_decomp_lhp_reorder );
	BOOST_UBLASX_TEST_DO( test_real_row_major_decomp_lhp_reorder );
	BOOST_UBLASX_TEST_DO( test_real_column_major_decomp_rhp_reorder );
	BOOST_UBLASX_TEST_DO( test_real_row_major_decomp_rhp_reorder );
	BOOST_UBLASX_TEST_DO( test_real_column_major_decomp_udi_reorder );
	BOOST_UBLASX_TEST_DO( test_real_row_major_decomp_udi_reorder );
	BOOST_UBLASX_TEST_DO( test_real_column_major_decomp_udo_reorder );
	BOOST_UBLASX_TEST_DO( test_real_row_major_decomp_udo_reorder );
	BOOST_UBLASX_TEST_DO( test_real_column_major_decomp_custom_reorder );
	BOOST_UBLASX_TEST_DO( test_real_row_major_decomp_custom_reorder );
	BOOST_UBLASX_TEST_DO( test_complex_column_major_decomp );
	BOOST_UBLASX_TEST_DO( test_complex_row_major_decomp );
	BOOST_UBLASX_TEST_DO( test_complex_column_major_decomp_lhp_reorder );
	BOOST_UBLASX_TEST_DO( test_complex_row_major_decomp_lhp_reorder );
	BOOST_UBLASX_TEST_DO( test_complex_column_major_decomp_rhp_reorder );
	BOOST_UBLASX_TEST_DO( test_complex_row_major_decomp_rhp_reorder );
	BOOST_UBLASX_TEST_DO( test_complex_column_major_decomp_udi_reorder );
	BOOST_UBLASX_TEST_DO( test_complex_row_major_decomp_udi_reorder );
	BOOST_UBLASX_TEST_DO( test_complex_column_major_decomp_udo_reorder );
	BOOST_UBLASX_TEST_DO( test_complex_row_major_decomp_udo_reorder );
	BOOST_UBLASX_TEST_DO( test_complex_column_major_decomp_custom_reorder );
	BOOST_UBLASX_TEST_DO( test_complex_row_major_decomp_custom_reorder );
	BOOST_UBLASX_TEST_DO( test_real_column_major_oo_decomp );
	BOOST_UBLASX_TEST_DO( test_real_row_major_oo_decomp );
	BOOST_UBLASX_TEST_DO( test_real_column_major_oo_lhp_reorder );
	BOOST_UBLASX_TEST_DO( test_real_row_major_oo_lhp_reorder );
	BOOST_UBLASX_TEST_DO( test_real_column_major_oo_rhp_reorder );
	BOOST_UBLASX_TEST_DO( test_real_row_major_oo_rhp_reorder );
	BOOST_UBLASX_TEST_DO( test_real_column_major_oo_udi_reorder );
	BOOST_UBLASX_TEST_DO( test_real_row_major_oo_udi_reorder );
	BOOST_UBLASX_TEST_DO( test_real_column_major_oo_udo_reorder );
	BOOST_UBLASX_TEST_DO( test_real_row_major_oo_udo_reorder );
	BOOST_UBLASX_TEST_DO( test_real_column_major_oo_custom_reorder );
	BOOST_UBLASX_TEST_DO( test_real_row_major_oo_custom_reorder );
	BOOST_UBLASX_TEST_DO( test_complex_column_major_oo_decomp );
	BOOST_UBLASX_TEST_DO( test_complex_row_major_oo_decomp );
	BOOST_UBLASX_TEST_DO( test_complex_column_major_oo_lhp_reorder );
	BOOST_UBLASX_TEST_DO( test_complex_row_major_oo_lhp_reorder );
	BOOST_UBLASX_TEST_DO( test_complex_column_major_oo_rhp_reorder );
	BOOST_UBLASX_TEST_DO( test_complex_row_major_oo_rhp_reorder );
	BOOST_UBLASX_TEST_DO( test_complex_column_major_oo_udi_reorder );
	BOOST_UBLASX_TEST_DO( test_complex_row_major_oo_udi_reorder );
	BOOST_UBLASX_TEST_DO( test_complex_column_major_oo_udo_reorder );
	BOOST_UBLASX_TEST_DO( test_complex_row_major_oo_udo_reorder );
	BOOST_UBLASX_TEST_DO( test_complex_column_major_oo_custom_reorder );
	BOOST_UBLASX_TEST_DO( test_complex_row_major_oo_custom_reorder );

	BOOST_UBLASX_TEST_END();
}
