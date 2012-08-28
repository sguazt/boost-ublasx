/**
 * \file libs/numeric/ublasx/test/triu.cpp
 *
 * \brief Test suite for the upper-triangular view operation.
 *
 * Copyright (c) 2011, Marco Guazzone
 *
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompwhiching file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */

#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublasx/detail/debug.hpp>
#include <boost/numeric/ublasx/operation/triu.hpp>
#include <complex>
#include <cstddef>
#include "libs/numeric/ublasx/test/utils.hpp"


namespace ublas = ::boost::numeric::ublas;
namespace ublasx = ::boost::numeric::ublasx;


static const double tol = 1e-5;


BOOST_UBLASX_TEST_DEF( real_square_matrix_row_major_keq0 )
{
	BOOST_UBLASX_DEBUG_TRACE( "Test Case: Real - Square - Row Major - k == 0" );

	typedef double value_type;
	typedef ublas::matrix<value_type,ublas::row_major> matrix_type;

	const ::std::size_t n(4);

	matrix_type A(n,n, value_type(1));


	matrix_type X;
	matrix_type E;

	X = ublasx::triu(A);
	E = ublasx::triangular_matrix<value_type,ublas::upper>(n,n);
	E(0,0) = E(0,1) = E(0,2) = E(0,3)
		   = E(1,1) = E(1,2) = E(1,3)
					= E(2,2) = E(2,3)
							 = E(3,3)
							 = value_type(1);

	BOOST_UBLASX_DEBUG_TRACE( "Input Matrix A=" << A );
	BOOST_UBLASX_DEBUG_TRACE( "triu(A)=" << X );
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( real_square_matrix_col_major_keq0 )
{
	BOOST_UBLASX_DEBUG_TRACE( "Test Case: Real - Square - Column Major - k == 0" );

	typedef double value_type;
	typedef ublas::matrix<value_type,ublas::column_major> matrix_type;

	const ::std::size_t n(4);

	matrix_type A(n,n, value_type(1));


	matrix_type X;
	matrix_type E;

	X = ublasx::triu(A);
	E = ublasx::triangular_matrix<value_type,ublas::upper>(n,n);
	E(0,0) = E(0,1) = E(0,2) = E(0,3)
		   = E(1,1) = E(1,2) = E(1,3)
					= E(2,2) = E(2,3)
							 = E(3,3)
							 = value_type(1);

	BOOST_UBLASX_DEBUG_TRACE( "Input Matrix A=" << A );
	BOOST_UBLASX_DEBUG_TRACE( "triu(A)=" << X );
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( real_square_matrix_row_major_kgt0 )
{
	BOOST_UBLASX_DEBUG_TRACE( "Test Case: Real - Square - Row Major - k > 0" );

	typedef double value_type;
	typedef ublas::matrix<value_type,ublas::row_major> matrix_type;

	const ::std::size_t n(4);

	matrix_type A(n,n, value_type(1));


	matrix_type X;
	matrix_type E;

	E = ublasx::triangular_matrix<value_type,ublas::upper>(n,n);
	E(0,0) = E(0,1) = E(0,2) = E(0,3)
		   = E(1,1) = E(1,2) = E(1,3)
					= E(2,2) = E(2,3)
							 = E(3,3)
							 = value_type(1);

	for (::std::size_t k = 1; k < n; ++k)
	{
		X = ublasx::triu(A, k);

		for (::std::size_t i = 0; i <= (n-k); ++i)
		{
			E(i,i+k-1) = value_type(0);
		}

		//BOOST_UBLASX_DEBUG_TRACE( "E=" << E );
		BOOST_UBLASX_DEBUG_TRACE( "triu(A," << k << ")=" << X );
		BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, n, n, tol );
	}
}


BOOST_UBLASX_TEST_DEF( real_square_matrix_col_major_kgt0 )
{
	BOOST_UBLASX_DEBUG_TRACE( "Test Case: Real - Square - Column Major - k > 0" );

	typedef double value_type;
	typedef ublas::matrix<value_type,ublas::column_major> matrix_type;

	const ::std::size_t n(4);

	matrix_type A(n,n, value_type(1));


	matrix_type X;
	matrix_type E;

	E = ublasx::triangular_matrix<value_type,ublas::upper>(n,n);
	E(0,0) = E(0,1) = E(0,2) = E(0,3)
		   = E(1,1) = E(1,2) = E(1,3)
					= E(2,2) = E(2,3)
							 = E(3,3)
							 = value_type(1);

	for (::std::size_t k = 1; k < n; ++k)
	{
		X = ublasx::triu(A, k);

		for (::std::size_t i = 0; i <= (n-k); ++i)
		{
			E(i,i+k-1) = value_type(0);
		}

		//BOOST_UBLASX_DEBUG_TRACE( "E=" << E );
		BOOST_UBLASX_DEBUG_TRACE( "triu(A," << k << ")=" << X );
		BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, n, n, tol );
	}
}


BOOST_UBLASX_TEST_DEF( real_square_matrix_row_major_klt0 )
{
	BOOST_UBLASX_DEBUG_TRACE( "Test Case: Real - Square - Row Major - k < 0" );

	typedef double value_type;
	typedef ublas::matrix<value_type,ublas::row_major> matrix_type;

	const ::std::size_t n(4);

	matrix_type A(n,n, value_type(1));


	matrix_type X;
	matrix_type E;

	E = ublasx::scalar_matrix<value_type>(n, n, value_type(1));

	for (::std::ptrdiff_t k = n-1; k >= 0; --k)
	{
		X = ublasx::triu(A, -k);

		for (::std::size_t i = k+1; i < n; ++i)
		{
			E(i,i-k-1) = value_type(0);
		}

		//BOOST_UBLASX_DEBUG_TRACE( "E=" << E );
		BOOST_UBLASX_DEBUG_TRACE( "triu(A," << -k << ")=" << X );
		BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, n, n, tol );
	}
}


BOOST_UBLASX_TEST_DEF( real_square_matrix_col_major_klt0 )
{
	BOOST_UBLASX_DEBUG_TRACE( "Test Case: Real - Square - Column Major - k < 0" );

	typedef double value_type;
	typedef ublas::matrix<value_type,ublas::column_major> matrix_type;

	const ::std::size_t n(4);

	matrix_type A(n,n, value_type(1));


	matrix_type X;
	matrix_type E;

	E = ublasx::scalar_matrix<value_type>(n, n, value_type(1));

	for (::std::ptrdiff_t k = n-1; k >= 0; --k)
	{
		X = ublasx::triu(A, -k);

		for (::std::size_t i = k+1; i < n; ++i)
		{
			E(i,i-k-1) = value_type(0);
		}

		//BOOST_UBLASX_DEBUG_TRACE( "E=" << E );
		BOOST_UBLASX_DEBUG_TRACE( "triu(A," << -k << ")=" << X );
		BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, n, n, tol );
	}
}


BOOST_UBLASX_TEST_DEF( real_horizontal_matrix_row_major_keq0 )
{
	BOOST_UBLASX_DEBUG_TRACE( "Test Case: Real - Horizontal - Row Major - k == 0" );

	typedef double value_type;
	typedef ublas::matrix<value_type,ublas::row_major> matrix_type;

	const ::std::size_t nr(4);
	const ::std::size_t nc(6);

	matrix_type A(nr,nc, value_type(1));


	matrix_type X;
	matrix_type E;

	X = ublasx::triu(A);
	E = ublasx::triangular_matrix<value_type,ublas::upper>(nr,nc);
	E(0,0) = E(0,1) = E(0,2) = E(0,3) = E(0,4) = E(0,5)
		   = E(1,1) = E(1,2) = E(1,3) = E(1,4) = E(1,5)
					= E(2,2) = E(2,3) = E(2,4) = E(2,5)
							 = E(3,3) = E(3,4) = E(3,5)
							 = value_type(1);

	BOOST_UBLASX_DEBUG_TRACE( "Input Matrix A=" << A );
	BOOST_UBLASX_DEBUG_TRACE( "triu(A)=" << X );
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, nr, nc, tol );
}


BOOST_UBLASX_TEST_DEF( real_horizontal_matrix_col_major_keq0 )
{
	BOOST_UBLASX_DEBUG_TRACE( "Test Case: Real - Horizontal - Column Major - k == 0" );

	typedef double value_type;
	typedef ublas::matrix<value_type,ublas::column_major> matrix_type;

	const ::std::size_t nr(4);
	const ::std::size_t nc(6);

	matrix_type A(nr,nc, value_type(1));


	matrix_type X;
	matrix_type E;

	X = ublasx::triu(A);
	E = ublasx::triangular_matrix<value_type,ublas::upper>(nr,nc);
	E(0,0) = E(0,1) = E(0,2) = E(0,3) = E(0,4) = E(0,5)
		   = E(1,1) = E(1,2) = E(1,3) = E(1,4) = E(1,5)
					= E(2,2) = E(2,3) = E(2,4) = E(2,5)
							 = E(3,3) = E(3,4) = E(3,5)
							 = value_type(1);

	BOOST_UBLASX_DEBUG_TRACE( "Input Matrix A=" << A );
	BOOST_UBLASX_DEBUG_TRACE( "triu(A)=" << X );
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, nr, nc, tol );
}


BOOST_UBLASX_TEST_DEF( real_horizontal_matrix_row_major_kgt0 )
{
	BOOST_UBLASX_DEBUG_TRACE( "Test Case: Real - Horizontal - Row Major - k > 0" );

	typedef double value_type;
	typedef ublas::matrix<value_type,ublas::row_major> matrix_type;

	const ::std::size_t nr(4);
	const ::std::size_t nc(6);

	matrix_type A(nr,nc, value_type(1));


	matrix_type X;
	matrix_type E;

	E = ublasx::triangular_matrix<value_type,ublas::upper>(nr,nc);
	E(0,0) = E(0,1) = E(0,2) = E(0,3) = E(0,4) = E(0,5)
		   = E(1,1) = E(1,2) = E(1,3) = E(1,4) = E(1,5)
					= E(2,2) = E(2,3) = E(2,4) = E(2,5)
							 = E(3,3) = E(3,4) = E(3,5)
							 = value_type(1);

	for (::std::size_t k = 1; k < nc; ++k)
	{
		X = ublasx::triu(A, k);

		for (::std::size_t i = 0; i <= ::std::min(nc-k,nr-1); ++i)
		{
			E(i,i+k-1) = value_type(0);
		}

		//BOOST_UBLASX_DEBUG_TRACE( "E=" << E );
		BOOST_UBLASX_DEBUG_TRACE( "triu(A," << k << ")=" << X );
		BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, nr, nc, tol );
	}
}


BOOST_UBLASX_TEST_DEF( real_horizontal_matrix_col_major_kgt0 )
{
	BOOST_UBLASX_DEBUG_TRACE( "Test Case: Real - Horizontal - Column Major - k > 0" );

	typedef double value_type;
	typedef ublas::matrix<value_type,ublas::column_major> matrix_type;

	const ::std::size_t nr(4);
	const ::std::size_t nc(6);

	matrix_type A(nr,nc, value_type(1));


	matrix_type X;
	matrix_type E;

	E = ublasx::triangular_matrix<value_type,ublas::upper>(nr,nc);
	E(0,0) = E(0,1) = E(0,2) = E(0,3) = E(0,4) = E(0,5)
		   = E(1,1) = E(1,2) = E(1,3) = E(1,4) = E(1,5)
					= E(2,2) = E(2,3) = E(2,4) = E(2,5)
							 = E(3,3) = E(3,4) = E(3,5)
							 = value_type(1);

	for (::std::size_t k = 1; k < nc; ++k)
	{
		X = ublasx::triu(A, k);

		for (::std::size_t i = 0; i <= ::std::min(nc-k,nr-1); ++i)
		{
			E(i,i+k-1) = value_type(0);
		}

		//BOOST_UBLASX_DEBUG_TRACE( "E=" << E );
		BOOST_UBLASX_DEBUG_TRACE( "triu(A," << k << ")=" << X );
		BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, nr, nc, tol );
	}
}


BOOST_UBLASX_TEST_DEF( real_horizontal_matrix_row_major_klt0 )
{
	BOOST_UBLASX_DEBUG_TRACE( "Test Case: Real - Horizontal - Row Major - k < 0" );

	typedef double value_type;
	typedef ublas::matrix<value_type,ublas::row_major> matrix_type;

	const ::std::size_t nr(4);
	const ::std::size_t nc(6);

	matrix_type A(nr,nc, value_type(1));


	matrix_type X;
	matrix_type E;

	E = ublasx::scalar_matrix<value_type>(nr, nc, value_type(1));

	for (::std::ptrdiff_t k = nr-1; k >= 0; --k)
	{
		X = ublasx::triu(A, -k);

		for (::std::size_t i = k+1; i <= ::std::min(nc-k,nr-1); ++i)
		{
			E(i,i-k-1) = value_type(0);
		}

		//BOOST_UBLASX_DEBUG_TRACE( "E=" << E );
		BOOST_UBLASX_DEBUG_TRACE( "triu(A," << -k << ")=" << X );
		BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, nr, nc, tol );
	}
}


BOOST_UBLASX_TEST_DEF( real_horizontal_matrix_col_major_klt0 )
{
	BOOST_UBLASX_DEBUG_TRACE( "Test Case: Real - Horizontal - Column Major - k < 0" );

	typedef double value_type;
	typedef ublas::matrix<value_type,ublas::column_major> matrix_type;

	const ::std::size_t nr(4);
	const ::std::size_t nc(6);

	matrix_type A(nr,nc, value_type(1));


	matrix_type X;
	matrix_type E;

	E = ublasx::scalar_matrix<value_type>(nr, nc, value_type(1));

	for (::std::ptrdiff_t k = nr-1; k >= 0; --k)
	{
		X = ublasx::triu(A, -k);

		for (::std::size_t i = k+1; i <= ::std::min(nc-k,nr-1); ++i)
		{
			E(i,i-k-1) = value_type(0);
		}

		//BOOST_UBLASX_DEBUG_TRACE( "E=" << E );
		BOOST_UBLASX_DEBUG_TRACE( "triu(A," << -k << ")=" << X );
		BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, nr, nc, tol );
	}
}


BOOST_UBLASX_TEST_DEF( real_vertical_matrix_row_major_keq0 )
{
	BOOST_UBLASX_DEBUG_TRACE( "Test Case: Real - Vertical - Row Major - k == 0" );

	typedef double value_type;
	typedef ublas::matrix<value_type,ublas::row_major> matrix_type;

	const ::std::size_t nr(6);
	const ::std::size_t nc(4);

	matrix_type A(nr,nc, value_type(1));


	matrix_type X;
	matrix_type E;

	X = ublasx::triu(A);
	E = ublasx::triangular_matrix<value_type,ublas::upper>(nr,nc);
	E(0,0) = E(0,1) = E(0,2) = E(0,3)
		   = E(1,1) = E(1,2) = E(1,3)
					= E(2,2) = E(2,3)
							 = E(3,3)
							 = value_type(1);

	BOOST_UBLASX_DEBUG_TRACE( "Input Matrix A=" << A );
	BOOST_UBLASX_DEBUG_TRACE( "triu(A)=" << X );
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, nr, nc, tol );
}


BOOST_UBLASX_TEST_DEF( real_vertical_matrix_col_major_keq0 )
{
	BOOST_UBLASX_DEBUG_TRACE( "Test Case: Real - Vertical - Column Major - k == 0" );

	typedef double value_type;
	typedef ublas::matrix<value_type,ublas::column_major> matrix_type;

	const ::std::size_t nr(6);
	const ::std::size_t nc(4);

	matrix_type A(nr,nc, value_type(1));


	matrix_type X;
	matrix_type E;

	X = ublasx::triu(A);
	E = ublasx::triangular_matrix<value_type,ublas::upper>(nr,nc);
	E(0,0) = E(0,1) = E(0,2) = E(0,3)
		   = E(1,1) = E(1,2) = E(1,3)
					= E(2,2) = E(2,3)
							 = E(3,3)
							 = value_type(1);

	BOOST_UBLASX_DEBUG_TRACE( "Input Matrix A=" << A );
	BOOST_UBLASX_DEBUG_TRACE( "triu(A)=" << X );
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, nr, nc, tol );
}


BOOST_UBLASX_TEST_DEF( real_vertical_matrix_row_major_kgt0 )
{
	BOOST_UBLASX_DEBUG_TRACE( "Test Case: Real - Vertical - Row Major - k > 0" );

	typedef double value_type;
	typedef ublas::matrix<value_type,ublas::row_major> matrix_type;

	const ::std::size_t nr(6);
	const ::std::size_t nc(4);

	matrix_type A(nr,nc, value_type(1));


	matrix_type X;
	matrix_type E;

	X = ublasx::triu(A);
	E = ublasx::triangular_matrix<value_type,ublas::upper>(nr,nc);
	E(0,0) = E(0,1) = E(0,2) = E(0,3)
		   = E(1,1) = E(1,2) = E(1,3)
					= E(2,2) = E(2,3)
							 = E(3,3)
							 = value_type(1);

	for (::std::size_t k = 1; k < nc; ++k)
	{
		X = ublasx::triu(A, k);

		for (::std::size_t i = 0; i <= ::std::min(nc-k,nr-1); ++i)
		{
			E(i,i+k-1) = value_type(0);
		}

		//BOOST_UBLASX_DEBUG_TRACE( "E=" << E );
		BOOST_UBLASX_DEBUG_TRACE( "triu(A," << k << ")=" << X );
		BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, nr, nc, tol );
	}
}


BOOST_UBLASX_TEST_DEF( real_vertical_matrix_col_major_kgt0 )
{
	BOOST_UBLASX_DEBUG_TRACE( "Test Case: Real - Vertical - Column Major - k > 0" );

	typedef double value_type;
	typedef ublas::matrix<value_type,ublas::column_major> matrix_type;

	const ::std::size_t nr(6);
	const ::std::size_t nc(4);

	matrix_type A(nr,nc, value_type(1));


	matrix_type X;
	matrix_type E;

	E = ublasx::triangular_matrix<value_type,ublas::upper>(nr,nc);
	E(0,0) = E(0,1) = E(0,2) = E(0,3)
		   = E(1,1) = E(1,2) = E(1,3)
					= E(2,2) = E(2,3)
							 = E(3,3)
							 = value_type(1);

	for (::std::size_t k = 1; k < nc; ++k)
	{
		X = ublasx::triu(A, k);

		for (::std::size_t i = 0; i <= ::std::min(nc-k,nr-1); ++i)
		{
			E(i,i+k-1) = value_type(0);
		}

		//BOOST_UBLASX_DEBUG_TRACE( "E=" << E );
		BOOST_UBLASX_DEBUG_TRACE( "triu(A," << k << ")=" << X );
		BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, nr, nc, tol );
	}
}


BOOST_UBLASX_TEST_DEF( real_vertical_matrix_row_major_klt0 )
{
	BOOST_UBLASX_DEBUG_TRACE( "Test Case: Real - Vertical - Row Major - k < 0" );

	typedef double value_type;
	typedef ublas::matrix<value_type,ublas::row_major> matrix_type;

	const ::std::size_t nr(6);
	const ::std::size_t nc(4);

	matrix_type A(nr,nc, value_type(1));


	matrix_type X;
	matrix_type E;

	E = ublasx::scalar_matrix<value_type>(nr, nc, value_type(1));

	for (::std::ptrdiff_t k = nr-1; k >= 0; --k)
	{
		X = ublasx::triu(A, -k);

		for (::std::size_t i = k+1; i < nr; ++i)
		{
			E(i,::std::min(i-k-1,nc-1)) = value_type(0);
		}

		//BOOST_UBLASX_DEBUG_TRACE( "E=" << E );
		BOOST_UBLASX_DEBUG_TRACE( "triu(A," << k << ")=" << X );
		BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, nr, nc, tol );
	}
}


BOOST_UBLASX_TEST_DEF( real_vertical_matrix_col_major_klt0 )
{
	BOOST_UBLASX_DEBUG_TRACE( "Test Case: Real - Vertical - Column Major - k < 0" );

	typedef double value_type;
	typedef ublas::matrix<value_type,ublas::column_major> matrix_type;

	const ::std::size_t nr(6);
	const ::std::size_t nc(4);

	matrix_type A(nr,nc, value_type(1));


	matrix_type X;
	matrix_type E;

	E = ublasx::scalar_matrix<value_type>(nr, nc, value_type(1));

	for (::std::ptrdiff_t k = nr-1; k >= 0; --k)
	{
		X = ublasx::triu(A, -k);

		for (::std::size_t i = k+1; i < nr; ++i)
		{
			E(i,::std::min(i-k-1,nc-1)) = value_type(0);
		}

		//BOOST_UBLASX_DEBUG_TRACE( "E=" << E );
		BOOST_UBLASX_DEBUG_TRACE( "triu(A," << k << ")=" << X );
		BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, nr, nc, tol );
	}
}


BOOST_UBLASX_TEST_DEF( complex_square_matrix_row_major_keq0 )
{
	BOOST_UBLASX_DEBUG_TRACE( "Test Case: Complex - Square - Row Major - k == 0" );

	typedef double real_type;
	typedef ::std::complex<real_type> value_type;
	typedef ublas::matrix<value_type,ublas::row_major> matrix_type;

	const ::std::size_t n(4);

	matrix_type A(n,n, value_type(1));


	matrix_type X;
	matrix_type E;

	X = ublasx::triu(A);
	E = ublasx::triangular_matrix<value_type,ublas::upper>(n,n);
	E(0,0) = E(0,1) = E(0,2) = E(0,3)
		   = E(1,1) = E(1,2) = E(1,3)
					= E(2,2) = E(2,3)
							 = E(3,3)
							 = value_type(1);

	BOOST_UBLASX_DEBUG_TRACE( "Input Matrix A=" << A );
	BOOST_UBLASX_DEBUG_TRACE( "triu(A)=" << X );
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( complex_square_matrix_col_major_keq0 )
{
	BOOST_UBLASX_DEBUG_TRACE( "Test Case: Complex - Square - Column Major - k == 0" );

	typedef double real_type;
	typedef ::std::complex<real_type> value_type;
	typedef ublas::matrix<value_type,ublas::column_major> matrix_type;

	const ::std::size_t n(4);

	matrix_type A(n,n, value_type(1));


	matrix_type X;
	matrix_type E;

	X = ublasx::triu(A);
	E = ublasx::triangular_matrix<value_type,ublas::upper>(n,n);
	E(0,0) = E(0,1) = E(0,2) = E(0,3)
		   = E(1,1) = E(1,2) = E(1,3)
					= E(2,2) = E(2,3)
							 = E(3,3)
							 = value_type(1);

	BOOST_UBLASX_DEBUG_TRACE( "Input Matrix A=" << A );
	BOOST_UBLASX_DEBUG_TRACE( "triu(A)=" << X );
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( complex_square_matrix_row_major_kgt0 )
{
	BOOST_UBLASX_DEBUG_TRACE( "Test Case: Complex - Square - Row Major - k > 0" );

	typedef double real_type;
	typedef ::std::complex<real_type> value_type;
	typedef ublas::matrix<value_type,ublas::row_major> matrix_type;

	const ::std::size_t n(4);

	matrix_type A(n,n, value_type(1));


	matrix_type X;
	matrix_type E;

	E = ublasx::triangular_matrix<value_type,ublas::upper>(n,n);
	E(0,0) = E(0,1) = E(0,2) = E(0,3)
		   = E(1,1) = E(1,2) = E(1,3)
					= E(2,2) = E(2,3)
							 = E(3,3)
							 = value_type(1);

	for (::std::size_t k = 1; k < n; ++k)
	{
		X = ublasx::triu(A, k);

		for (::std::size_t i = 0; i <= (n-k); ++i)
		{
			E(i,i+k-1) = value_type(0);
		}

		//BOOST_UBLASX_DEBUG_TRACE( "E=" << E );
		BOOST_UBLASX_DEBUG_TRACE( "triu(A," << k << ")=" << X );
		BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, n, n, tol );
	}
}


BOOST_UBLASX_TEST_DEF( complex_square_matrix_col_major_kgt0 )
{
	BOOST_UBLASX_DEBUG_TRACE( "Test Case: Complex - Square - Column Major - k > 0" );

	typedef double real_type;
	typedef ::std::complex<real_type> value_type;
	typedef ublas::matrix<value_type,ublas::column_major> matrix_type;

	const ::std::size_t n(4);

	matrix_type A(n,n, value_type(1));


	matrix_type X;
	matrix_type E;

	E = ublasx::triangular_matrix<value_type,ublas::upper>(n,n);
	E(0,0) = E(0,1) = E(0,2) = E(0,3)
		   = E(1,1) = E(1,2) = E(1,3)
					= E(2,2) = E(2,3)
							 = E(3,3)
							 = value_type(1);

	for (::std::size_t k = 1; k < n; ++k)
	{
		X = ublasx::triu(A, k);

		for (::std::size_t i = 0; i <= (n-k); ++i)
		{
			E(i,i+k-1) = value_type(0);
		}

		//BOOST_UBLASX_DEBUG_TRACE( "E=" << E );
		BOOST_UBLASX_DEBUG_TRACE( "triu(A," << k << ")=" << X );
		BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, n, n, tol );
	}
}


BOOST_UBLASX_TEST_DEF( complex_square_matrix_row_major_klt0 )
{
	BOOST_UBLASX_DEBUG_TRACE( "Test Case: Complex - Square - Row Major - k < 0" );

	typedef double real_type;
	typedef ::std::complex<real_type> value_type;
	typedef ublas::matrix<value_type,ublas::row_major> matrix_type;

	const ::std::size_t n(4);

	matrix_type A(n,n, value_type(1));


	matrix_type X;
	matrix_type E;

	E = ublasx::scalar_matrix<value_type>(n, n, value_type(1));

	for (::std::ptrdiff_t k = n-1; k >= 0; --k)
	{
		X = ublasx::triu(A, -k);

		for (::std::size_t i = k+1; i < n; ++i)
		{
			E(i,i-k-1) = value_type(0);
		}

		//BOOST_UBLASX_DEBUG_TRACE( "E=" << E );
		BOOST_UBLASX_DEBUG_TRACE( "triu(A," << -k << ")=" << X );
		BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, n, n, tol );
	}
}


BOOST_UBLASX_TEST_DEF( complex_square_matrix_col_major_klt0 )
{
	BOOST_UBLASX_DEBUG_TRACE( "Test Case: Complex - Square - Column Major - k < 0" );

	typedef double real_type;
	typedef ::std::complex<real_type> value_type;
	typedef ublas::matrix<value_type,ublas::column_major> matrix_type;

	const ::std::size_t n(4);

	matrix_type A(n,n, value_type(1));


	matrix_type X;
	matrix_type E;

	E = ublasx::scalar_matrix<value_type>(n, n, value_type(1));

	for (::std::ptrdiff_t k = n-1; k >= 0; --k)
	{
		X = ublasx::triu(A, -k);

		for (::std::size_t i = k+1; i < n; ++i)
		{
			E(i,i-k-1) = value_type(0);
		}

		//BOOST_UBLASX_DEBUG_TRACE( "E=" << E );
		BOOST_UBLASX_DEBUG_TRACE( "triu(A," << -k << ")=" << X );
		BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, n, n, tol );
	}
}


BOOST_UBLASX_TEST_DEF( complex_horizontal_matrix_row_major_keq0 )
{
	BOOST_UBLASX_DEBUG_TRACE( "Test Case: Complex - Horizontal - Row Major - k == 0" );

	typedef double real_type;
	typedef ::std::complex<real_type> value_type;
	typedef ublas::matrix<value_type,ublas::row_major> matrix_type;

	const ::std::size_t nr(4);
	const ::std::size_t nc(6);

	matrix_type A(nr,nc, value_type(1));


	matrix_type X;
	matrix_type E;

	X = ublasx::triu(A);
	E = ublasx::triangular_matrix<value_type,ublas::upper>(nr,nc);
	E(0,0) = E(0,1) = E(0,2) = E(0,3) = E(0,4) = E(0,5)
		   = E(1,1) = E(1,2) = E(1,3) = E(1,4) = E(1,5)
					= E(2,2) = E(2,3) = E(2,4) = E(2,5)
							 = E(3,3) = E(3,4) = E(3,5)
							 = value_type(1);

	BOOST_UBLASX_DEBUG_TRACE( "Input Matrix A=" << A );
	BOOST_UBLASX_DEBUG_TRACE( "triu(A)=" << X );
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, nr, nc, tol );
}


BOOST_UBLASX_TEST_DEF( complex_horizontal_matrix_col_major_keq0 )
{
	BOOST_UBLASX_DEBUG_TRACE( "Test Case: Complex - Horizontal - Column Major - k == 0" );

	typedef double real_type;
	typedef ::std::complex<real_type> value_type;
	typedef ublas::matrix<value_type,ublas::column_major> matrix_type;

	const ::std::size_t nr(4);
	const ::std::size_t nc(6);

	matrix_type A(nr,nc, value_type(1));


	matrix_type X;
	matrix_type E;

	X = ublasx::triu(A);
	E = ublasx::triangular_matrix<value_type,ublas::upper>(nr,nc);
	E(0,0) = E(0,1) = E(0,2) = E(0,3) = E(0,4) = E(0,5)
		   = E(1,1) = E(1,2) = E(1,3) = E(1,4) = E(1,5)
					= E(2,2) = E(2,3) = E(2,4) = E(2,5)
							 = E(3,3) = E(3,4) = E(3,5)
							 = value_type(1);

	BOOST_UBLASX_DEBUG_TRACE( "Input Matrix A=" << A );
	BOOST_UBLASX_DEBUG_TRACE( "triu(A)=" << X );
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, nr, nc, tol );
}


BOOST_UBLASX_TEST_DEF( complex_horizontal_matrix_row_major_kgt0 )
{
	BOOST_UBLASX_DEBUG_TRACE( "Test Case: Complex - Horizontal - Row Major - k > 0" );

	typedef double real_type;
	typedef ::std::complex<real_type> value_type;
	typedef ublas::matrix<value_type,ublas::row_major> matrix_type;

	const ::std::size_t nr(4);
	const ::std::size_t nc(6);

	matrix_type A(nr,nc, value_type(1));


	matrix_type X;
	matrix_type E;

	E = ublasx::triangular_matrix<value_type,ublas::upper>(nr,nc);
	E(0,0) = E(0,1) = E(0,2) = E(0,3) = E(0,4) = E(0,5)
		   = E(1,1) = E(1,2) = E(1,3) = E(1,4) = E(1,5)
					= E(2,2) = E(2,3) = E(2,4) = E(2,5)
							 = E(3,3) = E(3,4) = E(3,5)
							 = value_type(1);

	for (::std::size_t k = 1; k < nc; ++k)
	{
		X = ublasx::triu(A, k);

		for (::std::size_t i = 0; i <= ::std::min(nc-k,nr-1); ++i)
		{
			E(i,i+k-1) = value_type(0);
		}

		//BOOST_UBLASX_DEBUG_TRACE( "E=" << E );
		BOOST_UBLASX_DEBUG_TRACE( "triu(A," << k << ")=" << X );
		BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, nr, nc, tol );
	}
}


BOOST_UBLASX_TEST_DEF( complex_horizontal_matrix_col_major_kgt0 )
{
	BOOST_UBLASX_DEBUG_TRACE( "Test Case: Complex - Horizontal - Column Major - k > 0" );

	typedef double real_type;
	typedef ::std::complex<real_type> value_type;
	typedef ublas::matrix<value_type,ublas::column_major> matrix_type;

	const ::std::size_t nr(4);
	const ::std::size_t nc(6);

	matrix_type A(nr,nc, value_type(1));


	matrix_type X;
	matrix_type E;

	E = ublasx::triangular_matrix<value_type,ublas::upper>(nr,nc);
	E(0,0) = E(0,1) = E(0,2) = E(0,3) = E(0,4) = E(0,5)
		   = E(1,1) = E(1,2) = E(1,3) = E(1,4) = E(1,5)
					= E(2,2) = E(2,3) = E(2,4) = E(2,5)
							 = E(3,3) = E(3,4) = E(3,5)
							 = value_type(1);

	for (::std::size_t k = 1; k < nc; ++k)
	{
		X = ublasx::triu(A, k);

		for (::std::size_t i = 0; i <= ::std::min(nc-k,nr-1); ++i)
		{
			E(i,i+k-1) = value_type(0);
		}

		//BOOST_UBLASX_DEBUG_TRACE( "E=" << E );
		BOOST_UBLASX_DEBUG_TRACE( "triu(A," << k << ")=" << X );
		BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, nr, nc, tol );
	}
}


BOOST_UBLASX_TEST_DEF( complex_horizontal_matrix_row_major_klt0 )
{
	BOOST_UBLASX_DEBUG_TRACE( "Test Case: Complex - Horizontal - Row Major - k < 0" );

	typedef double real_type;
	typedef ::std::complex<real_type> value_type;
	typedef ublas::matrix<value_type,ublas::row_major> matrix_type;

	const ::std::size_t nr(4);
	const ::std::size_t nc(6);

	matrix_type A(nr,nc, value_type(1));


	matrix_type X;
	matrix_type E;

	E = ublasx::scalar_matrix<value_type>(nr, nc, value_type(1));

	for (::std::ptrdiff_t k = nr-1; k >= 0; --k)
	{
		X = ublasx::triu(A, -k);

		for (::std::size_t i = k+1; i <= ::std::min(nc-k,nr-1); ++i)
		{
			E(i,i-k-1) = value_type(0);
		}

		//BOOST_UBLASX_DEBUG_TRACE( "E=" << E );
		BOOST_UBLASX_DEBUG_TRACE( "triu(A," << -k << ")=" << X );
		BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, nr, nc, tol );
	}
}


BOOST_UBLASX_TEST_DEF( complex_horizontal_matrix_col_major_klt0 )
{
	BOOST_UBLASX_DEBUG_TRACE( "Test Case: Complex - Horizontal - Column Major - k < 0" );

	typedef double real_type;
	typedef ::std::complex<real_type> value_type;
	typedef ublas::matrix<value_type,ublas::column_major> matrix_type;

	const ::std::size_t nr(4);
	const ::std::size_t nc(6);

	matrix_type A(nr,nc, value_type(1));


	matrix_type X;
	matrix_type E;

	E = ublasx::scalar_matrix<value_type>(nr, nc, value_type(1));

	for (::std::ptrdiff_t k = nr-1; k >= 0; --k)
	{
		X = ublasx::triu(A, -k);

		for (::std::size_t i = k+1; i <= ::std::min(nc-k,nr-1); ++i)
		{
			E(i,i-k-1) = value_type(0);
		}

		//BOOST_UBLASX_DEBUG_TRACE( "E=" << E );
		BOOST_UBLASX_DEBUG_TRACE( "triu(A," << -k << ")=" << X );
		BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, nr, nc, tol );
	}
}


BOOST_UBLASX_TEST_DEF( complex_vertical_matrix_row_major_keq0 )
{
	BOOST_UBLASX_DEBUG_TRACE( "Test Case: Complex - Vertical - Row Major - k == 0" );

	typedef double real_type;
	typedef ::std::complex<real_type> value_type;
	typedef ublas::matrix<value_type,ublas::row_major> matrix_type;

	const ::std::size_t nr(6);
	const ::std::size_t nc(4);

	matrix_type A(nr,nc, value_type(1));


	matrix_type X;
	matrix_type E;

	X = ublasx::triu(A);
	E = ublasx::triangular_matrix<value_type,ublas::upper>(nr,nc);
	E(0,0) = E(0,1) = E(0,2) = E(0,3)
		   = E(1,1) = E(1,2) = E(1,3)
					= E(2,2) = E(2,3)
							 = E(3,3)
							 = value_type(1);

	BOOST_UBLASX_DEBUG_TRACE( "Input Matrix A=" << A );
	BOOST_UBLASX_DEBUG_TRACE( "triu(A)=" << X );
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, nr, nc, tol );
}


BOOST_UBLASX_TEST_DEF( complex_vertical_matrix_col_major_keq0 )
{
	BOOST_UBLASX_DEBUG_TRACE( "Test Case: Complex - Vertical - Column Major - k == 0" );

	typedef double real_type;
	typedef ::std::complex<real_type> value_type;
	typedef ublas::matrix<value_type,ublas::column_major> matrix_type;

	const ::std::size_t nr(6);
	const ::std::size_t nc(4);

	matrix_type A(nr,nc, value_type(1));


	matrix_type X;
	matrix_type E;

	X = ublasx::triu(A);
	E = ublasx::triangular_matrix<value_type,ublas::upper>(nr,nc);
	E(0,0) = E(0,1) = E(0,2) = E(0,3)
		   = E(1,1) = E(1,2) = E(1,3)
					= E(2,2) = E(2,3)
							 = E(3,3)
							 = value_type(1);

	BOOST_UBLASX_DEBUG_TRACE( "Input Matrix A=" << A );
	BOOST_UBLASX_DEBUG_TRACE( "triu(A)=" << X );
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, nr, nc, tol );
}


BOOST_UBLASX_TEST_DEF( complex_vertical_matrix_row_major_kgt0 )
{
	BOOST_UBLASX_DEBUG_TRACE( "Test Case: Complex - Vertical - Row Major - k > 0" );

	typedef double real_type;
	typedef ::std::complex<real_type> value_type;
	typedef ublas::matrix<value_type,ublas::row_major> matrix_type;

	const ::std::size_t nr(6);
	const ::std::size_t nc(4);

	matrix_type A(nr,nc, value_type(1));


	matrix_type X;
	matrix_type E;

	X = ublasx::triu(A);
	E = ublasx::triangular_matrix<value_type,ublas::upper>(nr,nc);
	E(0,0) = E(0,1) = E(0,2) = E(0,3)
		   = E(1,1) = E(1,2) = E(1,3)
					= E(2,2) = E(2,3)
							 = E(3,3)
							 = value_type(1);

	for (::std::size_t k = 1; k < nc; ++k)
	{
		X = ublasx::triu(A, k);

		for (::std::size_t i = 0; i <= ::std::min(nc-k,nr-1); ++i)
		{
			E(i,i+k-1) = value_type(0);
		}

		//BOOST_UBLASX_DEBUG_TRACE( "E=" << E );
		BOOST_UBLASX_DEBUG_TRACE( "triu(A," << k << ")=" << X );
		BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, nr, nc, tol );
	}
}


BOOST_UBLASX_TEST_DEF( complex_vertical_matrix_col_major_kgt0 )
{
	BOOST_UBLASX_DEBUG_TRACE( "Test Case: Complex - Vertical - Column Major - k > 0" );

	typedef double real_type;
	typedef ::std::complex<real_type> value_type;
	typedef ublas::matrix<value_type,ublas::column_major> matrix_type;

	const ::std::size_t nr(6);
	const ::std::size_t nc(4);

	matrix_type A(nr,nc, value_type(1));


	matrix_type X;
	matrix_type E;

	E = ublasx::triangular_matrix<value_type,ublas::upper>(nr,nc);
	E(0,0) = E(0,1) = E(0,2) = E(0,3)
		   = E(1,1) = E(1,2) = E(1,3)
					= E(2,2) = E(2,3)
							 = E(3,3)
							 = value_type(1);

	for (::std::size_t k = 1; k < nc; ++k)
	{
		X = ublasx::triu(A, k);

		for (::std::size_t i = 0; i <= ::std::min(nc-k,nr-1); ++i)
		{
			E(i,i+k-1) = value_type(0);
		}

		//BOOST_UBLASX_DEBUG_TRACE( "E=" << E );
		BOOST_UBLASX_DEBUG_TRACE( "triu(A," << k << ")=" << X );
		BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, nr, nc, tol );
	}
}


BOOST_UBLASX_TEST_DEF( complex_vertical_matrix_row_major_klt0 )
{
	BOOST_UBLASX_DEBUG_TRACE( "Test Case: Complex - Vertical - Row Major - k < 0" );

	typedef double real_type;
	typedef ::std::complex<real_type> value_type;
	typedef ublas::matrix<value_type,ublas::row_major> matrix_type;

	const ::std::size_t nr(6);
	const ::std::size_t nc(4);

	matrix_type A(nr,nc, value_type(1));


	matrix_type X;
	matrix_type E;

	E = ublasx::scalar_matrix<value_type>(nr, nc, value_type(1));

	for (::std::ptrdiff_t k = nr-1; k >= 0; --k)
	{
		X = ublasx::triu(A, -k);

		for (::std::size_t i = k+1; i < nr; ++i)
		{
			E(i,::std::min(i-k-1,nc-1)) = value_type(0);
		}

		//BOOST_UBLASX_DEBUG_TRACE( "E=" << E );
		BOOST_UBLASX_DEBUG_TRACE( "triu(A," << k << ")=" << X );
		BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, nr, nc, tol );
	}
}


BOOST_UBLASX_TEST_DEF( complex_vertical_matrix_col_major_klt0 )
{
	BOOST_UBLASX_DEBUG_TRACE( "Test Case: Complex - Vertical - Column Major - k < 0" );

	typedef double real_type;
	typedef ::std::complex<real_type> value_type;
	typedef ublas::matrix<value_type,ublas::column_major> matrix_type;

	const ::std::size_t nr(6);
	const ::std::size_t nc(4);

	matrix_type A(nr,nc, value_type(1));


	matrix_type X;
	matrix_type E;

	E = ublasx::scalar_matrix<value_type>(nr, nc, value_type(1));

	for (::std::ptrdiff_t k = nr-1; k >= 0; --k)
	{
		X = ublasx::triu(A, -k);

		for (::std::size_t i = k+1; i < nr; ++i)
		{
			E(i,::std::min(i-k-1,nc-1)) = value_type(0);
		}

		//BOOST_UBLASX_DEBUG_TRACE( "E=" << E );
		BOOST_UBLASX_DEBUG_TRACE( "triu(A," << k << ")=" << X );
		BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( X, E, nr, nc, tol );
	}
}


int main()
{
	BOOST_UBLASX_TEST_BEGIN();

	BOOST_UBLASX_TEST_DO( real_square_matrix_col_major_keq0 );
	BOOST_UBLASX_TEST_DO( real_square_matrix_row_major_keq0 );
	BOOST_UBLASX_TEST_DO( real_square_matrix_col_major_kgt0 );
	BOOST_UBLASX_TEST_DO( real_square_matrix_row_major_kgt0 );
	BOOST_UBLASX_TEST_DO( real_square_matrix_row_major_klt0 );
	BOOST_UBLASX_TEST_DO( real_square_matrix_col_major_klt0 );

	BOOST_UBLASX_TEST_DO( real_horizontal_matrix_row_major_keq0 );
	BOOST_UBLASX_TEST_DO( real_horizontal_matrix_col_major_keq0 );
	BOOST_UBLASX_TEST_DO( real_horizontal_matrix_row_major_kgt0 );
	BOOST_UBLASX_TEST_DO( real_horizontal_matrix_col_major_kgt0 );
	BOOST_UBLASX_TEST_DO( real_horizontal_matrix_row_major_klt0 );
	BOOST_UBLASX_TEST_DO( real_horizontal_matrix_col_major_klt0 );

	BOOST_UBLASX_TEST_DO( real_vertical_matrix_row_major_keq0 );
	BOOST_UBLASX_TEST_DO( real_vertical_matrix_col_major_keq0 );
	BOOST_UBLASX_TEST_DO( real_vertical_matrix_row_major_kgt0 );
	BOOST_UBLASX_TEST_DO( real_vertical_matrix_col_major_kgt0 );
	BOOST_UBLASX_TEST_DO( real_vertical_matrix_row_major_klt0 );
	BOOST_UBLASX_TEST_DO( real_vertical_matrix_col_major_klt0 );

	BOOST_UBLASX_TEST_DO( complex_square_matrix_col_major_keq0 );
	BOOST_UBLASX_TEST_DO( complex_square_matrix_row_major_keq0 );
	BOOST_UBLASX_TEST_DO( complex_square_matrix_col_major_kgt0 );
	BOOST_UBLASX_TEST_DO( complex_square_matrix_row_major_kgt0 );
	BOOST_UBLASX_TEST_DO( complex_square_matrix_row_major_klt0 );
	BOOST_UBLASX_TEST_DO( complex_square_matrix_col_major_klt0 );

	BOOST_UBLASX_TEST_DO( complex_horizontal_matrix_row_major_keq0 );
	BOOST_UBLASX_TEST_DO( complex_horizontal_matrix_col_major_keq0 );
	BOOST_UBLASX_TEST_DO( complex_horizontal_matrix_row_major_kgt0 );
	BOOST_UBLASX_TEST_DO( complex_horizontal_matrix_col_major_kgt0 );
	BOOST_UBLASX_TEST_DO( complex_horizontal_matrix_row_major_klt0 );
	BOOST_UBLASX_TEST_DO( complex_horizontal_matrix_col_major_klt0 );

	BOOST_UBLASX_TEST_DO( complex_vertical_matrix_row_major_keq0 );
	BOOST_UBLASX_TEST_DO( complex_vertical_matrix_col_major_keq0 );
	BOOST_UBLASX_TEST_DO( complex_vertical_matrix_row_major_kgt0 );
	BOOST_UBLASX_TEST_DO( complex_vertical_matrix_col_major_kgt0 );
	BOOST_UBLASX_TEST_DO( complex_vertical_matrix_row_major_klt0 );
	BOOST_UBLASX_TEST_DO( complex_vertical_matrix_col_major_klt0 );

	BOOST_UBLASX_TEST_END();
}
