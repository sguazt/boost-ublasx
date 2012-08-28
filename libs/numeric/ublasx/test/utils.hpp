/**
 *  \file lib/numeric/ublasx/test/util.hpp
 *
 *  \brief Utility macros/functions for testing and debugging purpose.
 *
 *  Copyright (c) 2009-2012, Marco Guazzone
 *
 *  Distributed under the Boost Software License, Version 1.0. (See
 *  accompanying file LICENSE_1_0.txt or copy at
 *  http://www.boost.org/LICENSE_1_0.txt)
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */

#ifndef BOOST_NUMERIC_UBLASX_TEST_UTILS_HPP
#define BOOST_NUMERIC_UBLASX_TEST_UTILS_HPP


#include <boost/numeric/ublas/fwd.hpp>
#include <boost/numeric/ublasx/detail/debug.hpp>
#include <boost/numeric/ublasx/detail/macro.hpp>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>


/// Define the beginning of a test suite.
#define BOOST_UBLASX_TEST_BEGIN() 	/* [BOOST_UBLASX_TEST_BEGIN] */ \
									{ /* Begin of Test Suite */ \
										unsigned int test_fails__(0) \
									/* [/BOOST_UBLASX_TEST_BEGIN] */


/// Define a test case \a x inside the current test suite.
#define BOOST_UBLASX_TEST_DEF(x) void BOOST_UBLASX_PASSTHROUGH(x)(unsigned int& test_fails__)


/// Call the test case \a x.
#define BOOST_UBLASX_TEST_DO(x) 	/* [BOOST_UBLASX_TEST_DO] */ \
								try \
								{ \
									BOOST_UBLASX_PASSTHROUGH(x)(test_fails__); \
								} \
								catch (::std::exception& e) \
								{ \
									++test_fails__; \
									BOOST_UBLASX_TEST_ERROR( e.what() ); \
								} \
								catch (...) \
								{ \
									++test_fails__; \
								} \
								/* [/BOOST_UBLASX_TEST_DO] */


/// Define the end of a test suite.
#define BOOST_UBLASX_TEST_END() 	/* [BOOST_UBLASX_TEST_END] */ \
								if (test_fails__ > 0) \
								{ \
									::std::cerr << "Number of failed tests: " << test_fails__ << ::std::endl; \
								} \
								else \
								{ \
									::std::cerr << "No failed test" << ::std::endl; \
								} \
								} /* End of test suite */ \
								/* [/BOOST_UBLASX_TEST_END] */


/// Check the truth of assertion \a x.
#define BOOST_UBLASX_TEST_CHECK(x)	/* [BOOST_UBLASX_TEST_CHECK] */ \
									if (!BOOST_UBLASX_EXPAND(x)) \
									{ \
										BOOST_UBLASX_TEST_ERROR( "Failed assertion: " << BOOST_UBLASX_STRINGIFY(x) ); \
										++test_fails__; \
									} \
									/* [/BOOST_UBLASX_TEST_CHECK] */


/// Check for the equality of \a x against \a y.
#define BOOST_UBLA_TEST_CHECK_EQUAL(x,y)	/* [BOOST_UBLA_TEST_CHECK_EQUAL] */ \
											if (!(BOOST_UBLAS_TEST_PARAM_EXPAND_(x) == BOOST_UBLAS_TEST_PARAM_EXPAND_(y))) \
											{ \
												BOOST_UBLAS_TEST_ERROR( "Failed assertion: (" << BOOST_UBLAS_TEST_STRINGIFY_(x) << " == " << BOOST_UBLAS_TEST_STRINGIFY_(y) << ")" ); \
												++test_fails__; \
											} \
											/* [/BOOST_UBLA_TEST_CHECK_EQUAL] */


/// Check that \a x and \a y are close with respect to a given precision.
#define BOOST_UBLASX_TEST_CHECK_CLOSE(x,y,e)	/* [BOOST_UBLASX_TEST_CHECK_CLOSE] */ \
											if (!::boost::numeric::ublasx::test::detail::close_to(BOOST_UBLASX_EXPAND(x), BOOST_UBLASX_EXPAND(y), BOOST_UBLASX_EXPAND(e))) \
											{ \
												BOOST_UBLASX_TEST_ERROR( "Failed assertion: abs(" << BOOST_UBLASX_STRINGIFY(x) << "-" << BOOST_UBLASX_STRINGIFY(y) << ") <= " << BOOST_UBLASX_STRINGIFY(e) << " [with " << BOOST_UBLASX_STRINGIFY(x) << " == " << BOOST_UBLASX_PASSTHROUGH(x) << ", " << BOOST_UBLASX_STRINGIFY(y) << " == " << BOOST_UBLASX_PASSTHROUGH(y) << " and " << BOOST_UBLASX_STRINGIFY(e) << " == " << BOOST_UBLASX_PASSTHROUGH(e) << "]" ); \
												++test_fails__; \
											} \
											/* [/BOOST_UBLASX_TEST_CHECK_CLOSE] */


/// Alias for macro \c BOOST_UBLASX_TEST_CHECK_CLOSE.
#define BOOST_UBLAS_TEST_CHECK_PRECISION(x,y,e)	BOOST_UBLASX_TEST_CHECK_CLOSE(x,y,e)


/// Check that \a x is close to \a y with respect to a given relative precision.
#define BOOST_UBLASX_TEST_CHECK_REL_CLOSE(x,y,e)	/* [BOOST_UBLASX_TEST_CHECK_REL_CLOSE] */ \
												if (!::boost::numeric::ublasx::test::detail::close_to(BOOST_UBLASX_EXPAND(x)/BOOST_UBLASX_EXPAND(y), 1.0, BOOST_UBLASX_EXPAND(e))) \
												{ \
													BOOST_UBLASX_TEST_ERROR( "Failed assertion: abs((" << BOOST_UBLASX_STRINGIFY(x) << "-" << BOOST_UBLASX_STRINGIFY(y) << ")/" << BOOST_UBLASX_STRINGIFY(y) << ") <= " << BOOST_UBLASX_STRINGIFY(e)  << " [with " << BOOST_UBLASX_STRINGIFY(x) << " == " << BOOST_UBLASX_PASSTHROUGH(x) << ", " << BOOST_UBLASX_STRINGIFY(y) << " == " << BOOST_UBLASX_PASSTHROUGH(y) << " and " << BOOST_UBLASX_STRINGIFY(e) << " == " << BOOST_UBLASX_PASSTHROUGH(e) << "]" ); \
													++test_fails__; \
												} \
												/* [/BOOST_UBLASX_TEST_CHECK_REL_CLOSE] */


/// Alias for macro \c BOOST_UBLASX_TEST_CHECK_REL_CLOSE.
#define BOOST_UBLAS_TEST_CHECK_REL_PRECISION(x,y,e)	BOOST_UBLASX_TEST_CHECK_REL_CLOSE(x,y,e)


/// Check that elements of \a x and \a y are equal.
#define BOOST_UBLASX_TEST_CHECK_VECTOR_EQ(x,y,n)	/* [BOOST_UBLASX_TEST_CHECK_VECTOR_EQ] */ \
												if (BOOST_UBLASX_EXPAND(n) > 0) \
												{ \
													::std::size_t n__ = BOOST_UBLASX_EXPAND(n); \
													for (::std::size_t i__ = n__; i__ > 0; --i__) \
													{ \
														if (!(BOOST_UBLASX_EXPAND(x)[n__-i__] == BOOST_UBLASX_EXPAND(y)[n__-i__])) \
														{ \
															BOOST_UBLASX_TEST_ERROR( "Failed assertion: (" << BOOST_UBLASX_STRINGIFY(x[i__]) << "==" << BOOST_UBLASX_STRINGIFY(y[i__]) << ")" << " [with " << BOOST_UBLASX_STRINGIFY(x[i__]) << " == " << BOOST_UBLASX_PASSTHROUGH(x)[n__-i__] << ", " << BOOST_UBLASX_STRINGIFY(y[i__]) << " == " << BOOST_UBLASX_PASSTHROUGH(y)[n__-i__] << ", " << BOOST_UBLASX_STRINGIFY(i__) << " == " << i__ << " and " << BOOST_UBLASX_STRINGIFY(n) << " == " << n__ << "]" ); \
															++test_fails__; \
														} \
													} \
												} \
												/* [/BOOST_UBLASX_TEST_CHECK_VECTOR_EQ] */


/// Check that elements of \a x and \a y are close with respect to a given precision.
#define BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE(x,y,n,e)	/* [BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE] */ \
														if (BOOST_UBLASX_EXPAND(n) > 0) \
														{ \
															::std::size_t n__ = BOOST_UBLASX_EXPAND(n); \
															for (::std::size_t i__ = n__; i__ > 0; --i__) \
															{ \
																if (!::boost::numeric::ublasx::test::detail::close_to(BOOST_UBLASX_EXPAND(x)[n__-i__], BOOST_UBLASX_EXPAND(y)[n__-i__], BOOST_UBLASX_EXPAND(e))) \
																{ \
																	BOOST_UBLASX_TEST_ERROR( "Failed assertion: abs(" << BOOST_UBLASX_STRINGIFY(x[i__]) << "-" << BOOST_UBLASX_STRINGIFY(y[i__]) << ") <= " << BOOST_UBLASX_STRINGIFY(e)  << " [with " << BOOST_UBLASX_STRINGIFY(x[i__]) << " == " << BOOST_UBLASX_PASSTHROUGH(x)[n__-i__] << ", " << BOOST_UBLASX_STRINGIFY(y[i__]) << " == " << BOOST_UBLASX_PASSTHROUGH(y)[n__-i__] << ", " << BOOST_UBLASX_STRINGIFY(i__) << " == " << i__ << ", " << BOOST_UBLASX_STRINGIFY(n) << " == " << n__ << " and " << BOOST_UBLASX_STRINGIFY(e) << " == " << BOOST_UBLASX_PASSTHROUGH(e) << "]" ); \
																	++test_fails__; \
																} \
															} \
														} \
														/* [/BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE] */


/// Check that elements of matrices \a x and \a y are equal.
#define BOOST_UBLASX_TEST_CHECK_MATRIX_EQ(x,y,nr,nc)	/* [BOOST_UBLASX_TEST_CHECK_MATRIX_EQ] */ \
													for (::std::size_t i__ = 0; i__ < BOOST_UBLASX_EXPAND(nr); ++i__) \
													{ \
														for (::std::size_t j__ = 0; j__ < BOOST_UBLASX_EXPAND(nc); ++j__) \
														{ \
															if (!(BOOST_UBLASX_EXPAND(x)(i__,j__) == BOOST_UBLASX_EXPAND(y)(i__,j__))) \
															{ \
																BOOST_UBLASX_TEST_ERROR( "Failed assertion: (" << BOOST_UBLASX_STRINGIFY(x(i__,j__)) << " == " << BOOST_UBLASX_STRINGIFY(y(i__,j__)) << ") [with " << BOOST_UBLASX_STRINGIFY(x(i__,j__)) << " == " << BOOST_UBLASX_PASSTHROUGH(x)(i__,j__) << ", " << BOOST_UBLASX_STRINGIFY(y(i__,j__)) << " == " << BOOST_UBLASX_PASSTHROUGH(y)(i__,j__) << ", " << BOOST_UBLASX_STRINGIFY(i__) << " == " << i__ << ", " << BOOST_UBLASX_STRINGIFY(j__) << " == " << BOOST_UBLASX_PASSTHROUGH(j__) << ", " << BOOST_UBLASX_STRINGIFY(nr) << " == " << BOOST_UBLASX_PASSTHROUGH(nr) << " and " << BOOST_UBLASX_STRINGIFY(nc) << " == " << BOOST_UBLASX_PASSTHROUGH(nc) << "]" ); \
																++test_fails__; \
															} \
														} \
													} \
													/* [/BOOST_UBLASX_TEST_CHECK_MATRIX_EQ] */


/// Check that elements of matrices \a x and \a y are close with respect to a give precision.
#define BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE(x,y,nr,nc,e)	/* [BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE] */ \
															::boost::numeric::ublasx::test::detail::check_matrix_close(BOOST_UBLASX_EXPAND(x), BOOST_UBLASX_EXPAND(y), BOOST_UBLASX_EXPAND(nr), BOOST_UBLASX_EXPAND(nc), BOOST_UBLASX_EXPAND(e), test_fails__); \
															/* [/BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE] */


/// Check that elements of matrices \a x and \a y are close with respect to a give precision.
#define BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE_IT(x,y,e)	/* [BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE] */ \
														::boost::numeric::ublasx::test::detail::check_matrix_close(BOOST_UBLASX_EXPAND(x), BOOST_UBLASX_EXPAND(y), BOOST_UBLASX_EXPAND(e), test_fails__); \
														/* [/BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE] */


/// Output the error message \a x.
#define BOOST_UBLASX_TEST_ERROR(x) ::std::cerr << "[Error (" << __FILE__ << ":" << __func__ << ":" << __LINE__ << ")>> " << BOOST_UBLASX_PASSTHROUGH(x) << ::std::endl


namespace boost { namespace numeric { namespace ublasx { namespace test { namespace detail {

template <typename T1, typename T2, typename T3>
BOOST_UBLAS_INLINE
bool close_to(T1 x, T2 y, T3 tol)
{
    if (::std::isnan(x) || ::std::isnan(y))
    {
        // According to IEEE, NaN are different event by itself
        return false;
    }
    return ::std::abs(x-y) <= (::std::max(::std::abs(x), ::std::abs(y))*tol);
}

template <typename T>
BOOST_UBLAS_INLINE
bool isnan(::std::complex<T> z)
{
	return (z != z) || ::std::isnan(z.real()) || ::std::isnan(z.imag());
}

template <typename T1, typename T2, typename T3>
BOOST_UBLAS_INLINE
bool close_to(::std::complex<T1> x, ::std::complex<T2> y, T3 tol)
{
    if (isnan(x) || isnan(y))
    {
        // According to IEEE, NaN are different event by itself
        return false;
    }
    return ::std::abs(x-y) <= (::std::max(::std::abs(x), ::std::abs(y))*tol);
}

template <
	typename Matrix1T,
	typename Matrix2T,
	typename SizeT,
	typename RealT,
	typename LongT
>
BOOST_UBLAS_INLINE
void check_matrix_close(::boost::numeric::ublas::matrix_expression<Matrix1T> const& x, ::boost::numeric::ublas::matrix_expression<Matrix2T> const& y, SizeT nr, SizeT nc, RealT e, LongT& test_fails)
{
	for (SizeT i = 0; i < nr; ++i)
	{
		for (SizeT j = 0; j < nc; ++j)
		{
			if (!close_to(x()(i,j), y()(i,j), e))
			{
				BOOST_UBLASX_TEST_ERROR( "Failed assertion: abs(" << BOOST_UBLASX_STRINGIFY(x(i,j)) << "-" << BOOST_UBLASX_STRINGIFY(y(i,j)) << ") <= " << BOOST_UBLASX_STRINGIFY(e)  << " [with " << BOOST_UBLASX_STRINGIFY(x(i,j)) << " == " << x()(i,j) << ", " << BOOST_UBLASX_STRINGIFY(y(i,j)) << " == " << y()(i,j) << ", " << BOOST_UBLASX_STRINGIFY(i) << " == " << i << ", " << BOOST_UBLASX_STRINGIFY(j) << " == " << j << ", " << BOOST_UBLASX_STRINGIFY(nr) << " == " << nr << ", " << BOOST_UBLASX_STRINGIFY(nc) << " == " << nc << " and " << BOOST_UBLASX_STRINGIFY(e) << " == " << e << "]" );
				++test_fails;
			}
		}
	}
}


template <
	typename Matrix1T,
	typename Matrix2T,
	typename RealT,
	typename LongT
>
BOOST_UBLAS_INLINE
void check_matrix_close(::boost::numeric::ublas::matrix_expression<Matrix1T> const& x, ::boost::numeric::ublas::matrix_expression<Matrix2T> const& y, RealT e, LongT& test_fails)
{
	for (typename Matrix1T::const_iterator1 row_it = x().begin1(); row_it != x().end1(); ++row_it)
	{
		for (typename Matrix1T::const_iterator2 col_it = row_it.begin(); col_it != row_it.end(); ++col_it)
		{
			typename Matrix1T::size_type i = col_it.index1();
			typename Matrix1T::size_type j = col_it.index2();

			if (!close_to(x()(i,j), y()(i,j), e))
			{
				BOOST_UBLASX_TEST_ERROR( "Failed assertion: abs(" << BOOST_UBLASX_STRINGIFY(x(i,j)) << "-" << BOOST_UBLASX_STRINGIFY(y(i,j)) << ") <= " << BOOST_UBLASX_STRINGIFY(e)  << " [with " << BOOST_UBLASX_STRINGIFY(x(i,j)) << " == " << x()(i,j) << ", " << BOOST_UBLASX_STRINGIFY(y(i,j)) << " == " << y()(i,j) << ", " << BOOST_UBLASX_STRINGIFY(i) << " == " << i << ", " << BOOST_UBLASX_STRINGIFY(j) << " == " << j << " and " << BOOST_UBLASX_STRINGIFY(e) << " == " << e << "]" );
				++test_fails;
			}
		}
	}
}

}}}}} // Namespace boost::numeric::ublasx::test::detail



#endif // BOOST_NUMERIC_UBLASX_TEST_UTILS_HPP
