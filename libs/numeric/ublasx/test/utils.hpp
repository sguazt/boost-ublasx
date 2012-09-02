/**
 * \file libs/numeric/ublasx/test/util.hpp
 *
 * \brief Utility macros/functions for testing and debugging purpose.
 *
 * Basic usage:
 * <pre>
 * BOOST_UBLASX_TEST_DEF( test_case_1 )
 * {
 *   // do your test stuff
 * }
 *
 * BOOST_UBLASX_TEST_DEF( test_case_2 )
 * {
 *   // do your test stuff
 * }
 *
 * // ...
 *
 * BOOST_UBLASX_TEST_DEF( test_case_n )
 * {
 *   // do your test stuff
 * }
 *
 * int main()
 * {
 *   BOOST_UBLASX_TEST_SUITE( "My Test Suite" ); // optional
 *
 *   BOOST_UBLASX_TEST_BEGIN();
 *     BOOST_UBLASX_TEST_DO( test_case_1 );
 *     BOOST_UBLASX_TEST_DO( test_case_2 );
 *     // ...
 *     BOOST_UBLASX_TEST_DO( test_case_n );
 *   BOOST_UBLASX_TEST_END();
 * }
 * </pre>
 * Inside each <em>test_case_<code>k</code></em> you can use the various
 * \c BOOST_UBLASX_TEST_CHECK* macros.
 *
 * <hr/>
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


#include <boost/numeric/ublas/detail/config.hpp>
#include <boost/numeric/ublas/traits.hpp>
#include <boost/numeric/ublasx/detail/debug.hpp>
#include <boost/numeric/ublasx/detail/macro.hpp>
#include <cmath>
#include <complex>
#include <cstddef>
#include <iostream>
#include <limits>
#include <stdexcept>


namespace boost { namespace numeric { namespace ublas { namespace test { namespace detail { namespace /*<unnamed>*/ {

/// Check if the given complex number is a NaN.
template <typename T>
BOOST_UBLAS_INLINE
bool isnan(::std::complex<T> const& z)
{
	// According to IEEE, NaN is different even by itself
	return (z != z) || ::std::isnan(z.real()) || ::std::isnan(z.imag());
}

/// Check if two (real) numbers are close each other (wrt a given tolerance).
template <typename T1, typename T2, typename T3>
BOOST_UBLAS_INLINE
bool close_to(T1 x, T2 y, T3 tol)
{
	typedef typename promote_traits<typename promote_traits<T1,T2>::promote_type,
									T3>::promote_type real_type;

    if (::std::isnan(x) || ::std::isnan(y))
    {
        // According to IEEE, NaN is different even by itself
        return false;
    }
    return ::std::abs(x-y) <= (::std::max(static_cast<real_type>(::std::abs(x)), static_cast<real_type>(::std::abs(y)))*tol);
}

/// Check if two complex numbers are close each other (wrt a given tolerance).
template <typename T1, typename T2, typename T3>
BOOST_UBLAS_INLINE
bool close_to(::std::complex<T1> const& x, ::std::complex<T2> const& y, T3 tol)
{
	typedef typename promote_traits<typename promote_traits<T1,T2>::promote_type,
									T3>::promote_type real_type;

    if (isnan(x) || isnan(y))
    {
        // According to IEEE, NaN is different even by itself
        return false;
    }
	::std::complex<real_type> xx(x);
	::std::complex<real_type> yy(y);
    return ::std::abs(xx-yy) <= (::std::max(::std::abs(xx), ::std::abs(yy))*tol);
}

/// Check if two (real) numbers are close each other (wrt a given tolerance).
template <typename T1, typename T2, typename T3>
BOOST_UBLAS_INLINE
bool rel_close_to(T1 x, T2 y, T3 tol)
{
	typedef typename promote_traits<typename promote_traits<T1,T2>::promote_type,
									T3>::promote_type real_type;

    if (::std::isnan(x) || ::std::isnan(y))
    {
        // According to IEEE, NaN is different even by itself
        return false;
    }
    return ::std::abs(x-y)/::std::abs(y) <= tol;
}

/// Check if two complex numbers are close each other (wrt a given tolerance).
template <typename T1, typename T2, typename T3>
BOOST_UBLAS_INLINE
bool rel_close_to(::std::complex<T1> const& x, ::std::complex<T2> const& y, T3 tol)
{
	typedef typename promote_traits<typename promote_traits<T1,T2>::promote_type,
									T3>::promote_type real_type;

    if (isnan(x) || isnan(y))
    {
        // According to IEEE, NaN is different even by itself
        return false;
    }
	::std::complex<real_type> xx(x);
	::std::complex<real_type> yy(y);
    return ::std::abs(xx-yy)/::std::abs(yy) <= tol;
}

}}}}}} // Namespace boost::numeric::ublas::test::detail::<unnamed>


/// Expand its argument \a x.
#define BOOST_UBLASX_TEST_EXPAND_(x) BOOST_UBLASX_PASSTHROUGH(x)


/// Expand its argument \a x inside parenthesis.
#define BOOST_UBLASX_TEST_EXPANDP_(x) BOOST_UBLASX_EXPAND(x)


/// Transform its argument \a x into a string.
#define BOOST_UBLASX_TEST_STRINGIFY_(x) #x


/// Concatenate its two \e string arguments \a x and \a y.
#define BOOST_UBLASX_TEST_JOIN_(x,y) x ## y


/// Define the name \a m of the entire test suite.
#define BOOST_UBLASX_TEST_SUITE(m) ::std::cerr << "--- Test Suite: " << BOOST_UBLASX_TEST_EXPAND_(m) << " ---" << ::std::endl;


/// Define the beginning of a test suite.
#define BOOST_UBLASX_TEST_BEGIN() 	/* [BOOST_UBLASX_TEST_BEGIN] */ \
									{ \
										/* Begin of Test Suite */ \
										::std::size_t test_fails__(0) \
									/* [/BOOST_UBLASX_TEST_BEGIN] */


/// Define a test case \a x inside the current test suite.
#define BOOST_UBLASX_TEST_DEF(x) static void BOOST_UBLASX_TEST_EXPAND_(x)(::std::size_t& test_fails__)


/// Call the test case \a x.
#define BOOST_UBLASX_TEST_DO(x) 	/* [BOOST_UBLASX_TEST_DO] */ \
								try \
								{ \
									BOOST_UBLASX_TEST_EXPAND_(x)(test_fails__); \
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


/// Output the message \a m.
/// Note: we don't use macro expansion inside parenthesis to let \a m be an
///  expression of the form <code>a &lt;&lt; b</code>.
#define BOOST_UBLASX_TEST_TRACE(m) ::std::cerr << "[Info>> " << BOOST_UBLASX_TEST_EXPAND_(m) << ::std::endl


/// Check the truth of assertion \a x.
#define BOOST_UBLASX_TEST_CHECK(x)	/* [BOOST_UBLASX_TEST_CHECK] */ \
									if (!BOOST_UBLASX_TEST_EXPANDP_(x)) \
									{ \
										BOOST_UBLASX_TEST_ERROR( "Failed assertion: " << BOOST_UBLASX_TEST_STRINGIFY_(x) ); \
										++test_fails__; \
									} \
									/* [/BOOST_UBLASX_TEST_CHECK] */


/// Check for the equality of \a x against \a y.
#define BOOST_UBLASX_TEST_CHECK_EQ(x,y)	/* [BOOST_UBLASX_TEST_CHECK_EQ] */ \
										if (!(BOOST_UBLASX_TEST_EXPANDP_(x) == BOOST_UBLASX_TEST_EXPANDP_(y))) \
										{ \
											BOOST_UBLASX_TEST_ERROR( "Failed assertion: (" << BOOST_UBLASX_TEST_STRINGIFY_(x) << " == " << BOOST_UBLASX_TEST_STRINGIFY_(y) << ")" ); \
											++test_fails__; \
										} \
										/* [/BOOST_UBLASX_TEST_CHECK_EQ] */


/// Alias for macro \c BOOST_UBLASX_TEST_CHECK_EQ (for backward compatibility).
#define BOOST_UBLASX_TEST_CHECK_EQUAL(x,y) BOOST_UBLASX_TEST_CHECK_EQ(x,y)


/// Check that \a x and \a y are close with respect to a given precision \a e.
#define BOOST_UBLASX_TEST_CHECK_CLOSE(x,y,e)	/* [BOOST_UBLASX_TEST_CHECK_CLOSE] */ \
											if (!::boost::numeric::ublas::test::detail::close_to(BOOST_UBLASX_TEST_EXPAND_(x), BOOST_UBLASX_TEST_EXPAND_(y), BOOST_UBLASX_TEST_EXPAND_(e))) \
											{ \
												BOOST_UBLASX_TEST_ERROR( "Failed assertion: abs(" << BOOST_UBLASX_TEST_STRINGIFY_(x) << "-" << BOOST_UBLASX_TEST_STRINGIFY_(y) << ") <= " << BOOST_UBLASX_TEST_STRINGIFY_(e) << " [with " << BOOST_UBLASX_TEST_STRINGIFY_(x) << " == " << BOOST_UBLASX_TEST_EXPANDP_(x) << ", " << BOOST_UBLASX_TEST_STRINGIFY_(y) << " == " << BOOST_UBLASX_TEST_EXPANDP_(y) << " and " << BOOST_UBLASX_TEST_STRINGIFY_(e) << " == " << BOOST_UBLASX_TEST_EXPANDP_(e) << "]" ); \
												++test_fails__; \
											} \
											/* [/BOOST_UBLASX_TEST_CHECK_CLOSE] */


/// Alias for macro \c BOOST_UBLASX_TEST_CHECK_CLOSE (for backward compatibility),
#define BOOST_UBLASX_TEST_CHECK_PRECISION(x,y,e) BOOST_UBLASX_TEST_CHECK_CLOSE(x,y,e)


/// Check that \a x is close to \a y with respect to a given relative precision \a e.
#define BOOST_UBLASX_TEST_CHECK_REL_CLOSE(x,y,e)	/* [BOOST_UBLASX_TEST_CHECK_REL_CLOSE] */ \
												if (!::boost::numeric::ublas::test::detail::rel_close_to(BOOST_UBLASX_TEST_EXPAND_(x), BOOST_UBLASX_TEST_EXPAND_(y), BOOST_UBLASX_TEST_EXPAND_(e))) \
												{ \
													BOOST_UBLASX_TEST_ERROR( "Failed assertion: abs((" << BOOST_UBLASX_TEST_STRINGIFY_(x) << "-" << BOOST_UBLASX_TEST_STRINGIFY_(y) << ")/" << BOOST_UBLASX_TEST_STRINGIFY_(y) << ") <= " << BOOST_UBLASX_TEST_STRINGIFY_(e)  << " [with " << BOOST_UBLASX_TEST_STRINGIFY_(x) << " == " << BOOST_UBLASX_TEST_EXPANDP_(x) << ", " << BOOST_UBLASX_TEST_STRINGIFY_(y) << " == " << BOOST_UBLASX_TEST_EXPANDP_(y) << " and " << BOOST_UBLASX_TEST_STRINGIFY_(e) << " == " << BOOST_UBLASX_TEST_EXPANDP_(e) << "]" ); \
													++test_fails__; \
												} \
												/* [/BOOST_UBLASX_TEST_CHECK_REL_CLOSE] */


/// Alias for macro \c BOOST_UBLASX_TEST_CHECK_REL_CLOSE (for backward compatibility),
#define BOOST_UBLASX_TEST_CHECK_REL_PRECISION(x,y,e) BOOST_UBLASX_TEST_CHECK_REL_CLOSE(x,y,e)


/// Check that elements of \a x and \a y are equal.
#define BOOST_UBLASX_TEST_CHECK_VECTOR_EQ(x,y,n)	/* [BOOST_UBLASX_TEST_CHECK_VECTOR_EQ] */ \
												if (BOOST_UBLASX_TEST_EXPANDP_(n) > 0) \
												{ \
													::std::size_t n__ = BOOST_UBLASX_TEST_EXPAND_(n); \
													for (::std::size_t i__ = n__; i__ > 0; --i__) \
													{ \
														if (!(BOOST_UBLASX_TEST_EXPANDP_(x)[n__-i__]==BOOST_UBLASX_TEST_EXPANDP_(y)[n__-i__])) \
														{ \
															BOOST_UBLASX_TEST_ERROR( "Failed assertion: (" << BOOST_UBLASX_TEST_STRINGIFY_(x[i__]) << "==" << BOOST_UBLASX_TEST_STRINGIFY_(y[i__]) << ")" << " [with " << BOOST_UBLASX_TEST_STRINGIFY_(x[i__]) << " == " << BOOST_UBLASX_TEST_EXPANDP_(x)[n__-i__] << ", " << BOOST_UBLASX_TEST_STRINGIFY_(y[i__]) << " == " << BOOST_UBLASX_TEST_EXPANDP_(y)[n__-i__] << ", " << BOOST_UBLASX_TEST_STRINGIFY_(i__) << " == " << i__ << " and " << BOOST_UBLASX_TEST_STRINGIFY_(n) << " == " << n__ << "]" ); \
															++test_fails__; \
														} \
													} \
												} \
												/* [/BOOST_UBLASX_TEST_CHECK_VECTOR_EQ] */


/// Check that elements of \a x and \a y are close with respect to a given precision \a e.
#define BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE(x,y,n,e)	/* [BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE] */ \
														if (BOOST_UBLASX_TEST_EXPANDP_(n) > 0) \
														{ \
															::std::size_t n__ = BOOST_UBLASX_TEST_EXPAND_(n); \
															for (::std::size_t i__ = n__; i__ > 0; --i__) \
															{ \
																if (!::boost::numeric::ublas::test::detail::close_to(BOOST_UBLASX_TEST_EXPANDP_(x)[n__-i__], BOOST_UBLASX_TEST_EXPANDP_(y)[n__-i__], BOOST_UBLASX_TEST_EXPANDP_(e))) \
																{ \
																	BOOST_UBLASX_TEST_ERROR( "Failed assertion: abs((" << BOOST_UBLASX_TEST_STRINGIFY_(x[i__]) << "-" << BOOST_UBLASX_TEST_STRINGIFY_(y[i__]) << ") <= " << BOOST_UBLASX_TEST_STRINGIFY_(e)  << " [with " << BOOST_UBLASX_TEST_STRINGIFY_(x[i__]) << " == " << BOOST_UBLASX_TEST_EXPANDP_(x)[n__-i__] << ", " << BOOST_UBLASX_TEST_STRINGIFY_(y[i__]) << " == " << BOOST_UBLASX_TEST_EXPANDP_(y)[n__-i__] << ", " << BOOST_UBLASX_TEST_STRINGIFY_(i__) << " == " << i__ << ", " << BOOST_UBLASX_TEST_STRINGIFY_(e) << " == " << BOOST_UBLASX_TEST_EXPANDP_(e) << " and " << BOOST_UBLASX_TEST_STRINGIFY_(n) << " == " << n__ << "]" ); \
																	++test_fails__; \
																} \
															} \
														} \
														/* [/BOOST_UBLASX_TEST_CHECK_VECTOR_CLOSE] */


/// Check that elements of \a x and \a y are close with respect to a given relative precision \a e.
#define BOOST_UBLASX_TEST_CHECK_VECTOR_REL_CLOSE(x,y,n,e)	/* [BOOST_UBLASX_TEST_CHECK_VECTOR_REL_CLOSE] */ \
														if (BOOST_UBLASX_TEST_EXPANDP_(n) > 0) \
														{ \
															::std::size_t n__ = BOOST_UBLASX_TEST_EXPAND_(n); \
															for (::std::size_t i__ = n__; i__ > 0; --i__) \
															{ \
																if (!::boost::numeric::ublas::test::detail::rel_close_to(BOOST_UBLASX_TEST_EXPANDP_(x)[n__-i__], BOOST_UBLASX_TEST_EXPANDP_(y)[n__-i__], BOOST_UBLASX_TEST_EXPANDP_(e))) \
																{ \
																	BOOST_UBLASX_TEST_ERROR( "Failed assertion: abs((" << BOOST_UBLASX_TEST_STRINGIFY_(x[i__]) << "-" << BOOST_UBLASX_TEST_STRINGIFY_(y[i__]) << ")/" << BOOST_UBLASX_TEST_STRINGIFY_(y[i__]) << ") <= " << BOOST_UBLASX_TEST_STRINGIFY_(e)  << " [with " << BOOST_UBLASX_TEST_STRINGIFY_(x[i__]) << " == " << BOOST_UBLASX_TEST_EXPANDP_(x)[n__-i__] << ", " << BOOST_UBLASX_TEST_STRINGIFY_(y[i__]) << " == " << BOOST_UBLASX_TEST_EXPANDP_(y)[n__-i__] << ", " << BOOST_UBLASX_TEST_STRINGIFY_(i__) << " == " << i__ << ", " << BOOST_UBLASX_TEST_STRINGIFY_(e) << " == " << BOOST_UBLASX_TEST_EXPANDP_(e) << " and " << BOOST_UBLASX_TEST_STRINGIFY_(n) << " == " << n__ << "]" ); \
																	++test_fails__; \
																} \
															} \
														} \
														/* [/BOOST_UBLASX_TEST_CHECK_VECTOR_REL_CLOSE] */


/// Check that elements of matrices \a x and \a y are equal.
#define BOOST_UBLASX_TEST_CHECK_MATRIX_EQ(x,y,nr,nc)	/* [BOOST_UBLASX_TEST_CHECK_MATRIX_EQ] */ \
													for (::std::size_t i__ = 0; i__ < BOOST_UBLASX_TEST_EXPANDP_(nr); ++i__) \
													{ \
														for (::std::size_t j__ = 0; j__ < BOOST_UBLASX_TEST_EXPANDP_(nc); ++j__) \
														{ \
															if (!(BOOST_UBLASX_TEST_EXPANDP_(x)(i__,j__)==BOOST_UBLASX_TEST_EXPANDP_(y)(i__,j__))) \
															{ \
																BOOST_UBLASX_TEST_ERROR( "Failed assertion: (" << BOOST_UBLASX_TEST_STRINGIFY_(x(i__,j__)) << " == " << BOOST_UBLASX_TEST_STRINGIFY_(y(i__,j__)) << ") [with " << BOOST_UBLASX_TEST_STRINGIFY_(x(i__,j__)) << " == " << BOOST_UBLASX_TEST_EXPANDP_(x)(i__,j__) << ", " << BOOST_UBLASX_TEST_STRINGIFY_(y(i__,j__)) << " == " << BOOST_UBLASX_TEST_EXPANDP_(y)(i__,j__) << ", " << BOOST_UBLASX_TEST_STRINGIFY_(i__) << " == " << i__ << ", " << BOOST_UBLASX_TEST_STRINGIFY_(j__) << " == " << BOOST_UBLASX_TEST_EXPANDP_(j__) << ", " << BOOST_UBLASX_TEST_STRINGIFY_(nr) << " == " << BOOST_UBLASX_TEST_EXPANDP_(nr) << " and " << BOOST_UBLASX_TEST_STRINGIFY_(nc) << " == " << BOOST_UBLASX_TEST_EXPANDP_(nc) << "]" ); \
																++test_fails__; \
															} \
														} \
													} \
													/* [/BOOST_UBLASX_TEST_CHECK_MATRIX_EQ] */


/// Check that elements of matrices \a x and \a y are close with respect to a given precision \a e.
#define BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE(x,y,nr,nc,e)	/* [BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE] */ \
															for (::std::size_t i__ = 0; i__ < BOOST_UBLASX_TEST_EXPANDP_(nr); ++i__) \
															{ \
																for (::std::size_t j__ = 0; j__ < BOOST_UBLASX_TEST_EXPANDP_(nc); ++j__) \
																{ \
																	if (!::boost::numeric::ublas::test::detail::close_to(BOOST_UBLASX_TEST_EXPANDP_(x)(i__,j__), BOOST_UBLASX_TEST_EXPANDP_(y)(i__,j__), BOOST_UBLASX_TEST_EXPANDP_(e))) \
																	{ \
																		BOOST_UBLASX_TEST_ERROR( "Failed assertion: abs(" << BOOST_UBLASX_TEST_STRINGIFY_(x(i__,j__)) << "-" << BOOST_UBLASX_TEST_STRINGIFY_(y(i__,j__)) << ") <= " << BOOST_UBLASX_TEST_STRINGIFY_(e)  << " [with " << BOOST_UBLASX_TEST_STRINGIFY_(x(i__,j__)) << " == " << BOOST_UBLASX_TEST_EXPANDP_(x)(i__,j__) << ", " << BOOST_UBLASX_TEST_STRINGIFY_(y(i__,j__)) << " == " << BOOST_UBLASX_TEST_EXPANDP_(y)(i__,j__) << ", " << BOOST_UBLASX_TEST_STRINGIFY_(i__) << " == " << i__ << ", " << BOOST_UBLASX_TEST_STRINGIFY_(j__) << " == " << BOOST_UBLASX_TEST_EXPANDP_(j__) << ", " << BOOST_UBLASX_TEST_STRINGIFY_(e) << " == " << BOOST_UBLASX_TEST_EXPANDP_(e) << ", " << BOOST_UBLASX_TEST_STRINGIFY_(nr) << " == " << BOOST_UBLASX_TEST_EXPANDP_(nr) << " and " << BOOST_UBLASX_TEST_STRINGIFY_(nc) << " == " << BOOST_UBLASX_TEST_EXPANDP_(nc) << "]" ); \
																		++test_fails__; \
																	} \
																} \
															} \
															/* [/BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE] */


/// Check that elements of matrices \a x and \a y are close with respect to a given relative precision \a e.
#define BOOST_UBLASX_TEST_CHECK_MATRIX_REL_CLOSE(x,y,nr,nc,e)	/* [BOOST_UBLASX_TEST_CHECK_MATRIX_REL_CLOSE] */ \
																for (::std::size_t i__ = 0; i__ < BOOST_UBLASX_TEST_EXPANDP_(nr); ++i__) \
																{ \
																	for (::std::size_t j__ = 0; j__ < BOOST_UBLASX_TEST_EXPANDP_(nc); ++j__) \
																	{ \
																		if (!::boost::numeric::ublas::test::detail::rel_close_to(BOOST_UBLASX_TEST_EXPANDP_(x)(i__,j__), BOOST_UBLASX_TEST_EXPANDP_(y)(i__,j__), BOOST_UBLASX_TEST_EXPANDP_(e))) \
																		{ \
																			BOOST_UBLASX_TEST_ERROR( "Failed assertion: abs((" << BOOST_UBLASX_TEST_STRINGIFY_(x(i__,j__)) << "-" << BOOST_UBLASX_TEST_STRINGIFY_(y(i__,j__)) << ")/" << BOOST_UBLASX_TEST_STRINGIFY_(y(i__,j__)) << ") <= " << BOOST_UBLASX_TEST_STRINGIFY_(e)  << " [with " << BOOST_UBLASX_TEST_STRINGIFY_(x(i__,j__)) << " == " << BOOST_UBLASX_TEST_EXPANDP_(x)(i__,j__) << ", " << BOOST_UBLASX_TEST_STRINGIFY_(y(i__,j__)) << " == " << BOOST_UBLASX_TEST_EXPANDP_(y)(i__,j__) << ", " << BOOST_UBLASX_TEST_STRINGIFY_(i__) << " == " << i__ << ", " << BOOST_UBLASX_TEST_STRINGIFY_(j__) << " == " << BOOST_UBLASX_TEST_EXPANDP_(j__) << ", " << BOOST_UBLASX_TEST_STRINGIFY_(nr) << " == " << BOOST_UBLASX_TEST_EXPANDP_(nr) << " and " << BOOST_UBLASX_TEST_STRINGIFY_(nc) << " == " << BOOST_UBLASX_TEST_EXPANDP_(nc) << "]" ); \
																			++test_fails__; \
																		} \
																	} \
																} \
																/* [/BOOST_UBLASX_TEST_CHECK_MATRIX_REL_CLOSE] */


///< Output the error message \a x.
#define BOOST_UBLASX_TEST_ERROR(x) ::std::cerr << "[Error (" << __FILE__ << ":" << __func__ << ":" << __LINE__ << ")>> " << BOOST_UBLASX_TEST_EXPAND_(x) << ::std::endl

#endif // BOOST_NUMERIC_UBLASX_TEST_UTILS_HPP
