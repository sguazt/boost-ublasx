/**
 * \file boost/numeric/ublasx/operation/eps.hpp
 *
 * \brief Floating-point relative accuracy.
 *
 * Given a scalar \f$x\f$, compute the positive distance from \f$|x|\f$ to
 * the next larger in magnitude floating point number of the same precision as
 * \f$x\f$.
 * Except for numbers whose absolute value is smaller than the smallest positive
 * normalied floating-point number representable by its type , if
 * \f$2^y \le |x| < 2^{y+1}\f$, then the \c eps(x) function return
 * \f$2^{y-d}\f$, where \f$d\f$ is the number of radix digits in the mantissa.
 *
 * For all X of class double such that abs(X) <= realmin, eps(X) = 2^(-1074). Similarly, for all X of class single such that abs(X) <= realmin('single'), eps(X) = 2^(-149).
 *
 * <hr&>
 *
 * Copyright (c) 2010, Marco Guazzone
 *
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */

#ifndef BOOST_NUMERIC_UBLASX_OPERATION_EPS_HPP
#define BOOST_NUMERIC_UBLASX_OPERATION_EPS_HPP


#include <boost/numeric/ublas/traits.hpp>
#include <cmath>
#include <limits>


namespace boost { namespace numeric { namespace ublasx {

using namespace ::boost::numeric::ublas;


/**
 * \brief Compute the distance from \c 1.0 to the next largest floating-point
 *  precision number.
 * \tparam RealT The floaint-point type.
 * \return The distance from \c 1.0 to the next largest floating-point
 *  precision number.
 */
template <typename RealT>
BOOST_UBLAS_INLINE
typename type_traits<RealT>::real_type eps()
{
	// NOTE: type_traits<>::real_type is used in case of RealT is a non real
	//       type (e.g., std::complex).

	return ::std::numeric_limits<typename type_traits<RealT>::real_type>::epsilon();
}


/**
 * \brief  Given a scalar \f$x\f$, compute the positive distance from
 *  \f$|x|\f$ to the next larger in magnitude floating point number of the
 *  same precision as \f$x\f$.
 * \tparam RealT The floaint-point type.
 * \param x A floating-point scalar value.
 * \return The positive distance from \f$|x|\f$ to the next larger in
 *  magnitude floating point number of the same precision as \f$a\f$.
 */
template <typename RealT>
BOOST_UBLAS_INLINE
typename type_traits<RealT>::real_type eps(RealT x)
{
	// NOTE: type_traits<>::real_type is used in case of RealT is a non real
	//       type (e.g., std::complex).

	typedef typename type_traits<RealT>::real_type real_type;
	real_type y = ::std::abs(x);

	if (y == ::std::numeric_limits<real_type>::infinity()
		||
		::std::isnan(y))
	{
		return ::std::numeric_limits<real_type>::quiet_NaN();
	}
	else if (y <= ::std::numeric_limits<real_type>::min())
	{
		return ::std::numeric_limits<real_type>::denorm_min();
	}
	else
	{
		int e;
		::std::frexp(y, &e);

		return ::std::ldexp(1, e - ::std::numeric_limits<real_type>::digits);
	}
}

}}} // Namespace boost::numeric::ublasx

#endif // BOOST_NUMERIC_UBLASX_OPERATION_EPS_HPP
