/**
 * \file boost/numeric/ublasx/operation/logspace.hpp
 *
 * \brief Logarithmically spaced vector
 *
 * Inspired by MATLAB's logspace function.
 *
 * <hr/>
 *
 * Copyright (c) 2015, Marco Guazzone
 * 
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 *
 * \author Marco Guazzone (marco.guazzone@gmail.com)
 */

#ifndef BOOST_NUMERIC_UBLASX_OPERATION_LOGSPACE_HPP
#define BOOST_NUMERIC_UBLASX_OPERATION_LOGSPACE_HPP


#include <boost/numeric/ublas/exception.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublasx/operation/element_pow.hpp>
#include <boost/numeric/ublasx/operation/linspace.hpp>
#include <cstddef>


namespace boost { namespace numeric { namespace ublasx {

using namespace ::boost::numeric::ublas;

/**
 * \brief Generates a logarithmically spaced vector.
 *
 * Generates \a n values logarithmically equally spaced between `pow(base,a)`
 * and `pow(base,b)`.
 * Note, in case \f$a<b\f$, generates a decreasing sequence.
 *
 * Inspired by MATLAB's logspace function.
 *
 * \param a The starting value of the logarithmically spaced sequence.
 * \param b The final value of the logarithmically spaced sequence.
 * \param n The number of values to generate
 * \param base The base of the logarithm
 * \return A vector of logarithmically spaced values in
 *  \f$[\mathrm{base}^a,\mathrm{base}^b]\f$; if `n=1`, returns `pow(base,b)`.
 *
 * \author Marco Guazzone (marco.guazzone@gmail.com)
 */
template <typename ValueT>
BOOST_UBLAS_INLINE
vector<ValueT> logspace(ValueT a, ValueT b, std::size_t n = 100, ValueT base = 10)
{
	// pre: n > 0
	BOOST_UBLAS_CHECK( n > 0,
					   bad_argument() );
	// pre: base > 0
	BOOST_UBLAS_CHECK( base > 0,
					   bad_argument() );

	return element_pow(base, linspace(a, b, n));
}

}}} // Namespace boost::numeric::ublasx


#endif // BOOST_NUMERIC_UBLASX_OPERATION_LOGSPACE_HPP
