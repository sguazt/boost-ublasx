/* vim: set tabstop=4 expandtab shiftwidth=4 softtabstop=4: */

/**
 * \file boost/numeric/ublasx/operation/realmax.hpp
 *
 * \brief Largest positive normalized floating-point number.
 *
 * \author Marco Guazzone (marco.guazzone@gmail.com)
 *
 * <hr/>
 *
 * Copyright (c) 2010, Marco Guazzone
 *
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef BOOST_NUMERIC_UBLASX_OPERATION_REALMAX_HPP
#define BOOST_NUMERIC_UBLASX_OPERATION_REALMAX_HPP


#include <boost/numeric/ublas/traits.hpp>
#include <limits>


namespace boost { namespace numeric { namespace ublasx {

using namespace boost::numeric::ublas;

/// Return the largest positive normalized floating-point number.
template <typename RealT>
BOOST_UBLAS_INLINE
typename type_traits<RealT>::real_type realmax()
{
    return ::std::numeric_limits<typename type_traits<RealT>::real_type>::max();
}

}}} // Namespace boost::numeric::ublasx

#endif // BOOST_NUMERIC_UBLASX_OPERATION_REALMAX_HPP
