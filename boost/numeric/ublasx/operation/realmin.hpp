/* vim: set tabstop=4 expandtab shiftwidth=4 softtabstop=4: */

/**
 * \file boost/numeric/ublasx/operation/realmin.hpp
 *
 * \brief Smallest positive normalized floating-point number.
 *
 * Copyright (c) 2010, Marco Guazzone
 *
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */

#ifndef BOOST_NUMERIC_UBLASX_OPERATION_REALMIN_HPP
#define BOOST_NUMERIC_UBLASX_OPERATION_REALMIN_HPP


#include <boost/numeric/ublas/fwd.hpp>
#include <boost/numeric/ublas/traits.hpp>
#include <limits>


namespace boost { namespace numeric { namespace ublasx {

using namespace ::boost::numeric::ublas;

/// Return the smallest positive normalized floating-point number.
template <typename RealT>
BOOST_UBLAS_INLINE
typename type_traits<RealT>::real_type realmin()
{
    return ::std::numeric_limits<typename type_traits<RealT>::real_type>::min();
}

}}} // Namespace boost::numeric::ublasx

#endif // BOOST_NUMERIC_UBLASX_OPERATION_REALMIN_HPP
