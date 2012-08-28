/**
 * \file boost/numeric/ublasx/detail/lapack.hpp
 *
 * \brief LAPACK definitions.
 *
 * Copyright (c) 2009-2010, Marco Guazzone
 *
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 *
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */

#ifndef BOOST_NUMERIC_UBLASX_DETAIL_LAPACK_HPP
#define BOOST_NUMERIC_UBLASX_DETAIL_LAPACK_HPP


#include <cstddef>


namespace boost { namespace numeric { namespace ublasx { namespace detail { namespace lapack {

/// The minimum dimension we can assign to LAPACK arrays.
static const std::size_t min_array_size = 1;

}}}}} // Namespace boost::numeric::ublasx::detail::lapack


#endif // BOOST_NUMERIC_UBLASX_DETAIL_LAPACK_HPP
