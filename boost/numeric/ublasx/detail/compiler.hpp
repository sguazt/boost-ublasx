/* vim: set tabstop=4 expandtab shiftwidth=4 softtabstop=4: */

/**
 * \file boost/numeric/ublasx/detail/compiler.hpp
 *
 * \brief Compiler-related code.
 *
 * Copyright (c) 2010, Marco Guazzone
 * 
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */

#ifndef BOOST_NUMERIC_UBLASX_DETAIL_COMPILER_HPP
#define BOOST_NUMERIC_UBLASX_DETAIL_COMPILER_HPP

namespace boost { namespace numeric { namespace ublasx { namespace detail {

/// Dummy function used to quiet the compiler about the unused variable warning.
template <typename T>
inline void suppress_unused_variable_warning(T const&) {}

}}}} // Namespace boost::numeric::ublasx::detail


/// Suppress warnings issued by the compiler for not using variable \a x.
#define BOOST_UBLASX_SUPPRESS_UNUSED_VARIABLE_WARNING(x) \
    ::boost::numeric::ublasx::detail::suppress_unused_variable_warning(x)


#endif // BOOST_NUMERIC_UBLASX_DETAIL_COMPILER_HPP
