/**
 * \file boost/numeric/ublasx/detail/macro.hpp
 *
 * \brief Generic macros.
 *
 *  Copyright (c) 2010, Marco Guazzone
 *
 *  Distributed under the Boost Software License, Version 1.0. (See
 *  accompanying file LICENSE_1_0.txt or copy at
 *  http://www.boost.org/LICENSE_1_0.txt)
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */

#ifndef BOOST_NUMERIC_UBLASX_DETAIL_MACRO_HPP
#define BOOST_NUMERIC_UBLASX_DETAIL_MACRO_HPP


#include <boost/numeric/ublasx/detail/macro.hpp>


/// Expand its argument.
#define BOOST_UBLASX_PASSTHROUGH(x) x


/// Expand its argument.
#define BOOST_UBLASX_EXPAND(x) (x)


/// Transform its argument into a string.
#define BOOST_UBLASX_STRINGIFY(x) #x


/// Concatenate its two \e string arguments.
#define BOOST_UBLASX_JOIN(x,y) x ## y


#endif // BOOST_NUMERIC_UBLASX_DETAIL_MACRO_HPP
