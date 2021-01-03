/* vim: set tabstop=4 expandtab shiftwidth=4 softtabstop=4: */

/**
 * \file boost/numeric/ublasx/detail/debug.hpp
 *
 * \brief Utility to help debugging.
 *
 *  Copyright (c) 2010, Marco Guazzone
 *
 *  Distributed under the Boost Software License, Version 1.0. (See
 *  accompanying file LICENSE_1_0.txt or copy at
 *  http://www.boost.org/LICENSE_1_0.txt)
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */

#ifndef BOOST_NUMERIC_UBLASX_DETAIL_DEBUG_HPP
#define BOOST_NUMERIC_UBLASX_DETAIL_DEBUG_HPP


#ifndef NDEBUG
//@{ for C99 and C++0x
//# include <cstdio>
//# define BOOST_UBLASX_DEBUG_WHERESTR_  "[file %s, function %s, line %d]: "
//# define BOOST_UBLASX_DEBUG_WHEREARG_  __FILE__, __func__, __LINE__
//# define BOOST_UBLASX_DEBUG_PRINT2_(...)       fprintf(stderr, __VA_ARGS__)
//# define BOOST_UBLASX_DEBUG_PRINT(fmt, ...)  BOOST_UBLASX_DEBUG_PRINT2_(BOOST_UBLASX_DEBUG_WHERESTR_ fmt, BOOST_UBLASX_DEBUG_WHEREARG_, __VA_ARGS__)
//@} for C99 and C++0x

#include <boost/numeric/ublasx/detail/macro.hpp>
#include <iostream>
#include <typeinfo>

/// Macro for telling whether we are in debug mode.
#   define BOOST_UBLASX_DEBUG /**/

/// Output the message \a x if in debug-mode; otherwise output nothing.
#   define BOOST_UBLASX_DEBUG_TRACE(x) ::std::cerr << "[Debug>> " << BOOST_UBLASX_PASSTHROUGH(x) << ::std::endl

/// Macro for setting flags \c x of the underlying debug output stream.
#   define BOOST_UBLASX_DEBUG_STREAM_SETFLAGS(x) std::cerr.setf((x))

/// Macro for converting the a type into a string.
#   define BOOST_UBLASX_DEBUG_TYPE2STR(x) typeid(x).name()

#else

/// Macro for telling that we are in debug mode.
#   undef BOOST_UBLASX_DEBUG

/// Output the message \a x if in debug-mode; otherwise output nothing.
#   define BOOST_UBLASX_DEBUG_TRACE(x) /**/

/// Macro for setting flags \c x of the underlying debug output stream.
#   define BOOST_UBLASX_DEBUG_STREAM_SETFLAGS(x) /**/

/// Macro for converting the a type into a string.
#   define BOOST_UBLASX_DEBUG_TYPE2STR(x) /**/

#endif // NDEBUG


#endif // BOOST_NUMERIC_UBLASX_DETAIL_DEBUG_HPP
