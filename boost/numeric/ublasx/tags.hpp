/**
 * \file boost/numeric/ublasx/tags.hpp
 *
 * \brief Tags.
 *
 * Copyright (c) 2009, Marco Guazzone
 *
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */

#ifndef BOOST_NUMERIC_UBLASX_TAG_HPP
#define BOOST_NUMERIC_UBLASX_TAG_HPP


#include <boost/numeric/ublas/tags.hpp>


namespace boost { namespace numeric { namespace ublasx { namespace tag {

/// \brief Tag for the major dimension.
using ::boost::numeric::ublas::tag::major;

/// \brief Tag for the minor dimension.
using ::boost::numeric::ublas::tag::minor;

/// \brief Tag for the leading dimension.
using ::boost::numeric::ublas::tag::leading;

}}}} // Namespace boost::numeric::ublasx::tag


#endif // BOOST_NUMERIC_UBLASX_TAG_HPP
