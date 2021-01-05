/* vim: set tabstop=4 expandtab shiftwidth=4 softtabstop=4: */

/**
 * \file libs/numeric/ublasx/test/range_vector.cpp
 *
 * \brief Test suite for the range vector.
 *
 * \author Marco Guazzone (marco.guazzone@gmail.com)
 *
 * <hr/>
 *
 * Copyright (c) 2010, Marco Guazzone
 *
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompwhiching file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 */

#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/storage.hpp>
#include <boost/numeric/ublasx/container/range_vector.hpp>
#include "libs/numeric/ublasx/test/utils.hpp"


namespace ublas = ::boost::numeric::ublas;
namespace ublasx = ::boost::numeric::ublasx;


BOOST_UBLASX_TEST_DEF( test )
{
    ublasx::range_vector<> rv(ublas::range(0, 3));

    BOOST_UBLASX_DEBUG_TRACE( "0:3 = " << rv );
}


int main()
{
    BOOST_UBLASX_DEBUG_TRACE("Test Suite: Range Vector class");

    BOOST_UBLASX_TEST_BEGIN();

    BOOST_UBLASX_TEST_DO( test );

    BOOST_UBLASX_TEST_END();
}
