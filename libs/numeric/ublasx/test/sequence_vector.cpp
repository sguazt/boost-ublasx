#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/storage.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublasx/container/sequence_vector.hpp>
#include "libs/numeric/ublasx/test/utils.hpp"


namespace ublas = ::boost::numeric::ublas;
namespace ublasx = ::boost::numeric::ublasx;


BOOST_UBLASX_TEST_DEF( creation_incr )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Creation - Increasing Sequence");

	typedef short value_type;

	const std::size_t n(3);

	ublasx::sequence_vector<value_type> res;
	ublas::vector<value_type> expect_res(n);

	res = ublasx::sequence_vector<value_type>(0, 2, n);

	value_type x(0);
	for (std::size_t i = 0; i < n; ++i)
	{
		expect_res(i) = x;
		x += 2;
	}

	BOOST_UBLASX_DEBUG_TRACE( "res = " << res );
	BOOST_UBLASX_DEBUG_TRACE( "expect res = " << expect_res );
	BOOST_UBLASX_TEST_CHECK_VECTOR_EQ( res, expect_res, n );
}


BOOST_UBLASX_TEST_DEF( creation_decr )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Creation - Decreasing Sequence");

	typedef short value_type;

	const std::size_t n(3);

	ublasx::sequence_vector<value_type> res;
	ublas::vector<value_type> expect_res(n);

	res = ublasx::sequence_vector<value_type>(0, -2, n);

	value_type x(0);
	for (std::size_t i = 0; i < n; ++i)
	{
		expect_res(i) = x;
		x -= 2;
	}

	BOOST_UBLASX_DEBUG_TRACE( "res = " << res );
	BOOST_UBLASX_DEBUG_TRACE( "expect res = " << expect_res );
	BOOST_UBLASX_TEST_CHECK_VECTOR_EQ( res, expect_res, n );
}


BOOST_UBLASX_TEST_DEF( creation_from_range )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Creation - Range");

	typedef short value_type;

	const std::size_t n(3);

	ublasx::sequence_vector<value_type> res;
	ublas::vector<value_type> expect_res(n);

	res = ublasx::sequence_vector<value_type>(ublas::range(4, 4+n));

	value_type x(4);
	for (std::size_t i = 0; i < n; ++i)
	{
		expect_res(i) = x;
		++x;
	}

	BOOST_UBLASX_DEBUG_TRACE( "res = " << res );
	BOOST_UBLASX_DEBUG_TRACE( "expect res = " << expect_res );
	BOOST_UBLASX_TEST_CHECK_VECTOR_EQ( res, expect_res, n );
}


BOOST_UBLASX_TEST_DEF( creation_from_slice )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Creation - Slice");

	typedef short value_type;

	const std::size_t n(3);

	ublasx::sequence_vector<value_type> res;
	ublas::vector<value_type> expect_res(n);

	res = ublasx::sequence_vector<value_type>(ublas::slice(5, 3, n));

	value_type x(5);
	for (std::size_t i = 0; i < n; ++i)
	{
		expect_res(i) = x;
		x += 3;
	}

	BOOST_UBLASX_DEBUG_TRACE( "res = " << res );
	BOOST_UBLASX_DEBUG_TRACE( "expect res = " << expect_res );
	BOOST_UBLASX_TEST_CHECK_VECTOR_EQ( res, expect_res, n );
}


int main()
{
	BOOST_UBLASX_DEBUG_TRACE("Test Suite: Sequence Vector class");

	BOOST_UBLASX_TEST_BEGIN();

	BOOST_UBLASX_TEST_DO( creation_incr );
	BOOST_UBLASX_TEST_DO( creation_decr );
	BOOST_UBLASX_TEST_DO( creation_from_range );
	BOOST_UBLASX_TEST_DO( creation_from_slice );

	BOOST_UBLASX_TEST_END();
}
