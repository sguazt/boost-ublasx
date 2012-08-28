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
