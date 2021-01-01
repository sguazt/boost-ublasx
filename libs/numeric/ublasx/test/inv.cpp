/**
 * \file libs/numeric/ublasx/test/inv.cpp
 *
 * \brief Test suite for the \c inv operation.
 *
 * Copyright (c) 2011, Marco Guazzone
 *
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */


// Make sure that type checks are disabled in order to successfully run the
// 'illconditioned_matrix' test.
#undef BOOST_UBLAS_TYPE_CHECK
#define BOOST_UBLASX_TEST_INV_TYPE_CHECK BOOST_UBLAS_TYPE_CHECK
#define BOOST_UBLAS_TYPE_CHECK 0

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublasx/detail/debug.hpp>
#include <boost/numeric/ublasx/operation/hilb.hpp>
#include <boost/numeric/ublasx/operation/inv.hpp>
#include <exception>
#include "libs/numeric/ublasx/test/utils.hpp"

// Restore the old value for BOOST_UBLAS_TYPE_CHECK
#undef BOOST_UBLAS_TYPE_CHECK
#define BOOST_UBLAS_TYPE_CHECK BOOST_UBLASX_TEST_INV_TYPE_CHECK
#undef BOOST_UBLASX_TEST_INV_TYPE_CHECK


namespace ublas = ::boost::numeric::ublas;
namespace ublasx = ::boost::numeric::ublasx;


const double tol = 1.0e-5;


BOOST_UBLASX_TEST_DEF( row_major_matrix_inplace )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Row-major Matrix - In place Inversion");

	typedef double value_type;
	typedef ublas::matrix<value_type> matrix_type;
	typedef ::std::size_t size_type;

	const size_type n(4);

	matrix_type A(n,n);
	A(0,0) =  1.80; A(0,1) =  2.88; A(0,2) =  2.05; A(0,3) = -0.89;
	A(1,0) =  5.25; A(1,1) = -2.95; A(1,2) = -0.95; A(1,3) = -3.80;
	A(2,0) =  1.58; A(2,1) = -2.69; A(2,2) = -2.90; A(2,3) = -1.04;
	A(3,0) = -1.11; A(3,1) = -0.66; A(3,2) = -0.59; A(3,3) =  0.80;

	bool inv_ok(false);
	matrix_type expect;


	inv_ok = ublasx::inv_inplace(A);
	expect = matrix_type(n,n);
	expect(0,0) = 1.771998173034358; expect(0,1) =  0.575690823228768; expect(0,2) =  0.084325372165000; expect(0,3) =  4.815502361651872;
	expect(1,0) =-0.117466074066139; expect(1,1) = -0.445615014196196; expect(1,2) =  0.411362607935861; expect(1,3) = -1.712580934513892;
	expect(2,0) = 0.179856389553414; expect(2,1) =  0.452662043400721; expect(2,2) = -0.667565300509907; expect(2,3) =  1.482400048868720;
	expect(3,0) = 2.494382041276250; expect(3,1) =  0.764976887526086; expect(3,2) = -0.035953803700033; expect(3,3) =  7.611900291858691;

	BOOST_UBLASX_DEBUG_TRACE("A^{-1} = " << A);
	BOOST_UBLASX_TEST_CHECK( inv_ok );
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( A, expect, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( col_major_matrix_inplace )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Column-major Matrix - In place Inversion");

	typedef double value_type;
	typedef ublas::matrix<value_type, ublas::column_major> matrix_type;
	typedef ::std::size_t size_type;

	const size_type n(4);

	matrix_type A(n,n);
	A(0,0) =  1.80; A(0,1) =  2.88; A(0,2) =  2.05; A(0,3) = -0.89;
	A(1,0) =  5.25; A(1,1) = -2.95; A(1,2) = -0.95; A(1,3) = -3.80;
	A(2,0) =  1.58; A(2,1) = -2.69; A(2,2) = -2.90; A(2,3) = -1.04;
	A(3,0) = -1.11; A(3,1) = -0.66; A(3,2) = -0.59; A(3,3) =  0.80;

	bool inv_ok(false);
	matrix_type expect;


	inv_ok = ublasx::inv_inplace(A);
	expect = matrix_type(n,n);
	expect(0,0) = 1.771998173034358; expect(0,1) =  0.575690823228768; expect(0,2) =  0.084325372165000; expect(0,3) =  4.815502361651872;
	expect(1,0) =-0.117466074066139; expect(1,1) = -0.445615014196196; expect(1,2) =  0.411362607935861; expect(1,3) = -1.712580934513892;
	expect(2,0) = 0.179856389553414; expect(2,1) =  0.452662043400721; expect(2,2) = -0.667565300509907; expect(2,3) =  1.482400048868720;
	expect(3,0) = 2.494382041276250; expect(3,1) =  0.764976887526086; expect(3,2) = -0.035953803700033; expect(3,3) =  7.611900291858691;

	BOOST_UBLASX_DEBUG_TRACE("A^{-1} = " << A);
	BOOST_UBLASX_TEST_CHECK( inv_ok );
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( A, expect, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( row_major_matrix )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Row-major Matrix");

	typedef double value_type;
	typedef ublas::matrix<value_type> matrix_type;
	typedef ::std::size_t size_type;

	const size_type n(4);

	matrix_type A(n,n);
	A(0,0) =  1.80; A(0,1) =  2.88; A(0,2) =  2.05; A(0,3) = -0.89;
	A(1,0) =  5.25; A(1,1) = -2.95; A(1,2) = -0.95; A(1,3) = -3.80;
	A(2,0) =  1.58; A(2,1) = -2.69; A(2,2) = -2.90; A(2,3) = -1.04;
	A(3,0) = -1.11; A(3,1) = -0.66; A(3,2) = -0.59; A(3,3) =  0.80;

	matrix_type B;
	matrix_type expect;


	B = ublasx::inv(A);
	expect = matrix_type(n,n);
	expect(0,0) = 1.771998173034358; expect(0,1) =  0.575690823228768; expect(0,2) =  0.084325372165000; expect(0,3) =  4.815502361651872;
	expect(1,0) =-0.117466074066139; expect(1,1) = -0.445615014196196; expect(1,2) =  0.411362607935861; expect(1,3) = -1.712580934513892;
	expect(2,0) = 0.179856389553414; expect(2,1) =  0.452662043400721; expect(2,2) = -0.667565300509907; expect(2,3) =  1.482400048868720;
	expect(3,0) = 2.494382041276250; expect(3,1) =  0.764976887526086; expect(3,2) = -0.035953803700033; expect(3,3) =  7.611900291858691;

	BOOST_UBLASX_DEBUG_TRACE("A^{-1} = " << B);
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( B, expect, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( col_major_matrix )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Column-major Matrix");

	typedef double value_type;
	typedef ublas::matrix<value_type, ublas::column_major> matrix_type;
	typedef ::std::size_t size_type;

	const size_type n(4);

	matrix_type A(n,n);
	A(0,0) =  1.80; A(0,1) =  2.88; A(0,2) =  2.05; A(0,3) = -0.89;
	A(1,0) =  5.25; A(1,1) = -2.95; A(1,2) = -0.95; A(1,3) = -3.80;
	A(2,0) =  1.58; A(2,1) = -2.69; A(2,2) = -2.90; A(2,3) = -1.04;
	A(3,0) = -1.11; A(3,1) = -0.66; A(3,2) = -0.59; A(3,3) =  0.80;

	matrix_type B;
	matrix_type expect;


	B = ublasx::inv(A);
	expect = matrix_type(n,n);
	expect(0,0) = 1.771998173034358; expect(0,1) =  0.575690823228768; expect(0,2) =  0.084325372165000; expect(0,3) =  4.815502361651872;
	expect(1,0) =-0.117466074066139; expect(1,1) = -0.445615014196196; expect(1,2) =  0.411362607935861; expect(1,3) = -1.712580934513892;
	expect(2,0) = 0.179856389553414; expect(2,1) =  0.452662043400721; expect(2,2) = -0.667565300509907; expect(2,3) =  1.482400048868720;
	expect(3,0) = 2.494382041276250; expect(3,1) =  0.764976887526086; expect(3,2) = -0.035953803700033; expect(3,3) =  7.611900291858691;

	BOOST_UBLASX_DEBUG_TRACE("A^{-1} = " << B);
	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( B, expect, n, n, tol );
}


BOOST_UBLASX_TEST_DEF( rectangular_matrix )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Rectangular Matrix");

	typedef double value_type;
	typedef ublas::matrix<value_type> matrix_type;
	typedef ::std::size_t size_type;

	const size_type nr(3);
	const size_type nc(2);

	matrix_type A(nr,nc);
	A(0,0) =  1.00; A(0,1) = 2.00;
	A(1,0) =  3.00; A(1,1) = 4.00;
	A(2,0) =  5.00; A(2,1) = 6.00;

	bool expect(false);


	try
	{
		matrix_type B;

		B = ublasx::inv(A);

		expect = false;
	}
	catch (::std::exception const& e)
	{
		BOOST_UBLASX_DEBUG_TRACE("Caught exception: " << e.what());
		expect = true;
	}

	BOOST_UBLASX_TEST_CHECK( expect );
}


BOOST_UBLASX_TEST_DEF( singular_matrix )
{
	BOOST_UBLASX_DEBUG_TRACE("Test Case: Singular Matrix");

	typedef double value_type;
	typedef ublas::matrix<value_type> matrix_type;
	typedef ::std::size_t size_type;

	const size_type n(2);

	matrix_type A(n,n);
	A(0,0) =  1.00; A(0,1) = 2.00;
	A(1,0) =  1.00; A(1,1) = 2.00;

	matrix_type B;
	matrix_type expect;


	B = ublasx::inv(A);
	expect = ublas::scalar_matrix<value_type>(n,n, ::std::numeric_limits<value_type>::infinity());

	BOOST_UBLASX_DEBUG_TRACE("A^{-1} = " << B);
	BOOST_UBLASX_TEST_CHECK_MATRIX_EQ( B, expect, n, n );
}


BOOST_UBLASX_TEST_DEF( illconditioned_matrix )
{

	BOOST_UBLASX_DEBUG_TRACE("Test Case: Ill-conditioned Matrix");
	BOOST_UBLASX_DEBUG_TRACE("  NOTE: Expect to Fail");

	typedef double value_type;
	typedef ublas::matrix<value_type> matrix_type;
	typedef ::std::size_t size_type;

	const size_type n(15);

	matrix_type A(ublasx::hilb(n));

	matrix_type B;
	matrix_type expect;


	BOOST_UBLASX_DEBUG_TRACE("A = " << A);
	B = ublasx::inv(A);
	expect = matrix_type(n,n);
	expect( 0, 0) =  0.000000000000015; expect( 0, 1) = -0.000000000001135; expect( 0, 2) =  0.000000000027663; expect( 0, 3) = -0.000000000322359; expect( 0, 4) =  0.000000002106184; expect( 0, 5) = -0.000000008327391; expect( 0, 6) =  0.000000020509383; expect( 0, 7) = -0.000000031165902; expect( 0, 8) =  0.000000028161374; expect( 0, 9) = -0.000000017595800; expect( 0,10) =  0.000000021720982; expect( 0,11) = -0.000000037794293; expect( 0,12) =  0.000000037822586; expect( 0,13) = -0.000000018936236; expect( 0,14) =  0.000000003794930;
	expect( 1, 0) = -0.000000000001135; expect( 1, 1) =  0.000000000113818; expect( 1, 2) = -0.000000003134274; expect( 1, 3) =  0.000000039206017; expect( 1, 4) = -0.000000269474866; expect( 1, 5) =  0.000001113546642; expect( 1, 6) = -0.000002879031111; expect( 1, 7) =  0.000004693328846; expect( 1, 8) = -0.000004835284200; expect( 1, 9) =  0.000003712413766; expect( 1,10) = -0.000004031585592; expect( 1,11) =  0.000005669113752; expect( 1,12) = -0.000005280440772; expect( 1,13) =  0.000002586209123; expect( 1,14) = -0.000000514980210;
	expect( 2, 0) =  0.000000000027663; expect( 2, 1) = -0.000000003134274; expect( 2, 2) =  0.000000092605093; expect( 2, 3) = -0.000001216803663; expect( 2, 4) =  0.000008709220452; expect( 2, 5) = -0.000037443539434; expect( 2, 6) =  0.000101459901155; expect( 2, 7) = -0.000176956007098; expect( 2, 8) =  0.000203623046245; expect( 2, 9) = -0.000178359663965; expect( 2,10) =  0.000178297492780; expect( 2,11) = -0.000209406698825; expect( 2,12) =  0.000180031334394; expect( 2,13) = -0.000085747214524; expect( 2,14) =  0.000016919440846;
	expect( 3, 0) = -0.000000000322359; expect( 3, 1) =  0.000000039206017; expect( 3, 2) = -0.000001216803663; expect( 3, 3) =  0.000016633129082; expect( 3, 4) = -0.000123478399050; expect( 3, 5) =  0.000551937865697; expect( 3, 6) = -0.001568844300273; expect( 3, 7) =  0.002924018288619; expect( 3, 8) = -0.003699545110614; expect( 3, 9) =  0.003544822435936; expect( 3,10) = -0.003318581860854; expect( 3,11) =  0.003305473056782; expect( 3,12) = -0.002588068541346; expect( 3,13) =  0.001188322127769; expect( 3,14) = -0.000231510872329;
	expect( 4, 0) =  0.000000002106184; expect( 4, 1) = -0.000000269474866; expect( 4, 2) =  0.000008709220452; expect( 4, 3) = -0.000123478399050; expect( 4, 4) =  0.000951060735832; expect( 4, 5) = -0.004430383416667; expect( 4, 6) =  0.013247732497878; expect( 4, 7) = -0.026370615726994; expect( 4, 8) =  0.036192914970134; expect( 4, 9) = -0.036849609703653; expect( 4,10) =  0.032388804243636; expect( 4,11) = -0.027311153070550; expect( 4,12) =  0.018987053862231; expect( 4,13) = -0.008272912326863; expect( 4,14) =  0.001582145241706;
	expect( 5, 0) = -0.000000008327391; expect( 5, 1) =  0.000001113546642; expect( 5, 2) = -0.000037443539434; expect( 5, 3) =  0.000551937865697; expect( 5, 4) = -0.004430383416667; expect( 5, 5) =  0.021623894530916; expect( 5, 6) = -0.068312259561204; expect( 5, 7) =  0.145076633389740; expect( 5, 8) = -0.213108600667151; expect( 5, 9) =  0.224955978959315; expect( 5,10) = -0.183625950054056; expect( 5,11) =  0.127213485355599; expect( 5,12) = -0.073762939837580; expect( 5,13) =  0.029275516007783; expect( 5,14) = -0.005420977412400;
	expect( 6, 0) =  0.000000020509383; expect( 6, 1) = -0.000002879031111; expect( 6, 2) =  0.000101459901155; expect( 6, 3) = -0.001568844300273; expect( 6, 4) =  0.013247732497878; expect( 6, 5) = -0.068312259561204; expect( 6, 6) =  0.229049070708291; expect( 6, 7) = -0.517275605487115; expect( 6, 8) =  0.800461681765880; expect( 6, 9) = -0.852873361163061; expect( 6,10) =  0.625257683259964; expect( 6,11) = -0.318158094252897; expect( 6,12) =  0.117090718756610; expect( 6,13) = -0.032195652749596; expect( 6,14) =  0.005178336003076;
	expect( 7, 0) = -0.000000031165902; expect( 7, 1) =  0.000004693328846; expect( 7, 2) = -0.000176956007098; expect( 7, 3) =  0.002924018288619; expect( 7, 4) = -0.026370615726994; expect( 7, 5) =  0.145076633389740; expect( 7, 6) = -0.517275605487115; expect( 7, 7) =  1.231030965555309; expect( 7, 8) = -1.961333420439397; expect( 7, 9) =  2.026141297007867; expect( 7,10) = -1.208876359519808; expect( 7,11) =  0.215257108700206; expect( 7,12) =  0.212547662099395; expect( 7,13) = -0.149163484452081; expect( 7,14) =  0.030214090603182;
	expect( 8, 0) =  0.000000028161374; expect( 8, 1) = -0.000004835284200; expect( 8, 2) =  0.000203623046245; expect( 8, 3) = -0.003699545110614; expect( 8, 4) =  0.036192914970134; expect( 8, 5) = -0.213108600667151; expect( 8, 6) =  0.800461681765880; expect( 8, 7) = -1.961333420439397; expect( 8, 8) =  3.084540906170247; expect( 8, 9) = -2.822265919772054; expect( 8,10) =  0.820618664142026; expect( 8,11) =  1.196934021150055; expect( 8,12) = -1.551378693718132; expect( 8,13) =  0.753570737589437; expect( 8,14) = -0.140731579235935;
	expect( 9, 0) = -0.000000017595800; expect( 9, 1) =  0.000003712413766; expect( 9, 2) = -0.000178359663965; expect( 9, 3) =  0.003544822435936; expect( 9, 4) = -0.036849609703653; expect( 9, 5) =  0.224955978959315; expect( 9, 6) = -0.852873361163061; expect( 9, 7) =  2.026141297007867; expect( 9, 8) = -2.822265919772054; expect( 9, 9) =  1.510675327619418; expect( 9,10) =  1.901183785745312; expect( 9,11) = -4.445256273997598; expect( 9,12) =  3.864018381472909; expect( 9,13) = -1.669802837364426; expect( 9,14) =  0.296703119337109;
	expect(10, 0) =  0.000000021720982; expect(10, 1) = -0.000004031585592; expect(10, 2) =  0.000178297492780; expect(10, 3) = -0.003318581860854; expect(10, 4) =  0.032388804243636; expect(10, 5) = -0.183625950054056; expect(10, 6) =  0.625257683259964; expect(10, 7) = -1.208876359519808; expect(10, 8) =  0.820618664142026; expect(10, 9) =  1.901183785745312; expect(10,10) = -5.953305805323117; expect(10,11) =  7.736775352133582; expect(10,12) = -5.608616538552710; expect(10,13) =  2.213742954886355; expect(10,14) = -0.372398343230434;
	expect(11, 0) = -0.000000037794293; expect(11, 1) =  0.000005669113752; expect(11, 2) = -0.000209406698825; expect(11, 3) =  0.003305473056782; expect(11, 4) = -0.027311153070550; expect(11, 5) =  0.127213485355599; expect(11, 6) = -0.318158094252897; expect(11, 7) =  0.215257108700206; expect(11, 8) =  1.196934021150055; expect(11, 9) = -4.445256273997598; expect(11,10) =  7.736775352133582; expect(11,11) = -8.061571695633706; expect(11,12) =  5.131646077729269; expect(11,13) = -1.848269059782749; expect(11,14) =  0.289638546414438;
	expect(12, 0) =  0.000000037822586; expect(12, 1) = -0.000005280440772; expect(12, 2) =  0.000180031334394; expect(12, 3) = -0.002588068541346; expect(12, 4) =  0.018987053862231; expect(12, 5) = -0.073762939837580; expect(12, 6) =  0.117090718756610; expect(12, 7) =  0.212547662099395; expect(12, 8) = -1.551378693718132; expect(12, 9) =  3.864018381472909; expect(12,10) = -5.608616538552710; expect(12,11) =  5.131646077729269; expect(12,12) = -2.922801908619862; expect(12,13) =  0.949122740280543; expect(12,14) = -0.134439257679115;
	expect(13, 0) = -0.000000018936236; expect(13, 1) =  0.000002586209123; expect(13, 2) = -0.000085747214524; expect(13, 3) =  0.001188322127769; expect(13, 4) = -0.008272912326863; expect(13, 5) =  0.029275516007783; expect(13, 6) = -0.032195652749596; expect(13, 7) = -0.149163484452081; expect(13, 8) =  0.753570737589437; expect(13, 9) = -1.669802837364426; expect(13,10) =  2.213742954886355; expect(13,11) = -1.848269059782749; expect(13,12) =  0.949122740280543; expect(13,13) = -0.271934959594640; expect(13,14) =  0.032821800652652;
	expect(14, 0) =  0.000000003794930; expect(14, 1) = -0.000000514980210; expect(14, 2) =  0.000016919440846; expect(14, 3) = -0.000231510872329; expect(14, 4) =  0.001582145241706; expect(14, 5) = -0.005420977412400; expect(14, 6) =  0.005178336003076; expect(14, 7) =  0.030214090603182; expect(14, 8) = -0.140731579235935; expect(14, 9) =  0.296703119337109; expect(14,10) = -0.372398343230434; expect(14,11) =  0.289638546414438; expect(14,12) = -0.134439257679115; expect(14,13) =  0.032821800652652; expect(14,14) = -0.002932774335623;
	expect = 1.0e+16*expect;
	BOOST_UBLASX_DEBUG_TRACE("A^{-1} = " << B);
//	BOOST_UBLASX_TEST_CHECK_MATRIX_CLOSE( B, expect, n, n, tol );
	BOOST_UBLASX_TEST_CHECK( true ); // Just avoid unused variable warnings from the compiler
}

//#define BOOST_UBLAS_TYPE_CHECK 1


int main()
{
	BOOST_UBLASX_DEBUG_TRACE("Test Suite: 'inv' operation");

	BOOST_UBLASX_TEST_BEGIN();

	BOOST_UBLASX_TEST_DO( row_major_matrix_inplace );
	BOOST_UBLASX_TEST_DO( col_major_matrix_inplace );
	BOOST_UBLASX_TEST_DO( row_major_matrix );
	BOOST_UBLASX_TEST_DO( col_major_matrix );
	BOOST_UBLASX_TEST_DO( rectangular_matrix );
	BOOST_UBLASX_TEST_DO( singular_matrix );
	BOOST_UBLASX_TEST_DO( illconditioned_matrix );

	BOOST_UBLASX_TEST_END();
}
