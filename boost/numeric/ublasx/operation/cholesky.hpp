/* vim: set tabstop=4 expandtab shiftwidth=4 softtabstop=4: */

/** 
 * \file cholesky.hpp
 *
 * \brief Cholesky decomposition
 * -   begin                : 2005-08-24
 * -   copyright            : (C) 2005 by Gunter Winkler, Konstantin Kutzkow
 * -   email                : guwi17@gmx.de
 *
 *  This library is free software; you can redistribute it and/or
 *  modify it under the terms of the GNU Lesser General Public
 *  License as published by the Free Software Foundation; either
 *  version 2.1 of the License, or (at your option) any later version.

 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *  Lesser General Public License for more details.

 *  You should have received a copy of the GNU Lesser General Public
 *  License along with this library; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 */


#ifndef BOOST_NUMERIC_UBLASX_OPERATION_CHOLESKY_HPP
#define BOOST_NUMERIC_UBLASX_OPERATION_CHOLESKY_HPP


#include <cassert>

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>

#include <boost/numeric/ublas/vector_expression.hpp>
#include <boost/numeric/ublas/matrix_expression.hpp>

#include <boost/numeric/ublas/triangular.hpp>

namespace boost { namespace numeric { namespace ublasx {

/** \brief decompose the symmetric positive definit matrix A into product L L^T.
 *
 * \param MATRIX type of input matrix 
 * \param TRIA type of lower triangular output matrix
 * \param A square symmetric positive definite input matrix (only the lower triangle is accessed)
 * \param L lower triangular output matrix 
 * \return nonzero if decompositon fails (the value ist 1 + the numer of the failing row)
 */
template < class MATRIX, class TRIA >
size_t cholesky_decompose(const MATRIX& A, TRIA& L)
{
  namespace ublas = ::boost::numeric::ublas;

  typedef typename MATRIX::value_type T;
  
  assert( A.size1() == A.size2() );
  assert( A.size1() == L.size1() );
  assert( A.size2() == L.size2() );

  const size_t n = A.size1();
  
  for (size_t k=0 ; k < n; k++) {
        
    double qL_kk = A(k,k) - ublas::inner_prod( ublas::project( ublas::row(L, k), ublas::range(0, k) ), ublas::project( ublas::row(L, k), ublas::range(0, k) ) );
    
    if (qL_kk <= 0) {
      return 1 + k;
    } else {
      double L_kk = ::std::sqrt( qL_kk );
      L(k,k) = L_kk;
      
      ublas::matrix_column<TRIA> cLk(L, k);
      ublas::project( cLk, ublas::range(k+1, n) )
        = ( ublas::project( ublas::column(A, k), ublas::range(k+1, n) )
            - ublas::prod( ublas::project(L, ublas::range(k+1, n), ublas::range(0, k)), 
                    ublas::project(ublas::row(L, k), ublas::range(0, k) ) ) ) / L_kk;
    }
  }
  return 0;      
}


/** \brief decompose the symmetric positive definit matrix A into product L L^T.
 *
 * \param MATRIX type of matrix A
 * \param A input: square symmetric positive definite matrix (only the lower triangle is accessed)
 * \param A output: the lower triangle of A is replaced by the cholesky factor
 * \return nonzero if decompositon fails (the value ist 1 + the numer of the failing row)
 */
template < class MATRIX >
size_t cholesky_decompose(MATRIX& A)
{
  namespace ublas = ::boost::numeric::ublas;

  typedef typename MATRIX::value_type T;
  
  const MATRIX& A_c(A);

  const size_t n = A.size1();
  
  for (size_t k=0 ; k < n; k++) {
        
    double qL_kk = A_c(k,k) - ublas::inner_prod( ublas::project( row(A_c, k), ublas::range(0, k) ),
                                          ublas::project( ublas::row(A_c, k), ublas::range(0, k) ) );
    
    if (qL_kk <= 0) {
      return 1 + k;
    } else {
      double L_kk = ::std::sqrt( qL_kk );
      
      ublas::matrix_column<MATRIX> cLk(A, k);
      ublas::project( cLk, ublas::range(k+1, n) )
        = ( ublas::project( ublas::column(A_c, k), ublas::range(k+1, n) )
            - ublas::prod( ublas::project(A_c, ublas::range(k+1, n), ublas::range(0, k)), 
                    ublas::project(ublas::row(A_c, k), ublas::range(0, k) ) ) ) / L_kk;
      A(k,k) = L_kk;
    }
  }
  return 0;      
}

#if 0
  using namespace ublas;

    // Operations:
    //  n * (n - 1) / 2 + n = n * (n + 1) / 2 multiplications,
    //  n * (n - 1) / 2 additions

    // Dense (proxy) case
    template<class E1, class E2>
    void inplace_solve (const matrix_expression<E1> &e1, vector_expression<E2> &e2,
                        lower_tag, column_major_tag) {
      std::cout << " is_lc ";
        typedef typename E2::size_type size_type;
        typedef typename E2::difference_type difference_type;
        typedef typename E2::value_type value_type;

        BOOST_UBLAS_CHECK (e1 ().size1 () == e1 ().size2 (), bad_size ());
        BOOST_UBLAS_CHECK (e1 ().size2 () == e2 ().size (), bad_size ());
        size_type size = e2 ().size ();
        for (size_type n = 0; n < size; ++ n) {
#ifndef BOOST_UBLAS_SINGULAR_CHECK
            BOOST_UBLAS_CHECK (e1 () (n, n) != value_type/*zero*/(), singular ());
#else
            if (e1 () (n, n) == value_type/*zero*/())
                singular ().raise ();
#endif
            value_type t = e2 () (n) / e1 () (n, n);
            e2 () (n) = t;
            if (t != value_type/*zero*/()) {
              project( e2 (), range(n+1, size) )
                .minus_assign( t * project( column( e1 (), n), range(n+1, size) ) );
            }
        }
    }
#endif





/** \brief decompose the symmetric positive definit matrix A into product L L^T.
 *
 * \param MATRIX type of matrix A
 * \param A input: square symmetric positive definite matrix (only the lower triangle is accessed)
 * \param A output: the lower triangle of A is replaced by the cholesky factor
 * \return nonzero if decompositon fails (the value ist 1 + the numer of the failing row)
 */
template < class MATRIX >
size_t incomplete_cholesky_decompose(MATRIX& A)
{
  namespace ublas = ::boost::numeric::ublas;

  typedef typename MATRIX::value_type T;
  
  // read access to a const matrix is faster
  const MATRIX& A_c(A);

  const size_t n = A.size1();
  
  for (size_t k=0 ; k < n; k++) {
    
    double qL_kk = A_c(k,k) - ublas::inner_prod( ublas::project( ublas::row( A_c, k ), ublas::range(0, k) ), ublas::project( ublas::row( A_c, k ), ublas::range(0, k) ) );
    
    if (qL_kk <= 0) {
      return 1 + k;
    } else {
      double L_kk = ::std::sqrt( qL_kk );

      // aktualisieren
      for (size_t i = k+1; i < A.size1(); ++i) {
        T* Aik = A.find_element(i, k);

        if (Aik != 0) {
          *Aik = ( *Aik - ublas::inner_prod( ublas::project( ublas::row( A_c, k ), range(0, k) ), ublas::project( ublas::row( A_c, i ), ublas::range(0, k) ) ) ) / L_kk;
        }
      }
        
      A(k,k) = L_kk;
    }
  }
        
  return 0;
}




/** \brief solve system L L^T x = b inplace
 *
 * \param L a triangular matrix
 * \param x input: right hand side b; output: solution x
 */
template < class TRIA, class VEC >
void
cholesky_solve(const TRIA& L, VEC& x, ublas::lower)
{
  namespace ublas = ::boost::numeric::ublas;
//   ::inplace_solve(L, x, lower_tag(), typename TRIA::orientation_category () );
  ublas::inplace_solve(L, x, lower_tag() );
  ublas::inplace_solve(trans(L), x, upper_tag());
}


}}} // Namespace boost::numeric::ublax


#endif // BOOST_NUMERIC_UBLASX_OPERATION_CHOLESKY_HPP
