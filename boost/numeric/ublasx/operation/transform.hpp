/* vim: set tabstop=4 expandtab shiftwidth=4 softtabstop=4: */

/**
 * \file boost/numeric/ublasx/operation/transform.hpp
 *
 * The \c transform operation.
 *
 * Copyright (c) 2010, Marco Guazzone
 *
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */

#ifndef BOOST_NUMERIC_UBLASX_OPERATION_TRANSFORM_HPP
#define BOOST_NUMERIC_UBLASX_OPERATION_TRANSFORM_HPP


#include <boost/numeric/ublas/expression_types.hpp>
//#include <boost/numeric/ublas/fwd.hpp>
//#include <boost/numeric/ublas/functional.hpp>
#include <boost/numeric/ublas/traits.hpp>
#include <boost/numeric/ublasx/expression/matrix_unary_functor.hpp>
#include <boost/numeric/ublasx/expression/vector_unary_functor.hpp>
//#include <boost/shared_ptr.hpp>//FIXME
//#include <boost/function.hpp>//FIXME


namespace boost { namespace numeric { namespace ublasx {

using namespace ::boost::numeric::ublas;


//namespace detail {
//
//template <typename T, typename F>
//struct scalar_unary_generic: public scalar_unary_functor<T>
//{
//  typedef typename scalar_unary_functor<T>::value_type value_type;
//  typedef typename scalar_unary_functor<T>::argument_type argument_type;
//  typedef typename scalar_unary_functor<T>::result_type result_type;
//  //typedef typename F::result_type result_type;
//
//  explicit scalar_unary_generic(F const& f)
//  : f_(f)
//  {
//  }
//
//
//  BOOST_UBLAS_INLINE
//  result_type operator()(argument_type t)
//  {
//std::cerr << "In scalar_unary_generic: " << t << std::endl;
//      return f_(t);
//  }
//
//  boost::function<result_type (argument_type)> f_;
//};
//
//} // Namespace detail


/**
 * \brief Apply a function to each element of a given vector expression.
 *
 * \tparam VectorExprT The type of the input vector expression.
 * \tparam UnaryFunctorT The type of the unary functor.
 *
 * \param ve The input vector expression.
 * \param f The unary functor to be applied to each vector element.
 * \return A vector expression representing the application of \a f to each
 *  element of \a ve.
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */
template <typename VectorExprT, typename UnaryFunctorT>
BOOST_UBLAS_INLINE
typename vector_unary_functor_traits<
    VectorExprT,
    typename UnaryFunctorT::result_type (typename UnaryFunctorT::argument_type)
>::result_type transform(vector_expression<VectorExprT> const& ve, UnaryFunctorT const& f)
{
    typedef typename vector_unary_functor_traits<
                VectorExprT,
                typename UnaryFunctorT::result_type (typename UnaryFunctorT::argument_type)
            >::expression_type expression_type;

    return expression_type(ve(), f);
}


/**
 * \brief Apply a function to each element of a given matrix expression.
 *
 * \tparam MatrixExprT The type of the input matrix expression.
 * \tparam UnaryFunctorT The type of the unary functor.
 *
 * \param me The input matrix expression.
 * \param f The unary functor to be applied to each matrix element.
 * \return A matrix expression representing the application of \a f to each
 *  element of \a me.
 *
 * \author Marco Guazzone, marco.guazzone@gmail.com
 */
template <typename MatrixExprT, typename UnaryFunctorT>
BOOST_UBLAS_INLINE
typename matrix_unary_functor_traits<
    MatrixExprT,
//  UnaryFunctorT
    typename UnaryFunctorT::result_type (typename UnaryFunctorT::argument_type)
//  detail::scalar_unary_generic<
//      typename matrix_traits<MatrixExprT>::value_type,
//      UnaryFunctorT
//  >
>::result_type transform(matrix_expression<MatrixExprT> const& me, UnaryFunctorT const& f)
{
//  typedef detail::scalar_unary_generic<
//          typename matrix_traits<MatrixExprT>::value_type,
//          UnaryFunctorT
//      > wrapper_functor_type;

    typedef typename matrix_unary_functor_traits<
            MatrixExprT,
//          wrapper_functor_type
            typename UnaryFunctorT::result_type (typename UnaryFunctorT::argument_type)
        >::expression_type expression_type;

    //return expression_type(me(), wrapper_functor_type(f));
    return expression_type(me(), f);
}

}}} // Namespace boost::numeric::ublasx


#endif // BOOST_NUMERIC_UBLASX_OPERATION_TRANSFORM_HPP
