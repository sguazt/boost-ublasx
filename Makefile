src_path=.
apidoc_path=libs/numeric/ublasx/doc/api
test_path=libs/numeric/ublasx/test
apidoc_path=libs/numeric/ublasx/doc/api

USER_CXXFLAGS=
USER_LDFLAGS=

-include ./user-config.mk

CXXFLAGS=-Wall -Wextra -pedantic -ansi -I$(src_path) $(USER_CXXFLAGS) -g -O0
LDFLAGS+=$(USER_LDFLAGS) -lm -llapack -lblas

CLEANER=rm -rf
DOXYGEN=doxygen


.PHONY: all apidoc clean msg ublasx


all: ublasx


ublasx:	msg \
		$(test_path)/abs \
		$(test_path)/all \
		$(test_path)/any \
		$(test_path)/arithmetic_ops \
		$(test_path)/balance \
		$(test_path)/begin_end \
		$(test_path)/cat \
		$(test_path)/cond \
		$(test_path)/cumsum \
		$(test_path)/diag \
		$(test_path)/dot \
		$(test_path)/eigen \
		$(test_path)/empty \
		$(test_path)/eps \
		$(test_path)/find \
		$(test_path)/for_each \
		$(test_path)/generalized_diagonal_matrix \
		$(test_path)/hold \
		$(test_path)/inv \
		$(test_path)/isinf \
		$(test_path)/layout_type \
		$(test_path)/log2 \
		$(test_path)/lsq \
		$(test_path)/lu \
		$(test_path)/matrix_diagonal_proxy \
		$(test_path)/max \
		$(test_path)/min \
		$(test_path)/mldivide \
		$(test_path)/num_columns \
		$(test_path)/num_rows \
		$(test_path)/pow2 \
		$(test_path)/ql \
		$(test_path)/qr \
		$(test_path)/qz \
		$(test_path)/rank \
		$(test_path)/rcond \
		$(test_path)/relational_ops \
		$(test_path)/rep \
		$(test_path)/reshape \
		$(test_path)/rot90 \
		$(test_path)/round \
		$(test_path)/seq \
		$(test_path)/sequence_vector \
		$(test_path)/size \
		$(test_path)/sqr \
		$(test_path)/sqrt \
		$(test_path)/sum \
		$(test_path)/svd \
		$(test_path)/test_utils \
		$(test_path)/trace \
		$(test_path)/transform \
		$(test_path)/tril \
		$(test_path)/triu \
		$(test_path)/which

msg:
	@echo "=== Building binary targets ==="


apidoc:
	@echo "=== Building API doc ==="
	@mkdir -p $(apidoc_path)
	@$(DOXYGEN) Doxyfile



clean: apidoc-clean build-clean


apidoc-clean:
	@echo "=== Cleaning doc files ==="
	@$(CLEANER) $(apidoc_path)


build-clean:
	@echo "=== Cleaning build files ==="
	@$(CLEANER)	$(test_path)/abs $(test_path)/abs.o \
				$(test_path)/all $(test_path)/all.o \
				$(test_path)/any $(test_path)/any.o \
				$(test_path)/balance $(test_path)/balance.o \
				$(test_path)/begin_end $(test_path)/begin_end.o \
				$(test_path)/cat $(test_path)/cat.o \
				$(test_path)/cond $(test_path)/cond.o \
				$(test_path)/cumsum $(test_path)/cumsum.o \
				$(test_path)/diag $(test_path)/diag.o \
				$(test_path)/dot $(test_path)/dot.o \
				$(test_path)/eigen $(test_path)/eigen.o \
				$(test_path)/empty $(test_path)/empty.o \
				$(test_path)/eps $(test_path)/eps.o \
				$(test_path)/find $(test_path)/find.o \
				$(test_path)/for_each $(test_path)/for_each.o \
				$(test_path)/generalized_diagonal_matrix $(test_path)/generalized_diagonal_matrix.o \
				$(test_path)/hold $(test_path)/hold.o \
				$(test_path)/inv $(test_path)/inv.o \
				$(test_path)/isinf $(test_path)/isinf.o \
				$(test_path)/layout_type $(test_path)/layout_type.o \
				$(test_path)/log2 $(test_path)/log2.o \
				$(test_path)/lsq $(test_path)/lsq.o \
				$(test_path)/lu $(test_path)/lu.o \
				$(test_path)/matrix_diagonal_proxy $(test_path)/matrix_diagonal_proxy.o \
				$(test_path)/max $(test_path)/max.o \
				$(test_path)/min $(test_path)/min.o \
				$(test_path)/mldivide $(test_path)/mldivide.o \
				$(test_path)/num_columns $(test_path)/num_rows.o \
				$(test_path)/num_rows $(test_path)/num_columns.o \
				$(test_path)/pow2 $(test_path)/pow2.o \
				$(test_path)/ql $(test_path)/ql.o \
				$(test_path)/qr $(test_path)/qr.o \
				$(test_path)/qz $(test_path)/qz.o \
				$(test_path)/rank $(test_path)/rank.o \
				$(test_path)/rcond $(test_path)/rcond.o \
				$(test_path)/relational_ops $(test_path)/relational_ops.o \
				$(test_path)/rep $(test_path)/rep.o \
				$(test_path)/reshape $(test_path)/reshape.o \
				$(test_path)/rot90 $(test_path)/rot90.o \
				$(test_path)/round $(test_path)/round.o \
				$(test_path)/seq $(test_path)/seq.o \
				$(test_path)/sequence_vector $(test_path)/sequence_vector.o \
				$(test_path)/size $(test_path)/size.o \
				$(test_path)/sqr $(test_path)/sqr.o \
				$(test_path)/sqrt $(test_path)/sqrt.o \
				$(test_path)/sum $(test_path)/sum.o \
				$(test_path)/svd $(test_path)/svd.o \
				$(test_path)/test_utils $(test_path)/test_utils.o \
				$(test_path)/trace $(test_path)/trace.o \
				$(test_path)/transform $(test_path)/transform.o \
				$(test_path)/tril $(test_path)/tril.o \
				$(test_path)/triu $(test_path)/triu.o \
				$(test_path)/which $(test_path)/which.o \
				$(apidoc_path)

