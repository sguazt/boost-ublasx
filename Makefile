src_path=.
apidoc_path=libs/numeric/ublasx/doc/api
test_path=libs/numeric/ublasx/test
apidoc_path=libs/numeric/ublasx/doc/api

USER_CXXFLAGS=
USER_LDFLAGS=

ifneq (,$(wildcard ./user-config.mk))
-include ./user-config.mk
endif

CXXFLAGS += -Wall -Wextra -pedantic -ansi
CXXFLAGS += -I$(src_path)
CXXFLAGS += $(USER_CXXFLAGS)
#CXXFLAGS += -g -O0
LDFLAGS += $(USER_LDFLAGS)
LDLIBS += -lm
ifneq (,$(USER_LDLIBS))
LDLIBS += $(USER_LDLIBS)
else
LDLIBS += -llapack -lblas
endif

DOXYGEN = doxygen

test_cases =	abs \
				all \
				any \
				arithmetic_ops \
				balance \
				begin_end \
				cat \
				cond \
				cumsum \
				diag \
				dot \
				eigen \
				element_pow \
				empty \
				eps \
				exp \
				find \
				for_each \
				generalized_diagonal_matrix \
				hold \
				inv \
				isinf \
				isfinite \
				layout_type \
				linspace \
				log \
				logspace \
				log2 \
				log10 \
				lsq \
				lu \
				matrix_diagonal_proxy \
				max \
				min \
				mldivide \
				num_columns \
				num_rows \
				pow \
				pow2 \
				ql \
				qr \
				qz \
				rank \
				rcond \
				relational_ops \
				rep \
				reshape \
				rot90 \
				round \
				seq \
				sequence_vector \
				sign \
				size \
				sqr \
				sqrt \
				sum \
				svd \
				tanh \
				test_utils \
				trace \
				transform \
				tril \
				triu \
				which

targets = $(addprefix $(test_path)/, $(test_cases))
objects = $(addsuffix .o, $(targets))


.PHONY: all apidoc clean msg ublasx


all: ublasx


ublasx:	msg $(targets)

msg:
	@echo "=== Building binary targets ==="


apidoc:
	@echo "=== Building API doc ==="
	@mkdir -p $(apidoc_path)
	@$(DOXYGEN) Doxyfile



clean: apidoc-clean build-clean


apidoc-clean:
	@echo "=== Cleaning doc files ==="
	@$(RM) $(apidoc_path)


build-clean:
	@echo "=== Cleaning build files ==="
	@$(RM) $(targets)
	@$(RM) $(objects)
