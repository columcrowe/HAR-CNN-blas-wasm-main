EMSDK_PATH := $(HOME)/emsdk
LFORT_PATH := /root/lfortran/inst/bin
LFORT_RUNTIME := /root/lfortran/inst/share/lfortran/lib/lfortran_runtime_wasm_emcc.o

SRC := classifier.f90
OBJ := classifier.o
JS_OUT := www/mnist.js

EMCC_FLAGS := \
    --target=wasm32-unknown-emscripten \
    -sSTACK_SIZE=50mb \
    -sINITIAL_MEMORY=4095mb \
    -sWASM_BIGINT=1 \
    -sALLOW_MEMORY_GROWTH=1 \
    --no-entry \
    -sEXPORTED_FUNCTIONS=['_classifier','_malloc','_free'] \
    -sEXPORTED_RUNTIME_METHODS=['HEAPF64']

all: $(JS_OUT)

$(OBJ): $(SRC)
	$(LFORT_PATH)/lfortran -c $< --generate-object-code --rtlib --target=wasm32-unknown-emscripten

$(JS_OUT): $(OBJ)
	emcc $(OBJ) $(LFORT_RUNTIME) $(EMCC_FLAGS) -o $@

clean:
	rm -f $(OBJ) $(JS_OUT)

.PHONY: all clean install-hooks

install-hooks:
	@echo "Installing Git hooks..."
	@cp pre-commit .git/hooks/pre-commit
	@chmod +x .git/hooks/pre-commit
	@sed -i 's/\r$$//' .git/hooks/pre-commit

