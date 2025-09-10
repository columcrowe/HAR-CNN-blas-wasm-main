# HAR CNN - A WebAssembly BLAS Demo

A WebAssembly BLAS Demo modified for human activity recognition (HAR) using IMU data.

LFortran used is https://github.com/lfortran/lfortran/tree/26c7a7976708f2d595a0ab64f0d531b43518f200.
<path-to-lfortran> -> /root/lfortran/inst/bin/
```console
% /root/lfortran/inst/bin/lfortran --version
```
export EMSDK_PATH=$HOME/emsdk
lfortran --show-asr .f90
<path-to-lfortran-runtime-library> -> /root/lfortran/inst/share/lfortran/lib/

Steps to generate `mnist.js` and `mnist.wasm`:
/root/lfortran/inst/bin/lfortran /root/lfortran/examples/expr2.f90 --target=wasm32-unknown-emscripten
```console
lfortran -c classifier.f90 --generate-object-code --rtlib --target=wasm32-unknown-emscripten
emcc --target=wasm32-unknown-emscripten -sSTACK_SIZE=50mb -sINITIAL_MEMORY=4095mb -sWASM_BIGINT=1 -sALLOW_MEMORY_GROWTH=1 -o www/mnist.js classifier.o <path-to-lfortran-runtime-library>/lfortran_runtime_wasm_emcc.o --no-entry -sEXPORTED_FUNCTIONS=['_classifier','_malloc','_free'] -sEXPORTED_RUNTIME_METHODS=['HEAPF64']
```
