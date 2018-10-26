CXX=nvcc
CXXFLAGS=-O3 -I$(CUDA_INSTALL_PATH)/include --std=c++14 --generate-code arch=compute_60,code=sm_60 --expt-relaxed-constexpr
LDFLAGS=-L$(CUDA_INSTALL_PATH)/lib64 -lcudart

PROCESS=functor_test

$(PROCESS): main.cu
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $(PROCESS) main.cu
clean:
	rm $(PROCESS)
