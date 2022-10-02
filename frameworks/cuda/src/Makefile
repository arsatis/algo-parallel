CU_APPS=hello hello_multithreaded
CU_UTIL=utils

all: ${CU_APPS}

%: %.cu
	nvcc -O2 -arch=sm_80 -o $@ $< ${CU_UTIL}.cu -lcudadevrt --relocatable-device-code true
clean:
	rm -f ${CU_APPS}