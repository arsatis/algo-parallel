CU_APPS=hello hello_multithreaded
CU_UTIL=utils

all: $(CU_APPS)

%: %.cu
	nvcc -O2 -arch=sm_80 -o $@ $< ${CU_UTIL}.cu -lcudadevrt --relocatable-device-code true
clean:
	del $(CU_APPS:=.exe) $(CU_APPS:=.exp) $(CU_APPS:=.lib)