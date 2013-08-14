CXX=g++
LIBS=`pkg-config --libs opencv`
FLAGS=-W -fpic -O3 -funroll-loops -fopenmp
FLAGS=-W -fpic  -g
ARCH=$(shell bash -c 'if [[ `uname -a` =~ "x86" ]]; then echo "__x86__"; else echo "__ARM__"; fi')
ifeq (${ARCH}, __ARM__)
FLAGS+= -mfpu=neon
endif

all: match

.cpp.o:
	${CXX} -D${ARCH} ${FLAGS} -c $< -o $@

match: matching.cpp matching.o depthMap.cpp depthMap.o
	${CXX} ${FLAGS} depthMap.o matching.o ${LIBS} -shared -o libmatching.so

slam: slam.cpp slam.o kalman.o kalman.cpp
	${CXX} ${FLAGS} slam.o kalman.o ${LIBS} -o slamtest

kalman: kalman.cpp kalman.o
	${CXX} ${FLAGS} kalman.o ${LIBS} -o kalmantest

sim: simulation.cpp kalman.cpp simulation.o kalman.o
	${CXX} ${FLAGS} kalman.o simulation.o ${LIBS} -o simul

clean:
	rm -rf libmatching.so matching.o depthMap.o