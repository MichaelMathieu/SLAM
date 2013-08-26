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


slam: slam.cpp slam.o kalman.o kalman.cpp mongoose.o mongoose.cpp visualize.o visualize.cpp
	${CXX} ${FLAGS} slam.o kalman.o mongoose.o visualize.o ${LIBS} -o slamtest

kalman: kalman.cpp kalman.o
	${CXX} ${FLAGS} kalman.o ${LIBS} -o kalmantest

sim: simulation.cpp kalman.cpp simulation.o kalman.o
	${CXX} ${FLAGS} kalman.o simulation.o ${LIBS} -o simul

clean:
	rm -rf *.o simul kalmantest slamtest