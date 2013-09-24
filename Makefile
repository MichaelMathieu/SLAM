CXX=g++
LIBS=`pkg-config --libs opencv`
FLAGS=-W -fpic -O3 -funroll-loops -fopenmp
FLAGS=-W -fpic  -g
SRCS=kalman.cpp mongoose.cpp visualize.cpp matching.cpp new_point.cpp cone.cpp slam.cpp
HEADERS=slam.hpp
OBJS = $(SRCS:.cpp=.o)
ARCH=$(shell bash -c 'if [[ `uname -a` =~ "x86" ]]; then echo "__x86__"; else echo "__ARM__"; fi')
ifeq (${ARCH}, __ARM__)
FLAGS+= -mfpu=neon
endif

%.o : %.cpp
	$(CXX) -M $(FLAGS) -o $*.P $<
	@cp $*.P $*.d; \
		sed -e 's/#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$//' \
			-e '/^$$/ d' -e 's/$$/ :/' < $*.P >> $*.d; \
		rm -f $*.P
	$(CXX) $(FLAGS) -c $< -o $@

-include *.d

slam: ${OBJS}
	${CXX} ${FLAGS} ${OBJS} ${LIBS} -o slamtest

kalman: kalman.cpp kalman.o
	${CXX} ${FLAGS} kalman.o ${LIBS} -o kalmantest

sim: simulation.cpp kalman.cpp simulation.o kalman.o
	${CXX} ${FLAGS} kalman.o simulation.o ${LIBS} -o simul

clean:
	rm -rf *.o *.d simul kalmantest slamtest