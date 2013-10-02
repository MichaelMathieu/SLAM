CXX=g++
LIBS=`pkg-config --libs opencv`
#FLAGS=-W -fpic -O3 -funroll-loops -fopenmp
FLAGS=-W -fpic -g
SRCS=kalman.cpp mongoose.cpp visualize.cpp matching.cpp new_point.cpp cone.cpp lineFeature.cpp feature.cpp quaternion.cpp random.cpp
MAIN=slam.cpp
SRCSTEST=tests.cpp test_matching.cpp test_project.cpp
OBJS=$(SRCS:.cpp=.o)
MAINOBJ=$(MAIN:.cpp=.o)
OBJSTEST=$(SRCSTEST:.cpp=.o)
OBJSTESTPREFFIX=$(addprefix ${TEST_FOLDER}/,${OBJSTEST})
TEST_FOLDER=tests

ARCH=$(shell bash -c 'if [[ `uname -a` =~ "x86" ]]; then echo "__x86__"; else echo "__ARM__"; fi')
ifeq (${ARCH}, __ARM__)
FLAGS+= -mfpu=neon
endif

random.o : FLAGS += -std=c++0x

%.o : %.cpp
	$(CXX) -M $(FLAGS) -o $*.P $<
	@cp $*.P $*.d; \
		sed -e 's/#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$//' \
			-e '/^$$/ d' -e 's/$$/ :/' < $*.P >> $*.d; \
		rm -f $*.P
	$(CXX) $(FLAGS) -c $< -o $@

-include *.d

slam: ${OBJS} ${MAINOBJ}
	${CXX} ${FLAGS} ${OBJS} ${MAINOBJ} ${LIBS} -o slamtest

test0: ${OBJSTESTPREFFIX}

test: ${OBJS}
	rm -f ${TEST_FOLDER}/*.o
	make test0
	${CXX} ${FLAGS} ${OBJS} ${OBJSTESTPREFFIX} ${LIBS} -o ${TEST_FOLDER}/unit_tests
	${TEST_FOLDER}/unit_tests

clean:
	rm -rf *.o *.d simul kalmantest slamtest