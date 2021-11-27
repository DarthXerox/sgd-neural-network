TARGETS = main
HEADERS = WeightLayer.h
BIN = main
CXXFLAGS = -fopenmp -Wall -Wextra  -std=c++17
CC= g++
OBJS = WeightLayer.o

all: $(TARGETS)

main: main.o WeightLayer.o
	$(CC) $(CXXFLAGS) -o main main.o WeightLayer.o
#main: main.o $(OBJS)
#	$(CC) $(CXXFLAGS) $^ -o $@

main.o: main.cpp $(HEADERS)
	$(CC) $(CXXFLAGS) -o $@ -c $<

%.o: %.cpp %.h
	$(CC) $(CXXFLAGS) -c $< -o $@


clean:
	-$(RM) $(OBJS) main.o

distclean: clean
	-$(RM) $(OBJS) $(TARGETS)

.PHONY: clean distclean
