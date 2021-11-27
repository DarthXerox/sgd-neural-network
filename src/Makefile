TARGETS = main
HEADERS = WeightLayer.h
BIN = main
CXXFLAGS  = -Wall -Werror -Wextra -pedantic -ansi -std=c++14
CC= g++

all: $(TARGETS)

main.o: $(HEADERS)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

%.o: %.c %.h
	$(CC) $(CXXFLAGS) $(CFLAGS) -c $< -o $@


clean:
	rm -f *.o
	rm $(BIN)