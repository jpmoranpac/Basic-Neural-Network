SRCS=$(wildcard *.cpp) $(wildcard */*.cpp)
OBJS=$(subst .cpp,.o,$(SRCS))

neural_network.out: $(OBJS)
	g++ -g -o neural_network.out $(OBJS)

clean:
	@rm -f $(OBJS)
	@rm -f neural_network.out