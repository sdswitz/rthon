CC = gcc
CFLAGS = -Wall -Wextra -std=c99 -pedantic -O2 -g
LDFLAGS = -lm

SOURCES = matrix.c qr_decomposition.c linear_regression.c
OBJECTS = $(SOURCES:.c=.o)
HEADERS = matrix.h qr_decomposition.h linear_regression.h

TARGET = test_regression
TEST_SOURCE = test_regression.c

.PHONY: all clean test

all: $(TARGET)

$(TARGET): $(OBJECTS) $(TEST_SOURCE)
	$(CC) $(CFLAGS) $(OBJECTS) $(TEST_SOURCE) -o $(TARGET) $(LDFLAGS)

%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

test: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(OBJECTS) $(TARGET)

debug: CFLAGS += -DDEBUG -g
debug: $(TARGET)

valgrind: $(TARGET)
	valgrind --leak-check=full --show-leak-kinds=all ./$(TARGET)

lint:
	cppcheck --enable=all --std=c99 $(SOURCES) $(TEST_SOURCE)

install-deps:
	@echo "Installing development dependencies..."
	@if command -v apt-get >/dev/null 2>&1; then \
		sudo apt-get update && sudo apt-get install -y build-essential valgrind cppcheck; \
	elif command -v yum >/dev/null 2>&1; then \
		sudo yum install -y gcc make valgrind cppcheck; \
	elif command -v brew >/dev/null 2>&1; then \
		brew install valgrind cppcheck; \
	else \
		echo "Please install gcc, make, valgrind, and cppcheck manually"; \
	fi