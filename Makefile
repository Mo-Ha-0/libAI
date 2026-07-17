CXX = g++
CXXFLAGS = -std=c++17 -O2 -march=native -ffast-math -fopenmp -Iinclude
LDFLAGS = -fopenmp -lm

SRC_DIR = src
OBJ_DIR = build/obj
EXAMPLES_DIR = examples
LIB_DIR = build

SOURCES = $(wildcard $(SRC_DIR)/layers/*.cpp) $(wildcard $(SRC_DIR)/optimizers/*.cpp) $(SRC_DIR)/network.cpp $(SRC_DIR)/trainer.cpp $(SRC_DIR)/tuning.cpp
OBJECTS = $(SOURCES:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)
LIBRARY = $(LIB_DIR)/liblibAI.a

.PHONY: all examples clean

all: $(LIBRARY) examples

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(LIBRARY): $(OBJECTS)
	@mkdir -p $(LIB_DIR)
	ar rcs $@ $^

examples: $(LIBRARY)
	$(CXX) $(CXXFLAGS) $(EXAMPLES_DIR)/simple_example.cpp -L$(LIB_DIR) -llibAI $(LDFLAGS) -o $(EXAMPLES_DIR)/simple_example
	$(CXX) $(CXXFLAGS) $(EXAMPLES_DIR)/iris_example.cpp -L$(LIB_DIR) -llibAI $(LDFLAGS) -o $(EXAMPLES_DIR)/iris_example

run_simple: examples
	./$(EXAMPLES_DIR)/simple_example

run_iris: examples
	./$(EXAMPLES_DIR)/iris_example

clean:
	rm -rf $(OBJ_DIR) $(LIB_DIR) $(EXAMPLES_DIR)/simple_example $(EXAMPLES_DIR)/iris_example
