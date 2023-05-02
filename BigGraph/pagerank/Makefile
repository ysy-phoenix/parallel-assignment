all: default

default: page_rank.cpp main.cpp
	g++ -I../ -std=c++11 -fopenmp -g -O3 -o pr main.cpp page_rank.cpp ../common/graph.cpp
#grade: page_rank.cpp grade.cpp
#	g++ -I../ -std=c++11 -fopenmp -g -O3 -o pr_grader grade.cpp page_rank.cpp ../common/graph.cpp
clean:
#	rm -rf pr pr_grader *~ *.*~
