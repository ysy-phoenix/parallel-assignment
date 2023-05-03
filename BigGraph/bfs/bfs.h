#ifndef BFS_H_
#define BFS_H_

//#define DEBUG

#include "../common/graph.h"

class solution {
public:
    int *distances;

    solution () : distances(nullptr) {};

    explicit solution(int numNodes) {
        this->distances = new int[numNodes];
    }

    ~solution() {
        delete[] this->distances;
    }
};

class vertex_set {
public:
    // # of vertices in the set
    int count{};
    // max size of buffer vertices
    int max_vertices;
    // array of vertex ids in set
    int *vertices;

    // default constructor
    vertex_set() : count(0), max_vertices(0), vertices(nullptr) {};

    inline void clear() {
        this->count = 0;
    }

    explicit vertex_set(int count) {
        this->max_vertices = count;
        this->vertices = new int[this->max_vertices];
        this->clear();
    }

    ~vertex_set() {
        delete[] this->vertices;
    }
};


void bfs_top_down(Graph &graph, solution &sol);

void bfs_bottom_up(Graph &graph, solution &sol);

void bfs_hybrid(Graph &graph, solution &sol);

#endif
