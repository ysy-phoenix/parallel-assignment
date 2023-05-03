#ifndef GRAPH_H_
#define GRAPH_H_

#include <string>
#include <vector>

using Vertex = int;

class Graph {
public:
    // default constructor
    Graph() : num_edges(0), num_nodes(0) {};

    // de constructor
    ~Graph() {
        delete[] this->incoming_starts;
        delete[] this->incoming_edges;
        delete[] this->outgoing_starts;
        delete[] this->outgoing_edges;
    };

    /* Getters */
    inline int get_num_nodes() const;

    inline int get_num_edges() const;

    inline Vertex* outgoing_begin(Vertex) const;

    inline Vertex* outgoing_end(Vertex) const;

    inline int outgoing_size(Vertex) const;

    inline Vertex* incoming_begin(Vertex) const;

    inline Vertex* incoming_end(Vertex) const;

    inline int incoming_size(Vertex) const;


    /* IO */
    void read_graph_file(std::ifstream &);

    void load_graph(const std::string &);

    void load_graph_binary(const std::string &);

    void store_graph_binary(const std::string &);

    friend std::ostream &operator<<(std::ostream &os, const Graph &graph);

    // member function
    void build_incoming();

    void get_meta_data(std::ifstream &);


private:
    // Number of edges in the graph
    int num_edges;
    // Number of vertices in the graph
    int num_nodes;

    // The node reached by first outgoing edge of i is given by
    // outgoing_edges[outgoing_starts[i]].  To iterate over all
    // outgoing edges, please see the top-down bfs implementation.
    int *outgoing_starts{};
    Vertex *outgoing_edges{};

    int *incoming_starts{};
    Vertex *incoming_edges{};
};


/* Included here to enable inlining. Don't look. */
#include "graph_internal.h"

#endif
