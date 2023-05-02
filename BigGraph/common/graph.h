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
    ~Graph() = default;

    /* Getters */
    inline int get_num_nodes() const;

    inline int get_num_edges() const;

    inline std::vector<Vertex>::const_iterator outgoing_begin(Vertex);

    inline std::vector<Vertex>::const_iterator outgoing_end(Vertex);

    inline int outgoing_size(Vertex);

    inline std::vector<Vertex>::const_iterator incoming_begin(Vertex);

    inline std::vector<Vertex>::const_iterator incoming_end(Vertex);

    inline int incoming_size(Vertex);


    /* IO */
    static void read_graph_file(std::ifstream &, std::vector<int> &, std::vector<Vertex> &);

    void load_graph(const std::string&);

    void load_graph_binary(const std::string&);

    void store_graph_binary(const std::string&);

    friend std::ostream &operator<<(std::ostream &os, const Graph &graph);

    // member function
    void build_outgoing_starts(std::vector<Vertex> &);

    void build_outgoing_edges(std::vector<Vertex> &);

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
    std::vector<int> outgoing_starts;
    std::vector<Vertex> outgoing_edges;

    std::vector<int> incoming_starts;
    std::vector<Vertex> incoming_edges;
};


/* Included here to enable inlining. Don't look. */
#include "graph_internal.h"

#endif
