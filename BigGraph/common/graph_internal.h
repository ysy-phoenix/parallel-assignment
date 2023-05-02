#ifndef GRAPH_INTERNAL_H_
#define GRAPH_INTERNAL_H_

#include <cstdlib>
#include "contracts.h"

inline int Graph::get_num_nodes() const {
    return this->num_nodes;
}

inline int Graph::get_num_edges() const {
    return this->num_edges;
}

inline std::vector<Vertex>::const_iterator Graph::outgoing_begin(Vertex v) {
    REQUIRES(0 <= v && v < this->num_edges);
    return this->outgoing_edges.cbegin() + this->outgoing_starts[v];
}

inline std::vector<Vertex>::const_iterator Graph::outgoing_end(Vertex v) {
    REQUIRES(0 <= v && v < this->num_edges);
    return this->outgoing_edges.cbegin() + this->outgoing_starts[v + 1];
}

inline int Graph::outgoing_size(Vertex v) {
    REQUIRES(0 <= v && v < this->num_nodes);
    return this->outgoing_starts[v + 1] - this->outgoing_starts[v];
}

inline std::vector<Vertex>::const_iterator Graph::incoming_begin(Vertex v) {
    REQUIRES(0 <= v && v < this->num_nodes);
    return this->incoming_edges.cbegin() + this->incoming_starts[v];
}

inline std::vector<Vertex>::const_iterator Graph::incoming_end(Vertex v) {
    REQUIRES(0 <= v && v < this->num_nodes);
    return this->incoming_edges.cbegin() + this->incoming_starts[v + 1];
}

inline int Graph::incoming_size(Vertex v) {
    REQUIRES(0 <= v && v < this->num_nodes);
    return this->incoming_starts[v + 1] - this->incoming_starts[v];
}

#endif // GRAPH_INTERNAL_H_
