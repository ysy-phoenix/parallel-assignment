#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cstdio>

#include "graph.h"

#define GRAPH_HEADER_TOKEN ((int) 0xDEADBEEF)


// Given an outgoing edge adjacency list representation for a directed
// graph, build an incoming adjacency list representation
void Graph::build_incoming() {

    #ifdef DEBUG
    std::cout << "Beginning build_incoming... (" << this->num_nodes << " nodes)" << std::endl;
    #endif

    std::vector<int> node_counts(this->num_nodes);
    std::vector<int> node_scatter(this->num_nodes);

    this->incoming_starts = new int[this->num_nodes + 1];
    this->incoming_edges = new Vertex[this->num_edges];

    // compute number of incoming edges per node
    int total_edges = 0;
    for (int i = 0; i < this->num_nodes; ++i) {
        int start_edge = this->outgoing_starts[i];
        int end_edge = this->outgoing_starts[i + 1];
        for (int j = start_edge; j < end_edge; ++j) {
            int target_node = this->outgoing_edges[j];
            ++node_counts[target_node];
            ++total_edges;
        }
    }
    #ifdef DEBUG
    std::cout << "Total edges: " << total_edges << std::endl;
    std::cout << "Computed incoming edge counts." << std::endl;
    #endif

    // build the starts vector
    this->incoming_starts[0] = 0;
    for (int i = 1; i <= this->num_nodes; ++i) {
        this->incoming_starts[i] = this->incoming_starts[i - 1] + node_counts[i - 1];
        #ifdef DEBUG
        std::cout << i << " : " << this->incoming_starts[i] << std::endl;
        #endif
    }
    #ifdef DEBUG
    std::cout << "Computed per-node incoming starts." << std::endl;
    #endif

    // now perform the scatter
    for (int i = 0; i < num_nodes; ++i) {
        int start_edge = this->outgoing_starts[i];
        int end_edge = this->outgoing_starts[i + 1];
        for (int j = start_edge; j < end_edge; ++j) {
            int target_node = this->outgoing_edges[j];
            this->incoming_edges[this->incoming_starts[target_node] + node_scatter[target_node]] = i;
            ++node_scatter[target_node];
        }
    }


    // verify
    #ifdef DEBUG
    std::cout << "Verifying graph..." << std::endl;

    for (int i = 0; i < num_nodes; ++i) {
        int start_node = this->outgoing_starts[i];
        int end_node = this->outgoing_starts[i + 1];
        for (int j = start_node; j < end_node; ++j) {

            bool verified = false;

            // make sure that vertex i is a neighbor of target_node
            int target_node = this->outgoing_edges[j];
            int j_start_edge = this->incoming_starts[target_node];
            int j_end_edge = this->incoming_starts[target_node + 1];
            for (int k = j_start_edge; k < j_end_edge; ++k) {
                if (this->incoming_edges[k] == i) {
                    verified = true;
                    break;
                }
            }

            if (!verified) {
                std::cerr << "Error: " << i << target_node << " did not verify" << std::endl;
            }
        }
    }

    std::cout <<"Done verifying" << std::endl;
    #endif
}

void Graph::get_meta_data(std::ifstream &file) {
    // going back to the beginning of the file
    file.clear();
    file.seekg(0, std::ios::beg);

    std::string buffer;
    std::getline(file, buffer);
    if ((buffer.compare(std::string("AdjacencyGraph")))) {
        std::cout << "Invalid input file" << buffer << std::endl;
        exit(1);
    }
    buffer.clear();

    do {
        std::getline(file, buffer);
    } while (buffer.empty() || buffer[0] == '#');
    this->num_nodes = strtol(buffer.c_str(), nullptr, 10);
    buffer.clear();

    do {
        std::getline(file, buffer);
    } while (buffer.empty() || buffer[0] == '#');
    this->num_edges = strtol(buffer.c_str(), nullptr, 10);
}

void Graph::read_graph_file(std::ifstream &file) {
    std::string buffer;
    bool started = true;
    int idx = 0;
    this->outgoing_starts = new int[this->num_nodes + 1];
    this->outgoing_edges = new Vertex[this->num_edges];

    while (!file.eof()) {
        buffer.clear();
        std::getline(file, buffer);

        if (!buffer.empty() && buffer[0] == '#') {
            started = false;
            this->outgoing_starts[idx++] = this->num_edges;
            idx = 0;
            continue;
        }

        std::stringstream parse(buffer);
        while (!parse.fail()) {
            int v;
            parse >> v;
            if (parse.fail()) {
                break;
            }
            if (started) {
                this->outgoing_starts[idx++] = v;
            } else {
                this->outgoing_edges[idx++] = v;
            }
        }
    }
}

std::ostream &operator<<(std::ostream &os, const Graph &graph) {

    os << "Graph pretty print:" << std::endl;
    os << "num_nodes = " << graph.num_nodes << std::endl;
    os << "num_edges = " << graph.num_edges << std::endl;

    for (int i = 0; i < graph.num_nodes; ++i) {

        int start_edge = graph.outgoing_starts[i];
        int end_edge = graph.outgoing_starts[i + 1];
        os << "node " << i << ": out = " << end_edge - start_edge << " : ";
        for (int j = start_edge; j < end_edge; ++j) {
            int target = graph.outgoing_edges[j];
            os << target << " ";
        }
        os << std::endl;

        start_edge = graph.incoming_starts[i];
        end_edge = graph.incoming_starts[i + 1];
        os << "         in = " << end_edge - start_edge << " : ";
        for (int j = start_edge; j < end_edge; ++j) {
            int target = graph.incoming_edges[j];
            os << target << " ";
        }
        os << std::endl;
    }
    return os;
}

void Graph::load_graph(const std::string &filename) {
    // open the file
    std::ifstream graph_file;
    graph_file.open(filename);
    this->get_meta_data(graph_file);

    // read the file
    read_graph_file(graph_file);

    this->build_incoming();

    #ifdef DEBUG
    std::cout << *this;
    #endif
}

void Graph::load_graph_binary(const std::string &filename) {
    const char* file = filename.c_str();
    FILE* input;
    fopen_s(&input, file, "rb");

    if (!input) {
        fprintf(stderr, "Could not open: %s\n", file);
        exit(1);
    }

    int header[3];

    if (fread(header, sizeof(int), 3, input) != 3) {
        fprintf(stderr, "Error reading header.\n");
        exit(1);
    }

    if (header[0] != GRAPH_HEADER_TOKEN) {
        fprintf(stderr, "Invalid graph file header. File may be corrupt.\n");
        exit(1);
    }

    this->num_nodes = header[1];
    this->num_edges = header[2];

    this->outgoing_starts = new int[this->num_nodes + 1];
    this->outgoing_edges = new Vertex[this->num_edges];

    if (fread(this->outgoing_starts, sizeof(int), this->num_nodes, input) != (size_t) this->num_nodes) {
        fprintf(stderr, "Error reading nodes.\n");
        exit(1);
    }
    this->outgoing_starts[this->num_nodes] = this->num_edges;

    if (fread(this->outgoing_edges, sizeof(int), this->num_edges, input) != (size_t) this->num_edges) {
        fprintf(stderr, "Error reading edges.\n");
        exit(1);
    }

    fclose(input);

    this->build_incoming();

    #ifdef DEBUG
    std::cout << *this;
    #endif
}

void Graph::store_graph_binary(const std::string &filename) {
    const char* file = filename.c_str();
    FILE* output;
    fopen_s(&output,file, "wb");

    if (!output) {
        fprintf(stderr, "Could not open: %s\n", file);
        exit(1);
    }

    int header[3];
    header[0] = GRAPH_HEADER_TOKEN;
    header[1] = this->num_nodes;
    header[2] = this->num_edges;

    if (fwrite(header, sizeof(int), 3, output) != 3) {
        fprintf(stderr, "Error writing header.\n");
        exit(1);
    }

    if (fwrite(this->outgoing_starts, sizeof(int), this->num_nodes, output) != (size_t) this->num_nodes) {
        fprintf(stderr, "Error writing nodes.\n");
        exit(1);
    }

    if (fwrite(this->outgoing_edges, sizeof(int), this->num_edges, output) != (size_t) this->num_edges) {
        fprintf(stderr, "Error writing edges.\n");
        exit(1);
    }

    fclose(output);
}
