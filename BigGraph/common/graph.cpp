#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>

#include "graph.h"

#define GRAPH_HEADER_TOKEN ((int) 0xDEADBEEF)


void Graph::build_outgoing_starts(std::vector<int> &starts) {
    this->outgoing_starts = std::move(starts);
    this->outgoing_starts.push_back(this->num_edges);
}

void Graph::build_outgoing_edges(std::vector<Vertex> &edges) {
    this->outgoing_edges = std::move(edges);
}

// Given an outgoing edge adjacency list representation for a directed
// graph, build an incoming adjacency list representation
void Graph::build_incoming() {

    #ifdef DEBUG
    std::cout << "Beginning build_incoming... (" << this->num_nodes << " nodes)" << std::endl;
    #endif

    std::vector<int> node_counts(this->num_nodes);
    std::vector<int> node_scatter(this->num_nodes);

    this->incoming_starts = std::vector<int>(this->num_nodes + 1);
    this->incoming_edges = std::vector<Vertex>(this->num_edges);

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

void Graph::read_graph_file(std::ifstream &file, std::vector<int> &starts, std::vector<Vertex> &edges) {
    std::string buffer;
    bool started = true;
    while (!file.eof()) {
        buffer.clear();
        std::getline(file, buffer);

        if (!buffer.empty() && buffer[0] == '#') {
            started = false;
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
                starts.push_back(v);
            } else {
                edges.push_back(v);
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
        os << "node " << i << ": out = " << end_edge - start_edge << ": ";
        for (int j = start_edge; j < end_edge; ++j) {
            int target = graph.outgoing_edges[j];
            os << target;
        }
        os << std::endl;

        start_edge = graph.incoming_starts[i];
        end_edge = graph.incoming_starts[i + 1];
        os << "         in = " << end_edge - start_edge << " : ";
        for (int j = start_edge; j < end_edge; ++j) {
            int target = graph.incoming_edges[j];
            os << target;
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
    std::vector<int> starts;
    std::vector<Vertex> edges;
    read_graph_file(graph_file, starts, edges);

    // build the graph
    this->build_outgoing_starts(starts);
    this->build_outgoing_edges(edges);

    this->build_incoming();

    #ifdef DEBUG
    std::cout << *this;
    #endif
}

void Graph::load_graph_binary(const std::string& filename) {
    int data;
    std::ifstream fin;

    fin.open(filename, std::ios::in | std::ios::binary);
    if (!fin.is_open()) {
        std::cerr << "Could not open: " << filename << std::endl;
        exit(1);
    }

    // read header
    if (!fin.read((char*)&data, sizeof(data))) {
        std::cerr << "Error reading header." << std::endl;
    }
    if (data != GRAPH_HEADER_TOKEN) {
        std::cerr << "Invalid graph file header. File may be corrupt." << std::endl;
    }

    // read number of nodes
    if (!fin.read((char*)&data, sizeof(data))) {
        std::cerr << "Error reading number of nodes." << std::endl;
    }
    this->num_nodes = data;

    // read number of edges
    if (!fin.read((char*)&data, sizeof(data))) {
        std::cerr << "Error reading number of edges." << std::endl;
    }
    this->num_edges = data;

    // read outgoing starts
    for (int i = 0; i < this->num_nodes; ++i) {
        if (!fin.read((char*)&data, sizeof(data))) {
            std::cerr << "Error reading node " << i << std::endl;
        }
        this->outgoing_starts.push_back(data);
    }
    this->outgoing_starts.push_back(this->num_edges);

    // read outgoing edges
    for (int i = 0; i < this->num_edges; ++i) {
        if (!fin.read((char*)&data, sizeof(data))) {
            std::cerr << "Error reading edge " << i << std::endl;
        }
        this->outgoing_edges.push_back(data);
    }

    fin.close();

    this->build_incoming();

    #ifdef DEBUG
    std::cout << *this;
    #endif
}

void Graph::store_graph_binary(const std::string &filename) {

    std::ofstream fout;
    int data;

    fout.open(filename, std::ios::out | std::ios::binary);
    if (!fout.is_open()) {
        std::cerr << "Could not open: " << filename << std::endl;
        exit(1);
    }

    // write header
    data = GRAPH_HEADER_TOKEN;
    if (!fout.write((char*)&data, sizeof(data))) {
        std::cerr << "Error writing header." << std::endl;
    }

    // write number of nodes
    data = this->num_nodes;
    if (!fout.write((char*)&data, sizeof(data))) {
        std::cerr << "Error writing number of nodes." << std::endl;
    }

    // write number of edges
    data = this->num_edges;
    if (!fout.write((char*)&data, sizeof(data))) {
        std::cerr << "Error writing number of edges." << std::endl;
    }

    // write outgoing starts
    for (int i = 0; i < this->num_nodes; ++i) {
        data = this->outgoing_starts[i];
        if (!fout.write((char*)&data, sizeof(data))) {
            std::cerr << "Error writing node " << i << std::endl;
        }
    }

    // write outgoing edges
    for (int i = 0; i < this->num_edges; ++i) {
        data = this->outgoing_edges[i];
        if (!fout.write((char*)&data, sizeof(data))) {
            std::cerr << "Error writing edge " << i << std::endl;
        }
    }

    fout.close();
}
