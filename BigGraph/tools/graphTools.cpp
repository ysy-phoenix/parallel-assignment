#include <algorithm>
#include <climits>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>


#include "../common/graph.h"

#define CMD_TEXT2BIN      "text2bin"
#define CMD_INFO          "info"
#define CMD_PRINT         "print"
#define CMD_NO_OUT_EDGES  "noout"
#define CMD_NO_IN_EDGES   "noin"
#define CMD_EDGE_STATS    "edgestats"


void print_help(const char *binary_name) {
    std::cerr << "Usage: " << binary_name << " cmd args\n";
    std::cerr << "Use '" << binary_name << " cmd' to get command-specific help.\n";
    std::cerr << "\n";
    std::cerr << "Valid cmds are:\n\n"
              << CMD_TEXT2BIN << ": text file to binary file conversion\n"
              << CMD_INFO << ": print graph metadata\n"
              << CMD_PRINT << ": print graph topology (careful with big graphs)\n"
              << CMD_NO_OUT_EDGES << ": detect vertices with no outgoing edges\n"
              << CMD_NO_IN_EDGES << ": detect vertices with no incoming edges\n"
              << CMD_EDGE_STATS << ": print stats on graph edges: e.g., min/max edges per node, etc.\n";
}

int main(int argc, char *argv[]) {

    if (argc < 2) {
        print_help(argv[0]);
        exit(1);
    }

    std::string cmd = std::string(argv[1]);

    if (cmd == CMD_TEXT2BIN) {

        if (argc < 4) {
            std::cerr << "Usage: " << argv[0] << " " << cmd << " textfilename binfilename\n";
            std::cerr << "Converts a graph from text file format to binary file format\n";
            exit(1);
        }

        std::string inputFilename = std::string(argv[2]);
        std::string outputFilename = std::string(argv[3]);

        Graph g;
        std::cout << "Loading graph: " << inputFilename << "\n";
        g.load_graph(inputFilename);
        std::cout << "Done loading.\n";
        g.store_graph_binary(outputFilename);

    } else if (cmd == CMD_INFO) {
        if (argc < 3) {
            std::cerr << "Usage: " << argv[0] << " " << cmd << " filename\n";
            std::cerr << "Pretty-prints graph info (num vertices, num edges)\n";
            exit(1);
        }

        std::string inputFilename = std::string(argv[2]);

        Graph g;
        std::cout << "Loading graph: " << inputFilename << "\n";
        g.load_graph_binary(inputFilename);
        std::cout << "Done loading.\n";

        std::cout << "Num vertices: " << g.get_num_nodes() << "\n";
        std::cout << "Num edges:    " << g.get_num_edges() << "\n";

    } else if (cmd == CMD_PRINT) {

        if (argc < 3) {
            std::cerr << "Usage: " << argv[0] << " " << cmd << " filename\n";
            std::cerr << "Pretty-prints graph, including edge information (be careful with large graphs)\n";
            exit(1);
        }

        std::string inputFilename = std::string(argv[2]);

        Graph g;
        std::cout << "Loading graph: " << inputFilename << "\n";
        g.load_graph_binary(inputFilename);
        std::cout << "Done loading.\n";
        std::cout << g;

    } else if (cmd == CMD_NO_OUT_EDGES) {

        if (argc < 3) {
            std::cerr << "Usage: " << argv[0] << " " << cmd << " filename\n";
            std::cerr << "Lists all vertices without outgoing edges.\n";
            exit(1);
        }

        std::string inputFilename = std::string(argv[2]);

        Graph g;
        std::cout << "Loading graph: " << inputFilename << "\n";
        g.load_graph_binary(inputFilename);
        std::cout << "Done loading.\n";

        std::vector<Vertex> zero_outgoing;

        for (int i = 0; i < g.get_num_nodes(); ++i) {
            if (g.outgoing_size(i) == 0) {
                zero_outgoing.push_back(i);
            }
        }

        std::cout << "Nodes with no outgoing edges:\n";
        for (int node : zero_outgoing) {
            std::cout << node << " ";
        }
        std::cout << "\n";
        std::cout << zero_outgoing.size() << " of " << g.get_num_nodes() << " nodes have zero outgoing edges ("
                  << std::setprecision(2)
                  << 100.0 * static_cast<double>(zero_outgoing.size()) / g.get_num_nodes() << "%).\n";

    } else if (cmd == CMD_NO_IN_EDGES) {

        if (argc < 3) {
            std::cerr << "Usage: " << argv[0] << " " << cmd << " filename\n";
            std::cerr << "Lists all edges without incoming edges.\n";
            exit(1);
        }

        std::string inputFilename = std::string(argv[2]);

        Graph g;
        std::cout << "Loading graph: " << inputFilename << "\n";
        g.load_graph_binary(inputFilename);
        std::cout << "Done loading.\n";

        std::vector<Vertex> zero_incoming;

        for (int i = 0; i < g.get_num_nodes(); ++i) {
            if (g.incoming_size(i) == 0) {
                zero_incoming.push_back(i);
            }
        }

        std::cout << "Nodes with no incoming edges:\n";
        for (int node : zero_incoming) {
            std::cout << node << " ";
        }
        std::cout << "\n";
        std::cout << zero_incoming.size() << " of " << g.get_num_nodes() << " nodes have zero incoming edges ("
                  << std::setprecision(2)
                  << 100.0 * static_cast<double>(zero_incoming.size()) / g.get_num_nodes() << "%).\n";

    } else if (cmd ==CMD_EDGE_STATS) {

        if (argc < 3) {
            std::cerr << "Usage: " << argv[0] << " " << cmd << " filename\n";
            std::cerr << "Print basic stats about edges.\n";
            exit(1);
        }

        std::string inputFilename = std::string(argv[2]);

        Graph g;
        std::cout << "Loading graph: " << inputFilename << "\n";
        g.load_graph_binary(inputFilename);
        std::cout << "Done loading. Now analyzing graph...\n";

        unsigned int total_incoming = 0;
        unsigned int total_outgoing = 0;
        unsigned int min_outgoing = INT_MAX;
        unsigned int max_outgoing = 0;
        unsigned int min_incoming = INT_MAX;
        unsigned int max_incoming = 0;
        bool is_symmetric = true;

        for (int i = 0; i < g.get_num_nodes(); ++i) {

            unsigned int num_incoming = g.incoming_size(i);
            unsigned int num_outgoing = g.outgoing_size(i);

            min_outgoing = std::min(min_outgoing, num_outgoing);
            max_outgoing = std::max(max_outgoing, num_outgoing);
            total_outgoing += num_outgoing;

            min_incoming = std::min(min_incoming, num_incoming);
            max_incoming = std::max(max_incoming, num_incoming);
            total_incoming += num_incoming;

            // check graph for sanity, and test for symmetric directed
            // edges
            auto out_begin = g.outgoing_begin(i);
            auto out_end = g.outgoing_end(i);
            for (auto v = out_begin; v != out_end; ++v) {

                Vertex target = *v;

                // sanity check. vertex i has an outgoing edge to target
                // (v->target), therefore target better have an
                // incoming edge from i.
                bool found_matching = false;
                auto sanity_begin = g.incoming_begin(target);
                auto sanity_end = g.incoming_end(target);
                for (auto v2 = sanity_begin; v2 != sanity_end; ++v2) {
                    Vertex i2 = *v2;
                    if (i == i2) {
                        found_matching = true;
                        break;
                    }
                }
                if (!found_matching) {
                    std::cerr << "GRAPH DID NOT PASS SANITY CHECK:\n"
                              << "vertex " << i << " has outgoing edge to " << target << ",\n but "
                              << "vertex " << target << " has no incoming edge from " << i << "\n";

                    // abort on a failed sanity check
                    exit(1);
                }

                // symmetry test: vertex i has an outgoing edge to
                // target (i->target), so check to see if there's an
                // incoming edge from target as well (target->i).
                bool found_symmetric = false;
                auto in_start = g.incoming_begin(i);
                auto in_end = g.incoming_end(i);

                for (auto v2 = in_start; v2 != in_end; ++v2) {

                    Vertex target2 = *v2;

                    if (target == target2) {
                        found_symmetric = true;
                        break;
                    }
                }
                if (!found_symmetric) {
                    is_symmetric = false;
                }
            }

        }

        float avg_outgoing = (float) total_outgoing / (float) g.get_num_nodes();
        float avg_incoming = (float) total_incoming / (float) g.get_num_nodes();

        std::cout << "=========================================================\n";
        std::cout << "Edge statistics for this graph:\n";
        std::cout << "=========================================================\n";
        std::cout << "The graph " << ((is_symmetric) ? "IS " : "IS NOT ") << "symmetric.\n";
        std::cout << "Outgoing edges: total = " << total_outgoing
                  << " avg = " << avg_outgoing
                  << " min = " << min_outgoing
                  << " max = " << max_outgoing << "\n";

        std::cout << "Incoming edges: total = " << total_incoming
                  << " avg = " << avg_incoming
                  << " min = " << min_incoming
                  << " max = " << max_incoming << "\n";
    } else {
        print_help(argv[0]);
    }

    return 0;
}
