#include <cstdio>
#include <omp.h>
#include <string>

#include <iostream>
#include <sstream>
#include <vector>

#include "../common/graph.h"

#include "page_rank.h"

#define PageRankDampening 0.3f
#define PageRankConvergence 1e-7

int main(int argc, char **argv) {
    std::string graph_filename;

    if (argc < 2) {
        std::cerr << "Usage: <path/to/graph/file> [num_threads]\n";
        std::cerr << "  To run results for all thread counts: <path/to/graph/file>\n";
        std::cerr << "  Run with a certain number of threads: <path/to/graph/file> <num_threads>\n";
        exit(1);
    }

    int thread_count = -1;
    if (argc == 3) {
        thread_count = std::stol(argv[2], nullptr, 10);
    }

    graph_filename = argv[1];

    Graph g;

    std::cout << "----------------------------------------------------------" << std::endl;
    std::cout << "Max system threads = " << omp_get_max_threads() << std::endl;
    if (thread_count > 0) {
        thread_count = std::min(thread_count, omp_get_max_threads());
        std::cout << "Running with " << thread_count << " threads" << std::endl;
    }
    std::cout << "----------------------------------------------------------" << std::endl;

    std::cout << "Loading graph..." << std::endl;
    g.load_graph_binary(graph_filename);
    std::cout << "Graph stats:" << std::endl;
    std::cout << "  Edges: " << g.get_num_edges() << std::endl;
    std::cout << "  Nodes: " << g.get_num_nodes() << std::endl;

    auto numNodes = g.get_num_nodes();
    //If we want to run on all threads
    if (thread_count <= -1) {
        //Static num_threads to get consistent usage across trials
        int max_threads = omp_get_max_threads();

        std::vector<int> num_threads;

        // dynamic num_threads
        for (int i = 1; i < max_threads; i *= 2) {
            num_threads.push_back(i);
        }
        num_threads.push_back(max_threads);
        auto n_usage = num_threads.size();

        auto *sol1 = new double[numNodes];

        double pagerank_base;
        double pagerank_time;

        std::stringstream timing;

        timing << "Threads  Time (Speedup)\n";

        //Loop through num_threads values;
        for (int i = 0; i < n_usage; ++i) {
            std::cout << "----------------------------------------------------------" << std::endl;
            std::cout << "Running with " << num_threads[i] << " threads" << std::endl;
            //Set thread count
            omp_set_num_threads(num_threads[i]);

            //Run implementations
            double start = omp_get_wtime();
            pageRank(g, sol1, PageRankDampening, PageRankConvergence);
            pagerank_time = omp_get_wtime() - start;

            // record single thread times in order to report speedup
            if (num_threads[i] == 1) {
                pagerank_base = pagerank_time;
            }

            char buf[1024];
            sprintf(buf, "%4d:   %.4f (%.4fx)\n",
                    num_threads[i], pagerank_time, pagerank_base / pagerank_time);

            timing << buf;
        }

        std::cout << "----------------------------------------------------------" << std::endl;
        std::cout << "Your Code: Timing Summary" << std::endl;
        std::cout << timing.str();
        std::cout << "----------------------------------------------------------" << std::endl;
        delete[] sol1;
    }
    else { // Run the code with only one thread count and only report speedup
        auto *sol1 = new double[numNodes];

        double pagerank_time;

        std::stringstream timing;

        timing << "Threads  Time\n";

        // Loop through assignment values;
        std::cout << "Running with " << thread_count << " threads" << std::endl;
        // Set thread count
        omp_set_num_threads(thread_count);

        // Run implementations
        double start = omp_get_wtime();
        pageRank(g, sol1, PageRankDampening, PageRankConvergence);
        pagerank_time = omp_get_wtime() - start;

        char buf[1024];

        sprintf(buf, "%4d:   %.4f\n",
                thread_count, pagerank_time);

        timing << buf;

        std::cout << "----------------------------------------------------------" << std::endl;
        std::cout << "Your Code: Timing Summary" << std::endl;
        std::cout << timing.str();
        delete[] sol1;
    }

    return 0;
}
