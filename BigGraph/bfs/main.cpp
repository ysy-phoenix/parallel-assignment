#include <cstdio>
#include <omp.h>
#include <string>

#include <iostream>
#include <sstream>
#include <vector>

#include "../common/graph.h"
#include "bfs.h"

int main(int argc, char **argv) {
    std::string graph_filename;

    if (argc < 2) {
        std::cerr << "Usage: <path/to/graph/file> [num_threads]\n";
        std::cerr << "  To run results for all thread counts: <path/to/graph/file>\n";
        std::cerr
                << "  Run with a certain number of threads (no correctness run): <path/to/graph/file> <num_threads>\n";
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
        //Static assignment to get consistent usage across trials
        int max_threads = omp_get_max_threads();

        //static num threads
        std::vector<int> num_threads;

        //dynamic num_threads
        for (int i = 1; i < max_threads; i *= 2) {
            num_threads.push_back(i);
        }
        num_threads.push_back(max_threads);
        auto n_usage = num_threads.size();

        solution sol1(numNodes), sol2(numNodes), sol3(numNodes);

        double hybrid_base, top_base, bottom_base;
        double hybrid_time, top_time, bottom_time;

        double start;
        std::stringstream timing;
        timing << "Threads  Top Down          Bottom Up         Hybrid\n";

        //Loop through assignment values;
        for (int i = 0; i < n_usage; ++i) {
            std::cout << "----------------------------------------------------------" << std::endl;
            std::cout << "Running with " << num_threads[i] << " threads" << std::endl;
            //Set thread count
            omp_set_num_threads(num_threads[i]);

            //Run implementations
            start = omp_get_wtime();
            bfs_top_down(g, sol1);
            top_time = omp_get_wtime() - start;


            //Run implementations
            start = omp_get_wtime();
            bfs_bottom_up(g, sol2);
            bottom_time = omp_get_wtime() - start;


            start = omp_get_wtime();
            bfs_hybrid(g, sol3);
            hybrid_time = omp_get_wtime() - start;


            if (i == 0) {
                hybrid_base = hybrid_time;
                top_base = top_time;
                bottom_base = bottom_time;
            }

            char buf[1024];

            sprintf(buf, "%4d:    %2.2f (%.2fx)      %2.2f (%.2fx)      %2.2f (%.2fx)\n",
                    num_threads[i], top_time, top_base / top_time, bottom_time,
                    bottom_base / bottom_time, hybrid_time, hybrid_base / hybrid_time);

            timing << buf;
        }

        std::cout << "----------------------------------------------------------" << std::endl;
        std::cout << "Your Code: Timing Summary" << std::endl;
        std::cout << timing.str();
        std::cout << "----------------------------------------------------------" << std::endl;
    }
        //Run the code with only one thread count and only report speedup
    else {
        solution sol1(numNodes), sol2(numNodes), sol3(numNodes);

        double hybrid_time, top_time, bottom_time;

        double start;
        std::stringstream timing;

        timing << "Threads   Top Down    Bottom Up       Hybrid\n";

        //Loop through assignment values;
        std::cout << "Running with " << thread_count << " threads" << std::endl;
        //Set thread count
        omp_set_num_threads(thread_count);

        //Run implementations
        start = omp_get_wtime();
        bfs_top_down(g, sol1);
        top_time = omp_get_wtime() - start;


        //Run implementations
        start = omp_get_wtime();
        bfs_bottom_up(g, sol2);
        bottom_time = omp_get_wtime() - start;


        start = omp_get_wtime();
        bfs_hybrid(g, sol3);
        hybrid_time = omp_get_wtime() - start;


        char buf[1024];

        sprintf(buf, "%4d:     %8.2f     %8.2f     %8.2f\n",
                thread_count, top_time, bottom_time, hybrid_time);

        timing << buf;

        std::cout << "----------------------------------------------------------" << std::endl;
        std::cout << "Your Code: Timing Summary" << std::endl;
        std::cout << timing.str();
    }

    return 0;
}
