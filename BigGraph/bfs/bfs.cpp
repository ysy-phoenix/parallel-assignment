#include "bfs.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstddef>
#include <omp.h>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1
#define THRESHOLD 10000000

//#define VERBOSE

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
void top_down_step(Graph &g, vertex_set *frontier, vertex_set *new_frontier, int *distances) {
    int numNodes = g.get_num_nodes();
    #pragma omp parallel default(none) shared(g, frontier, new_frontier, distances, numNodes)
    {
        int local_count = 0;
        int size = numNodes / omp_get_num_threads();
        auto *local_frontier = new int[size];

        #pragma omp for schedule(dynamic, 200)
        for (int i = 0; i < frontier->count; ++i) {
            int node = frontier->vertices[i];
            const auto start = g.outgoing_begin(node);
            const auto end = g.outgoing_end(node);
            // attempt to add all neighbors to the new frontier
            for (auto neighbor = start; neighbor != end; ++neighbor) {
                int outgoing = *neighbor;
                if (distances[outgoing] == NOT_VISITED_MARKER &&
                    __sync_bool_compare_and_swap(&distances[outgoing], NOT_VISITED_MARKER, distances[node] + 1)) {
                    local_frontier[local_count++] = outgoing;
                }
            }
        }

        int start_idx = __sync_fetch_and_add(&new_frontier->count, local_count);
        memcpy(new_frontier->vertices + start_idx, local_frontier, local_count * sizeof(int));
        delete[] local_frontier;
    }
}

// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph &graph, solution &sol) {
    int numNodes = graph.get_num_nodes();
    vertex_set list1(numNodes);
    vertex_set list2(numNodes);

    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    #pragma omp parallel for default (none) shared (sol, numNodes)
    for (int i = 0; i < numNodes; ++i) {
        sol.distances[i] = NOT_VISITED_MARKER;
    }

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol.distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0) {

#ifdef VERBOSE
        double start_time = omp_get_wtime();
#endif

        new_frontier->clear();

        top_down_step(graph, frontier, new_frontier, sol.distances);

#ifdef VERBOSE
        double end_time = omp_get_wtime();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        std::swap(frontier, new_frontier);
    }
}

void bottom_up_step(Graph& g, vertex_set *frontier, vertex_set *new_frontier, int *distances, int iteration) {
    int numNodes = g.get_num_nodes();
    #pragma omp parallel default(none) shared(g, frontier, new_frontier, distances, iteration, numNodes)
    {
        int local_count = 0;
        int size = numNodes / omp_get_num_threads();
//        auto *local_frontier = new int[size];

        #pragma omp for schedule(dynamic, 200)
        for (int i = 0; i < numNodes; ++i) {
            if (distances[i] != NOT_VISITED_MARKER) {
                continue;
            }
            auto start = g.incoming_begin(i);
            auto end = g.incoming_end(i);

            for (auto neighbor = start; neighbor != end; ++neighbor) {
                int incoming = *neighbor;
                if (distances[incoming] == iteration) {
                    distances[i] = iteration + 1;
//                    local_frontier[local_count++] = i;
                    ++local_count;
                    break;
                }
            }
        }

        int start_idx = __sync_fetch_and_add(&new_frontier->count, local_count);
//        memcpy(new_frontier->vertices + start_idx, local_frontier, local_count * sizeof(int));
//        delete[] local_frontier;
    }
}

// As was done in the top-down case,
// create subroutine bottom_up_step() that is called in
// each step of the BFS process.
// initialize all nodes to NOT_VISITED
void bfs_bottom_up(Graph& graph, solution &sol) {
    int numNodes = graph.get_num_nodes();
    vertex_set list1(numNodes);
    vertex_set list2(numNodes);

    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    for (int i = 0; i < numNodes; ++i) {
        sol.distances[i] = NOT_VISITED_MARKER;
    }

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol.distances[ROOT_NODE_ID] = 0;
    int iteration = 0;

    while (frontier->count != 0) {

#ifdef VERBOSE
        double start_time = omp_get_wtime();
#endif

        new_frontier->clear();

        bottom_up_step(graph, frontier, new_frontier, sol.distances, iteration);

#ifdef VERBOSE
        double end_time = omp_get_wtime();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        std::swap(frontier, new_frontier);
        ++iteration;
    }
}

void bfs_hybrid(Graph &graph, solution &sol) {
    int numNodes = graph.get_num_nodes();
    vertex_set list1(numNodes);
    vertex_set list2(numNodes);

    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    for (int i = 0; i < numNodes; ++i) {
        sol.distances[i] = NOT_VISITED_MARKER;
    }

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol.distances[ROOT_NODE_ID] = 0;
    int iteration = 0;

    while (frontier->count != 0) {

#ifdef VERBOSE
        double start_time = omp_get_wtime();
#endif

        new_frontier->clear();
        if (frontier->count >= THRESHOLD) {
            bottom_up_step(graph, frontier, new_frontier, sol.distances, iteration);
        } else {
            top_down_step(graph, frontier, new_frontier, sol.distances);
        }

#ifdef VERBOSE
        double end_time = omp_get_wtime();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        std::swap(frontier, new_frontier);
        ++iteration;
    }

}


/**
30m
Threads  Top Down          Bottom Up         Hybrid
   1:    0.40 (1.00x)      0.95 (1.00x)      0.40 (1.00x)
   2:    0.21 (1.89x)      0.43 (2.21x)      0.21 (1.88x)
   4:    0.13 (3.07x)      0.28 (3.34x)      0.15 (2.62x)
   8:    0.10 (3.82x)      0.18 (5.30x)      0.11 (3.49x)
  12:    0.09 (4.27x)      0.15 (6.25x)      0.09 (4.27x)

 200m
 Threads  Top Down          Bottom Up         Hybrid
   1:    10.26 (1.00x)      17.64 (1.00x)      5.69 (1.00x)
   2:    5.67 (1.81x)      12.60 (1.40x)      3.47 (1.64x)
   4:    4.15 (2.47x)      7.84 (2.25x)      2.26 (2.52x)
   8:    2.76 (3.72x)      5.25 (3.36x)      1.62 (3.52x)
  12:    2.43 (4.22x)      4.92 (3.59x)      1.53 (3.73x)

random 500m
Threads  Top Down          Bottom Up         Hybrid
   1:    24.05 (1.00x)      78.24 (1.00x)      6.61 (1.00x)
   2:    12.81 (1.88x)      47.38 (1.65x)      4.30 (1.54x)
   4:    8.31 (2.89x)      32.43 (2.41x)      3.38 (1.95x)
   8:    5.67 (4.25x)      23.32 (3.35x)      2.46 (2.68x)

 */
