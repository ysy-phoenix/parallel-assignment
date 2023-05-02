#include "page_rank.h"

#include <cstdlib>
#include <cmath>
#include <omp.h>


// pageRank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is num_nodes(g))
// damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//
void pageRank(Graph &g, double *solution, double damping, double convergence) {
    // initialize vertex weights to uniform probability. Double
    // precision scores are used to avoid underflow for large graphs

    int numNodes = g.get_num_nodes();
    double equal_prob = 1.0 / numNodes;

    #pragma omp parallel for default(none), shared(solution, numNodes, equal_prob)
    for (int i = 0; i < numNodes; ++i) {
        solution[i] = equal_prob;
    }

    /*
       Basic page rank pseudocode is provided below:

       // initialization: see example code above
       score_old[vi] = 1/numNodes;

       while (!converged) {

         // compute score_new[vi] for all nodes vi:
         score_new[vi] = sum over all nodes vj reachable from incoming edges
                            { score_old[vj] / number of edges leaving vj  }
         score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / numNodes;

         score_new[vi] += sum over all nodes v in graph with no outgoing edges
                            { damping * score_old[v] / numNodes }

         // compute how much per-node scores have changed
         // quit once algorithm has converged

         global_diff = sum over all nodes vi { abs(score_new[vi] - score_old[vi]) };
         converged = (global_diff < convergence)
       }

     */

    auto *sink_nodes = new int[numNodes];
    int sink_nodes_num = 0;
    for (Vertex i = 0; i < numNodes; ++i) {
        if (g.outgoing_size(i) == 0) {
            sink_nodes[sink_nodes_num++] = i;
        }
    }
    bool converged = false;
    double rest = (1.0 - damping) / numNodes;
    auto *score_new = new double[numNodes];
    while (!converged) {
        double randomJumpPr = 0.0;

        #pragma omp parallel for reduction (+:randomJumpPr) schedule(dynamic, 200) default(none), shared(solution, sink_nodes, sink_nodes_num, damping, numNodes)
        for (int j = 0; j < sink_nodes_num; ++j) {
            randomJumpPr += solution[sink_nodes[j]];
        }
        randomJumpPr *= damping;
        randomJumpPr /= numNodes;

        #pragma omp parallel for schedule(dynamic, 200) default(none), shared(solution, score_new, damping, rest, randomJumpPr, numNodes, g)
        for (int i = 0; i < numNodes; ++i) {
            // Vertex is type defined to an int. Vertex* points into g.outgoing_edges[]
            score_new[i] = 0;
            const auto start = g.incoming_begin(i);
            const auto end = g.incoming_end(i);
            for (auto v = start; v != end; ++v) {
                int vv = *v;
                score_new[i] += solution[vv] / g.outgoing_size(vv);
            }
            score_new[i] = (damping * score_new[i]) + rest + randomJumpPr;
        }

        double global_diff = 0;

        #pragma omp parallel for reduction (+:global_diff) schedule(dynamic, 200) default(none), shared(solution, score_new, numNodes, convergence)
        for (int i = 0; i < numNodes; ++i) {
            global_diff += abs(score_new[i] - solution[i]);
            solution[i] = score_new[i];
        }
        converged = (global_diff < convergence);
    }
    free(score_new);
}
