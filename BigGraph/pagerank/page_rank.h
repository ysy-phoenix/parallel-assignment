#ifndef PAGE_RANK_H_
#define PAGE_RANK_H_

#include "../common/graph.h"

void pageRank(Graph &g, double *solution, double damping, double convergence);
void test();
#endif /* PAGE_RANK_H_ */
