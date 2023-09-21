#ifndef TC_SYNC_KERNEL_CUH
#define TC_SYNC_KERNEL_CUH

#include "../shared/graph.cuh"
#include "../shared/globals.hpp"
void printNeb(uint source,uint* array ,int size){
    printf("%u的邻居：",source);
    for(int i=0;i<size;i++){
        printf("%u,",array[i]);
    }
    Graph<OutEdge> graph;
    printf("\n");
}
__global__ void tc_sync_kernel(Graph<OutEdge> g){
    unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;
    uint source=tId;
    for(uint i=0;i<g.d_outDegree[source];i++){
        //这里的edgelist还得改，先都存在device里
        //uint dest=g.edgeList[source+i].end;
        //if(source>dest) return ;
        //求两个点的邻居的交集
        printNeb(source,g.N(source),g.d_outDegree[source]);
        
    }
}

#endif	//	