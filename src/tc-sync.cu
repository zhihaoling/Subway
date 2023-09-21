#include "../shared/globals.hpp"
#include "../shared/timer.hpp"
#include "../shared/argument_parsing.cuh"
#include "../shared/graph.cuh"
#include "../shared/subgraph.cuh"
#include "../shared/partitioner.cuh"
#include "../shared/subgraph_generator.cuh"
#include "../shared/gpu_error_check.cuh"
#include "../shared/gpu_kernels.cuh"
#include "../shared/subway_utilities.hpp"

#include <cuda_runtime.h>

//得到邻居
uint* N(uint source,OutEdge* edgeList,uint* nodePointer,uint* outDegree)
{
    uint* result=new uint[outDegree[source]*sizeof(OutEdge)];
	for(int i=0;i<outDegree[source];i++){
		result[i]=edgeList[nodePointer[source]+i].end;
	}
	return result;
}

void printNeb(uint source,uint* array ,int size){
    printf("%u的邻居：",source);
    for(int i=0;i<size;i++){
        printf("%u,",array[i]);
    }
    printf("\n");
}
__global__ void tc_sync_kernel(unsigned int numNodes,
							unsigned int *d_nodesPointer,
							OutEdge *d_edgeList,
							unsigned int *d_outDegree)
{
    unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;
	if(tId>=numNodes) return ;
    uint source=tId;
	printf("this is thread %u,and its outdegree size is %u\n",tId,d_outDegree[source]);
    for(uint i=0;i<d_outDegree[source];i++){
        //边终点
        uint dest=d_edgeList[d_nodesPointer[source]+i].end;
		//symmetry
        if(source>dest) continue ;
		//__syncthreads();
		printf("src:%u dest:%u \n",source,dest);
        //求两个点的邻居的交集
        //printNeb<<<1,1>>>(source,N(source,d_edgeList,d_nodesPointer,d_outDegree),d_outDegree[source]);
    }
	return ;
}

int main(int argc, char** argv)
{
	cudaFree(0);

	ArgumentParser arguments(argc, argv, false, false);
	
	Timer timer;
	timer.Start();
	
	Graph<OutEdge> graph(arguments.input, false);
	graph.ReadGraph();

	float readtime = timer.Finish();
	cout << "Graph Reading finished in " << readtime/1000 << " (s).\n";
	
	cout<<"graph.outDegree[0]="<<graph.outDegree[0]<<endl;

	// OutEdge *edgeList;
	// memcpy(edgeList,graph.edgeList,graph.num_edges*sizeof(OutEdge));
	// cout<<"edgeList:"<<endl;
	// for(int i=0;i<graph.num_edges;i++){
	// 	cout<<graph.edgeList[i].end<<" ";
	// }
	// cout<<endl<<"nodePointer:"<<endl;
	// for(int i=0;i<graph.num_nodes;i++){
	// 	cout<<graph.nodePointer[i]<<" ";
	// }
	// cout<<endl;

	uint* d_nodesPointer;
	OutEdge* d_edgeList;
	cudaMalloc(&d_nodesPointer,sizeof(uint)*(graph.num_nodes));
	cudaMalloc(&d_edgeList,sizeof(OutEdge)*graph.num_edges);
	cudaMemcpy(d_nodesPointer, graph.nodePointer,sizeof(uint)*(graph.num_nodes), cudaMemcpyHostToDevice);
	cudaMemcpy(d_edgeList, graph.edgeList, sizeof(OutEdge)*graph.num_edges, cudaMemcpyHostToDevice);

	cudaMemcpy(graph.d_outDegree, graph.outDegree, sizeof(uint)*(graph.num_nodes), cudaMemcpyHostToDevice);

	tc_sync_kernel<<<graph.num_nodes/512+1,512>>>(graph.num_nodes,d_nodesPointer,d_edgeList,graph.d_outDegree);
	
	cudaDeviceSynchronize();
	
	cout<<"cpu is finished"<<endl;

	return 0;

/*
	for(unsigned int i=0; i<graph.num_nodes; i++)
	{
		graph.value[i] = i;
		graph.label1[i] = false;
		graph.label2[i] = true;
	}

	gpuErrorcheck(cudaMemcpy(graph.d_outDegree, graph.outDegree, graph.num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(graph.d_value, graph.value, graph.num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(graph.d_label1, graph.label1, graph.num_nodes * sizeof(bool), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(graph.d_label2, graph.label2, graph.num_nodes * sizeof(bool), cudaMemcpyHostToDevice));
	
	Subgraph<OutEdge> subgraph(graph.num_nodes, graph.num_edges);
	
	SubgraphGenerator<OutEdge> subgen(graph);
	
	subgen.generate(graph, subgraph);


	Partitioner<OutEdge> partitioner;
	
	timer.Start();
	
	uint itr = 0;
		
	while (subgraph.numActiveNodes>0)
	{
		itr++;
		
		partitioner.partition(subgraph, subgraph.numActiveNodes);
		// a super iteration
		for(int i=0; i<partitioner.numPartitions; i++)
		{
			cudaDeviceSynchronize();
			gpuErrorcheck(cudaMemcpy(subgraph.d_activeEdgeList, subgraph.activeEdgeList + partitioner.fromEdge[i], (partitioner.partitionEdgeSize[i]) * sizeof(OutEdge), cudaMemcpyHostToDevice));
			cudaDeviceSynchronize();

			moveUpLabels<<< partitioner.partitionNodeSize[i]/512 + 1 , 512 >>>(subgraph.d_activeNodes, graph.d_label1, graph.d_label2, partitioner.partitionNodeSize[i], partitioner.fromNode[i]);

			cc_kernel<<< partitioner.partitionNodeSize[i]/512 + 1 , 512 >>>(partitioner.partitionNodeSize[i],
													partitioner.fromNode[i],
													partitioner.fromEdge[i],
													subgraph.d_activeNodes,
													subgraph.d_activeNodesPointer,
													subgraph.d_activeEdgeList,
													graph.d_outDegree,
													graph.d_value, 
													//d_finished,
													graph.d_label1,
													graph.d_label2);

			cudaDeviceSynchronize();
			gpuErrorcheck( cudaPeekAtLastError() );	
		}
		
		subgen.generate(graph, subgraph);
			
	}	
	
	float runtime = timer.Finish();
	cout << "Processing finished in " << runtime/1000 << " (s).\n";
	
	cout << "Number of iterations = " << itr << endl;
	
	gpuErrorcheck(cudaMemcpy(graph.value, graph.d_value, graph.num_nodes*sizeof(uint), cudaMemcpyDeviceToHost));
	
	utilities::PrintResults(graph.value, min(30, graph.num_nodes));
			
	if(arguments.hasOutput)
		utilities::SaveResults(arguments.output, graph.value, graph.num_nodes);
*/

}

