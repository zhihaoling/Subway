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


//并行地提取并存放end
__global__ void getEdgesEnd(OutEdge *d_edgeList,uint* to,uint size){
	unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;
	if(tId>=size) return;
	to[tId]=d_edgeList[tId].end;
	__syncthreads();
	if(tId==size-1){
		printf("edgesEnd: ");
		for(int i=0;i<size;i++) printf("%u ",to[i]);
		printf("\n");
	}
	return ;
}

//hash表求交集个数，总node数为n，复杂度为O(n)
__device__ uint set_intersaction(uint* set1,uint *set2,uint size1,uint size2,uint totalSize){
	uint* hash=NULL;
	uint result=0;
	cudaError_t cudaState=cudaMalloc((void**)&hash,(totalSize+10)*sizeof(uint)) ;
	if(cudaState!=cudaSuccess) printf("%s\n", cudaGetErrorString(cudaState));	
	memset(hash,0,totalSize*sizeof(uint));
	for(uint i=0;i<size1;i++) hash[set1[i]]+=1;
	for(uint i=0;i<size2;i++){
		hash[set2[i]]+=1;
	} 
	for(int i=0;i<totalSize;i++){
		if(hash[i]==2) result++;
	}
	cudaFree(hash);
	return result;
}
__global__ void tc_sync_kernel(unsigned int numNodes,
							unsigned int *d_nodesPointer,
							unsigned int *d_edgesEnd,
							unsigned int *d_outDegree,
							unsigned int *result)
{
    unsigned int tId = blockDim.x * blockIdx.x + threadIdx.x;
	if(tId>=numNodes) return ;
    uint source=tId;
	printf("this is thread %u,and its outdegree size is %u\n",tId,d_outDegree[source]);
    for(uint i=0;i<d_outDegree[source];i++){
        //边终点
        uint dest=d_edgesEnd[d_nodesPointer[source]+i];
		//symmetry
        if(source>dest) continue ;
		//printf("src:%u dest:%u \n",source,dest);
        //求两个点的邻居的交集
		uint* sourceNset;
		uint* destNset;
		cudaMalloc((void**)&sourceNset,sizeof(uint)*d_outDegree[source]);
		cudaMalloc((void**)&destNset,sizeof(uint)*d_outDegree[dest]);
		//这里基于d_edgesEnd的偏移量不需要*sizeof(uint)
		memcpy(sourceNset,d_edgesEnd+d_nodesPointer[source],sizeof(uint)*d_outDegree[source]);
		memcpy(destNset,d_edgesEnd+d_nodesPointer[dest],sizeof(uint)*d_outDegree[dest]);
		uint thisAdd=set_intersaction(sourceNset,
										destNset,
										d_outDegree[source],
										d_outDegree[dest],
										numNodes);
		printf("%u --> %u ,thisAdd=%u \n",source,dest,thisAdd);
		//原子加法神中神
		atomicAdd(result,thisAdd);
		
		cudaFree(sourceNset);
		cudaFree(destNset);
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
	// cout<<"graph.outDegree[0]="<<graph.outDegree[0]<<endl;
	uint* h_result=new uint;
	uint* d_nodesPointer;
	uint* d_edgesEnd;
	OutEdge* d_edgeList;
	uint* d_result;
	cudaMalloc(&d_nodesPointer,sizeof(uint)*(graph.num_nodes));
	cudaMalloc(&d_edgeList,sizeof(OutEdge)*graph.num_edges);
	cudaMalloc(&d_edgesEnd,sizeof(uint)*graph.num_edges);
	cudaMalloc(&d_result,sizeof(uint));
	cudaMemcpy(d_nodesPointer, graph.nodePointer,sizeof(uint)*(graph.num_nodes), cudaMemcpyHostToDevice);
	cudaMemcpy(d_edgeList, graph.edgeList, sizeof(OutEdge)*graph.num_edges, cudaMemcpyHostToDevice);
	cudaMemcpy(graph.d_outDegree, graph.outDegree, sizeof(uint)*(graph.num_nodes), cudaMemcpyHostToDevice);
	*h_result=0;
	cudaMemcpy(&d_result,h_result,sizeof(uint),cudaMemcpyHostToDevice);


	timer.Start();
	getEdgesEnd<<<(graph.num_edges+512-1)/512,512>>>(d_edgeList,d_edgesEnd,graph.num_edges);
	readtime = timer.Finish();
	cout << " getEdgesEnd() finished in " << readtime/1000 << " (s).\n";

	printf("nodePorinter: ");
	for(int i=0;i<graph.num_nodes;i++) printf("%u ",graph.nodePointer[i]);
	printf("\n");


	tc_sync_kernel<<<graph.num_nodes/512+1,512>>>(graph.num_nodes,d_nodesPointer,d_edgesEnd,graph.d_outDegree,d_result);
	cudaDeviceSynchronize();

	
	cudaMemcpy(h_result,d_result,sizeof(uint),cudaMemcpyDeviceToHost);
	//消除重复计算的边
	*h_result/=3;
	
	cout<<"The counting result : "<<*h_result<<endl;

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

