#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>

#include <unistd.h>

#include <cuda_profiler_api.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/extrema.h>

#define VERTEX_BLOCK_SIZE 1000
#define EDGE_BLOCK_SIZE 1000
#define VERTEX_PREALLOCATE_LIST_SIZE 2000
#define EDGE_PREALLOCATE_LIST_SIZE 150000

// Issues faced
// Passing structures to kernel seems to create issues

// Inserts
// -> Vertices => 1 thread per vertex block

struct graph_properties {

    unsigned long xDim;
    unsigned long yDim;
    unsigned long total_edges;

};

struct vertex_preallocated_queue {

    long front;
    long rear;
    unsigned long count;
    struct vertex_block *vertex_block_address[VERTEX_PREALLOCATE_LIST_SIZE];

};

struct edge_preallocated_queue {

    long front;
    long rear;
    unsigned long count;
    struct edge_block *edge_block_address[EDGE_PREALLOCATE_LIST_SIZE];

};

// below are structures for the data structure

struct edge {

    unsigned long destination_vertex;
    // unsigned long weight;
    // unsigned long timestamp;

};

struct edge_block {

    // Array of Structures (AoS) for each edge block
    struct edge edge_block_entry[EDGE_BLOCK_SIZE];
    unsigned long active_edge_count;
    struct edge_block *next;

};

struct adjacency_sentinel {

    unsigned long edge_block_count;
    unsigned long active_edge_count;
    unsigned long last_insert_edge_offset;

    struct edge_block *last_insert_edge_block;
    struct edge_block *next;

};

struct vertex_block {

    // Structure of Array (SoA) for each vertex block
    unsigned long vertex_id[VERTEX_BLOCK_SIZE];
    struct adjacency_sentinel *vertex_adjacency[VERTEX_BLOCK_SIZE];
    unsigned long active_vertex_count;
    struct vertex_block *next;

    // adjacency sentinel
    unsigned long edge_block_count[VERTEX_BLOCK_SIZE];
    unsigned long last_insert_edge_offset[VERTEX_BLOCK_SIZE];

};

struct vertex_dictionary_sentinel {

    unsigned long vertex_block_count;
    unsigned long vertex_count;
    unsigned long last_insert_vertex_offset;

    struct vertex_block *last_insert_vertex_block;
    struct vertex_block *next;

};

// global variables
__device__ struct vertex_dictionary_sentinel d_v_d_sentinel;
__device__ struct vertex_preallocated_queue d_v_queue;

__device__ struct adjacency_sentinel d_a_sentinel;
__device__ struct edge_preallocated_queue d_e_queue;

__device__ void push_to_vertex_preallocate_queue(struct vertex_block *device_vertex_block) {

    if( (d_v_queue.rear + 1) % VERTEX_PREALLOCATE_LIST_SIZE == d_v_queue.front ) {
        printf("Vertex queue Full, front = %ld, rear = %ld\n", d_v_queue.front, d_v_queue.rear);

        return;
    }
    else if (d_v_queue.front == -1)
        d_v_queue.front = 0;

    d_v_queue.rear = (d_v_queue.rear + 1) % VERTEX_PREALLOCATE_LIST_SIZE;
    d_v_queue.vertex_block_address[d_v_queue.rear] = device_vertex_block;
    d_v_queue.count++;

    printf("Inserted %p to the vertex queue, front = %ld, rear = %ld\n", d_v_queue.vertex_block_address[d_v_queue.rear], d_v_queue.front, d_v_queue.rear);

}

__device__ struct vertex_block* pop_from_vertex_preallocate_queue(unsigned long pop_count) {

    if(d_v_queue.front == -1) {
        printf("Vertex queue empty, front = %ld, rear = %ld\n", d_v_queue.front, d_v_queue.rear);
        return NULL;
    }
    else {

        struct vertex_block *device_vertex_block = d_v_queue.vertex_block_address[d_v_queue.front];
        d_v_queue.vertex_block_address[d_v_queue.front] = NULL;
        d_v_queue.count -= pop_count;
        printf("Popped %p from the vertex queue, front = %ld, rear = %ld\n", device_vertex_block, d_v_queue.front, d_v_queue.rear);
        
        if(d_v_queue.front == d_v_queue.rear) {

            d_v_queue.front = -1;
            d_v_queue.rear = -1;

        }
        else
            d_v_queue.front = (d_v_queue.front + 1) % VERTEX_PREALLOCATE_LIST_SIZE;

        return device_vertex_block;
    }

}

__device__ void push_to_edge_preallocate_queue(struct edge_block *device_edge_block) {

    if( (d_e_queue.rear + 1) % EDGE_PREALLOCATE_LIST_SIZE == d_e_queue.front ) {
        printf("Edge queue Full, front = %ld, rear = %ld\n", d_e_queue.front, d_e_queue.rear);

        return;
    }
    else if (d_e_queue.front == -1)
        d_e_queue.front = 0;

    d_e_queue.rear = (d_e_queue.rear + 1) % EDGE_PREALLOCATE_LIST_SIZE;
    d_e_queue.edge_block_address[d_e_queue.rear] = device_edge_block;
    d_e_queue.count++;

    printf("Inserted %p to the edge queue, front = %ld, rear = %ld\n", d_e_queue.edge_block_address[d_e_queue.rear], d_e_queue.front, d_e_queue.rear);

}

__device__ struct edge_block* pop_from_edge_preallocate_queue(unsigned long pop_count) {

    if(d_e_queue.front == -1) {
        printf("Edge queue empty, front = %ld, rear = %ld\n", d_e_queue.front, d_e_queue.rear);
        return NULL;
    }
    else {

        struct edge_block *device_edge_block = d_e_queue.edge_block_address[d_e_queue.front];
        d_e_queue.edge_block_address[d_e_queue.front] = NULL;
        d_e_queue.count -= pop_count;
        printf("Popped %p from the edge queue, front = %ld, rear = %ld\n", device_edge_block, d_e_queue.front, d_e_queue.rear);
        
        if(d_e_queue.front == d_e_queue.rear) {

            d_e_queue.front = -1;
            d_e_queue.rear = -1;

        }
        else
            d_e_queue.front = (d_e_queue.front + 1) % EDGE_PREALLOCATE_LIST_SIZE;

        return device_edge_block;
    }

}

// __global__ void push_preallocate_list_to_device_queue_kernel(struct vertex_block** d_vertex_preallocate_list, struct edge_block** d_edge_preallocate_list, struct adjacency_sentinel** d_adjacency_sentinel_list, unsigned long vertex_blocks_count_init, unsigned long *edge_blocks_count_init, unsigned long total_edge_blocks_count_init) {
__global__ void push_preallocate_list_to_device_queue_kernel(struct vertex_block* d_vertex_preallocate_list, struct edge_block* d_edge_preallocate_list, struct adjacency_sentinel** d_adjacency_sentinel_list, unsigned long vertex_blocks_count_init, unsigned long *edge_blocks_count_init, unsigned long total_edge_blocks_count_init) {

    // some inits, don't run the below two initializations again or queue will fail
    d_v_queue.front = -1;
    d_v_queue.rear = -1;
    d_e_queue.front = -1;
    d_e_queue.rear = -1;

    printf("Pushing vertex blocks to vertex queue\n");

    for(unsigned long i = 0 ; i < vertex_blocks_count_init ; i++) {
    
        // printf("%lu -> %p\n", i, d_vertex_preallocate_list[i]);
        printf("%lu -> %p\n", i, d_vertex_preallocate_list + i);

        // d_vertex_preallocate_list[i]->active_vertex_count = 1909;

        // push_to_vertex_preallocate_queue(d_vertex_preallocate_list[i]);
        push_to_vertex_preallocate_queue(d_vertex_preallocate_list + i);

    }

    printf("Pushing edge blocks to edge queue\n");

    for(unsigned long i = 0 ; i < total_edge_blocks_count_init ; i++) {
    
        // printf("%lu -> %p\n", i, d_edge_preallocate_list[i]);
        printf("%lu -> %p\n", i, d_edge_preallocate_list + i);

        // push_to_edge_preallocate_queue(d_edge_preallocate_list[i]);
        push_to_edge_preallocate_queue(d_edge_preallocate_list + i);

    }

}

// __device__ volatile unsigned int current;
// __device__ unsigned int lockvar = 0;

// __global__ void vertex_dictionary_init(struct vertex_block** d_vertex_preallocate_list, struct adjacency_sentinel** d_adjacency_sentinel_list, unsigned long vertex_blocks_count_init, struct graph_properties *d_graph_prop, unsigned long size, unsigned long *edge_blocks_count_init) {
__global__ void vertex_dictionary_init(struct vertex_block* d_vertex_preallocate_list, struct adjacency_sentinel* d_adjacency_sentinel_list, unsigned long vertex_blocks_count_init, struct graph_properties *d_graph_prop, unsigned long size, unsigned long *edge_blocks_count_init) {
	
    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int current;
    __shared__ unsigned int lockvar;
    lockvar = 0;

    __syncthreads();

    d_v_d_sentinel.vertex_count = 99;

    if(id < vertex_blocks_count_init) {

        // critical section start



        struct vertex_block *device_vertex_block;

        do {

            current = atomicCAS(&lockvar, 0, 1);
            // current = 0;
            if(current == 0) {
            
                device_vertex_block = pop_from_vertex_preallocate_queue(1);
            
                lockvar = 0;

                printf("ID = %lu in vertex CS\n", id);
            }

        } while(current != 0);

        // critical section end

        // assigning first vertex block to the vertex sentinel node
        if(id == 0)
            d_v_d_sentinel.next = device_vertex_block;

        printf("%lu\n", id);
        printf("---------\n");

        printf("%lu -> %p\n", id, device_vertex_block);

        unsigned long start_index = id * VERTEX_BLOCK_SIZE;
        unsigned long end_index = (start_index + VERTEX_BLOCK_SIZE - 1) < size ? start_index + VERTEX_BLOCK_SIZE : size; 
        unsigned long j = 0;

        // device_vertex_block->active_vertex_count = 0;

        for( unsigned long i = start_index ; i < end_index ; i++ ) {

            device_vertex_block->vertex_id[j] = i + 1;
            device_vertex_block->active_vertex_count++;
    
            // device_vertex_block->vertex_adjacency[j] = d_adjacency_sentinel_list[i];
            device_vertex_block->vertex_adjacency[j] = d_adjacency_sentinel_list + i;

            device_vertex_block->edge_block_count[j] = edge_blocks_count_init[i];

            printf("%lu from thread %lu, start = %lu and end = %lu\n", device_vertex_block->vertex_id[j], id, start_index, end_index);
            j++;

        }

        // device_vertex_block->vertex_adjacency = 

        // optimization needed
        // if(id != size - 1)
        //     d_vertex_preallocate_list[id]->next = d_vertex_preallocate_list[id + 1];
        // else
        //     d_vertex_preallocate_list[id]->next = NULL;

        // if(id != size - 1)
        if(id < vertex_blocks_count_init - 1)
            (d_vertex_preallocate_list + id)->next = d_vertex_preallocate_list + id + 1;
        else
            (d_vertex_preallocate_list + id)->next = NULL;

        // struct edge_block *prev, *curr;

        // prev = NULL;
        // curr = NULL;


        // for(unsigned long i = 0 ; i < edge_blocks_required ; i++) {

        //     curr = device_edge_block[i];

        //     if(prev != NULL)
        //         prev->next = curr;

        //     prev = curr;
            
        // }


    }

}

__device__ unsigned int lockvar = 0;

// __global__ void adjacency_list_init(struct edge_block** d_edge_preallocate_list, unsigned long *d_edge_blocks_count_init, struct graph_properties *d_graph_prop, unsigned long *d_source, unsigned long *d_destination, unsigned long total_edge_blocks_count_init, unsigned long vertex_size, unsigned long edge_size) {
__global__ void adjacency_list_init(struct edge_block* d_edge_preallocate_list, unsigned long *d_edge_blocks_count_init, struct graph_properties *d_graph_prop, unsigned long *d_source, unsigned long *d_destination, unsigned long total_edge_blocks_count_init, unsigned long vertex_size, unsigned long edge_size) {
	
    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int current;
    // __shared__ unsigned int lockvar;
    // lockvar = 0;

    __syncthreads();

    if(id < vertex_size) {

        unsigned long current_vertex = id + 1;
        unsigned long index = 0;
        struct vertex_block *ptr = d_v_d_sentinel.next;

        // locating vertex in the vertex dictionary
        while(ptr != NULL) {

            unsigned long flag = 1;

            for(unsigned long i = 0 ; i < VERTEX_BLOCK_SIZE ; i++) {

                if(ptr->vertex_id[i] == current_vertex) {
                    index = i;
                    flag = 0;
                    printf("Matched at %lu\n", index);
                    break;
                }

            }

            if(flag)
                ptr = ptr->next;
            else
                break;
                

        }

        if(ptr != NULL)
            printf("ID = %lu, Vertex = %lu, Adjacency Sentinel = %p and edge blocks = %lu\n", id, ptr->vertex_id[index], ptr->vertex_adjacency[index], ptr->edge_block_count[index]);

        unsigned long edge_blocks_required = ptr->edge_block_count[index];

        // critical section start

        // temporary fix, this can't be a constant sized one
        struct edge_block *device_edge_block[100];


        do {

            current = atomicCAS(&lockvar, 0, 1);

            if(current == 0) {
            
                for(unsigned long i = 0 ; i < edge_blocks_required ; i++) {
                    device_edge_block[i] = pop_from_edge_preallocate_queue(1);
                    printf("ID = %lu in edge CS\n", id);
                }
            
                lockvar = 0;

                
            }

        } while(current != 0);

        // critical section end
        if(threadIdx.x == 0)
            printf("ID\tIteration\tGPU address\n");
        
        for(unsigned long i = 0 ; i < edge_blocks_required ; i++)
            printf("%lu\t%lu\t\t%p\n", id, i, device_edge_block[i]);

        // adding the edge blocks
        struct adjacency_sentinel *vertex_adjacency = ptr->vertex_adjacency[index];

        if(edge_blocks_required > 0) {

            struct edge_block *prev, *curr;

            prev = NULL;
            curr = NULL;


            for(unsigned long i = 0 ; i < edge_blocks_required ; i++) {

                curr = device_edge_block[i];

                if(prev != NULL)
                    prev->next = curr;

                prev = curr;
                
            }

            if(edge_blocks_required > 0) {
                vertex_adjacency->next = device_edge_block[0];
                curr->next = NULL;
            }

            unsigned long edge_block_entry_count = 0;
            unsigned long edge_block_counter = 0;

            curr = vertex_adjacency->next;
            vertex_adjacency->active_edge_count = 0;

            for(unsigned long i = 0 ; i < edge_size ; i++) {

                if(d_source[i] == current_vertex){

                    printf("Source = %lu and Destination = %lu\n", d_source[i], d_destination[i]);

                    // insert here
                    curr->edge_block_entry[edge_block_entry_count].destination_vertex = d_destination[i];
                    vertex_adjacency->active_edge_count++;
                    
                    if(++edge_block_entry_count >= EDGE_BLOCK_SIZE) {

                        curr = curr->next;
                        edge_block_counter++;
                        edge_block_entry_count = 0;
                    }

                }

            }

        }

    }

}

// __global__ void printKernel(struct vertex_block** d_vertex_preallocate_list, unsigned long size) {
__global__ void printKernel(struct vertex_block* d_vertex_preallocate_list, unsigned long size) {

    printf("Printing Linked List\n");

    struct vertex_block *ptr = d_v_d_sentinel.next;

    unsigned long vertex_block = 0;

    while(ptr != NULL) {

        printf("Vertex Block = %lu\n", vertex_block);

        for(unsigned long i = 0 ; i < VERTEX_BLOCK_SIZE ; i++) {

            // printf("%lu -> , edge blocks = %lu, edge sentinel = %p, ", ptr->vertex_id[i], ptr->edge_block_count[i], ptr->vertex_adjacency[i]);

            if((ptr->vertex_adjacency[i] != NULL) && (ptr->vertex_id[i] != 0)) {
                printf("%lu -> , edge blocks = %lu, edge sentinel = %p, active edge count = %lu, destination vertices -> ", ptr->vertex_id[i], ptr->edge_block_count[i], ptr->vertex_adjacency[i], ptr->vertex_adjacency[i]->active_edge_count);
            
                struct edge_block *itr = ptr->vertex_adjacency[i]->next;
                unsigned long edge_block_entry_count = 0;

                for(unsigned long j = 0 ; j < ptr->vertex_adjacency[i]->active_edge_count ; j++) {
                                
                    printf("%lu ", itr->edge_block_entry[edge_block_entry_count].destination_vertex);
                
                    // edge_block_entry_count++;

                    if(++edge_block_entry_count >= EDGE_BLOCK_SIZE) {
                        itr = itr->next;
                        edge_block_entry_count = 0;
                    }
                }
                printf("\n");            
            
            }printf("\n");


        }

        ptr = ptr->next;
        vertex_block++;
        printf("\n");

    }

    printf("VDS = %lu\n", d_v_d_sentinel.vertex_count);

}

void readFile(char *fileLoc, struct graph_properties *h_graph_prop, thrust::host_vector <unsigned long> &h_source, thrust::host_vector <unsigned long> &h_destination) {

    FILE* ptr;
    char buff[100];
    ptr = fopen(fileLoc, "a+");
 
    if (NULL == ptr) {
        printf("file can't be opened \n");
    }

    int dataFlag = 0;
    int index = 0;
 
    while (fgets(buff, 100, ptr) != NULL) {

        // ignore commented lines
        if(buff[0] == '%')
            continue;
        // reading edge values
        else if (dataFlag) {
    
            // printf("%s", buff);
            unsigned long source;
            unsigned long destination;
            int i = 0, j = 0;
            char temp[100];

            while( buff[i] != ' ' ) 
                temp[j++] = buff[i++];
            source = strtol( temp, NULL, 10);
            // printf("%lu ", nodeID);
            memset( temp, '\0', 100);

            i++;
            j=0;
            while( buff[i] != ' ' )
                temp[j++] = buff[i++];
            destination = strtol( temp, NULL, 10);
            // printf("%.8Lf ", x);
            memset( temp, '\0', 100);
            // h_edges[i] = ;
            // printf("%lu and %lu\n", source, destination);

            h_source[index] = source;
            h_destination[index++] = destination;

        }
        // reading xDim, yDim, and total_edges
        else {

            unsigned long xDim, yDim, total_edges;

            int i = 0,j = 0;
            char temp[100];

            while( buff[i] != ' ' ) 
                temp[j++] = buff[i++];
            xDim = strtol( temp, NULL, 10);
            // printf("%lu ", nodeID);
            memset( temp, '\0', 100);

            i++;
            j=0;
            while( buff[i] != ' ' )
                temp[j++] = buff[i++];
            yDim = strtol( temp, NULL, 10);
            // printf("%.8Lf ", x);
            memset( temp, '\0', 100);

            i++;
            j=0;
            while( buff[i] !='\0' )
                temp[j++] = buff[i++];
            total_edges = strtol( temp, NULL, 10);
            // printf("%.8Lf\n", y);
            memset( temp, '\0', 100);

            printf("xDim = %lu, yDim = %lu, Total Edges = %lu\n", xDim, yDim, total_edges);
            h_graph_prop->xDim = xDim;
            h_graph_prop->yDim = yDim;
            h_graph_prop->total_edges = total_edges;

            h_source.resize(h_graph_prop->total_edges);
            h_destination.resize(h_graph_prop->total_edges);

            dataFlag = 1;

        }

    }
    
    fclose(ptr);

}

int main(void) {

    // char fileLoc[20] = "input.mtx";
    char fileLoc[30] = "inf-luxembourg_osm.mtx";

    // some random inits

    struct graph_properties *h_graph_prop = (struct graph_properties*)malloc(sizeof(struct graph_properties));
    thrust::host_vector <unsigned long> h_source(1);
    thrust::host_vector <unsigned long> h_destination(1);
    thrust::host_vector <unsigned long> h_source_degree(1);

    // reading file, after function call h_source has data on source vertex and h_destination on destination vertex
    // both represent the edge data on host
    readFile(fileLoc, h_graph_prop, h_source, h_destination);

    unsigned long vertex_size = h_graph_prop->xDim;
    unsigned long edge_size = h_graph_prop->total_edges;

    h_source_degree.resize(h_graph_prop->xDim);
    thrust::fill(h_source_degree.begin(), h_source_degree.end(), 0);

    std::cout << "Check, " << h_source.size() << " and " << h_destination.size() << std::endl;

    // sleep(5);

    for(unsigned long i = 0 ; i < h_graph_prop->total_edges ; i++) {
     
        printf("%lu and %lu\n", h_source[i], h_destination[i]);
        h_source_degree[h_source[i] - 1]++;
    
    }

    unsigned long zero_count = 0;
    std::cout << "Printing degree of each source vertex" << std::endl;
    for(unsigned long i = 0 ; i < h_graph_prop->xDim ; i++) {
        // printf("%lu ", h_source_degree[i]);
        if(h_source_degree[i] == 0)
            zero_count++;
    }
    printf("zero count is %lu\n", zero_count);

    // sleep(5);

    thrust::host_vector <unsigned long> ::iterator iter = thrust::max_element(h_source.begin(), h_source.end());

    printf("xDim = %lu, yDim = %lu, Total Edges = %lu\n", h_graph_prop -> xDim, h_graph_prop -> yDim, h_graph_prop -> total_edges);
    std::cout << "Check, " << h_source.size() << " and " << h_destination.size() << std::endl;
    printf("Max element is %lu\n", *iter);

    // vertex block code start

    unsigned long vertex_blocks_count_init = ceil( double(h_graph_prop -> xDim) / VERTEX_BLOCK_SIZE);

    printf("Vertex blocks needed = %lu\n", vertex_blocks_count_init);

    struct vertex_dictionary_sentinel *device_vertex_sentinel;
    cudaMalloc(&device_vertex_sentinel, sizeof(struct vertex_dictionary_sentinel));

    thrust::host_vector <struct vertex_block *> h_vertex_preallocate_list(VERTEX_PREALLOCATE_LIST_SIZE);
    thrust::host_vector <struct edge_block *> h_edge_preallocate_list(EDGE_PREALLOCATE_LIST_SIZE);
    thrust::host_vector <struct adjacency_sentinel *> h_adjacency_sentinel_list(VERTEX_PREALLOCATE_LIST_SIZE);
    
    
    struct graph_properties *d_graph_prop;
    cudaMalloc(&d_graph_prop, sizeof(struct graph_properties));
	cudaMemcpy(d_graph_prop, &h_graph_prop, sizeof(struct graph_properties), cudaMemcpyHostToDevice);

    // Individual mallocs or one single big malloc to reduce overhead??
    std::cout << "Checkpoint" << std::endl;

    // for(unsigned long i = 0 ; i < vertex_blocks_count_init ; i++) {

    //     struct vertex_block *device_vertex_block;
    //     cudaMalloc(&device_vertex_block, sizeof(struct vertex_block));
    //     h_vertex_preallocate_list[i] = device_vertex_block;     

    // }

    struct vertex_block *device_vertex_block;
    cudaMalloc((struct vertex_block**)&device_vertex_block, vertex_blocks_count_init * sizeof(struct vertex_block));

    // allocate only necessary vertex sentinel nodes, running out of memory

    // for(unsigned long i = 0 ; i < vertex_size ; i++) {
    //     std::cout << i << std::endl;
    //     struct adjacency_sentinel *device_adjacency_sentinel;
    //     cudaMalloc(&device_adjacency_sentinel, sizeof(struct adjacency_sentinel));
    //     h_adjacency_sentinel_list[i] = device_adjacency_sentinel;

    // }

    struct adjacency_sentinel *device_adjacency_sentinel;
    cudaMalloc((struct adjacency_sentinel**)&device_adjacency_sentinel, vertex_size * sizeof(struct adjacency_sentinel));

    cudaDeviceSynchronize();

    std::cout << std::endl << "Host side" << std::endl << "Vertex blocks calculation" << std::endl << "Vertex block\t" << "GPU address" << std::endl;
    for(unsigned long i = 0 ; i < vertex_blocks_count_init ; i++)
        // printf("%lu\t\t%p\n", i, h_vertex_preallocate_list[i]);  
        printf("%lu\t\t%p\n", i, device_vertex_block + i);  
    
    thrust::host_vector <unsigned long> h_edge_blocks_count_init(h_graph_prop->xDim);
    
    // sleep(5);

    // unsigned long k = 0;
    unsigned long total_edge_blocks_count_init = 0;
    std::cout << "Edge blocks calculation" << std::endl << "Source\tEdge block count\tGPU address" << std::endl;
    for(unsigned long i = 0 ; i < h_graph_prop->xDim ; i++) {

        unsigned long edge_blocks = ceil(double(h_source_degree[i]) / EDGE_BLOCK_SIZE);
        h_edge_blocks_count_init[i] = edge_blocks;
        total_edge_blocks_count_init += edge_blocks;

        // for(unsigned long j = 0 ; j < h_edge_blocks_count_init[i] ; j++) {

        //     struct edge_block *device_edge_block;
        //     cudaMalloc(&device_edge_block, sizeof(struct edge_block));
        //     h_edge_preallocate_list[k++] = device_edge_block;

        //     std::cout << i + 1 << "\t" << edge_blocks << "\t\t\t" << device_edge_block << std::endl; 

        // }
    }

    printf("Total edge blocks needed = %lu\n", total_edge_blocks_count_init);
    // sleep(5);

    struct edge_block *device_edge_block;
    cudaMalloc((struct edge_block**)&device_edge_block, total_edge_blocks_count_init * sizeof(struct edge_block));

    thrust::device_vector <struct vertex_block *> d_vertex_preallocate_list(VERTEX_PREALLOCATE_LIST_SIZE);
    thrust::device_vector <struct edge_block *> d_edge_preallocate_list(EDGE_PREALLOCATE_LIST_SIZE);
    thrust::device_vector <struct adjacency_sentinel *> d_adjacency_sentinel_list(VERTEX_PREALLOCATE_LIST_SIZE);
    thrust::device_vector <unsigned long> d_source(h_graph_prop->total_edges);
    thrust::device_vector <unsigned long> d_destination(h_graph_prop->total_edges);
    thrust::device_vector <unsigned long> d_edge_blocks_count_init(h_graph_prop->xDim);
  
    thrust::copy(h_vertex_preallocate_list.begin(), h_vertex_preallocate_list.end(), d_vertex_preallocate_list.begin());
    thrust::copy(h_edge_preallocate_list.begin(), h_edge_preallocate_list.end(), d_edge_preallocate_list.begin());
    thrust::copy(h_adjacency_sentinel_list.begin(), h_adjacency_sentinel_list.end(), d_adjacency_sentinel_list.begin());
    thrust::copy(h_source.begin(), h_source.end(), d_source.begin());
    thrust::copy(h_destination.begin(), h_destination.end(), d_destination.begin());
    thrust::copy(h_edge_blocks_count_init.begin(), h_edge_blocks_count_init.end(), d_edge_blocks_count_init.begin());
 
    struct vertex_block** dvpl = thrust::raw_pointer_cast(d_vertex_preallocate_list.data());
    struct edge_block** depl = thrust::raw_pointer_cast(d_edge_preallocate_list.data());
    struct adjacency_sentinel** dapl = thrust::raw_pointer_cast(d_adjacency_sentinel_list.data());
    unsigned long* source = thrust::raw_pointer_cast(d_source.data());
    unsigned long* destination = thrust::raw_pointer_cast(d_destination.data());
    unsigned long* ebci = thrust::raw_pointer_cast(d_edge_blocks_count_init.data());



    printf("GPU side\n");
    // push_preallocate_list_to_device_queue_kernel<<< 1, 1>>>(dvpl, depl, dapl, vertex_blocks_count_init, ebci, total_edge_blocks_count_init);
    push_preallocate_list_to_device_queue_kernel<<< 1, 1>>>(device_vertex_block, device_edge_block, dapl, vertex_blocks_count_init, ebci, total_edge_blocks_count_init);
    cudaDeviceSynchronize();

    // sleep(5);

    // Pass raw array and its size to kernel
    unsigned long thread_blocks;
    thread_blocks = ceil(double(vertex_blocks_count_init) / 1024);
    std::cout << "Thread blocks vertex init = " << thread_blocks << std::endl;
    // vertex_dictionary_init<<< thread_blocks, 1024>>>(dvpl, dapl, vertex_blocks_count_init, d_graph_prop, vertex_size, ebci);
    vertex_dictionary_init<<< thread_blocks, 1024>>>(device_vertex_block, device_adjacency_sentinel, vertex_blocks_count_init, d_graph_prop, vertex_size, ebci);

    thread_blocks = ceil(double(h_graph_prop->xDim) / 1024);
    std::cout << "Thread blocks edge init = " << thread_blocks << std::endl;

    // sleep(5);

    // adjacency_list_init<<< thread_blocks, 1024>>>(depl, ebci, d_graph_prop, source, destination, total_edge_blocks_count_init, vertex_size, edge_size);
    adjacency_list_init<<< thread_blocks, 1024>>>(device_edge_block, ebci, d_graph_prop, source, destination, total_edge_blocks_count_init, vertex_size, edge_size);

    cudaDeviceSynchronize();

    printKernel<<< 1, 1>>>(device_vertex_block, vertex_size);
    cudaDeviceSynchronize();

    printf("xDim = %lu, yDim = %lu, Total Edges = %lu\n", h_graph_prop -> xDim, h_graph_prop -> yDim, h_graph_prop -> total_edges);

	// // Cleanup
	// cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
	// printf("%d\n", c);
	return 0;
}