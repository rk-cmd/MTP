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

#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#define THREADS_PER_BLOCK 1024
#define VERTEX_BLOCK_SIZE 150000
#define EDGE_BLOCK_SIZE 10
#define VERTEX_PREALLOCATE_LIST_SIZE 2000
#define EDGE_PREALLOCATE_LIST_SIZE 1500000

// Issues faced
// Too big of an edge_preallocate_list giving errors

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

struct adjacency_sentinel_new {

    unsigned long edge_block_count;
    unsigned long active_edge_count;
    unsigned long last_insert_edge_offset;

    struct edge_block *last_insert_edge_block;
    struct edge_block *edge_block_address[100];

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

struct vertex_dictionary_structure {

    // Structure of Array (SoA) for vertex dictionary
    unsigned long vertex_id[VERTEX_BLOCK_SIZE];
    struct adjacency_sentinel_new *vertex_adjacency[VERTEX_BLOCK_SIZE];
    unsigned long edge_block_count[VERTEX_BLOCK_SIZE];
    unsigned long active_vertex_count;

};

struct vertex_dictionary_structure_new {

    // Structure of Array (SoA) for vertex dictionary
    unsigned long vertex_id[VERTEX_BLOCK_SIZE];
    struct adjacency_sentinel_new *vertex_adjacency[VERTEX_BLOCK_SIZE];
    unsigned long edge_block_count[VERTEX_BLOCK_SIZE];
    unsigned long active_vertex_count;

};

// global variables
__device__ struct vertex_dictionary_sentinel d_v_d_sentinel;
__device__ struct vertex_preallocated_queue d_v_queue;

__device__ struct adjacency_sentinel d_a_sentinel;
__device__ struct edge_preallocated_queue d_e_queue;

// __device__ void push_to_vertex_preallocate_queue(struct vertex_block *device_vertex_block) {

//     if( (d_v_queue.rear + 1) % VERTEX_PREALLOCATE_LIST_SIZE == d_v_queue.front ) {
//         printf("Vertex queue Full, front = %ld, rear = %ld\n", d_v_queue.front, d_v_queue.rear);

//         return;
//     }
//     else if (d_v_queue.front == -1)
//         d_v_queue.front = 0;

//     d_v_queue.rear = (d_v_queue.rear + 1) % VERTEX_PREALLOCATE_LIST_SIZE;
//     d_v_queue.vertex_block_address[d_v_queue.rear] = device_vertex_block;
//     d_v_queue.count++;

//     printf("Inserted %p to the vertex queue, front = %ld, rear = %ld\n", d_v_queue.vertex_block_address[d_v_queue.rear], d_v_queue.front, d_v_queue.rear);

// }

// __device__ struct vertex_block* pop_from_vertex_preallocate_queue(unsigned long pop_count) {

//     if(d_v_queue.front == -1) {
//         printf("Vertex queue empty, front = %ld, rear = %ld\n", d_v_queue.front, d_v_queue.rear);
//         return NULL;
//     }
//     else {

//         struct vertex_block *device_vertex_block = d_v_queue.vertex_block_address[d_v_queue.front];
//         d_v_queue.vertex_block_address[d_v_queue.front] = NULL;
//         d_v_queue.count -= pop_count;
//         printf("Popped %p from the vertex queue, front = %ld, rear = %ld\n", device_vertex_block, d_v_queue.front, d_v_queue.rear);
        
//         if(d_v_queue.front == d_v_queue.rear) {

//             d_v_queue.front = -1;
//             d_v_queue.rear = -1;

//         }
//         else
//             d_v_queue.front = (d_v_queue.front + 1) % VERTEX_PREALLOCATE_LIST_SIZE;

//         return device_vertex_block;
//     }

// }

__device__ struct vertex_block* parallel_pop_from_vertex_preallocate_queue(unsigned long pop_count, unsigned long id) {

    struct vertex_block *device_vertex_block;



    if((d_v_queue.count < pop_count) || (d_v_queue.front == -1)) {
        // printf("Vertex queue empty, front = %ld, rear = %ld\n", d_v_queue.front, d_v_queue.rear);        
        return NULL;
    }

    else {

        device_vertex_block = d_v_queue.vertex_block_address[d_v_queue.front + id];
        d_v_queue.vertex_block_address[d_v_queue.front + id] = NULL;
        // printf("Popped %p from the vertex queue, placeholders front = %ld, rear = %ld\n", device_vertex_block, d_v_queue.front, d_v_queue.rear);

    }

    __syncthreads();

    if(id == 0) {

        d_v_queue.count -= pop_count;

        // printf("Vertex Queue before, front = %ld, rear = %ld\n", d_v_queue.front, d_v_queue.rear);
        
        if((d_v_queue.front + pop_count - 1) % VERTEX_PREALLOCATE_LIST_SIZE == d_v_queue.rear) {

            d_v_queue.front = -1;
            d_v_queue.rear = -1;

        }
        else
            d_v_queue.front = (d_v_queue.front + pop_count) % VERTEX_PREALLOCATE_LIST_SIZE;

        // printf("Vertex Queue before, front = %ld, rear = %ld\n", d_v_queue.front, d_v_queue.rear);

    }

    return device_vertex_block;
}

// __device__ void push_to_edge_preallocate_queue(struct edge_block *device_edge_block) {

//     if( (d_e_queue.rear + 1) % EDGE_PREALLOCATE_LIST_SIZE == d_e_queue.front ) {
//         printf("Edge queue Full, front = %ld, rear = %ld\n", d_e_queue.front, d_e_queue.rear);

//         return;
//     }
//     else if (d_e_queue.front == -1)
//         d_e_queue.front = 0;

//     d_e_queue.rear = (d_e_queue.rear + 1) % EDGE_PREALLOCATE_LIST_SIZE;
//     d_e_queue.edge_block_address[d_e_queue.rear] = device_edge_block;
//     d_e_queue.count++;

//     printf("Inserted %p to the edge queue, front = %ld, rear = %ld\n", d_e_queue.edge_block_address[d_e_queue.rear], d_e_queue.front, d_e_queue.rear);

// }

// __device__ struct edge_block* pop_from_edge_preallocate_queue(unsigned long pop_count) {

//     if(d_e_queue.front == -1) {
//         printf("Edge queue empty, front = %ld, rear = %ld\n", d_e_queue.front, d_e_queue.rear);
//         return NULL;
//     }
//     else {

//         struct edge_block *device_edge_block = d_e_queue.edge_block_address[d_e_queue.front];
//         d_e_queue.edge_block_address[d_e_queue.front] = NULL;
//         d_e_queue.count -= pop_count;
//         printf("Popped %p from the edge queue, front = %ld, rear = %ld\n", device_edge_block, d_e_queue.front, d_e_queue.rear);
        
//         if(d_e_queue.front == d_e_queue.rear) {

//             d_e_queue.front = -1;
//             d_e_queue.rear = -1;

//         }
//         else
//             d_e_queue.front = (d_e_queue.front + 1) % EDGE_PREALLOCATE_LIST_SIZE;

//         return device_edge_block;
//     }

// }

__device__ unsigned k1counter = 0;
__device__ unsigned k2counter = 0;

__device__ void parallel_pop_from_edge_preallocate_queue(struct edge_block** device_edge_block, unsigned long pop_count, unsigned long* d_prefix_sum_edge_blocks, unsigned long id, unsigned long thread_blocks) {



    if((d_e_queue.count < pop_count) || (d_e_queue.front == -1)) {
        // printf("Edge queue empty, front = %ld, rear = %ld\n", d_e_queue.front, d_e_queue.rear);        
        // return NULL;
    }

    else {

        unsigned long start_index = d_prefix_sum_edge_blocks[id - 1];
        unsigned long end_index = d_prefix_sum_edge_blocks[id];
        unsigned long j = 0;

        for(unsigned long i = start_index ; i < end_index ; i++) {

            device_edge_block[j] = d_e_queue.edge_block_address[d_e_queue.front + i];
            d_e_queue.edge_block_address[d_e_queue.front + i] = NULL;
            // printf("Popped %p from the edge queue, placeholders front = %ld, rear = %ld\n", device_edge_block[j], d_e_queue.front, d_e_queue.rear);

            j++;
        }

    }

    __syncthreads();


}

// __global__ void push_preallocate_list_to_device_queue_kernel(struct vertex_block** d_vertex_preallocate_list, struct edge_block** d_edge_preallocate_list, struct adjacency_sentinel** d_adjacency_sentinel_list, unsigned long vertex_blocks_count_init, unsigned long *edge_blocks_count_init, unsigned long total_edge_blocks_count_init) {
// __global__ void push_preallocate_list_to_device_queue_kernel(struct vertex_block* d_vertex_preallocate_list, struct edge_block* d_edge_preallocate_list, struct adjacency_sentinel** d_adjacency_sentinel_list, unsigned long vertex_blocks_count_init, unsigned long *edge_blocks_count_init, unsigned long total_edge_blocks_count_init) {

//     // some inits, don't run the below two initializations again or queue will fail
//     d_v_queue.front = -1;
//     d_v_queue.rear = -1;
//     d_e_queue.front = -1;
//     d_e_queue.rear = -1;

//     printf("Pushing vertex blocks to vertex queue\n");

//     for(unsigned long i = 0 ; i < vertex_blocks_count_init ; i++) {
    
//         // printf("%lu -> %p\n", i, d_vertex_preallocate_list[i]);
//         printf("%lu -> %p\n", i, d_vertex_preallocate_list + i);

//         // d_vertex_preallocate_list[i]->active_vertex_count = 1909;

//         // push_to_vertex_preallocate_queue(d_vertex_preallocate_list[i]);
//         push_to_vertex_preallocate_queue(d_vertex_preallocate_list + i);

//     }

//     printf("Pushing edge blocks to edge queue\n");

//     for(unsigned long i = 0 ; i < total_edge_blocks_count_init ; i++) {
    
//         // printf("%lu -> %p\n", i, d_edge_preallocate_list[i]);
//         printf("%lu -> %p\n", i, d_edge_preallocate_list + i);

//         // push_to_edge_preallocate_queue(d_edge_preallocate_list[i]);
//         push_to_edge_preallocate_queue(d_edge_preallocate_list + i);

//     }

// }

// __device__ unsigned long d_search_flag;

__global__ void data_structure_init(struct vertex_dictionary_structure *device_vertex_dictionary) {

    d_v_queue.front = -1;
    d_v_queue.rear = -1;
    d_v_queue.count = 0;
    d_e_queue.front = -1;
    d_e_queue.rear = -1;
    d_e_queue.count = 0;

    device_vertex_dictionary->active_vertex_count = 0;
    // d_search_flag = 0;

}

// __global__ void parallel_push_vertex_preallocate_list_to_device_queue(struct vertex_block* d_vertex_preallocate_list, struct adjacency_sentinel** d_adjacency_sentinel_list, unsigned long vertex_blocks_count_init) {

//     unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

//     if(id < vertex_blocks_count_init) {


//         printf("%lu -> %p\n", id, d_vertex_preallocate_list + id);


//         unsigned long free_blocks = VERTEX_PREALLOCATE_LIST_SIZE - d_v_queue.count;

//         if( (free_blocks < vertex_blocks_count_init) || (d_v_queue.rear + vertex_blocks_count_init) % VERTEX_PREALLOCATE_LIST_SIZE == d_v_queue.front ) {
//             printf("Vertex queue Full, front = %ld, rear = %ld\n", d_v_queue.front, d_v_queue.rear);

//             return;
//         }


//         d_v_queue.vertex_block_address[id] = d_vertex_preallocate_list + id;



//     }

// }

__global__ void parallel_push_edge_preallocate_list_to_device_queue(struct edge_block* d_edge_preallocate_list, struct adjacency_sentinel** d_adjacency_sentinel_list, unsigned long *edge_blocks_count_init, unsigned long total_edge_blocks_count_init) {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < total_edge_blocks_count_init) {


        // printf("%lu -> %p\n", id, d_edge_preallocate_list + id);



        unsigned long free_blocks = EDGE_PREALLOCATE_LIST_SIZE - d_e_queue.count;

        if( (free_blocks < total_edge_blocks_count_init) || (d_e_queue.rear + total_edge_blocks_count_init) % EDGE_PREALLOCATE_LIST_SIZE == d_e_queue.front ) {
            // printf("Edge queue Full, front = %ld, rear = %ld\n", d_e_queue.front, d_e_queue.rear);

            return;
        }

        d_e_queue.edge_block_address[id] = d_edge_preallocate_list + id;


    }

}

__global__ void parallel_push_edge_preallocate_list_to_device_queue_v1(struct edge_block* d_edge_preallocate_list, struct adjacency_sentinel_new** d_adjacency_sentinel_list, unsigned long *edge_blocks_count_init, unsigned long total_edge_blocks_count_init) {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < total_edge_blocks_count_init) {


        // printf("%lu -> %p\n", id, d_edge_preallocate_list + id);



        unsigned long free_blocks = EDGE_PREALLOCATE_LIST_SIZE - d_e_queue.count;

        if( (free_blocks < total_edge_blocks_count_init) || (d_e_queue.rear + total_edge_blocks_count_init) % EDGE_PREALLOCATE_LIST_SIZE == d_e_queue.front ) {
            // printf("Edge queue Full, front = %ld, rear = %ld\n", d_e_queue.front, d_e_queue.rear);

            return;
        }

        d_e_queue.edge_block_address[id] = d_edge_preallocate_list + id;


    }

}

__global__ void parallel_push_queue_update(unsigned long vertex_blocks_count_init, unsigned long total_edge_blocks_count_init) {

    if (d_v_queue.front == -1)
        d_v_queue.front = 0;

    d_v_queue.rear = (d_v_queue.rear + vertex_blocks_count_init) % VERTEX_PREALLOCATE_LIST_SIZE;
    // d_v_queue.vertex_block_address[d_v_queue.rear] = device_vertex_block;
    d_v_queue.count += vertex_blocks_count_init;

    if (d_e_queue.front == -1)
        d_e_queue.front = 0;

    d_e_queue.rear = (d_e_queue.rear + total_edge_blocks_count_init) % EDGE_PREALLOCATE_LIST_SIZE;
    // d_v_queue.vertex_block_address[d_v_queue.rear] = device_vertex_block;
    d_e_queue.count += total_edge_blocks_count_init;

}


// __global__ void vertex_dictionary_init(struct vertex_block* d_vertex_preallocate_list, struct adjacency_sentinel* d_adjacency_sentinel_list, unsigned long vertex_blocks_count_init, struct graph_properties *d_graph_prop, unsigned long size, unsigned long *edge_blocks_count_init) {
	
//     unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

//     unsigned int current;
//     __shared__ unsigned int lockvar;
//     lockvar = 0;

//     __syncthreads();

//     d_v_d_sentinel.vertex_count = 99;

//     if(id < vertex_blocks_count_init) {

//         // critical section start



//         struct vertex_block *device_vertex_block;



        

//         device_vertex_block = parallel_pop_from_vertex_preallocate_queue( vertex_blocks_count_init, id);

//         // parallel test code end

//         // assigning first vertex block to the vertex sentinel node
//         if(id == 0)
//             d_v_d_sentinel.next = device_vertex_block;

//         printf("%lu\n", id);
//         printf("---------\n");

//         printf("%lu -> %p\n", id, device_vertex_block);

//         unsigned long start_index = id * VERTEX_BLOCK_SIZE;
//         unsigned long end_index = (start_index + VERTEX_BLOCK_SIZE - 1) < size ? start_index + VERTEX_BLOCK_SIZE : size; 
//         unsigned long j = 0;

//         // device_vertex_block->active_vertex_count = 0;

//         for( unsigned long i = start_index ; i < end_index ; i++ ) {

//             device_vertex_block->vertex_id[j] = i + 1;
//             device_vertex_block->active_vertex_count++;
    
//             // device_vertex_block->vertex_adjacency[j] = d_adjacency_sentinel_list[i];
//             device_vertex_block->vertex_adjacency[j] = d_adjacency_sentinel_list + i;

//             device_vertex_block->edge_block_count[j] = edge_blocks_count_init[i];

//             printf("%lu from thread %lu, start = %lu and end = %lu\n", device_vertex_block->vertex_id[j], id, start_index, end_index);
//             j++;

//         }


//         if(id < vertex_blocks_count_init - 1)
//             (d_vertex_preallocate_list + id)->next = d_vertex_preallocate_list + id + 1;
//         else
//             (d_vertex_preallocate_list + id)->next = NULL;


//     }

// }

// __global__ void parallel_vertex_dictionary_init(struct adjacency_sentinel* d_adjacency_sentinel_list, unsigned long vertex_blocks_count_init, struct graph_properties *d_graph_prop, unsigned long vertex_size, unsigned long *edge_blocks_count_init, struct vertex_dictionary_structure *device_vertex_dictionary) {
	
//     unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

//     if(id < vertex_size) {



//         if(id == 0)
//             device_vertex_dictionary->active_vertex_count += vertex_size;

//         device_vertex_dictionary->vertex_id[id] = id + 1;
//         // device_vertex_dictionary->active_vertex_count++;
//         device_vertex_dictionary->vertex_adjacency[id] = d_adjacency_sentinel_list + id;
//         device_vertex_dictionary->edge_block_count[id] = edge_blocks_count_init[id];




//     }

// }

__global__ void parallel_vertex_dictionary_init_v1(struct adjacency_sentinel_new* d_adjacency_sentinel_list, unsigned long vertex_blocks_count_init, struct graph_properties *d_graph_prop, unsigned long vertex_size, unsigned long *edge_blocks_count_init, struct vertex_dictionary_structure *device_vertex_dictionary) {
	
    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < vertex_size) {



        if(id == 0)
            device_vertex_dictionary->active_vertex_count += vertex_size;

        device_vertex_dictionary->vertex_id[id] = id + 1;
        // device_vertex_dictionary->active_vertex_count++;
        device_vertex_dictionary->vertex_adjacency[id] = d_adjacency_sentinel_list + id;
        device_vertex_dictionary->edge_block_count[id] = edge_blocks_count_init[id];




    }

    // __syncthreads();

    // if(id == 0)
    //     printf("Checkpoint VD\n");

}

__device__ unsigned int lockvar = 0;


// __global__ void adjacency_list_init(struct edge_block** d_edge_preallocate_list, unsigned long *d_edge_blocks_count_init, struct graph_properties *d_graph_prop, unsigned long *d_source, unsigned long *d_destination, unsigned long total_edge_blocks_count_init, unsigned long vertex_size, unsigned long edge_size) {
// __global__ void adjacency_list_init(struct edge_block* d_edge_preallocate_list, unsigned long *d_edge_blocks_count_init, struct graph_properties *d_graph_prop, unsigned long *d_source, unsigned long *d_destination, unsigned long total_edge_blocks_count_init, unsigned long vertex_size, unsigned long edge_size, unsigned long *d_prefix_sum_edge_blocks, unsigned long thread_blocks) {

	
//     unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

//     unsigned int current;
//     // __shared__ unsigned int lockvar;
//     // lockvar = 0;

//     __syncthreads();

//     // printf("K1 counter = %u\n", atomicAdd((unsigned *)&k1counter, 1));

//     if(id < vertex_size) {

//         printf("Prefix sum of %ld is %ld\n", id, d_prefix_sum_edge_blocks[id]);

//         unsigned long current_vertex = id + 1;
//         unsigned long index = 0;
//         struct vertex_block *ptr = d_v_d_sentinel.next;

//         // locating vertex in the vertex dictionary
//         while(ptr != NULL) {

//             unsigned long flag = 1;

//             for(unsigned long i = 0 ; i < VERTEX_BLOCK_SIZE ; i++) {

//                 if(ptr->vertex_id[i] == current_vertex) {
//                     index = i;
//                     flag = 0;
//                     printf("Matched at %lu\n", index);
//                     break;
//                 }

//             }

//             if(flag)
//                 ptr = ptr->next;
//             else
//                 break;
                

//         }

//         if(ptr != NULL)
//             printf("ID = %lu, Vertex = %lu, Adjacency Sentinel = %p and edge blocks = %lu\n", id, ptr->vertex_id[index], ptr->vertex_adjacency[index], ptr->edge_block_count[index]);

//         unsigned long edge_blocks_required = ptr->edge_block_count[index];

//         // critical section start

//         // temporary fix, this can't be a constant sized one
//         struct edge_block *device_edge_block[100];


//         // for(unsigned long i = 0 ; i < edge_blocks_required ; i++) {
//         parallel_pop_from_edge_preallocate_queue( device_edge_block, total_edge_blocks_count_init, d_prefix_sum_edge_blocks, id, thread_blocks);

//         // critical section end
//         if(threadIdx.x == 0)
//             printf("ID\tIteration\tGPU address\n");
        
//         for(unsigned long i = 0 ; i < edge_blocks_required ; i++)
//             printf("%lu\t%lu\t\t%p\n", id, i, device_edge_block[i]);

//         // adding the edge blocks
//         struct adjacency_sentinel *vertex_adjacency = ptr->vertex_adjacency[index];

//         if(edge_blocks_required > 0) {

//             struct edge_block *prev, *curr;

//             prev = NULL;
//             curr = NULL;


//             for(unsigned long i = 0 ; i < edge_blocks_required ; i++) {

//                 curr = device_edge_block[i];

//                 if(prev != NULL)
//                     prev->next = curr;

//                 prev = curr;
                
//             }

//             if(edge_blocks_required > 0) {
//                 vertex_adjacency->next = device_edge_block[0];
//                 curr->next = NULL;
//             }

//             unsigned long edge_block_entry_count = 0;
//             unsigned long edge_block_counter = 0;

//             curr = vertex_adjacency->next;
//             vertex_adjacency->active_edge_count = 0;

//             for(unsigned long i = 0 ; i < edge_size ; i++) {

//                 if(d_source[i] == current_vertex){

//                     // printf("Source = %lu and Destination = %lu\n", d_source[i], d_destination[i]);

//                     // insert here
//                     curr->edge_block_entry[edge_block_entry_count].destination_vertex = d_destination[i];
//                     vertex_adjacency->active_edge_count++;
                    
//                     if(++edge_block_entry_count >= EDGE_BLOCK_SIZE) {

//                         curr = curr->next;
//                         edge_block_counter++;
//                         edge_block_entry_count = 0;
//                     }

//                 }

//             }

//         }

//     }

// }

// __global__ void adjacency_list_init_modded(struct edge_block* d_edge_preallocate_list, unsigned long *d_edge_blocks_count_init, struct graph_properties *d_graph_prop, unsigned long *d_source, unsigned long *d_destination, unsigned long total_edge_blocks_count_init, unsigned long vertex_size, unsigned long edge_size, unsigned long *d_prefix_sum_edge_blocks, unsigned long thread_blocks, struct vertex_dictionary_structure *device_vertex_dictionary) {
	
//     unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

//     unsigned int current;
//     // __shared__ unsigned int lockvar;
//     // lockvar = 0;

//     __syncthreads();

//     // printf("K1 counter = %u\n", atomicAdd((unsigned *)&k1counter, 1));

//     if(id < vertex_size) {

//         printf("Prefix sum of %ld is %ld\n", id, d_prefix_sum_edge_blocks[id]);

//         unsigned long current_vertex = id + 1;

//         unsigned long edge_blocks_required = device_vertex_dictionary->edge_block_count[id];

//         // critical section start

//         // temporary fix, this can't be a constant sized one
//         struct edge_block *device_edge_block[100];


//         parallel_pop_from_edge_preallocate_queue( device_edge_block, total_edge_blocks_count_init, d_prefix_sum_edge_blocks, id, thread_blocks);

//         // critical section end
//         if(threadIdx.x == 0)
//             printf("ID\tIteration\tGPU address\n");
        
//         for(unsigned long i = 0 ; i < edge_blocks_required ; i++)
//             printf("%lu\t%lu\t\t%p\n", id, i, device_edge_block[i]);

//         // adding the edge blocks
//         // struct adjacency_sentinel *vertex_adjacency = ptr->vertex_adjacency[index];
//         struct adjacency_sentinel *vertex_adjacency = device_vertex_dictionary->vertex_adjacency[id];

//         if(edge_blocks_required > 0) {

//             struct edge_block *prev, *curr;

//             prev = NULL;
//             curr = NULL;


//             for(unsigned long i = 0 ; i < edge_blocks_required ; i++) {

//                 curr = device_edge_block[i];

//                 if(prev != NULL)
//                     prev->next = curr;

//                 prev = curr;
                
//             }

//             if(edge_blocks_required > 0) {
//                 vertex_adjacency->next = device_edge_block[0];
//                 curr->next = NULL;
//             }

//             unsigned long edge_block_entry_count = 0;
//             unsigned long edge_block_counter = 0;

//             curr = vertex_adjacency->next;
//             vertex_adjacency->active_edge_count = 0;

//             for(unsigned long i = 0 ; i < edge_size ; i++) {

//                 if(d_source[i] == current_vertex){

//                     // printf("Source = %lu and Destination = %lu\n", d_source[i], d_destination[i]);

//                     // insert here
//                     curr->edge_block_entry[edge_block_entry_count].destination_vertex = d_destination[i];
//                     vertex_adjacency->active_edge_count++;
                    
//                     if(++edge_block_entry_count >= EDGE_BLOCK_SIZE) {

//                         curr = curr->next;
//                         edge_block_counter++;
//                         edge_block_entry_count = 0;
//                     }

//                 }

//             }

//         }

//     }

// }

// __global__ void adjacency_list_init_modded_v1(struct edge_block* d_edge_preallocate_list, unsigned long *d_edge_blocks_count_init, struct graph_properties *d_graph_prop, unsigned long *d_source, unsigned long *d_destination, unsigned long total_edge_blocks_count_init, unsigned long vertex_size, unsigned long edge_size, unsigned long *d_prefix_sum_edge_blocks, unsigned long *d_prefix_sum_vertex_degrees, unsigned long thread_blocks, struct vertex_dictionary_structure *device_vertex_dictionary) {
	
//     unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

//     unsigned int current;
//     // __shared__ unsigned int lockvar;
//     // lockvar = 0;

//     __syncthreads();


//     if(id < vertex_size) {

//         printf("Prefix sum of %ld is %ld\n", id, d_prefix_sum_edge_blocks[id]);

//         unsigned long current_vertex = id + 1;

//         // unsigned long edge_blocks_required = ptr->edge_block_count[index];
//         unsigned long edge_blocks_required = device_vertex_dictionary->edge_block_count[id];

//         // critical section start

//         // temporary fix, this can't be a constant sized one
//         struct edge_block *device_edge_block[100];




//         // for(unsigned long i = 0 ; i < edge_blocks_required ; i++) {
//         parallel_pop_from_edge_preallocate_queue( device_edge_block, total_edge_blocks_count_init, d_prefix_sum_edge_blocks, id, thread_blocks);


//         // critical section end
//         if(threadIdx.x == 0)
//             printf("ID\tIteration\tGPU address\n");
        
//         for(unsigned long i = 0 ; i < edge_blocks_required ; i++)
//             printf("%lu\t%lu\t\t%p\n", id, i, device_edge_block[i]);

//         // adding the edge blocks
//         // struct adjacency_sentinel *vertex_adjacency = ptr->vertex_adjacency[index];
//         struct adjacency_sentinel_new *vertex_adjacency = device_vertex_dictionary->vertex_adjacency[id];

//         if(edge_blocks_required > 0) {

//             struct edge_block *prev, *curr;

//             prev = NULL;
//             curr = NULL;


//             for(unsigned long i = 0 ; i < edge_blocks_required ; i++) {

//                 curr = device_edge_block[i];

//                 if(prev != NULL)
//                     prev->next = curr;

//                 prev = curr;
                
//             }

//             if(edge_blocks_required > 0) {
//                 vertex_adjacency->next = device_edge_block[0];
//                 curr->next = NULL;
//             }

//             unsigned long edge_block_entry_count = 0;
//             unsigned long edge_block_counter = 0;

//             curr = vertex_adjacency->next;
//             vertex_adjacency->active_edge_count = 0;

//             unsigned long start_index;

//             if(current_vertex != 1)
//                 start_index = d_prefix_sum_vertex_degrees[current_vertex - 2];
//             else
//                 start_index = 0;

//             unsigned long end_index = d_prefix_sum_vertex_degrees[current_vertex - 1];

//             printf("Current vertex = %lu, start = %lu, end = %lu\n", current_vertex, start_index, end_index);



//             for(unsigned long i = start_index ; i < end_index ; i++) {

//                 // if(d_source[i] == current_vertex){

//                     // printf("Source = %lu and Destination = %lu\n", d_source[i], d_destination[i]);

//                     // insert here
//                     curr->edge_block_entry[edge_block_entry_count].destination_vertex = d_destination[i];
//                     vertex_adjacency->active_edge_count++;
                    
//                     if(++edge_block_entry_count >= EDGE_BLOCK_SIZE) {

//                         curr = curr->next;
//                         edge_block_counter++;
//                         edge_block_entry_count = 0;
//                     }

//                 // }

//             }

//         }

//     }

// }

__global__ void adjacency_list_init_modded_v2(struct edge_block* d_edge_preallocate_list, unsigned long *d_edge_blocks_count_init, struct graph_properties *d_graph_prop, unsigned long *d_source, unsigned long *d_destination, unsigned long total_edge_blocks_count_init, unsigned long vertex_size, unsigned long edge_size, unsigned long *d_prefix_sum_edge_blocks, unsigned long *d_prefix_sum_vertex_degrees, unsigned long thread_blocks, struct vertex_dictionary_structure *device_vertex_dictionary) {
	
    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int current;
    // __shared__ unsigned int lockvar;
    // lockvar = 0;

    __syncthreads();


    if(id < vertex_size) {

        // printf("Prefix sum of %ld is %ld\n", id, d_prefix_sum_edge_blocks[id]);

        unsigned long current_vertex = id + 1;

        // unsigned long edge_blocks_required = ptr->edge_block_count[index];
        unsigned long edge_blocks_required = device_vertex_dictionary->edge_block_count[id];

        // critical section start

        // temporary fix, this can't be a constant sized one
        struct edge_block *device_edge_block[100];




        // for(unsigned long i = 0 ; i < edge_blocks_required ; i++) {
        parallel_pop_from_edge_preallocate_queue( device_edge_block, total_edge_blocks_count_init, d_prefix_sum_edge_blocks, id, thread_blocks);


        // critical section end
        // if(threadIdx.x == 0)
        //     printf("ID\tIteration\tGPU address\n");
        
        // for(unsigned long i = 0 ; i < edge_blocks_required ; i++)
        //     printf("%lu\t%lu\t\t%p\n", id, i, device_edge_block[i]);

        // adding the edge blocks
        // struct adjacency_sentinel *vertex_adjacency = ptr->vertex_adjacency[index];
        struct adjacency_sentinel_new *vertex_adjacency = device_vertex_dictionary->vertex_adjacency[id];

        if(edge_blocks_required > 0) {

            // one thread inserts sequentially all destination vertices of a source vertex.

            struct edge_block *prev, *curr;

            // prev = NULL;
            // curr = NULL;


            // for(unsigned long i = 0 ; i < edge_blocks_required ; i++) {

            //     curr = device_edge_block[i];

            //     if(prev != NULL)
            //         prev->next = curr;

            //     prev = curr;
                
            // }

            // if(edge_blocks_required > 0) {
            //     vertex_adjacency->next = device_edge_block[0];
            //     curr->next = NULL;
            // }

            unsigned long edge_block_entry_count = 0;
            unsigned long edge_block_counter = 0;

            // curr = vertex_adjacency->next;
            vertex_adjacency->active_edge_count = 0;

            unsigned long start_index;

            if(current_vertex != 1)
                start_index = d_prefix_sum_vertex_degrees[current_vertex - 2];
            else
                start_index = 0;

            unsigned long end_index = d_prefix_sum_vertex_degrees[current_vertex - 1];

            // printf("Current vertex = %lu, start = %lu, end = %lu\n", current_vertex, start_index, end_index);

            // unsigned long current_edge_block = 0;
            curr = device_edge_block[edge_block_counter];
            vertex_adjacency->edge_block_address[edge_block_counter] = curr;

            // __syncthreads();

            if(id == 0)
                printf("Checkpoint AL beg\n");

            for(unsigned long i = start_index ; i < end_index ; i++) {

                // if(d_source[i] == current_vertex){

                    // printf("Source = %lu and Destination = %lu\n", d_source[i], d_destination[i]);

                    // insert here
                    curr->edge_block_entry[edge_block_entry_count].destination_vertex = d_destination[i];
                    vertex_adjacency->active_edge_count++;
                    
                    if(++edge_block_entry_count >= EDGE_BLOCK_SIZE) {

                        // curr = curr->next;
                        // edge_block_counter++;
                        edge_block_entry_count = 0;
                        curr = device_edge_block[++edge_block_counter];
                        vertex_adjacency->edge_block_address[edge_block_counter] = curr;

                    }

                // }

            }

        }

    }

    // __syncthreads();

    // if(id == 0)
    //     printf("Checkpoint AL\n");

}

__global__ void update_edge_queue(unsigned long pop_count) {


    d_e_queue.count -= pop_count;

    printf("Edge Queue before, front = %ld, rear = %ld\n", d_e_queue.front, d_e_queue.rear);
    
    if((d_e_queue.front + pop_count - 1) % EDGE_PREALLOCATE_LIST_SIZE == d_e_queue.rear) {

        d_e_queue.front = -1;
        d_e_queue.rear = -1;

    }
    else
        d_e_queue.front = (d_e_queue.front + pop_count) % EDGE_PREALLOCATE_LIST_SIZE;

    printf("Edge Queue before, front = %ld, rear = %ld\n", d_e_queue.front, d_e_queue.rear);


}

__global__ void search_edge_kernel(struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long vertex_size, unsigned long total_search_threads, unsigned long search_source, unsigned long search_destination, unsigned long *d_search_flag) {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < total_search_threads) {

        unsigned long edge_block = id / EDGE_BLOCK_SIZE;
        unsigned long edge_block_entry = id % EDGE_BLOCK_SIZE;

        // printf("Search kernel launch test for source %lu and destination %lu, edge_block %lu and entry %lu\n", search_source, search_destination, edge_block, edge_block_entry);

        struct adjacency_sentinel_new *edge_sentinel = device_vertex_dictionary->vertex_adjacency[search_source - 1];
        unsigned long extracted_value;
        extracted_value = edge_sentinel->edge_block_address[edge_block]->edge_block_entry[edge_block_entry].destination_vertex;

        if(extracted_value == search_destination) {
            *d_search_flag = 1;
            printf("Edge exists\n");
        }

    }

}

// __global__ void printKernel(struct vertex_block** d_vertex_preallocate_list, unsigned long size) {
// __global__ void printKernel(struct vertex_block* d_vertex_preallocate_list, unsigned long size) {

//     printf("Printing Linked List\n");

//     struct vertex_block *ptr = d_v_d_sentinel.next;

//     unsigned long vertex_block = 0;

//     while(ptr != NULL) {

//         printf("Vertex Block = %lu\n", vertex_block);

//         for(unsigned long i = 0 ; i < VERTEX_BLOCK_SIZE ; i++) {

//             // printf("%lu -> , edge blocks = %lu, edge sentinel = %p, ", ptr->vertex_id[i], ptr->edge_block_count[i], ptr->vertex_adjacency[i]);

//             if((ptr->vertex_adjacency[i] != NULL) && (ptr->vertex_id[i] != 0)) {
//                 printf("%lu -> , edge blocks = %lu, edge sentinel = %p, active edge count = %lu, destination vertices -> ", ptr->vertex_id[i], ptr->edge_block_count[i], ptr->vertex_adjacency[i], ptr->vertex_adjacency[i]->active_edge_count);
            
//                 struct edge_block *itr = ptr->vertex_adjacency[i]->next;
//                 unsigned long edge_block_entry_count = 0;

//                 for(unsigned long j = 0 ; j < ptr->vertex_adjacency[i]->active_edge_count ; j++) {
                                
//                     printf("%lu ", itr->edge_block_entry[edge_block_entry_count].destination_vertex);
                
//                     // edge_block_entry_count++;

//                     if(++edge_block_entry_count >= EDGE_BLOCK_SIZE) {
//                         itr = itr->next;
//                         edge_block_entry_count = 0;
//                     }
//                 }
//                 printf("\n");            
            
//             }


//         }

//         ptr = ptr->next;
//         vertex_block++;
//         printf("\n");

//     }

//     printf("VDS = %lu\n", d_v_d_sentinel.vertex_count);
//     printf("K2 counter = %u\n", k2counter);

// }

// __global__ void printKernelmodded(struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long size) {

//     printf("Printing Linked List\n");

//     // struct vertex_block *ptr = d_v_d_sentinel.next;

//     unsigned long vertex_block = 0;

//     for(unsigned long i = 0 ; i < device_vertex_dictionary->active_vertex_count ; i++) {

//         // printf("Checkpoint\n");

//         if((device_vertex_dictionary->vertex_adjacency[i] != NULL) && (device_vertex_dictionary->vertex_id[i] != 0)) {
//             printf("%lu -> , edge blocks = %lu, edge sentinel = %p, active edge count = %lu, destination vertices -> ", device_vertex_dictionary->vertex_id[i], device_vertex_dictionary->edge_block_count[i], device_vertex_dictionary->vertex_adjacency[i], device_vertex_dictionary->vertex_adjacency[i]->active_edge_count);

//             struct edge_block *itr = device_vertex_dictionary->vertex_adjacency[i]->next;
//             unsigned long edge_block_entry_count = 0;

//             for(unsigned long j = 0 ; j < device_vertex_dictionary->vertex_adjacency[i]->active_edge_count ; j++) {
                            
//                 printf("%lu ", itr->edge_block_entry[edge_block_entry_count].destination_vertex);
            
//                 // edge_block_entry_count++;

//                 if(++edge_block_entry_count >= EDGE_BLOCK_SIZE) {
//                     itr = itr->next;
//                     edge_block_entry_count = 0;
//                 }
//             }
//             printf("\n");            
    
//         }



//     }


//     printf("VDS = %lu\n", d_v_d_sentinel.vertex_count);
//     printf("K2 counter = %u\n", k2counter);

// }

__global__ void printKernelmodded_v1(struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long size) {

    printf("Printing Linked List\n");

    // struct vertex_block *ptr = d_v_d_sentinel.next;

    unsigned long vertex_block = 0;

    for(unsigned long i = 0 ; i < device_vertex_dictionary->active_vertex_count ; i++) {

        // printf("Checkpoint\n");

        if((device_vertex_dictionary->vertex_adjacency[i] != NULL) && (device_vertex_dictionary->vertex_id[i] != 0)) {
            printf("%lu -> , edge blocks = %lu, edge sentinel = %p, active edge count = %lu, destination vertices -> ", device_vertex_dictionary->vertex_id[i], device_vertex_dictionary->edge_block_count[i], device_vertex_dictionary->vertex_adjacency[i], device_vertex_dictionary->vertex_adjacency[i]->active_edge_count);

            unsigned long edge_block_counter = 0;
            unsigned long edge_block_entry_count = 0;
            struct edge_block *itr = device_vertex_dictionary->vertex_adjacency[i]->edge_block_address[edge_block_counter];

            for(unsigned long j = 0 ; j < device_vertex_dictionary->vertex_adjacency[i]->active_edge_count ; j++) {
                            
                printf("%lu ", itr->edge_block_entry[edge_block_entry_count].destination_vertex);
            
                // edge_block_entry_count++;

                if(++edge_block_entry_count >= EDGE_BLOCK_SIZE) {
                    // itr = itr->next;
                    itr = device_vertex_dictionary->vertex_adjacency[i]->edge_block_address[++edge_block_counter];
                    edge_block_entry_count = 0;
                }
            }
            printf("\n");            
    
        }



    }


    printf("VDS = %lu\n", d_v_d_sentinel.vertex_count);
    printf("K2 counter = %u\n", k2counter);

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

            // below part makes it an undirected graph
            h_source[index] = destination;
            h_destination[index++] = source;

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
            // total edges doubles since undirected graph
            h_graph_prop->total_edges = total_edges * 2;

            h_source.resize(h_graph_prop->total_edges);
            h_destination.resize(h_graph_prop->total_edges);

            dataFlag = 1;

        }

    }
    
    fclose(ptr);

}

int main(void) {

    // char fileLoc[20] = "input.mtx";
    // char fileLoc[30] = "inf-luxembourg_osm.mtx";
    // char fileLoc[30] = "rgg_n_2_16_s0.mtx";
    char fileLoc[30] = "delaunay_n17.mtx";
    // char fileLoc[30] = "fe-ocean.mtx";
    // char fileLoc[30] = "kron_g500-logn16.mtx";

    // some random inits
    unsigned long choice = 1;
    printf("Please enter structure of edge blocks\n1. Unsorted\n2. Sorted\n");
    scanf("%lu", &choice);

    clock_t section1, section2, section2a, section3, search_times;

    struct graph_properties *h_graph_prop = (struct graph_properties*)malloc(sizeof(struct graph_properties));
    thrust::host_vector <unsigned long> h_source(1);
    thrust::host_vector <unsigned long> h_destination(1);
    thrust::host_vector <unsigned long> h_source_degree(1);
    thrust::host_vector <unsigned long> h_prefix_sum_vertex_degrees(1);
    thrust::host_vector <unsigned long> h_prefix_sum_edge_blocks(1);

    // reading file, after function call h_source has data on source vertex and h_destination on destination vertex
    // both represent the edge data on host
    section1 = clock();
    readFile(fileLoc, h_graph_prop, h_source, h_destination);
    section1 = clock() - section1;

    unsigned long vertex_size = h_graph_prop->xDim;
    unsigned long edge_size = h_graph_prop->total_edges;

    h_source_degree.resize(h_graph_prop->xDim);
    h_prefix_sum_vertex_degrees.resize(vertex_size);
    h_prefix_sum_edge_blocks.resize(h_graph_prop->xDim);
    thrust::fill(h_source_degree.begin(), h_source_degree.end(), 0);

    // std::cout << "Check, " << h_source.size() << " and " << h_destination.size() << std::endl;

    // sleep(5);

    section2 = clock();

    for(unsigned long i = 0 ; i < h_graph_prop->total_edges ; i++) {
     
        // printf("%lu and %lu\n", h_source[i], h_destination[i]);
        h_source_degree[h_source[i] - 1]++;
    
    }

    section2 = clock() - section2;

    // std::cout << std::endl << "After sorting" << std::endl << std::endl;

    // sort test start

    section2a = clock();

    unsigned long* h_source_ptr = thrust::raw_pointer_cast(h_source.data());
    unsigned long* h_destination_ptr = thrust::raw_pointer_cast(h_destination.data());
    // thrust::sort_by_key(thrust::host, thrust::host_ptr<unsigned long> (h_source), thrust::host_ptr<unsigned long> (h_source + edge_size), thrust::host_ptr<unsigned long> (h_destination));
    thrust::sort_by_key(thrust::host, thrust::raw_pointer_cast(h_source.data()), thrust::raw_pointer_cast(h_source.data()) + edge_size, thrust::raw_pointer_cast(h_destination.data()));



    if(choice == 2) {

        unsigned long current = h_source[0];
        unsigned long start_index = 0, end_index = 0;
        for(unsigned long i = 0 ; i < 25 ; i++) {

            if((current != h_source[i])) {

                // std::cout << "start_index is " << start_index << " and end_index is " << end_index << std::endl;
                thrust::sort_by_key(thrust::host, h_destination_ptr + start_index, h_destination_ptr + end_index, h_source_ptr + start_index);
                start_index = end_index;
                current = h_source[i];

            }
            
            end_index++;

        }

        // std::cout << "start_index is " << start_index << " and end_index is " << end_index << std::endl;
        thrust::sort_by_key(thrust::host, h_destination_ptr + start_index, h_destination_ptr + end_index, h_source_ptr + start_index);

    }

    section2a = clock() - section2a;

    // after sort
    // for(unsigned long i = 0 ; i < h_graph_prop->total_edges ; i++) {
     
    //     printf("%lu and %lu\n", h_source[i], h_destination[i]);
    //     // h_source_degree[h_source[i] - 1]++;
    
    // }
    // printf("After final sort\n");

    // for(unsigned long i = 0 ; i < h_graph_prop->total_edges ; i++) {
     
    //     printf("%lu and %lu\n", h_source[i], h_destination[i]);
    //     // h_source_degree[h_source[i] - 1]++;
    
    // }    

    // sort test end

    section3 = clock();

    unsigned long zero_count = 0;
    // std::cout << "Printing degree of each source vertex" << std::endl;
    for(unsigned long i = 0 ; i < h_graph_prop->xDim ; i++) {
        // printf("%lu ", h_source_degree[i]);
        if(h_source_degree[i] == 0)
            zero_count++;
    }
    // sleep(15);
    printf("zero count is %lu\n", zero_count);

    h_prefix_sum_vertex_degrees[0] = h_source_degree[0];
    // printf("Prefix sum array vertex degrees\n%ld ", h_prefix_sum_vertex_degrees[0]);
    for(unsigned long i = 1 ; i < h_graph_prop->xDim ; i++) {

        h_prefix_sum_vertex_degrees[i] += h_prefix_sum_vertex_degrees[i-1] + h_source_degree[i];
        // printf("%ld ", h_prefix_sum_vertex_degrees[i]);

    }
    // printf("\n");
    

    // sleep(5);

    thrust::host_vector <unsigned long> ::iterator iter = thrust::max_element(h_source.begin(), h_source.end());

    // printf("xDim = %lu, yDim = %lu, Total Edges = %lu\n", h_graph_prop -> xDim, h_graph_prop -> yDim, h_graph_prop -> total_edges);
    // std::cout << "Check, " << h_source.size() << " and " << h_destination.size() << std::endl;
    // printf("Max element is %lu\n", *iter);

    // vertex block code start

    unsigned long vertex_blocks_count_init = ceil( double(h_graph_prop -> xDim) / VERTEX_BLOCK_SIZE);

    printf("Vertex blocks needed = %lu\n", vertex_blocks_count_init);

    struct vertex_dictionary_sentinel *device_vertex_sentinel;
    cudaMalloc(&device_vertex_sentinel, sizeof(struct vertex_dictionary_sentinel));

    // thrust::host_vector <struct vertex_block *> h_vertex_preallocate_list(VERTEX_PREALLOCATE_LIST_SIZE);
    thrust::host_vector <struct edge_block *> h_edge_preallocate_list(EDGE_PREALLOCATE_LIST_SIZE);
    // thrust::host_vector <struct adjacency_sentinel *> h_adjacency_sentinel_list(VERTEX_PREALLOCATE_LIST_SIZE);
    thrust::host_vector <struct adjacency_sentinel_new *> h_adjacency_sentinel_list(VERTEX_PREALLOCATE_LIST_SIZE);
    
    
    struct graph_properties *d_graph_prop;
    cudaMalloc(&d_graph_prop, sizeof(struct graph_properties));
	cudaMemcpy(d_graph_prop, &h_graph_prop, sizeof(struct graph_properties), cudaMemcpyHostToDevice);

    // Individual mallocs or one single big malloc to reduce overhead??
    // std::cout << "Checkpoint" << std::endl;

    // for(unsigned long i = 0 ; i < vertex_blocks_count_init ; i++) {

    //     struct vertex_block *device_vertex_block;
    //     cudaMalloc(&device_vertex_block, sizeof(struct vertex_block));
    //     h_vertex_preallocate_list[i] = device_vertex_block;     

    // }

    // struct vertex_block *device_vertex_block;
    // cudaMalloc((struct vertex_block**)&device_vertex_block, vertex_blocks_count_init * sizeof(struct vertex_block));

    // allocate only necessary vertex sentinel nodes, running out of memory

    // for(unsigned long i = 0 ; i < vertex_size ; i++) {
    //     std::cout << i << std::endl;
    //     struct adjacency_sentinel *device_adjacency_sentinel;
    //     cudaMalloc(&device_adjacency_sentinel, sizeof(struct adjacency_sentinel));
    //     h_adjacency_sentinel_list[i] = device_adjacency_sentinel;

    // }

    struct vertex_dictionary_structure *device_vertex_dictionary;
    cudaMalloc(&device_vertex_dictionary, sizeof(struct vertex_dictionary_structure));

    // struct adjacency_sentinel *device_adjacency_sentinel;
    // cudaMalloc((struct adjacency_sentinel**)&device_adjacency_sentinel, vertex_size * sizeof(struct adjacency_sentinel));
    struct adjacency_sentinel_new *device_adjacency_sentinel;
    cudaMalloc((struct adjacency_sentinel_new**)&device_adjacency_sentinel, vertex_size * sizeof(struct adjacency_sentinel_new));

    cudaDeviceSynchronize();

    // std::cout << std::endl << "Host side" << std::endl << "Vertex blocks calculation" << std::endl << "Vertex block\t" << "GPU address" << std::endl;
    // for(unsigned long i = 0 ; i < vertex_blocks_count_init ; i++)
    //     // printf("%lu\t\t%p\n", i, h_vertex_preallocate_list[i]);  
    //     printf("%lu\t\t%p\n", i, device_vertex_block + i);  
    
    thrust::host_vector <unsigned long> h_edge_blocks_count_init(h_graph_prop->xDim);
    
    // sleep(5);

    // unsigned long k = 0;
    unsigned long total_edge_blocks_count_init = 0;
    // std::cout << "Edge blocks calculation" << std::endl << "Source\tEdge block count\tGPU address" << std::endl;
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

    h_prefix_sum_edge_blocks[0] = h_edge_blocks_count_init[0];
    // printf("Prefix sum array edge blocks\n%ld ", h_prefix_sum_edge_blocks[0]);
    for(unsigned long i = 1 ; i < h_graph_prop->xDim ; i++) {

        h_prefix_sum_edge_blocks[i] += h_prefix_sum_edge_blocks[i-1] + h_edge_blocks_count_init[i];
        // printf("%ld ", h_prefix_sum_edge_blocks[i]);

    }
    // printf("\n");

    struct edge_block *device_edge_block;
    cudaMalloc((struct edge_block**)&device_edge_block, total_edge_blocks_count_init * sizeof(struct edge_block));

    // thrust::device_vector <struct vertex_block *> d_vertex_preallocate_list(VERTEX_PREALLOCATE_LIST_SIZE);
    thrust::device_vector <struct edge_block *> d_edge_preallocate_list(EDGE_PREALLOCATE_LIST_SIZE);
    // thrust::device_vector <struct adjacency_sentinel *> d_adjacency_sentinel_list(VERTEX_PREALLOCATE_LIST_SIZE);
    thrust::device_vector <struct adjacency_sentinel_new *> d_adjacency_sentinel_list(VERTEX_PREALLOCATE_LIST_SIZE);
    thrust::device_vector <unsigned long> d_source(h_graph_prop->total_edges);
    thrust::device_vector <unsigned long> d_destination(h_graph_prop->total_edges);
    thrust::device_vector <unsigned long> d_edge_blocks_count_init(h_graph_prop->xDim);
    thrust::device_vector <unsigned long> d_prefix_sum_edge_blocks(h_graph_prop->xDim);
    thrust::device_vector <unsigned long> d_prefix_sum_vertex_degrees(vertex_size);
  
    // thrust::copy(h_vertex_preallocate_list.begin(), h_vertex_preallocate_list.end(), d_vertex_preallocate_list.begin());
    // thrust::copy(h_edge_preallocate_list.begin(), h_edge_preallocate_list.end(), d_edge_preallocate_list.begin());
    thrust::copy(h_adjacency_sentinel_list.begin(), h_adjacency_sentinel_list.end(), d_adjacency_sentinel_list.begin());
    thrust::copy(h_source.begin(), h_source.end(), d_source.begin());
    thrust::copy(h_destination.begin(), h_destination.end(), d_destination.begin());
    thrust::copy(h_edge_blocks_count_init.begin(), h_edge_blocks_count_init.end(), d_edge_blocks_count_init.begin());
    thrust::copy(h_prefix_sum_edge_blocks.begin(), h_prefix_sum_edge_blocks.end(), d_prefix_sum_edge_blocks.begin());
    thrust::copy(h_prefix_sum_vertex_degrees.begin(), h_prefix_sum_vertex_degrees.end(), d_prefix_sum_vertex_degrees.begin());

    // struct vertex_block** dvpl = thrust::raw_pointer_cast(d_vertex_preallocate_list.data());
    // struct edge_block** depl = thrust::raw_pointer_cast(d_edge_preallocate_list.data());
    // struct adjacency_sentinel** dapl = thrust::raw_pointer_cast(d_adjacency_sentinel_list.data());
    struct adjacency_sentinel_new** dapl = thrust::raw_pointer_cast(d_adjacency_sentinel_list.data());
    unsigned long* source = thrust::raw_pointer_cast(d_source.data());
    unsigned long* destination = thrust::raw_pointer_cast(d_destination.data());
    unsigned long* ebci = thrust::raw_pointer_cast(d_edge_blocks_count_init.data());
    unsigned long* pseb = thrust::raw_pointer_cast(d_prefix_sum_edge_blocks.data());
    unsigned long* psvd = thrust::raw_pointer_cast(d_prefix_sum_vertex_degrees.data());

    section3 = clock() - section3;

    clock_t total_time, time_req, push_to_queues_time, vd_time, al_time;
    time_req = clock();

    printf("GPU side\n");

    unsigned long thread_blocks;


    // Parallelize this
    // push_preallocate_list_to_device_queue_kernel<<< 1, 1>>>(device_vertex_block, device_edge_block, dapl, vertex_blocks_count_init, ebci, total_edge_blocks_count_init);
    
    thread_blocks = ceil(double(vertex_blocks_count_init) / THREADS_PER_BLOCK);    

    push_to_queues_time = clock();

    data_structure_init<<< 1, 1>>>(device_vertex_dictionary);
    // parallel_push_vertex_preallocate_list_to_device_queue<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_block, dapl, vertex_blocks_count_init);

    thread_blocks = ceil(double(total_edge_blocks_count_init) / THREADS_PER_BLOCK);

    // parallel_push_edge_preallocate_list_to_device_queue<<< thread_blocks, THREADS_PER_BLOCK>>>(device_edge_block, dapl, ebci, total_edge_blocks_count_init);
    parallel_push_edge_preallocate_list_to_device_queue_v1<<< thread_blocks, THREADS_PER_BLOCK>>>(device_edge_block, dapl, ebci, total_edge_blocks_count_init);

    parallel_push_queue_update<<< 1, 1>>>(vertex_blocks_count_init, total_edge_blocks_count_init);
    cudaDeviceSynchronize();

    push_to_queues_time = clock() - push_to_queues_time;


    // sleep(5);

    // Pass raw array and its size to kernel
    // thread_blocks = ceil(double(vertex_blocks_count_init) / THREADS_PER_BLOCK);
    // std::cout << "Thread blocks vertex init = " << thread_blocks << std::endl;
    // vertex_dictionary_init<<< thread_blocks, THREADS_PER_BLOCK>>>(dvpl, dapl, vertex_blocks_count_init, d_graph_prop, vertex_size, ebci);
    vd_time = clock();
    // vertex_dictionary_init<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_block, device_adjacency_sentinel, vertex_blocks_count_init, d_graph_prop, vertex_size, ebci);
    thread_blocks = ceil(double(vertex_size) / THREADS_PER_BLOCK);
    std::cout << "Thread blocks vertex init = " << thread_blocks << std::endl;
    // parallel_vertex_dictionary_init<<< thread_blocks, THREADS_PER_BLOCK>>>(device_adjacency_sentinel, vertex_blocks_count_init, d_graph_prop, vertex_size, ebci, device_vertex_dictionary);
    parallel_vertex_dictionary_init_v1<<< thread_blocks, THREADS_PER_BLOCK>>>(device_adjacency_sentinel, vertex_blocks_count_init, d_graph_prop, vertex_size, ebci, device_vertex_dictionary);


    cudaDeviceSynchronize();
    vd_time = clock() - vd_time;

    thread_blocks = ceil(double(h_graph_prop->xDim) / THREADS_PER_BLOCK);
    std::cout << "Thread blocks edge init = " << thread_blocks << std::endl;

    // sleep(5);

    // adjacency_list_init<<< thread_blocks, THREADS_PER_BLOCK>>>(depl, ebci, d_graph_prop, source, destination, total_edge_blocks_count_init, vertex_size, edge_size);
    al_time = clock();
    // adjacency_list_init<<< thread_blocks, THREADS_PER_BLOCK>>>(device_edge_block, ebci, d_graph_prop, source, destination, total_edge_blocks_count_init, vertex_size, edge_size, pseb, thread_blocks);
    // adjacency_list_init_modded<<< thread_blocks, THREADS_PER_BLOCK>>>(device_edge_block, ebci, d_graph_prop, source, destination, total_edge_blocks_count_init, vertex_size, edge_size, pseb, thread_blocks, device_vertex_dictionary);
    // adjacency_list_init_modded_v1<<< thread_blocks, THREADS_PER_BLOCK>>>(device_edge_block, ebci, d_graph_prop, source, destination, total_edge_blocks_count_init, vertex_size, edge_size, pseb, psvd, thread_blocks, device_vertex_dictionary);
    adjacency_list_init_modded_v2<<< thread_blocks, THREADS_PER_BLOCK>>>(device_edge_block, ebci, d_graph_prop, source, destination, total_edge_blocks_count_init, vertex_size, edge_size, pseb, psvd, thread_blocks, device_vertex_dictionary);

    // Seperate kernel for updating queues due to performance issues for global barriers
    update_edge_queue<<< 1, 1>>>(total_edge_blocks_count_init);

    cudaDeviceSynchronize();
    al_time = clock() - al_time;
    time_req = clock() - time_req;
    // sleep(5);

    total_time = push_to_queues_time + vd_time + al_time;

    // printKernel<<< 1, 1>>>(device_vertex_block, vertex_size);
    // printKernelmodded<<< 1, 1>>>(device_vertex_dictionary, vertex_size);
    printKernelmodded_v1<<< 1, 1>>>(device_vertex_dictionary, vertex_size);

    cudaDeviceSynchronize();

    unsigned long exitFlag = 1;
    unsigned long menuChoice;
    unsigned long h_search_flag, *d_search_flag;
    cudaMalloc(&d_search_flag, sizeof(unsigned long));

    while(exitFlag) {

        std::cout << "Please enter any of the below options" << std::endl << "1. Search for and edge" << std::endl << "2. Exit" << std::endl;
        scanf("%lu", &menuChoice);

        switch(menuChoice) {

            case 1  :
                        std::cout << "Enter the source and destination vertices respectively" << std::endl;
                        unsigned long search_source, search_destination, total_search_threads;
                        scanf("%lu %lu", &search_source, &search_destination);
                        std::cout << "Edge blocks count for " << search_source << " is " << h_edge_blocks_count_init[search_source - 1] << std::endl;
                        
                        search_times = clock();

                        total_search_threads = h_edge_blocks_count_init[search_source - 1] * EDGE_BLOCK_SIZE;
                        thread_blocks = ceil(double(total_search_threads) / THREADS_PER_BLOCK);

                        search_edge_kernel<<< thread_blocks, 1024>>>(device_vertex_dictionary, vertex_size, total_search_threads, search_source, search_destination, d_search_flag);
                        
                        cudaMemcpy(&h_search_flag, d_search_flag, sizeof(unsigned long),cudaMemcpyDeviceToHost);
                        cudaDeviceSynchronize();
                        search_times = clock() - search_times;

                        std::cout << "Search result was " << h_search_flag << " with time " << (float)search_times/CLOCKS_PER_SEC << " seconds" << std::endl;
                        h_search_flag = 0;
                        cudaMemcpy(d_search_flag, &h_search_flag, sizeof(unsigned long), cudaMemcpyHostToDevice);
                        cudaDeviceSynchronize();
                        break;

            case 2  :   
                        exitFlag = 0;
                        break;

            default :;

        }

    }

    printf("xDim = %lu, yDim = %lu, Total Edges = %lu\n", h_graph_prop -> xDim, h_graph_prop -> yDim, h_graph_prop -> total_edges);
    std::cout << "Queues: " << (float)push_to_queues_time/CLOCKS_PER_SEC << " seconds" << std::endl;
    std::cout << "Vertex Dictionary: " << (float)vd_time/CLOCKS_PER_SEC << " seconds" << std::endl;
    std::cout << "Adjacency List: " << (float)al_time/CLOCKS_PER_SEC << " seconds" << std::endl;

    std::cout << "Time taken: " << (float)time_req/CLOCKS_PER_SEC << " seconds" << std::endl;
    std::cout << "Added time: " << (float)total_time/CLOCKS_PER_SEC << " seconds" << std::endl;
    std::cout << "Read file : " << (float)section1/CLOCKS_PER_SEC << " seconds" << std::endl;
    std::cout << "Degree    : " << (float)section2/CLOCKS_PER_SEC << " seconds" << std::endl;
    std::cout << "Sorting   : " << (float)section2a/CLOCKS_PER_SEC << " seconds" << std::endl;
    std::cout << "Prefix sum, cudaMalloc, and cudaMemcpy: " << (float)section3/CLOCKS_PER_SEC << " seconds" << std::endl;      
	// // Cleanup
	// cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
	// printf("%d\n", c);
	return 0;
}