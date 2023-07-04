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
#define VERTEX_BLOCK_SIZE 17000000
#define EDGE_BLOCK_SIZE 128
#define VERTEX_PREALLOCATE_LIST_SIZE 2000
#define EDGE_PREALLOCATE_LIST_SIZE 59000000
#define BATCH_SIZE 100000

#define PREALLOCATE_EDGE_BLOCKS 100
#define SEARCH_BLOCKS_COUNT 100
#define INFTY 1000000000

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
    struct edge_block *lptr;
    struct edge_block *rptr;

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
    // struct edge_block *edge_block_address[100];
    struct edge_block *edge_block_address;

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

            // thrust::copy(h_source_degrees_new.begin(), h_source_degrees_new.end(), d_source_degrees_new.begin());
            // thrust::copy(h_csr_offset_new.begin(), h_csr_offset_new.end(), d_csr_offset_new.begin());
            // thrust::copy(h_csr_edges_new.begin(), h_csr_edges_new.end(), d_csr_edges_new.begin());
            // thrust::copy(h_edge_blocks_count.begin(), h_edge_blocks_count.end(), d_edge_blocks_count.begin());
            // thrust::copy(h_prefix_sum_edge_blocks_new.begin(), h_prefix_sum_edge_blocks_new.end(), d_prefix_sum_edge_blocks_new.begin());

struct batch_update_data {

    unsigned long *csr_offset;
    unsigned long *csr_edges;
    unsigned long *prefix_sum_edge_blocks;

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

__device__ void parallel_pop_from_edge_preallocate_queue(struct edge_block** device_edge_block, unsigned long pop_count, unsigned long* d_prefix_sum_edge_blocks, unsigned long id, unsigned long thread_blocks, unsigned long edge_blocks_used_present, unsigned long edge_blocks_required) {



    if((d_e_queue.count < pop_count) || (d_e_queue.front == -1)) {
        ;
        // printf("Edge queue empty, front = %ld, rear = %ld\n", d_e_queue.front, d_e_queue.rear);        
        // return NULL;
    }

    else {

        unsigned long start_index;
        if(id == 0)
            start_index = 0;
        else
            start_index = d_prefix_sum_edge_blocks[id - 1];


        // unsigned long end_index = d_prefix_sum_edge_blocks[id];
        unsigned long end_index = start_index + edge_blocks_required;

        if(start_index < end_index)
            device_edge_block[0] = d_e_queue.edge_block_address[d_e_queue.front + start_index];

        // unsigned long j = 0;

        // // printf("Thread #%lu, start_index is %lu and end_index is %lu\n", id, start_index, end_index);

        // for(unsigned long i = start_index ; i < end_index ; i++) {

        //     device_edge_block[j] = d_e_queue.edge_block_address[d_e_queue.front + i];
        //     d_e_queue.edge_block_address[d_e_queue.front + i] = NULL;

        //     // printf("Popped %p from the edge queue, placeholders front = %ld, rear = %ld\n", device_edge_block[j], d_e_queue.front, d_e_queue.rear);

        //     j++;
        // }

        // if(id == 0) {
        //     printf("Popped %p from the edge queue, start_index = %ld, end_index = %ld\n", device_edge_block[j-1], start_index, end_index);
        //     // printf("Queue address = %p, index is %lu\n", d_e_queue.edge_block_address[0], d_e_queue.front + start_index);
        // }

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
    // printf("At data structure init \n");
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

__global__ void parallel_push_edge_preallocate_list_to_device_queue_v1(struct edge_block* d_edge_preallocate_list, unsigned long total_edge_blocks_count_init) {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < total_edge_blocks_count_init) {


        // printf("%lu -> %p\n", id, d_edge_preallocate_list + id);



        unsigned long free_blocks = EDGE_PREALLOCATE_LIST_SIZE - d_e_queue.count;

        if( (free_blocks < total_edge_blocks_count_init) || (d_e_queue.rear + total_edge_blocks_count_init) % EDGE_PREALLOCATE_LIST_SIZE == d_e_queue.front ) {
            // printf("Edge queue Full, front = %ld, rear = %ld\n", d_e_queue.front, d_e_queue.rear);

            return;
        }

        d_e_queue.edge_block_address[id] = d_edge_preallocate_list + id;

        // if(id == 0) {
        //     printf("%lu -> %p\n", id, d_edge_preallocate_list + id);
        //     printf("Queue is %p\n", d_e_queue.edge_block_address[0]);
        // }
    }

}

__global__ void parallel_push_queue_update(unsigned long total_edge_blocks_count_init) {

    // if (d_v_queue.front == -1)
    //     d_v_queue.front = 0;

    // d_v_queue.rear = (d_v_queue.rear + vertex_blocks_count_init) % VERTEX_PREALLOCATE_LIST_SIZE;
    // // d_v_queue.vertex_block_address[d_v_queue.rear] = device_vertex_block;
    // d_v_queue.count += vertex_blocks_count_init;

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

__global__ void parallel_vertex_dictionary_init_v1(struct adjacency_sentinel_new* d_adjacency_sentinel_list, unsigned long vertex_size, unsigned long *edge_blocks_count_init, struct vertex_dictionary_structure *device_vertex_dictionary) {
	
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

// __global__ void adjacency_list_init_modded_v2(struct edge_block* d_edge_preallocate_list, unsigned long *d_edge_blocks_count_init, unsigned long *d_source, unsigned long *d_destination, unsigned long total_edge_blocks_count_init, unsigned long vertex_size, unsigned long edge_size, unsigned long *d_prefix_sum_edge_blocks, unsigned long *d_prefix_sum_vertex_degrees, unsigned long thread_blocks, struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long batch_number, unsigned long batch_size) {
	
//     unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

//     unsigned int current;
//     // __shared__ unsigned int lockvar;
//     // lockvar = 0;

//     __syncthreads();


//     if(id < vertex_size) {

//         // printf("Prefix sum of %ld is %ld\n", id, d_prefix_sum_edge_blocks[id]);

//         unsigned long current_vertex = id + 1;

//         // unsigned long edge_blocks_required = ptr->edge_block_count[index];
//         unsigned long edge_blocks_required = device_vertex_dictionary->edge_block_count[id];

//         // critical section start

//         struct edge_block *device_edge_block_base;

//         if(batch_number == 0) {
//             // temporary fix, this can't be a constant sized one
//             // for(unsigned long i = 0 ; i < edge_blocks_required ; i++) {
//             struct edge_block *device_edge_block[4];
//             parallel_pop_from_edge_preallocate_queue( device_edge_block, total_edge_blocks_count_init, d_prefix_sum_edge_blocks, id, thread_blocks);
//             device_edge_block_base = device_edge_block[0];
//         }

//         // critical section end
//         // if(threadIdx.x == 0)
//         //     printf("ID\tIteration\tGPU address\n");
        
//         // if(id == 0) {

//         //     for(unsigned long i = 0 ; i < batch_size ; i++)
//         //         printf("%lu and %lu\n", d_source[i], d_destination[i]);

//         // }

//         // for(unsigned long i = 0 ; i < edge_blocks_required ; i++)
//         //     printf("%lu\t%lu\t\t%p\n", id, i, device_edge_block[i]);

//         // adding the edge blocks
//         // struct adjacency_sentinel *vertex_adjacency = ptr->vertex_adjacency[index];
//         struct adjacency_sentinel_new *vertex_adjacency = device_vertex_dictionary->vertex_adjacency[id];

//         if(edge_blocks_required > 0) {

//             // one thread inserts sequentially all destination vertices of a source vertex.

//             struct edge_block *prev, *curr;

//             // prev = NULL;
//             // curr = NULL;


//             // for(unsigned long i = 0 ; i < edge_blocks_required ; i++) {

//             //     curr = device_edge_block[i];

//             //     if(prev != NULL)
//             //         prev->next = curr;

//             //     prev = curr;
                
//             // }

//             // if(edge_blocks_required > 0) {
//             //     vertex_adjacency->next = device_edge_block[0];
//             //     curr->next = NULL;
//             // }

//             // unsigned long edge_block_entry_count = 0;
//             // unsigned long edge_block_counter = 0;
//             unsigned long edge_block_entry_count;
//             unsigned long edge_block_counter;

//             if(batch_number == 0) {

//                 edge_block_entry_count = 0;
//                 edge_block_counter = 0;
//                 vertex_adjacency->active_edge_count = 0;

//                 // curr = device_edge_block[edge_block_counter];
//                 curr = device_edge_block_base;
//                 // vertex_adjacency->edge_block_address[edge_block_counter] = curr;
//                 vertex_adjacency->edge_block_address = curr;
//                 vertex_adjacency->last_insert_edge_block = curr;

//             }

//             else {

//                 edge_block_entry_count = vertex_adjacency->last_insert_edge_offset;
//                 edge_block_counter = vertex_adjacency->edge_block_count;
//                 curr = vertex_adjacency->last_insert_edge_block;
//                 curr->active_edge_count = 0;
                

//             }

//             // curr = vertex_adjacency->next;

//             unsigned long start_index;

//             if(current_vertex != 1)
//                 start_index = d_prefix_sum_vertex_degrees[current_vertex - 2];
//             else
//                 start_index = 0;

//             unsigned long end_index = d_prefix_sum_vertex_degrees[current_vertex - 1];

//             // printf("Current vertex = %lu, start = %lu, end = %lu\n", current_vertex, start_index, end_index);

//             // unsigned long current_edge_block = 0;
//             // curr = device_edge_block[edge_block_counter];
//             // // vertex_adjacency->edge_block_address[edge_block_counter] = curr;
//             // vertex_adjacency->edge_block_address = curr;
//             // vertex_adjacency->last_insert_edge_block = curr;

//             // __syncthreads();

//             // if(id == 0)
//             //     printf("Checkpoint AL beg\n");

//             // unsigned long edge_counter = 0;

//             // if(id == 0) {
//             //     // printf("Edge counter is %lu\n", vertex_adjacency->active_edge_count);
//             //     printf("Start index is %lu and end index is %lu\n", start_index, end_index);
//             //     printf("Device edge block address is %p\n", curr);
//             // }

//             for(unsigned long i = start_index ; i < end_index ; i++) {

//                 // printf("Checkpoint 0\n");

//                 // if(d_source[i] == current_vertex){

//                 // printf("Thread = %lu and Source = %lu and Destination = %lu\n", id, d_source[i], d_destination[i]);
//                 // printf("Checkpoint 1\n");

//                 // insert here

//                 // if(curr == NULL)
//                 //     printf("Hit here at %lu\n", id);

//                 curr->edge_block_entry[edge_block_entry_count].destination_vertex = d_destination[i];                    
//                 // printf("Checkpoint 2\n");
                
//                 // if(id == 0) {
//                 //     printf("Entry is %lu\n", curr->edge_block_entry[edge_block_entry_count].destination_vertex);
//                 // }
//                 // printf("Entry is %lu\n", curr->edge_block_entry[edge_block_entry_count].destination_vertex);
//                 vertex_adjacency->active_edge_count++;
//                 curr->active_edge_count++;
//                 vertex_adjacency->last_insert_edge_offset++;
//                 // printf("Checkpoint 3\n");

//                 if(++edge_block_entry_count >= EDGE_BLOCK_SIZE) {

//                     // curr = curr->next;
//                     // edge_block_counter++;
//                     edge_block_entry_count = 0;
//                     curr = curr + 1;
//                     // curr = device_edge_block[++edge_block_counter];
//                     curr->active_edge_count = 0;
//                     vertex_adjacency->last_insert_edge_block = curr;
//                     vertex_adjacency->edge_block_count++;
//                     vertex_adjacency->last_insert_edge_offset = 0;
//                     // vertex_adjacency->edge_block_address[edge_block_counter] = curr;

//                 }

//                 // }

//             }

//             // printf("Success\n");

//         }

//     }

//     // __syncthreads();

//     // if(id == 0)
//     //     printf("Checkpoint AL\n");

// }

// __global__ void adjacency_list_init_modded_v3(struct edge_block* d_edge_preallocate_list, unsigned long *d_edge_blocks_count_init, unsigned long total_edge_blocks_count_init, unsigned long vertex_size, unsigned long edge_size, unsigned long *d_prefix_sum_edge_blocks, unsigned long thread_blocks, struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long batch_number, unsigned long batch_size, unsigned long start_index_batch, unsigned long end_index_batch, unsigned long* d_csr_offset, unsigned long* d_csr_edges) {
	
//     unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

//     unsigned int current;
//     // __shared__ unsigned int lockvar;
//     // lockvar = 0;

//     __syncthreads();


//     if(id < vertex_size) {

//         // printf("Prefix sum of %ld is %ld\n", id, d_prefix_sum_edge_blocks[id]);

//         unsigned long current_vertex = id + 1;

//         // unsigned long edge_blocks_required = ptr->edge_block_count[index];
//         unsigned long edge_blocks_required = device_vertex_dictionary->edge_block_count[id];

//         // critical section start

//         struct edge_block *device_edge_block_base;

//         if(batch_number == 0) {
//             // temporary fix, this can't be a constant sized one
//             // for(unsigned long i = 0 ; i < edge_blocks_required ; i++) {
//             struct edge_block *device_edge_block[4];
//             parallel_pop_from_edge_preallocate_queue( device_edge_block, total_edge_blocks_count_init, d_prefix_sum_edge_blocks, id, thread_blocks);
//             device_edge_block_base = device_edge_block[0];
//         }

//         // critical section end
//         // if(threadIdx.x == 0)
//         //     printf("ID\tIteration\tGPU address\n");
        
//         // if(id == 0) {

//         //     for(unsigned long i = 0 ; i < batch_size ; i++)
//         //         printf("%lu and %lu\n", d_source[i], d_destination[i]);

//         // }

//         // for(unsigned long i = 0 ; i < edge_blocks_required ; i++)
//         //     printf("%lu\t%lu\t\t%p\n", id, i, device_edge_block[i]);

//         // adding the edge blocks
//         // struct adjacency_sentinel *vertex_adjacency = ptr->vertex_adjacency[index];
//         struct adjacency_sentinel_new *vertex_adjacency = device_vertex_dictionary->vertex_adjacency[id];

//         if(edge_blocks_required > 0) {

//             // one thread inserts sequentially all destination vertices of a source vertex.

//             struct edge_block *prev, *curr;

//             // prev = NULL;
//             // curr = NULL;


//             // for(unsigned long i = 0 ; i < edge_blocks_required ; i++) {

//             //     curr = device_edge_block[i];

//             //     if(prev != NULL)
//             //         prev->next = curr;

//             //     prev = curr;
                
//             // }

//             // if(edge_blocks_required > 0) {
//             //     vertex_adjacency->next = device_edge_block[0];
//             //     curr->next = NULL;
//             // }

//             // unsigned long edge_block_entry_count = 0;
//             // unsigned long edge_block_counter = 0;
//             unsigned long edge_block_entry_count;
//             unsigned long edge_block_counter;

//             if(batch_number == 0) {

//                 edge_block_entry_count = 0;
//                 edge_block_counter = 0;
//                 vertex_adjacency->active_edge_count = 0;

//                 // curr = device_edge_block[edge_block_counter];
//                 curr = device_edge_block_base;
//                 // vertex_adjacency->edge_block_address[edge_block_counter] = curr;
//                 vertex_adjacency->edge_block_address = curr;
//                 vertex_adjacency->last_insert_edge_block = curr;

//             }

//             else {

//                 edge_block_entry_count = vertex_adjacency->last_insert_edge_offset;
//                 edge_block_counter = vertex_adjacency->edge_block_count;
//                 curr = vertex_adjacency->last_insert_edge_block;
//                 curr->active_edge_count = 0;
                

//             }

//             // curr = vertex_adjacency->next;

//             // test code v3 start

//             unsigned long start_index = start_index_batch;
//             unsigned long end_index = end_index_batch;

//             if(start_index_batch <= d_csr_offset[id])
//                 start_index = d_csr_offset[id];
//             // else
//             //     end_index = end_index_batch;

//             // test code v3 end

//             if(end_index_batch >= d_csr_offset[id + 1])
//                 end_index = d_csr_offset[id + 1];
//             // else
//             //     end_index = end_index_batch;

//             // test code v3 end

//             // unsigned long start_index;

//             // if(current_vertex != 1)
//             //     start_index = d_prefix_sum_vertex_degrees[current_vertex - 2];
//             // else
//             //     start_index = 0;

//             // unsigned long end_index = d_prefix_sum_vertex_degrees[current_vertex - 1];

//             // printf("Current vertex = %lu, start = %lu, end = %lu\n", current_vertex, start_index, end_index);

//             // unsigned long current_edge_block = 0;
//             // curr = device_edge_block[edge_block_counter];
//             // // vertex_adjacency->edge_block_address[edge_block_counter] = curr;
//             // vertex_adjacency->edge_block_address = curr;
//             // vertex_adjacency->last_insert_edge_block = curr;

//             // __syncthreads();

//             // if(id == 0)
//             //     printf("Checkpoint AL beg\n");

//             // unsigned long edge_counter = 0;

//             // if(id == 0) {
//             //     // printf("Edge counter is %lu\n", vertex_adjacency->active_edge_count);
//             //     printf("Start index is %lu and end index is %lu\n", start_index, end_index);
//             //     printf("Device edge block address is %p\n", curr);
//             // }

//             for(unsigned long i = start_index ; i < end_index ; i++) {

//                 // printf("Checkpoint 0\n");

//                 // if(d_source[i] == current_vertex){

//                 // printf("Thread = %lu and Source = %lu and Destination = %lu\n", id, d_source[i], d_destination[i]);
//                 // printf("Checkpoint 1\n");

//                 // insert here

//                 // if(curr == NULL)
//                 //     printf("Hit here at %lu\n", id);
//                 // curr->edge_block_entry[edge_block_entry_count].destination_vertex = d_destination[i];                    
//                 curr->edge_block_entry[edge_block_entry_count].destination_vertex = d_csr_edges[i];                    
//                 // printf("Checkpoint 2\n");
                
//                 // if(id == 0) {
//                 //     printf("Entry is %lu\n", curr->edge_block_entry[edge_block_entry_count].destination_vertex);
//                 // }
//                 // printf("Entry is %lu\n", curr->edge_block_entry[edge_block_entry_count].destination_vertex);
//                 vertex_adjacency->active_edge_count++;
//                 curr->active_edge_count++;
//                 vertex_adjacency->last_insert_edge_offset++;
//                 // printf("Checkpoint 3\n");

//                 if(++edge_block_entry_count >= EDGE_BLOCK_SIZE) {

//                     // curr = curr->next;
//                     // edge_block_counter++;
//                     edge_block_entry_count = 0;
//                     curr = curr + 1;
//                     // curr = device_edge_block[++edge_block_counter];
//                     curr->active_edge_count = 0;
//                     vertex_adjacency->last_insert_edge_block = curr;
//                     vertex_adjacency->edge_block_count++;
//                     vertex_adjacency->last_insert_edge_offset = 0;
//                     // vertex_adjacency->edge_block_address[edge_block_counter] = curr;

//                 }

//                 // }

//             }

//             // printf("Success\n");

//         }

//     }

//     // __syncthreads();

//     // if(id == 0)
//     //     printf("Checkpoint AL\n");

// }

__device__ unsigned long traversal_string(unsigned long val, unsigned long *length) {

    unsigned long temp = val;
    unsigned long bit_string = 0;
    *length = 0;

    while(temp > 1) {

        // bit_string = ((temp % 2) * pow(10, iteration++)) + bit_string;
        if(temp % 2)
            bit_string = (bit_string * 10) + 2;
        else
            bit_string = (bit_string * 10) + 1;

        // bit_string = (bit_string * 10) + (temp % 2);
        temp = temp / 2;
        // (*length)++;
        // *length = *length + 1;


    }

    // printf("%lu\n", iteration);

    return bit_string;
}

__device__ void insert_edge_block_to_CBT(struct edge_block *root, unsigned long bit_string, unsigned long length, struct edge_block *new_block) {

    struct edge_block *curr = root;

    // if(length > 0) {

        // for(unsigned long i = 0 ; i < length - 1 ; i++) {
        for( ; bit_string > 10  ; bit_string /= 10) {

            // if(bit_string % 2)
            //     curr = curr->rptr;
            // else
            //     curr = curr->lptr;
            
            if(bit_string % 2)
                curr = curr->lptr;
            else
                curr = curr->rptr;

            // bit_string /= 10;

        }

    // }

    new_block->lptr = NULL;
    new_block->rptr = NULL;

    // printf("Checkpoint\n");

    if(bit_string % 2)
        curr->lptr = new_block;
    else
        curr->rptr = new_block;

}

__device__ void inorderTraversalTemp(struct edge_block *root) {

    if(root == NULL) {
        printf("Hit\n");
        return;
    }

    else {

        printf("Hit 1\n");
        printf("Root %p contents are %lu and %lu, pointers are %p and %p, active_edge_count is %lu\n", root, root->edge_block_entry[0].destination_vertex, root->edge_block_entry[1].destination_vertex, root->lptr, root->rptr, root->active_edge_count);
        printf("Hit 2\n");

        inorderTraversalTemp(root->lptr);

        printf("\nedge block edge count = %lu, ", root->active_edge_count);

        for(unsigned long j = 0 ; j < root->active_edge_count ; j++) {
                        
            printf("%lu ", root->edge_block_entry[j].destination_vertex);
        
            // edge_block_entry_count++;

            // if(++edge_block_entry_count >= EDGE_BLOCK_SIZE) {
            //     // itr = itr->next;
            //     // itr = device_vertex_dictionary->vertex_adjacency[i]->edge_block_address[++edge_block_counter];
            //     itr = itr + 1;
            //     edge_block_entry_count = 0;
            // }
        }

        // printf("\n");

        // inorderTraversalTemp(root->rptr);

    }

    // unsigned long edge_block_counter = 0;
    // unsigned long edge_block_entry_count = 0;
    // // struct edge_block *itr = device_vertex_dictionary->vertex_adjacency[i]->edge_block_address[edge_block_counter];
    // struct edge_block *itr = device_vertex_dictionary->vertex_adjacency[i]->edge_block_address;


    // for(unsigned long j = 0 ; j < device_vertex_dictionary->vertex_adjacency[i]->active_edge_count ; j++) {
                    
    //     printf("%lu ", itr->edge_block_entry[edge_block_entry_count].destination_vertex);
    
    //     // edge_block_entry_count++;

    //     if(++edge_block_entry_count >= EDGE_BLOCK_SIZE) {
    //         // itr = itr->next;
    //         // itr = device_vertex_dictionary->vertex_adjacency[i]->edge_block_address[++edge_block_counter];
    //         itr = itr + 1;
    //         edge_block_entry_count = 0;
    //     }
    // }

}

__global__ void adjacency_list_init_modded_v4(struct edge_block* d_edge_preallocate_list, unsigned long *d_edge_blocks_count_init, unsigned long total_edge_blocks_count_init, unsigned long vertex_size, unsigned long edge_size, unsigned long *d_prefix_sum_edge_blocks, unsigned long thread_blocks, struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long batch_number, unsigned long batch_size, unsigned long start_index_batch, unsigned long end_index_batch, unsigned long* d_csr_offset, unsigned long* d_csr_edges) {
	
    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int current;
    // __shared__ unsigned int lockvar;
    // lockvar = 0;

    __syncthreads();


    if(id < vertex_size) {

        // printf("Prefix sum of %ld is %ld\n", id, d_prefix_sum_edge_blocks[id]);

        unsigned long current_vertex = id + 1;

        // unsigned long edge_blocks_required = ptr->edge_block_count[index];
        // unsigned long edge_blocks_required = device_vertex_dictionary->edge_block_count[id];
        unsigned long edge_blocks_required = 0;

        // critical section start

        struct edge_block *device_edge_block_base;

        // v4 test code start

        struct adjacency_sentinel_new *vertex_adjacency = device_vertex_dictionary->vertex_adjacency[id];
        struct edge_block *curr;

        unsigned long space_remaining = 0;
        unsigned long edge_blocks_used_present = 0;

        if((vertex_adjacency->edge_block_address != NULL) && (vertex_adjacency->last_insert_edge_offset != 0)) {

            space_remaining = EDGE_BLOCK_SIZE - device_vertex_dictionary->vertex_adjacency[id]->last_insert_edge_offset;

        }

        edge_blocks_used_present = device_vertex_dictionary->vertex_adjacency[id]->edge_block_count;

        // printf("%lu\n", edge_blocks_required);
        unsigned long start_index = start_index_batch;
        unsigned long end_index = end_index_batch;

        unsigned long new_edges_count = 0;
        // if(end_index > start_index)
        //     new_edges_count = end_index - start_index;

        if(start_index_batch <= d_csr_offset[id])
            start_index = d_csr_offset[id];
        // else
        //     end_index = end_index_batch;

        // test code v3 end

        if(end_index_batch >= d_csr_offset[id + 1])
            end_index = d_csr_offset[id + 1];

        // below check since end_index and start_index are unsigned longs
        if(end_index > start_index) {

            // if((end_index - start_index) > space_remaining) {
                new_edges_count = end_index - start_index;
                // edge_blocks_required = ceil(double((end_index - start_index) - space_remaining) / EDGE_BLOCK_SIZE);
                if(new_edges_count > space_remaining)
                    edge_blocks_required = ceil(double(new_edges_count - space_remaining) / EDGE_BLOCK_SIZE);
                // else
                //     edge_blocks_required = ceil(double(new_edges_count - space_remaining) / EDGE_BLOCK_SIZE);
            // }

        }

        else
            edge_blocks_required = 0;

        // printf("thread #%lu, start_index is %lu, end_index is %lu, end_index_batch is %lu, d_csr_offset[id+1] is %lu, edge_blocks_required is %lu, edge_blocks_used is %lu, space_remaining is %lu\n", id, start_index, end_index, end_index_batch, d_csr_offset[id + 1], edge_blocks_required, edge_blocks_used_present, space_remaining);

        // printf("Checkpointer 1\n");

        // printf("%lu\n", edge_blocks_required);

        // v4 test code end

        struct edge_block *device_edge_block[PREALLOCATE_EDGE_BLOCKS];
        
        if(edge_blocks_required > 0) {

            parallel_pop_from_edge_preallocate_queue( device_edge_block, total_edge_blocks_count_init, d_prefix_sum_edge_blocks, id, thread_blocks, edge_blocks_used_present, edge_blocks_required);
            vertex_adjacency->edge_block_count += edge_blocks_required;            
            device_edge_block_base = device_edge_block[0];

        }




        // printf("Checkpointer 2\n");

        // create complete binary tree for first time
        if((edge_blocks_required > 0) && (vertex_adjacency->edge_block_address == NULL)) {

            curr = device_edge_block[0];
            // vertex_adjacency->edge_block_address = curr;

            // unsigned long array[] = {1,2,3,4,5,6,7,8,9,10,11,12};
            // unsigned long len = 12;

            // struct node *addresses = (struct node*)malloc(len * sizeof(struct node));

            // unsigned long inserted = 0;

            // struct node *root = addresses;
            // struct node *curr = root;

            unsigned long i = 0;
            unsigned long curr_index = 0;

            if(edge_blocks_required > 1) {

                for(i = 0 ; i < (edge_blocks_required / 2) - 1 ; i++) {

                    // curr->value = array[i];
                    curr->lptr = *(device_edge_block + (2 * i) + 1);
                    curr->rptr = *(device_edge_block + (2 * i) + 2);

                    // printf("Inserted internal node %p\n", curr);

                    curr_index++;
                    curr = *(device_edge_block + curr_index);

                }

                if(edge_blocks_required % 2) {

                    // curr->value = array [i];
                    curr->lptr = *(device_edge_block + (2 * i) + 1);
                    curr->rptr = *(device_edge_block + (2 * i++) + 2);

                    // printf("Inserted internal node v1 %p\n", curr);

                }
                else {
                    // curr->value = array [i];
                    curr->lptr = *(device_edge_block + (2 * i++) + 1);  

                    // printf("Inserted internal node v2 %p\n", curr);
                
                }

                curr_index++;
                curr = *(device_edge_block + curr_index);
            
            }

            // printf("Checkpoint %lu\n", edge_blocks_required);

            for( ; i < edge_blocks_required ; i++) {

                // curr->value = array[i];
                curr->lptr = NULL;
                curr->rptr = NULL;

                // printf("Inserted leaf node %p\n", curr);

                curr_index++;
                curr = *(device_edge_block + curr_index);

            }    

        }

        else if((edge_blocks_required > 0) && (vertex_adjacency->edge_block_address != NULL)) {

            // printf("Checkpoint update %lu\n", edge_blocks_required);

            for(unsigned long i = 0 ; i < edge_blocks_required ; i++) {

                unsigned long current_edge_block = vertex_adjacency->edge_block_count - edge_blocks_required + i + 1;
                unsigned long length = 0;
                unsigned long bit_string = traversal_string(current_edge_block, &length);

                // printf("Disputed address %p\n", device_edge_block[i]);

                // printf("Checkpoint cbt1 for thread #%lu, bit_string for %lu is %lu with length %lu\n", id, current_edge_block, bit_string, length);
                insert_edge_block_to_CBT(vertex_adjacency->edge_block_address, bit_string, length, device_edge_block[i]);
                // printf("Checkpoint cbt2\n");

            }

        }

        // if(batch_number == 0) {
        //     // temporary fix, this can't be a constant sized one
        //     // for(unsigned long i = 0 ; i < edge_blocks_required ; i++) {
        //     struct edge_block *device_edge_block[4];
        //     parallel_pop_from_edge_preallocate_queue( device_edge_block, total_edge_blocks_count_init, d_prefix_sum_edge_blocks, id, thread_blocks);
        //     device_edge_block_base = device_edge_block[0];
        // }

        // critical section end
        // if(threadIdx.x == 0)
        //     printf("ID\tIteration\tGPU address\n");
        
        // if(id == 0) {

        //     for(unsigned long i = 0 ; i < batch_size ; i++)
        //         printf("%lu and %lu\n", d_source[i], d_destination[i]);

        // }

        // for(unsigned long i = 0 ; i < edge_blocks_required ; i++)
        //     printf("%lu\t%lu\t\t%p\n", id, i, device_edge_block[i]);

        // adding the edge blocks
        // struct adjacency_sentinel *vertex_adjacency = ptr->vertex_adjacency[index];

        if(new_edges_count > 0) {

            // one thread inserts sequentially all destination vertices of a source vertex.

            // struct edge_block *prev, *curr;

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

            // unsigned long edge_block_entry_count = 0;
            // unsigned long edge_block_counter = 0;
            unsigned long edge_block_entry_count;
            unsigned long edge_block_counter;


            if(vertex_adjacency->edge_block_address == NULL) {

                edge_block_entry_count = 0;
                edge_block_counter = 0;
                vertex_adjacency->active_edge_count = 0;
                // vertex_adjacency->edge_block_count = 1;

                // curr = device_edge_block[edge_block_counter];
                curr = device_edge_block[0];
                curr->active_edge_count = 0;
                // vertex_adjacency->edge_block_address[edge_block_counter] = curr;
                vertex_adjacency->edge_block_address = curr;
                vertex_adjacency->last_insert_edge_block = curr;

            }

            else {

                edge_block_entry_count = vertex_adjacency->last_insert_edge_offset;
                // edge_block_counter = vertex_adjacency->edge_block_count;
                edge_block_counter = 0;
                curr = vertex_adjacency->last_insert_edge_block;
                // curr->active_edge_count = 0;
                
                if(space_remaining == 0) {
                    curr = device_edge_block[0];
                    curr->active_edge_count = 0;
                    edge_block_entry_count = 0;
                    edge_block_counter = 1;
                }
                // else {
                //     curr = vertex_adjacency->last_insert_edge_block;
                // }
            }

            // curr = vertex_adjacency->next;

            // test code v3 start

            // unsigned long start_index = start_index_batch;
            // unsigned long end_index = end_index_batch;

            // if(start_index_batch <= d_csr_offset[id])
            //     start_index = d_csr_offset[id];
            // // else
            // //     end_index = end_index_batch;

            // // test code v3 end

            // if(end_index_batch >= d_csr_offset[id + 1])
            //     end_index = d_csr_offset[id + 1];
            // else
            //     end_index = end_index_batch;

            // test code v3 end

            // unsigned long start_index;

            // if(current_vertex != 1)
            //     start_index = d_prefix_sum_vertex_degrees[current_vertex - 2];
            // else
            //     start_index = 0;

            // unsigned long end_index = d_prefix_sum_vertex_degrees[current_vertex - 1];

            // printf("Current vertex = %lu, start = %lu, end = %lu\n", current_vertex, start_index, end_index);

            // unsigned long current_edge_block = 0;
            // curr = device_edge_block[edge_block_counter];
            // // vertex_adjacency->edge_block_address[edge_block_counter] = curr;
            // vertex_adjacency->edge_block_address = curr;
            // vertex_adjacency->last_insert_edge_block = curr;

            // __syncthreads();

            // if(id == 0)
            //     printf("Checkpoint AL beg\n");

            // unsigned long edge_counter = 0;

            // if(id == 0) {
            //     // printf("Edge counter is %lu\n", vertex_adjacency->active_edge_count);
            //     printf("Start index is %lu and end index is %lu\n", start_index, end_index);
            //     printf("Device edge block address is %p\n", curr);
            // }

            // printf("Checkpoint\n");

            for(unsigned long i = start_index ; i < end_index ; i++) {

                // printf("Checkpoint 0\n");

                // if(d_source[i] == current_vertex){

                // printf("Thread = %lu and Source = %lu and Destination = %lu\n", id, id + 1, d_csr_edges[i]);
                // printf("Checkpoint 1\n");

                // insert here

                // if(curr == NULL)
                //     printf("Hit here at %lu\n", id);
                // curr->edge_block_entry[edge_block_entry_count].destination_vertex = d_destination[i];                    
                curr->edge_block_entry[edge_block_entry_count].destination_vertex = d_csr_edges[i];                    
                // printf("Checkpoint 2\n");
                
                // if(id == 0) {
                // }
                // printf("Entry is %lu\n", curr->edge_block_entry[edge_block_entry_count].destination_vertex);
                vertex_adjacency->active_edge_count++;
                curr->active_edge_count++;
                vertex_adjacency->last_insert_edge_offset++;
                // printf("Checkpoint 3\n");

                // printf("Entry is %lu for thread #%lu at %p, counter is %lu and %lu\n", curr->edge_block_entry[edge_block_entry_count].destination_vertex, id, curr, curr->active_edge_count, vertex_adjacency->edge_block_address->active_edge_count);


                if((i + 1 < end_index) && (++edge_block_entry_count >= EDGE_BLOCK_SIZE) && (edge_block_counter < edge_blocks_required)) {

                    // curr = curr->next;
                    // edge_block_counter++;
                    edge_block_entry_count = 0;


                    if((space_remaining != 0) && (edge_block_counter == 0))
                        curr = device_edge_block[0];
                    else
                        curr = curr + 1;

                    // printf("Hit for thread #%lu at %p\n", id, curr);

                    // curr = device_edge_block[++edge_block_counter];
                    ++edge_block_counter;
                    curr->active_edge_count = 0;
                    vertex_adjacency->last_insert_edge_block = curr;
                    // vertex_adjacency->edge_block_count++;
                    vertex_adjacency->last_insert_edge_offset = 0;
                    // vertex_adjacency->edge_block_address[edge_block_counter] = curr;

                }

                // }

            }

            // printf("Debug code start\n");
            // struct edge_block *curr = device_edge_block[0];

            // curr->rptr = NULL;
            // curr->lptr->lptr = NULL;
            // curr->lptr->rptr = NULL;

            // printf("Root %p contents are %lu and %lu, pointers are %p and %p, active_edge_count is %lu\n", curr, curr->edge_block_entry[0].destination_vertex, curr->edge_block_entry[1].destination_vertex, curr->lptr, curr->rptr, curr->active_edge_count);
            // curr = curr->lptr;
            // printf("Lptr %p contents are %lu and %lu, pointers are %p and %p, active_edge_count is %lu\n", curr, curr->edge_block_entry[0].destination_vertex, curr->edge_block_entry[1].destination_vertex, curr->lptr, curr->rptr, curr->active_edge_count);


            // inorderTraversalTemp(device_edge_block[0]);

            // printf("Success\n");

        }
        // printf("Checkpoint final thread#%lu\n", id);

    }

    // __syncthreads();

    // if(id == 0)

}

__global__ void adjacency_list_init_modded_v5(struct edge_block* d_edge_preallocate_list, unsigned long *d_edge_blocks_count_init, unsigned long total_edge_blocks_count_batch, unsigned long vertex_size, unsigned long edge_size, unsigned long *d_prefix_sum_edge_blocks, unsigned long thread_blocks, struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long batch_number, unsigned long batch_size, unsigned long start_index_batch, unsigned long end_index_batch, unsigned long* d_csr_offset, unsigned long* d_csr_edges) {
	
    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int current;

    __syncthreads();


    if(id < vertex_size) {

        unsigned long current_vertex = id + 1;

        unsigned long edge_blocks_required = 0;

        // printf("Checkpoint for id %lu\n", id);

        // critical section start

        struct edge_block *device_edge_block_base;

        // v4 test code start

        struct adjacency_sentinel_new *vertex_adjacency = device_vertex_dictionary->vertex_adjacency[id];
        struct edge_block *curr;

        unsigned long space_remaining = 0;
        unsigned long edge_blocks_used_present = 0;

        if((vertex_adjacency->edge_block_address != NULL) && (vertex_adjacency->last_insert_edge_offset != 0)) {

            space_remaining = EDGE_BLOCK_SIZE - device_vertex_dictionary->vertex_adjacency[id]->last_insert_edge_offset;
            // printf("id=%lu, last_insert_edge_offset is %lu\n", id, device_vertex_dictionary->vertex_adjacency[id]->last_insert_edge_offset);
        }

        edge_blocks_used_present = device_vertex_dictionary->vertex_adjacency[id]->edge_block_count;

        // printf("%lu\n", edge_blocks_required);
        unsigned long start_index = start_index_batch;
        unsigned long end_index = end_index_batch;

        unsigned long new_edges_count = 0;

        // if(start_index_batch <= d_csr_offset[id])
        //     start_index = d_csr_offset[id];

        // if(end_index_batch >= d_csr_offset[id + 1])
        //     end_index = d_csr_offset[id + 1];

        start_index = d_csr_offset[id];
        end_index = d_csr_offset[id + 1];

        // below check since end_index and start_index are unsigned longs
        if(end_index > start_index) {

                new_edges_count = end_index - start_index;
                // edge_blocks_required = ceil(double((end_index - start_index) - space_remaining) / EDGE_BLOCK_SIZE);
                if(new_edges_count > space_remaining)
                    edge_blocks_required = ceil(double(new_edges_count - space_remaining) / EDGE_BLOCK_SIZE);

        }

        else
            edge_blocks_required = 0;

        // struct edge_block *device_edge_block[PREALLOCATE_EDGE_BLOCKS];
        struct edge_block *device_edge_block[1];
        
        if(edge_blocks_required > 0) {

            parallel_pop_from_edge_preallocate_queue( device_edge_block, total_edge_blocks_count_batch, d_prefix_sum_edge_blocks, id, thread_blocks, edge_blocks_used_present, edge_blocks_required);
            vertex_adjacency->edge_block_count += edge_blocks_required;            
            device_edge_block_base = device_edge_block[0];

            // printf("Checkpoint ID is %lu, device edge block base is %p\n", id, device_edge_block_base);

        }



        // printf("thread #%lu, start_index is %lu, end_index is %lu, end_index_batch is %lu, d_csr_offset[id+1] is %lu, edge_blocks_required is %lu, edge_blocks_used is %lu, space_remaining is %lu\n", id, start_index, end_index, end_index_batch, d_csr_offset[id + 1], edge_blocks_required, edge_blocks_used_present, space_remaining);

        // create complete binary tree for first time
        if((edge_blocks_required > 0) && (vertex_adjacency->edge_block_address == NULL)) {

            curr = device_edge_block[0];

            unsigned long i = 0;
            unsigned long curr_index = 0;

            if(edge_blocks_required > 1) {

                for(i = 0 ; i < (edge_blocks_required / 2) - 1 ; i++) {

                    // curr->value = array[i];
                    curr->lptr = *device_edge_block + (2 * i) + 1;
                    curr->rptr = *device_edge_block + (2 * i) + 2;

                    // printf("Inserted internal node %p\n", curr);

                    curr_index++;
                    curr = *device_edge_block + curr_index;

                }

                if(edge_blocks_required % 2) {

                    // curr->value = array [i];
                    curr->lptr = *device_edge_block + (2 * i) + 1;
                    curr->rptr = *device_edge_block + (2 * i++) + 2;

                    // printf("Inserted internal node v1 %p for id %lu\n", curr, id);

                }
                else {
                    // curr->value = array [i];
                    curr->lptr = *device_edge_block + (2 * i++) + 1;  

                    // printf("Inserted internal node v2 %p for id %lu, lptr is %p\n", curr, id, *device_edge_block + (2 * (i - 1)) + 1);
                
                }

                curr_index++;
                curr = *device_edge_block + curr_index;
            
            }

            // printf("Checkpoint %lu\n", edge_blocks_required);

            for( ; i < edge_blocks_required ; i++) {

                // curr->value = array[i];
                curr->lptr = NULL;
                curr->rptr = NULL;

                // printf("Inserted leaf node %p\n", curr);

                curr_index++;
                curr = *device_edge_block + curr_index;

            }    

        }

        else if((edge_blocks_required > 0) && (vertex_adjacency->edge_block_address != NULL)) {

            // printf("Checkpoint update %lu\n", edge_blocks_required);

            for(unsigned long i = 0 ; i < edge_blocks_required ; i++) {

                unsigned long current_edge_block = vertex_adjacency->edge_block_count - edge_blocks_required + i + 1;
                unsigned long length = 0;
                unsigned long bit_string = traversal_string(current_edge_block, &length);

                // printf("Disputed address %p for id %lu\n", *device_edge_block + i, id);

                // printf("Checkpoint cbt1 for thread #%lu, bit_string for %lu is %lu with length %lu\n", id, current_edge_block, bit_string, length);
                insert_edge_block_to_CBT(vertex_adjacency->edge_block_address, bit_string, length, *device_edge_block + i);
                // printf("Checkpoint cbt2\n");

            }

        }


        if(new_edges_count > 0) {

            unsigned long edge_block_entry_count;
            unsigned long edge_block_counter;

            if(vertex_adjacency->edge_block_address == NULL) {

                edge_block_entry_count = 0;
                edge_block_counter = 0;
                vertex_adjacency->active_edge_count = 0;
                // vertex_adjacency->edge_block_count = 1;

                // curr = device_edge_block[edge_block_counter];
                curr = device_edge_block[0];
                curr->active_edge_count = 0;
                // vertex_adjacency->edge_block_address[edge_block_counter] = curr;
                vertex_adjacency->edge_block_address = curr;
                vertex_adjacency->last_insert_edge_block = curr;

            }

            else {

                edge_block_entry_count = vertex_adjacency->last_insert_edge_offset;
                // edge_block_counter = vertex_adjacency->edge_block_count;
                edge_block_counter = 0;
                curr = vertex_adjacency->last_insert_edge_block;
                // curr->active_edge_count = 0;
                
                if(space_remaining == 0) {
                    curr = device_edge_block[0];
                    curr->active_edge_count = 0;
                    vertex_adjacency->last_insert_edge_block = curr;
                    edge_block_entry_count = 0;
                    edge_block_counter = 1;
                }
                // else {
                //     curr = vertex_adjacency->last_insert_edge_block;
                // }
            }

            // goto exit_insert;


            for(unsigned long i = start_index ; i < end_index ; i++) {

                // if((curr == NULL) || (vertex_adjacency == NULL) || (edge_block_entry_count >= EDGE_BLOCK_SIZE) || (i >= batch_size))
                //     printf("Hit disupte null\n");
                   
                // if(edge_block_entry_count >= EDGE_BLOCK_SIZE)
                //     printf("Hit dispute margin\n");

                curr->edge_block_entry[edge_block_entry_count].destination_vertex = d_csr_edges[i];    

                // if(d_csr_edges[i] == 0)
                //     printf("Hit dispute\n");                

                vertex_adjacency->active_edge_count++;
                curr->active_edge_count++;
                vertex_adjacency->last_insert_edge_offset++;
                // printf("Checkpoint 3\n");

                // printf("Entry is %lu for thread #%lu at %p, counter is %lu and %lu\n", curr->edge_block_entry[edge_block_entry_count].destination_vertex, id, curr, curr->active_edge_count, vertex_adjacency->edge_block_address->active_edge_count);

                // edge_block_entry_count++;


                // if(edge_block_entry_count >= EDGE_BLOCK_SIZE)
                //     edge_block_entry_count = 0;

                // continue;


                // if((i + 1 < end_index) && (++edge_block_entry_count >= EDGE_BLOCK_SIZE) && (edge_block_counter < edge_blocks_required)) {
                if((++edge_block_entry_count >= EDGE_BLOCK_SIZE) && (i + 1 < end_index) && (edge_block_counter < edge_blocks_required)) {



                    // curr = curr->next;
                    // edge_block_counter++;
                    edge_block_entry_count = 0;


                    // printf("at dispute\n");
                    // if((curr == NULL) || (device_edge_block[0] == NULL) || (vertex_adjacency == NULL))
                    //     printf("Dispute caught\n");

                    if((space_remaining != 0) && (edge_block_counter == 0))
                        curr = device_edge_block[0];
                    else
                        curr = curr + 1;



                    // printf("Hit for thread #%lu at %p\n", id, curr);

                    // curr = device_edge_block[++edge_block_counter];
                    ++edge_block_counter;
                    curr->active_edge_count = 0;
                    vertex_adjacency->last_insert_edge_block = curr;
                    // vertex_adjacency->edge_block_count++;
                    vertex_adjacency->last_insert_edge_offset = 0;
                    // vertex_adjacency->edge_block_address[edge_block_counter] = curr;



                }

                // }

            }


            if(vertex_adjacency->last_insert_edge_offset == EDGE_BLOCK_SIZE)
                vertex_adjacency->last_insert_edge_offset = 0;

        }
        // printf("Checkpoint final thread#%lu\n", id);

    }

    // exit_insert:

    // __syncthreads();

    // if(id == 0) {
    //     printf("Checkpoint final\n");
    // }

}

__global__ void update_edge_queue(unsigned long pop_count) {


    d_e_queue.count -= pop_count;

    // printf("Edge Queue before, front = %ld, rear = %ld\n", d_e_queue.front, d_e_queue.rear);
    
    if((d_e_queue.front + pop_count - 1) % EDGE_PREALLOCATE_LIST_SIZE == d_e_queue.rear) {

        d_e_queue.front = -1;
        d_e_queue.rear = -1;

    }
    else
        d_e_queue.front = (d_e_queue.front + pop_count) % EDGE_PREALLOCATE_LIST_SIZE;

    // printf("Edge Queue before, front = %ld, rear = %ld\n", d_e_queue.front, d_e_queue.rear);


}

__global__ void search_edge_kernel(struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long vertex_size, unsigned long total_search_threads, unsigned long search_source, unsigned long search_destination, unsigned long *d_search_flag) {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < total_search_threads) {

        unsigned long edge_block = id / EDGE_BLOCK_SIZE;
        unsigned long edge_block_entry = id % EDGE_BLOCK_SIZE;

        // printf("Search kernel launch test for source %lu and destination %lu, edge_block %lu and entry %lu\n", search_source, search_destination, edge_block, edge_block_entry);

        struct adjacency_sentinel_new *edge_sentinel = device_vertex_dictionary->vertex_adjacency[search_source - 1];
        unsigned long extracted_value;

        struct edge_block *itr = edge_sentinel->edge_block_address + (edge_block);
        extracted_value = itr->edge_block_entry[edge_block_entry].destination_vertex;

        // extracted_value = edge_sentinel->edge_block_address[edge_block]->edge_block_entry[edge_block_entry].destination_vertex;

        if(extracted_value == search_destination) {
            *d_search_flag = 1;
            // printf("Edge exists\n");
        }

    }

}

__device__ struct edge_block *search_blocks[SEARCH_BLOCKS_COUNT];
__device__ unsigned long search_index = 0;

__device__ void inorderTraversal_search(struct edge_block *root) {

    if(root == NULL)
        return;
    else {

        inorderTraversal_search(root->lptr);
        search_blocks[search_index++] = root;
        inorderTraversal_search(root->rptr);

    }

}

__global__ void search_pre_processing(struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long curr_source) {

    struct edge_block *root = device_vertex_dictionary->vertex_adjacency[curr_source - 1]->edge_block_address;

    for(unsigned long i = 0 ; i < device_vertex_dictionary->edge_block_count[curr_source - 1] ; i++)
        search_blocks[i] = NULL;
    search_index = 0;

    inorderTraversal_search(root);

}

__global__ void search_edge_kernel_v1(struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long vertex_size, unsigned long total_search_threads, unsigned long search_source, unsigned long search_destination, unsigned long *d_search_flag) {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < total_search_threads) {

        __syncthreads();

        *d_search_flag = 0;

        unsigned long edge_block = id / EDGE_BLOCK_SIZE;
        unsigned long edge_block_entry = id % EDGE_BLOCK_SIZE;

        // printf("Search kernel launch test for source %lu and destination %lu, edge_block %lu and entry %lu\n", search_source, search_destination, edge_block, edge_block_entry);

        // struct adjacency_sentinel_new *edge_sentinel = device_vertex_dictionary->vertex_adjacency[search_source - 1];
        unsigned long extracted_value;

        // struct edge_block *itr = edge_sentinel->edge_block_address + (edge_block);
        // extracted_value = itr->edge_block_entry[edge_block_entry].destination_vertex;
        extracted_value = search_blocks[edge_block]->edge_block_entry[edge_block_entry].destination_vertex;

        // extracted_value = edge_sentinel->edge_block_address[edge_block]->edge_block_entry[edge_block_entry].destination_vertex;

        if(extracted_value == search_destination) {
            *d_search_flag = 1;
            // printf("Edge exists\n");
        }

    }

}



__global__ void delete_edge_kernel(struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long vertex_size, unsigned long total_search_threads, unsigned long search_source, unsigned long search_destination, unsigned long *d_search_flag) {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < total_search_threads) {

        __syncthreads();

        unsigned long edge_block = id / EDGE_BLOCK_SIZE;
        unsigned long edge_block_entry = id % EDGE_BLOCK_SIZE;

        // printf("Search kernel launch test for source %lu and destination %lu, edge_block %lu and entry %lu\n", search_source, search_destination, edge_block, edge_block_entry);

        // struct adjacency_sentinel_new *edge_sentinel = device_vertex_dictionary->vertex_adjacency[search_source - 1];
        unsigned long extracted_value;

        // struct edge_block *itr = edge_sentinel->edge_block_address + (edge_block);
        // extracted_value = itr->edge_block_entry[edge_block_entry].destination_vertex;
        extracted_value = search_blocks[edge_block]->edge_block_entry[edge_block_entry].destination_vertex;

        // extracted_value = edge_sentinel->edge_block_address[edge_block]->edge_block_entry[edge_block_entry].destination_vertex;

        if(extracted_value == search_destination) {
            *d_search_flag = 1;
            search_blocks[edge_block]->edge_block_entry[edge_block_entry].destination_vertex = 0;
            // printf("Edge exists\n");
        }

    }

}

// __device__ struct edge_block *delete_blocks[SEARCH_BLOCKS_COUNT];
// __device__ unsigned long delete_index[] = 0;

__device__ struct edge_block *stack_traversal[1];
__device__ unsigned long stack_top = 0;

__device__ void preorderTraversal_batch_delete(struct edge_block *root, unsigned long start_index, unsigned long end_index, unsigned long *d_csr_edges) {

    if(root == NULL)
        return;
    else {

        struct edge_block *temp = root;
        // struct stack *s_temp = NULL;
        int flag = 1;
        while (flag) {			//Loop run untill temp is null and stack is empty
            
            if (temp) {
            
                // printf ("%p ", temp);

                for(unsigned long i = 0 ; i < EDGE_BLOCK_SIZE ; i++) {

                    for(unsigned long j = start_index ; j < end_index ; j++) {

                        if(temp->edge_block_entry[i].destination_vertex == d_csr_edges[j])
                            temp->edge_block_entry[i].destination_vertex = 0;
                            // if(j >= BATCH_SIZE)
                            //     printf("Hit dispute\n");
                    
                    }
                }

                stack_traversal[stack_top++] = root;
                // push (&s_temp, temp);
                temp = temp->lptr;

            }
            else {
            
                if (stack_top) {
                    
                        // temp = pop (&s_temp);
                        temp = stack_traversal[--stack_top] = root;
                        temp = temp->rptr;

                    }
                else
                    flag = 0;
            }
        }

        // preorderTraversal_batch_delete(root->lptr, start_index, end_index, d_csr_edges);

        // // search_blocks[search_index++] = root;

        // for(unsigned long i = 0 ; i < EDGE_BLOCK_SIZE ; i++) {

        //     for(unsigned long j = start_index ; j < end_index ; j++) {

        //         // if(root->edge_block_entry[i].destination_vertex == d_csr_edges[j])
        //             root->edge_block_entry[i].destination_vertex = 0;
        //             if(j == BATCH_SIZE)
        //                 printf("Hit dispute\n");
            
        //     }
        // }


        // preorderTraversal_batch_delete(root->rptr, start_index, end_index, d_csr_edges);

    }

}

__device__ void inorderTraversal_batch_delete(struct edge_block *root, unsigned long start_index, unsigned long end_index, unsigned long *d_csr_edges) {

    if(root == NULL)
        return;
    else {

        inorderTraversal_batch_delete(root->lptr, start_index, end_index, d_csr_edges);

        // search_blocks[search_index++] = root;

        for(unsigned long i = 0 ; i < EDGE_BLOCK_SIZE ; i++) {

            for(unsigned long j = start_index ; j < end_index ; j++) {

                // if(root->edge_block_entry[i].destination_vertex == d_csr_edges[j])
                    root->edge_block_entry[i].destination_vertex = 0;
                    if(j == BATCH_SIZE)
                        printf("Hit dispute\n");
            
            }
        }


        inorderTraversal_batch_delete(root->rptr, start_index, end_index, d_csr_edges);

    }

}


// __device__ struct edge_block *delete_blocks[VERTEX_BLOCK_SIZE][2];
__device__ struct edge_block *delete_blocks_v1[EDGE_PREALLOCATE_LIST_SIZE];
__device__ unsigned long delete_index[VERTEX_BLOCK_SIZE];
// __device__ unsigned long delete_index_blocks[VERTEX_BLOCK_SIZE];
// __device__ unsigned long delete_source[EDGE_PREALLOCATE_LIST_SIZE];
__device__ unsigned long delete_source[1];
// __device__ unsigned long delete_source_counter[EDGE_PREALLOCATE_LIST_SIZE];

__device__ void inorderTraversal_batch_delete_v1(struct edge_block *root, unsigned long id, unsigned long offset, unsigned long *d_prefix_sum_edge_blocks) {

    if(root == NULL)
        return;
    else {

        inorderTraversal_batch_delete_v1(root->lptr, id, offset, d_prefix_sum_edge_blocks);

        // search_blocks[search_index++] = root;

        // delete_blocks[id][delete_index[id]] = root;

        // if(id != 0)
            // delete_blocks_v1[d_prefix_sum_edge_blocks[id] + delete_index[id]] = root;
        // else
        delete_blocks_v1[delete_index[id] + offset] = root;
        delete_source[delete_index[id]++ + offset] = id;
        // delete_source_counter[delete_index[id]] = delete_index[id]++; 


        // printf("id is %lu, delete_blocks is %p and delete_source is %lu and delete_source_counter is %lu\n", id, delete_blocks[id][delete_index[id] - 1], delete_source[delete_index[id] - 1 + offset], delete_source_counter[delete_index[id] - 1]);
        
        // delete_source[id] = id;

        // for(unsigned long i = 0 ; i < EDGE_BLOCK_SIZE ; i++) {

        //     for(unsigned long j = start_index ; j < end_index ; j++) {

        //         // if(root->edge_block_entry[i].destination_vertex == d_csr_edges[j])
        //             root->edge_block_entry[i].destination_vertex = 0;
        //             if(j == BATCH_SIZE)
        //                 printf("Hit dispute\n");
            
        //     }
        // }


        inorderTraversal_batch_delete_v1(root->rptr, id, offset, d_prefix_sum_edge_blocks);

    }

}

__global__ void batched_delete_preprocessing(struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long vertex_size, unsigned long *d_csr_offset, unsigned long *d_prefix_sum_edge_blocks, unsigned long *d_source_degrees) {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if((id < vertex_size) && (device_vertex_dictionary->vertex_adjacency[id]->edge_block_address != NULL)) {

        struct edge_block *root = device_vertex_dictionary->vertex_adjacency[id]->edge_block_address;

        // for(unsigned long i = 0 ; i < device_vertex_dictionary->edge_block_count[id] ; i++)
        //     search_blocks[i] = NULL;
        delete_index[id] = 0;
        // unsigned long offset = d_csr_offset[id];
        unsigned long offset = 0;
        if(id != 0)
        //     offset = 0;
        // else
            offset = d_prefix_sum_edge_blocks[id - 1];

        // unsigned long start_index = d_csr_offset[id];
        // unsigned long end_index = d_csr_offset[id + 1];

        // for(unsigned long i = start_index; i < end_index ; i++)
        // delete_source[delete_index[id]++] = id;

        inorderTraversal_batch_delete_v1(root, id, offset, d_prefix_sum_edge_blocks);



        // printf("ID is %lu, source is %lu, and address is %p\n", id, id, delete_blocks_v1[offset]);

    }

}

__global__ void batched_delete_kernel_v1(struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long vertex_size, unsigned long total_search_threads, unsigned long *d_csr_offset, unsigned long *d_csr_edges, unsigned long *d_prefix_sum_edge_blocks) {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    // if(id < vertex_size) {
    if(id < total_search_threads) {

        // __syncthreads();

        // *d_search_flag = 0;


        // unsigned long edge_block_entry = edge_block_count % EDGE_BLOCK_SIZE;




        // struct adjacency_sentinel_new *edge_sentinel = device_vertex_dictionary->vertex_adjacency[search_source - 1];
        unsigned long extracted_value;
        unsigned long curr_source = delete_source[id];
        unsigned long start_index = d_csr_offset[curr_source];
        unsigned long end_index = d_csr_offset[curr_source + 1];

        // if(id == 0) {
        //     for(unsigned long i = 0 ; i < 10 ; i++) {
        //         printf("%lu\n", delete_source[i]);
        //     }
        //     printf("\n");
        //     for(unsigned long i = 0 ; i < 10 ; i++) {

        //         if(delete_blocks[i][0] != NULL)
        //             printf("%p\n", delete_blocks[i][0]);

        //     }
        //     printf("\n");
        // }

        
        unsigned long offset = 0;
        unsigned long edge_block_count = d_prefix_sum_edge_blocks[curr_source];
        if(curr_source != 0) {
        //     edge_block_count = d_prefix_sum_edge_blocks[curr_source];
        //     offset = 0;
        // }
        // else {
            edge_block_count = d_prefix_sum_edge_blocks[curr_source] - d_prefix_sum_edge_blocks[curr_source - 1];
            offset = d_prefix_sum_edge_blocks[curr_source - 1];
        }
        // unsigned long edge_block = delete_source_counter[curr_source];

        unsigned long edge_block = id - offset;

        // unsigned long edge_block = d_prefix_sum_edge_blocks;
        // if(curr_source != 0)
        //     edge_block = d_prefix_sum_edge_blocks[curr_source] - d_prefix_sum_edge_blocks[curr_source - 1];
        // if(curr_source == 0)
        // edge_block = id;
        // else
        // edge_block = id - d_prefix_sum_edge_blocks[curr_source - 1];

        // unsigned long tester = 0;
        // if(curr_source != 0)
        //     tester = id - offset;

        // printf("ID is %lu, source is %lu, edge_block is %lu, test is %lu and address is %p\n", id, curr_source, edge_block, id - offset, delete_blocks_v1[offset + edge_block]);

        // printf("Search kernel launch test for id %lu, edge_block_address %p, edge_block_count %lu, source %lu, edge_block %lu, start_index %lu and end_index %lu\n", id, delete_blocks[curr_source][edge_block], edge_block_count, curr_source, edge_block, start_index, end_index);


        // printf("ID is %lu, start_index is %lu, end_index is %lu\n", id, start_index, end_index);

        // struct edge_block *root = device_vertex_dictionary->vertex_adjacency[curr_source]->edge_block_address;

        // struct edge_block *itr = edge_sentinel->edge_block_address + (edge_block);
        // extracted_value = itr->edge_block_entry[edge_block_entry].destination_vertex;

        for(unsigned long j = 0 ; j < EDGE_BLOCK_SIZE ; j++) {



            for(unsigned long i = start_index ; i < end_index ; i++) {


                // delete_blocks_v1[d_prefix_sum_edge_blocks[curr_source] + edge_block]
                // if((delete_blocks[curr_source][edge_block] != NULL) && (delete_blocks[curr_source][edge_block]->edge_block_entry[j].destination_vertex == d_csr_edges[i]))
                //     delete_blocks[curr_source][edge_block]->edge_block_entry[j].destination_vertex = 0;

                if((delete_blocks_v1[offset + edge_block] != NULL) && (delete_blocks_v1[offset + edge_block]->edge_block_entry[j].destination_vertex == d_csr_edges[i]))
                    delete_blocks_v1[offset + edge_block]->edge_block_entry[j].destination_vertex = 0;

            
            }

        }

    }

    // if(id == 0)
    //     printf("Checkpoint final\n");

}

__device__ void preorderTraversal_batch_delete_edge_centric(struct edge_block *root, unsigned long id, unsigned long offset, unsigned long *d_csr_offset) {

    if(root == NULL)
        return;
    else {
        delete_blocks_v1[delete_index[id]++ + offset] = root;

        preorderTraversal_batch_delete_edge_centric(root->lptr, id, offset, d_csr_offset);

        // search_blocks[search_index++] = root;

        // delete_blocks[id][delete_index[id]] = root;

        // if(id != 0)
            // delete_blocks_v1[d_prefix_sum_edge_blocks[id] + delete_index[id]] = root;
        // else
        // delete_source[delete_index[id]++ + offset] = id;
        // delete_source_counter[delete_index[id]] = delete_index[id]++; 


        // printf("id is %lu, delete_blocks is %p and delete_source is %lu and delete_source_counter is %lu\n", id, delete_blocks[id][delete_index[id] - 1], delete_source[delete_index[id] - 1 + offset], delete_source_counter[delete_index[id] - 1]);
        
        // delete_source[id] = id;

        // for(unsigned long i = 0 ; i < EDGE_BLOCK_SIZE ; i++) {

        //     for(unsigned long j = start_index ; j < end_index ; j++) {

        //         // if(root->edge_block_entry[i].destination_vertex == d_csr_edges[j])
        //             root->edge_block_entry[i].destination_vertex = 0;
        //             if(j == BATCH_SIZE)
        //                 printf("Hit dispute\n");
            
        //     }
        // }


        preorderTraversal_batch_delete_edge_centric(root->rptr, id, offset, d_csr_offset);

    }

}

__global__ void batched_delete_preprocessing_edge_centric(struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long vertex_size, unsigned long *d_csr_offset, unsigned long *d_prefix_sum_edge_blocks, unsigned long *d_source_degrees) {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if((id < vertex_size) && (device_vertex_dictionary->vertex_adjacency[id]->edge_block_address != NULL)) {

        struct edge_block *root = device_vertex_dictionary->vertex_adjacency[id]->edge_block_address;

        // for(unsigned long i = 0 ; i < device_vertex_dictionary->edge_block_count[id] ; i++)
        //     search_blocks[i] = NULL;
        delete_index[id] = 0;
        // unsigned long offset = d_csr_offset[id];
        unsigned long offset = 0;
        if(id != 0)
        //     offset = 0;
        // else
            offset = d_prefix_sum_edge_blocks[id - 1];

        // unsigned long start_index = d_csr_offset[id];
        // unsigned long end_index = d_csr_offset[id + 1];

        // for(unsigned long i = start_index; i < end_index ; i++)
        // delete_source[delete_index[id]++] = id;

        preorderTraversal_batch_delete_edge_centric(root, id, offset, d_csr_offset);

        // printf("ID is %lu, source is %lu, and address is %p\n", id, id, delete_blocks_v1[offset]);

    }

    // if(id == 0)
    //     printf("Checkpoint final preprocessing\n");

}

__global__ void batched_delete_kernel_edge_centric(struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long vertex_size, unsigned long total_search_threads, unsigned long *d_csr_offset, unsigned long *d_csr_edges, unsigned long *d_prefix_sum_edge_blocks, unsigned long *d_source, unsigned long *d_destination) {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    // if(id < vertex_size) {
    if(id < total_search_threads) {

        // __syncthreads();

        // *d_search_flag = 0;


        // unsigned long edge_block_entry = edge_block_count % EDGE_BLOCK_SIZE;

        // if(id == 0) {

        //     for(unsigned long i = 0 ; i < BATCH_SIZE ; i++) {

        //         printf("Source is %lu and Destination is %lu\n", d_source[i], d_destination[i]);

        //     }

        // }


        // struct adjacency_sentinel_new *edge_sentinel = device_vertex_dictionary->vertex_adjacency[search_source - 1];
        // unsigned long extracted_value;
        unsigned long curr_source = d_source[id] - 1;
        // unsigned long start_index = d_csr_offset[curr_source];
        // unsigned long end_index = d_csr_offset[curr_source + 1];

        // if(id == 0) {
        //     for(unsigned long i = 0 ; i < 10 ; i++) {
        //         printf("%lu\n", delete_source[i]);
        //     }
        //     printf("\n");
        //     for(unsigned long i = 0 ; i < 10 ; i++) {

        //         if(delete_blocks[i][0] != NULL)
        //             printf("%p\n", delete_blocks[i][0]);

        //     }
        //     printf("\n");
        // }

        
        unsigned long offset = 0;
        unsigned long edge_block_count = device_vertex_dictionary->vertex_adjacency[curr_source]->edge_block_count;
        // unsigned long edge_block_count = d_prefix_sum_edge_blocks[curr_source];
        if(curr_source != 0) {
        //     edge_block_count = d_prefix_sum_edge_blocks[curr_source];
        //     offset = 0;
        // }
        // else {
            // edge_block_count = d_prefix_sum_edge_blocks[curr_source] - d_prefix_sum_edge_blocks[curr_source - 1];
            offset = d_prefix_sum_edge_blocks[curr_source - 1];
        }
        // unsigned long edge_block = delete_source_counter[curr_source];

        

        // unsigned long edge_block = d_prefix_sum_edge_blocks;
        // if(curr_source != 0)
        //     edge_block = d_prefix_sum_edge_blocks[curr_source] - d_prefix_sum_edge_blocks[curr_source - 1];
        // if(curr_source == 0)
        // edge_block = id;
        // else
        // edge_block = id - d_prefix_sum_edge_blocks[curr_source - 1];

        // unsigned long tester = 0;
        // if(curr_source != 0)
        //     tester = id - offset;

        // unsigned long delete_edge = d_csr_edges[id];

        // printf("ID is %lu, source is %lu, edge_block_count is %lu, edge_block is %lu, delete_edge is %lu and address is %p\n", id, d_source[id], edge_block_count, edge_block, d_destination[id], delete_blocks_v1[offset]);

        // printf("Search kernel launch test for id %lu, edge_block_count %lu, source %lu, edge_block %lu, start_index %lu and end_index %lu\n", id, edge_block_count, curr_source, edge_block, start_index, end_index);


        // printf("ID is %lu, start_index is %lu, end_index is %lu\n", id, start_index, end_index);

        // struct edge_block *root = device_vertex_dictionary->vertex_adjacency[curr_source]->edge_block_address;

        // struct edge_block *itr = edge_sentinel->edge_block_address + (edge_block);
        // extracted_value = itr->edge_block_entry[edge_block_entry].destination_vertex;

        // unsigned long breakFlag = 0;

        // unsigned long *destination_entry;

        for(unsigned long edge_block = 0 ; edge_block < edge_block_count ; edge_block++) {
        
            // printf("ID is %lu, source is %lu, edge_block_count is %lu, edge_block is %lu, delete_edge is %lu and address is %p\n", id, d_source[id], edge_block_count, edge_block, d_destination[id], delete_blocks_v1[offset + edge_block]);


            if(delete_blocks_v1[offset + edge_block] != NULL) {

                // unsigned long edge_block_entry_count = delete_blocks_v1[offset + edge_block]->active_edge_count;

                // for(unsigned long j = 0 ; j < delete_blocks_v1[offset + edge_block]->active_edge_count ; j++) {
                for(unsigned long j = 0 ; j < EDGE_BLOCK_SIZE ; j++) {



                    // for(unsigned long i = start_index ; i < end_index ; i++) {


                        // delete_blocks_v1[d_prefix_sum_edge_blocks[curr_source] + edge_block]
                        // if((delete_blocks[curr_source][edge_block] != NULL) && (delete_blocks[curr_source][edge_block]->edge_block_entry[j].destination_vertex == d_csr_edges[i]))
                        //     delete_blocks[curr_source][edge_block]->edge_block_entry[j].destination_vertex = 0;

                    // if(delete_blocks_v1[offset + edge_block] != NULL) {

                        // destination_entry = &(delete_blocks_v1[offset + edge_block]->edge_block_entry[j].destination_vertex);

                        // if(*destination_entry == d_destination[id])
                        //     *destination_entry = INFTY;
                        // else if(*destination_entry == 0)
                        //     goto exit_delete;


                        if(delete_blocks_v1[offset + edge_block]->edge_block_entry[j].destination_vertex == d_destination[id])
                            delete_blocks_v1[offset + edge_block]->edge_block_entry[j].destination_vertex = INFTY;

                        else if(delete_blocks_v1[offset + edge_block]->edge_block_entry[j].destination_vertex == 0)
                            goto exit_delete;


                    // }

                    // if((delete_blocks_v1[offset + edge_block] != NULL) && (delete_blocks_v1[offset + edge_block]->edge_block_entry[j].destination_vertex == d_destination[id]))
                    //     delete_blocks_v1[offset + edge_block]->edge_block_entry[j].destination_vertex = INFTY;
                    
                    // else if((delete_blocks_v1[offset + edge_block] != NULL) && (delete_blocks_v1[offset + edge_block]->edge_block_entry[j].destination_vertex == 0)) {


                    //     goto exit_delete;
                    //     // breakFlag = 1;
                    //     // break;

                    // }


                    
                    // }

                }

            }

            // if(breakFlag)
                // break;

            // printf("\n");
        
        }

        exit_delete:

    }

    // if(id == 0)
    //     printf("Checkpoint final\n");

}

__global__ void batched_delete_kernel_edge_centric_parallelized(struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long vertex_size, unsigned long total_search_threads, unsigned long batch_size, unsigned long *d_csr_offset, unsigned long *d_csr_edges, unsigned long *d_prefix_sum_edge_blocks, unsigned long *d_source, unsigned long *d_destination) {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    // if(id < vertex_size) {
    if(id < total_search_threads) {

        // __syncthreads();

        // unsigned long curr_source = d_source[id] - 1;
        unsigned long curr_source = d_source[id / EDGE_BLOCK_SIZE] - 1;

        unsigned long edge_block_entry = id % EDGE_BLOCK_SIZE;

        // goto exit_delete_v1;


        unsigned long offset = 0;
        unsigned long edge_block_count = device_vertex_dictionary->vertex_adjacency[curr_source]->edge_block_count;
        // unsigned long edge_block_count = d_prefix_sum_edge_blocks[curr_source];
        // unsigned long edge_block = 0;


        if(curr_source != 0) {
        //     edge_block_count = d_prefix_sum_edge_blocks[curr_source];
        //     offset = 0;
        // }
        // else {
            
            // edge_block_count = d_prefix_sum_edge_blocks[curr_source] - d_prefix_sum_edge_blocks[curr_source - 1];
            offset = d_prefix_sum_edge_blocks[curr_source - 1];
        }


        // if(id == 32)


        for(unsigned long edge_block = 0; edge_block < edge_block_count ; edge_block++) {
        
            // printf("ID is %lu, source is %lu, edge_block_count is %lu, edge_block is %lu, delete_edge is %lu and address is %p\n", id, curr_source, edge_block_count, edge_block, d_destination[id], delete_blocks_v1[offset + edge_block]);
            // if(curr_source == 4)
            //     printf("ID is %lu, source is %lu, destination is %lu, entry is %lu\n", id, curr_source + 1, d_destination[id / EDGE_BLOCK_SIZE], delete_blocks_v1[offset + edge_block]->edge_block_entry[edge_block_entry].destination_vertex);


            if(delete_blocks_v1[offset + edge_block] != NULL) {

                // for(unsigned long j = 0 ; j < EDGE_BLOCK_SIZE ; j++) {

                        if(delete_blocks_v1[offset + edge_block]->edge_block_entry[edge_block_entry].destination_vertex == d_destination[id / EDGE_BLOCK_SIZE])
                            delete_blocks_v1[offset + edge_block]->edge_block_entry[edge_block_entry].destination_vertex = INFTY;

                        // else if(delete_blocks_v1[offset + edge_block]->edge_block_entry[edge_block_entry].destination_vertex == 0)
                        //     goto exit_delete;

                // }

            }
        
        }

        exit_delete_v1:

    }

}

__global__ void batched_delete_kernel(struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long vertex_size, unsigned long total_search_threads, unsigned long *d_csr_offset, unsigned long *d_csr_edges) {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < vertex_size) {

        // __syncthreads();

        // unsigned long search_source;
        // for(unsigned long i = 0 ; i < vertex_size + 1 ; i++) {

        //     if(id < d_csr_offset[i]) {
        //         search_source = i - 1;
        //         break;
        //     }

        // }
        // unsigned long index = id - d_csr_offset[search_source];
        // unsigned long search_destination = d_csr_edges[id];
        unsigned long start_index = d_csr_offset[id];
        unsigned long end_index = d_csr_offset[id + 1];


        // printf("ID is %lu, start_index is %lu, end_index is %lu\n", id, start_index, end_index);

        struct edge_block *root = device_vertex_dictionary->vertex_adjacency[id]->edge_block_address;
        if((root != NULL) && (root->active_edge_count > 0) && (start_index < end_index))
            inorderTraversal_batch_delete(root, start_index, end_index, d_csr_edges);
            // preorderTraversal_batch_delete(root, start_index, end_index, d_csr_edges);

        // unsigned long edge_block = id / EDGE_BLOCK_SIZE;
        // unsigned long edge_block_entry = id % EDGE_BLOCK_SIZE;

        // printf("Search kernel launch test for source %lu and destination %lu, edge_block %lu and entry %lu\n", search_source, search_destination, edge_block, edge_block_entry);

        // struct adjacency_sentinel_new *edge_sentinel = device_vertex_dictionary->vertex_adjacency[search_source - 1];
        // unsigned long extracted_value;

        // // struct edge_block *itr = edge_sentinel->edge_block_address + (edge_block);
        // // extracted_value = itr->edge_block_entry[edge_block_entry].destination_vertex;
        // extracted_value = search_blocks[edge_block]->edge_block_entry[edge_block_entry].destination_vertex;

        // // extracted_value = edge_sentinel->edge_block_address[edge_block]->edge_block_entry[edge_block_entry].destination_vertex;

        // if(extracted_value == search_destination) {
        //     *d_search_flag = 1;
        //     search_blocks[edge_block]->edge_block_entry[edge_block_entry].destination_vertex = 0;
        //     // printf("Edge exists\n");
        // }

    }

    // if(id == 0)
    //     printf("Checkpoint final\n");

}

__global__ void correctness_check_kernel(struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long vertex_size, unsigned long edge_size, unsigned long *d_csr_offset, unsigned long *d_csr_edges, unsigned long *d_correctness_flag) {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < vertex_size) {

        // if(id == 0) {
        //     printf("Device csr offset\n");
        //     for(unsigned long i = 0 ; i < vertex_size + 1 ; i++)
        //         printf("%lu ", d_csr_offset[i]);
        //     printf("\n");
        // }

        unsigned long existsFlag = 0;

        // for(unsigned long i = 0 ; i < edge_size ; i++)
        //     printf("%lu and %lu\n", d_c_source[i], d_c_destination[i]);
        
        // printf("*-------------*\n");

        unsigned long start_index = d_csr_offset[id];
        unsigned long end_index = d_csr_offset[id + 1];
        unsigned long curr_source = id + 1;

        // printf("Source=%lu, Start index=%lu, End index=%lu\n", curr_source, start_index, end_index);

        for(unsigned long i = start_index ; i < end_index ; i++) {

            // unsigned long curr_source = d_c_source[i];
            unsigned long curr_destination = d_csr_edges[i];

            // printf("%lu and %lu\n", curr_source, curr_destination);

            existsFlag = 0;

            // here checking at curr_source - 1, since source vertex 10 would be stored at index 9
            if((device_vertex_dictionary->vertex_adjacency[curr_source - 1] != NULL) && (device_vertex_dictionary->vertex_id[curr_source - 1] != 0)) {
                // printf("%lu -> , edge blocks = %lu, edge sentinel = %p, active edge count = %lu, destination vertices -> ", device_vertex_dictionary->vertex_id[i], device_vertex_dictionary->edge_block_count[i], device_vertex_dictionary->vertex_adjacency[i], device_vertex_dictionary->vertex_adjacency[i]->active_edge_count);

                unsigned long edge_block_counter = 0;
                unsigned long edge_block_entry_count = 0;
                // struct edge_block *itr = device_vertex_dictionary->vertex_adjacency[curr_source - 1]->edge_block_address[edge_block_counter];
                struct edge_block *itr = device_vertex_dictionary->vertex_adjacency[curr_source - 1]->edge_block_address + (edge_block_counter);

                for(unsigned long j = 0 ; j < device_vertex_dictionary->vertex_adjacency[curr_source - 1]->active_edge_count ; j++) {
                                
                    // printf("%lu ", itr->edge_block_entry[edge_block_entry_count].destination_vertex);
                
                    // edge_block_entry_count++;

                    if(itr->edge_block_entry[edge_block_entry_count].destination_vertex == curr_destination) {
                        existsFlag = 1;
                        // printf("Found %lu and %lu\n", curr_source, curr_destination);
                        break;
                    }

                    if(++edge_block_entry_count >= EDGE_BLOCK_SIZE) {
                        // itr = itr->next;
                        // itr = device_vertex_dictionary->vertex_adjacency[curr_source - 1]->edge_block_address[++edge_block_counter];
                        itr = device_vertex_dictionary->vertex_adjacency[curr_source - 1]->edge_block_address + (++edge_block_counter);
                        edge_block_entry_count = 0;
                    }
                }

                if(!existsFlag) {
                    // printf("Issue at id=%lu destination=%lu\n", id, curr_destination);
                    break;
                }

                // printf("\n");    

            }

        }

        if(!existsFlag)
            *d_correctness_flag = 1;
        else
            *d_correctness_flag = 0;

        // printf("*---------------*\n\n");

    }



}

__device__ unsigned long find_edge(struct edge_block *root, unsigned long curr_source, unsigned long curr_destination) {

    unsigned long existsFlag = 0;
    unsigned long edge_block_counter = 0;
    unsigned long edge_block_entry_count = 0;
    // struct edge_block *itr = device_vertex_dictionary->vertex_adjacency[curr_source - 1]->edge_block_address[edge_block_counter];
    // struct edge_block *itr = device_vertex_dictionary->vertex_adjacency[curr_source - 1]->edge_block_address + (edge_block_counter);

    for(unsigned long j = 0 ; j < root->active_edge_count ; j++) {
                    
        // printf("%lu ", itr->edge_block_entry[edge_block_entry_count].destination_vertex);
    
        // edge_block_entry_count++;

        if(root->edge_block_entry[edge_block_entry_count++].destination_vertex == curr_destination) {
            existsFlag = 1;
            // printf("%lu found at %lu in source %lu\n", curr_destination, edge_block_entry_count, curr_source);
            // printf("Found %lu and %lu\n", curr_source, curr_destination);
            break;
        }

        // if(++edge_block_entry_count >= EDGE_BLOCK_SIZE) {
        //     // itr = itr->next;
        //     // itr = device_vertex_dictionary->vertex_adjacency[curr_source - 1]->edge_block_address[++edge_block_counter];
        //     itr = device_vertex_dictionary->vertex_adjacency[curr_source - 1]->edge_block_address + (++edge_block_counter);
        //     edge_block_entry_count = 0;
        // }
    }

    unsigned long existsFlag_lptr = 0;
    unsigned long existsFlag_rptr = 0;

    if(root->lptr != NULL)
        existsFlag_lptr = find_edge(root->lptr, curr_source, curr_destination);
    if(root->rptr != NULL)
        existsFlag_rptr = find_edge(root->rptr, curr_source, curr_destination);

    return (existsFlag || existsFlag_lptr || existsFlag_rptr);
}

__global__ void correctness_check_kernel_v1(struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long vertex_size, unsigned long edge_size, unsigned long *d_csr_offset, unsigned long *d_csr_edges, unsigned long *d_correctness_flag) {

    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < vertex_size) {

        // if(id == 0) {
        //     printf("Device csr offset\n");
        //     for(unsigned long i = 0 ; i < vertex_size + 1 ; i++)
        //         printf("%lu ", d_csr_offset[i]);
        //     printf("\n");
        // }

        unsigned long existsFlag = 0;

        // for(unsigned long i = 0 ; i < edge_size ; i++)
        //     printf("%lu and %lu\n", d_c_source[i], d_c_destination[i]);
        
        // printf("*-------------*\n");

        unsigned long start_index = d_csr_offset[id];
        unsigned long end_index = d_csr_offset[id + 1];
        unsigned long curr_source = id + 1;

        // printf("Source=%lu, Start index=%lu, End index=%lu\n", curr_source, start_index, end_index);

        for(unsigned long i = start_index ; i < end_index ; i++) {

            // unsigned long curr_source = d_c_source[i];
            unsigned long curr_destination = d_csr_edges[i];

            // printf("%lu and %lu\n", curr_source, curr_destination);

            existsFlag = 0;

            // here checking at curr_source - 1, since source vertex 10 would be stored at index 9
            if((device_vertex_dictionary->vertex_adjacency[curr_source - 1] != NULL) && (device_vertex_dictionary->vertex_id[curr_source - 1] != 0)) {
                // printf("%lu -> , edge blocks = %lu, edge sentinel = %p, active edge count = %lu, destination vertices -> ", device_vertex_dictionary->vertex_id[i], device_vertex_dictionary->edge_block_count[i], device_vertex_dictionary->vertex_adjacency[i], device_vertex_dictionary->vertex_adjacency[i]->active_edge_count);

                existsFlag = find_edge(device_vertex_dictionary->vertex_adjacency[curr_source - 1]->edge_block_address, curr_source, curr_destination);
                // printf("%lu found in source %lu\n", curr_destination, curr_source);
                
                // if(!existsFlag)
                //     printf("Issue at id=%lu destination=%lu\n", id, curr_destination);

                // unsigned long edge_block_counter = 0;
                // unsigned long edge_block_entry_count = 0;
                // // struct edge_block *itr = device_vertex_dictionary->vertex_adjacency[curr_source - 1]->edge_block_address[edge_block_counter];
                // struct edge_block *itr = device_vertex_dictionary->vertex_adjacency[curr_source - 1]->edge_block_address + (edge_block_counter);

                // for(unsigned long j = 0 ; j < device_vertex_dictionary->vertex_adjacency[curr_source - 1]->active_edge_count ; j++) {
                                
                //     // printf("%lu ", itr->edge_block_entry[edge_block_entry_count].destination_vertex);
                
                //     // edge_block_entry_count++;

                //     if(itr->edge_block_entry[edge_block_entry_count].destination_vertex == curr_destination) {
                //         existsFlag = 1;
                //         // printf("Found %lu and %lu\n", curr_source, curr_destination);
                //         break;
                //     }

                //     if(++edge_block_entry_count >= EDGE_BLOCK_SIZE) {
                //         // itr = itr->next;
                //         // itr = device_vertex_dictionary->vertex_adjacency[curr_source - 1]->edge_block_address[++edge_block_counter];
                //         itr = device_vertex_dictionary->vertex_adjacency[curr_source - 1]->edge_block_address + (++edge_block_counter);
                //         edge_block_entry_count = 0;
                //     }
                // }

                if(!existsFlag) {
                    printf("Issue at id=%lu, source=%lu and destination=%lu\n", id, curr_source, curr_destination);
                    break;
                }

                // printf("\n");    

            }
            // else if(device_vertex_dictionary->vertex_adjacency[curr_source - 1]->active_edge_count == 0)
            //     existsFlag = 1;

        }

        // this means degree of that source vertex is 0
        if(start_index == end_index)
            existsFlag = 1;

        if(!existsFlag)
            *d_correctness_flag = 1;
        else
            *d_correctness_flag = 0;

        // printf("*---------------*\n\n");

    }



}

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
            // struct edge_block *itr = device_vertex_dictionary->vertex_adjacency[i]->edge_block_address[edge_block_counter];
            struct edge_block *itr = device_vertex_dictionary->vertex_adjacency[i]->edge_block_address;

            for(unsigned long j = 0 ; j < device_vertex_dictionary->vertex_adjacency[i]->active_edge_count ; j++) {
                            
                printf("%lu ", itr->edge_block_entry[edge_block_entry_count].destination_vertex);
            
                // edge_block_entry_count++;

                if(++edge_block_entry_count >= EDGE_BLOCK_SIZE) {
                    // itr = itr->next;
                    // itr = device_vertex_dictionary->vertex_adjacency[i]->edge_block_address[++edge_block_counter];
                    itr = itr + 1;
                    edge_block_entry_count = 0;
                }
            }
            printf("\n");            
    
        }



    }


    printf("VDS = %lu\n", d_v_d_sentinel.vertex_count);
    printf("K2 counter = %u\n", k2counter);

}

__device__ void preorderTraversal(struct edge_block *root) {

    if(root == NULL)
        return;
    
    else {

        printf("\nedge block edge count = %lu, %p, ", root->active_edge_count, root);

        for(unsigned long j = 0 ; j < root->active_edge_count ; j++) {
                        
            printf("%lu ", root->edge_block_entry[j].destination_vertex);
        
            // edge_block_entry_count++;

            // if(++edge_block_entry_count >= EDGE_BLOCK_SIZE) {
            //     // itr = itr->next;
            //     // itr = device_vertex_dictionary->vertex_adjacency[i]->edge_block_address[++edge_block_counter];
            //     itr = itr + 1;
            //     edge_block_entry_count = 0;
            // }
        }

        preorderTraversal(root->lptr);



        // printf("\n");

        preorderTraversal(root->rptr);

    }

    // unsigned long edge_block_counter = 0;
    // unsigned long edge_block_entry_count = 0;
    // // struct edge_block *itr = device_vertex_dictionary->vertex_adjacency[i]->edge_block_address[edge_block_counter];
    // struct edge_block *itr = device_vertex_dictionary->vertex_adjacency[i]->edge_block_address;


    // for(unsigned long j = 0 ; j < device_vertex_dictionary->vertex_adjacency[i]->active_edge_count ; j++) {
                    
    //     printf("%lu ", itr->edge_block_entry[edge_block_entry_count].destination_vertex);
    
    //     // edge_block_entry_count++;

    //     if(++edge_block_entry_count >= EDGE_BLOCK_SIZE) {
    //         // itr = itr->next;
    //         // itr = device_vertex_dictionary->vertex_adjacency[i]->edge_block_address[++edge_block_counter];
    //         itr = itr + 1;
    //         edge_block_entry_count = 0;
    //     }
    // }

}

__global__ void printKernelmodded_v2(struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long size) {

    printf("Printing Linked List\n");
    printf("Active Vertex Count = %lu\n", device_vertex_dictionary->active_vertex_count);

    // struct vertex_block *ptr = d_v_d_sentinel.next;

    unsigned long vertex_block = 0;

    // printf("%lu hidden value\n", device_vertex_dictionary->vertex_adjacency[4]->edge_block_address->active_edge_count);

    for(unsigned long i = 0 ; i < device_vertex_dictionary->active_vertex_count ; i++) {

        // printf("Checkpoint\n");

        if((device_vertex_dictionary->vertex_adjacency[i] != NULL) && (device_vertex_dictionary->vertex_id[i] != 0)) {
            printf("%lu -> , edge blocks = %lu, edge sentinel = %p, root = %p, active edge count = %lu, destination vertices -> ", device_vertex_dictionary->vertex_id[i], device_vertex_dictionary->vertex_adjacency[i]->edge_block_count, device_vertex_dictionary->vertex_adjacency[i], device_vertex_dictionary->vertex_adjacency[i]->edge_block_address, device_vertex_dictionary->vertex_adjacency[i]->active_edge_count);

            preorderTraversal(device_vertex_dictionary->vertex_adjacency[i]->edge_block_address);
            printf("\n");   

            // unsigned long edge_block_counter = 0;
            // unsigned long edge_block_entry_count = 0;
            // // struct edge_block *itr = device_vertex_dictionary->vertex_adjacency[i]->edge_block_address[edge_block_counter];
            // struct edge_block *itr = device_vertex_dictionary->vertex_adjacency[i]->edge_block_address;


            // for(unsigned long j = 0 ; j < device_vertex_dictionary->vertex_adjacency[i]->active_edge_count ; j++) {
                            
            //     printf("%lu ", itr->edge_block_entry[edge_block_entry_count].destination_vertex);
            
            //     // edge_block_entry_count++;

            //     if(++edge_block_entry_count >= EDGE_BLOCK_SIZE) {
            //         // itr = itr->next;
            //         // itr = device_vertex_dictionary->vertex_adjacency[i]->edge_block_address[++edge_block_counter];
            //         itr = itr + 1;
            //         edge_block_entry_count = 0;
            //     }
            // }
            // printf("\n");            
    
        }

        // else {

        //     printf("Hit\n");

        // }



    }


    printf("VDS = %lu\n", d_v_d_sentinel.vertex_count);
    printf("K2 counter = %u\n", k2counter);

}

__global__ void cbt_stats(struct vertex_dictionary_structure *device_vertex_dictionary, unsigned long size) {

    printf("Printing Bucket Stats\n");
    unsigned long tree_height[100];
    unsigned long max_height = 0;

    for(unsigned long i = 0 ; i < device_vertex_dictionary->active_vertex_count ; i++) {

        if((device_vertex_dictionary->vertex_adjacency[i] != NULL) && (device_vertex_dictionary->vertex_id[i] != 0)) {
            // printf("%lu -> , edge blocks = %lu, edge sentinel = %p, root = %p, active edge count = %lu, destination vertices -> ", device_vertex_dictionary->vertex_id[i], device_vertex_dictionary->edge_block_count[i], device_vertex_dictionary->vertex_adjacency[i], device_vertex_dictionary->vertex_adjacency[i]->edge_block_address, device_vertex_dictionary->vertex_adjacency[i]->active_edge_count);

            unsigned long height = floor(log2(ceil(double(device_vertex_dictionary->vertex_adjacency[i]->active_edge_count) / EDGE_BLOCK_SIZE)));
            tree_height[height]++;
            if(height > max_height)
                max_height = height;
            // inorderTraversal(device_vertex_dictionary->vertex_adjacency[i]->edge_block_address);
            // printf("\n");   

        }

    }

    for(unsigned long i = 0 ; i <= max_height ; i++) {
        printf("Height %lu has %lu vertices\n", i, tree_height[i]);
        tree_height[i] = 0;
    }


}

void readFile(char *fileLoc, struct graph_properties *h_graph_prop, thrust::host_vector <unsigned long> &h_source, thrust::host_vector <unsigned long> &h_destination, thrust::host_vector <unsigned long> &h_source_degrees) {

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
            h_source_degrees[source - 1]++;

            // below part makes it an undirected graph
            h_source[index] = destination;
            h_destination[index++] = source;
            h_source_degrees[destination - 1]++;

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
            h_source_degrees.resize(h_graph_prop->xDim);


            dataFlag = 1;

        }

    }
    
    fclose(ptr);

}

void generateCSR(unsigned long vertex_size, unsigned long edge_size, thrust::host_vector <unsigned long> &h_source, thrust::host_vector <unsigned long> &h_destination, thrust::host_vector <unsigned long> &h_csr_offset, thrust::host_vector <unsigned long> &h_csr_edges, thrust::host_vector <unsigned long> &h_source_degrees) {

    h_csr_offset[0] = 0;
    h_csr_offset[1] = h_source_degrees[0];
    for(unsigned long i = 2 ; i < vertex_size + 1 ; i++) 
        h_csr_offset[i] += h_csr_offset[i-1] + h_source_degrees[i - 1];

    thrust::host_vector <unsigned long> index(vertex_size);
    thrust::fill(index.begin(), index.end(), 0);

    // std::cout << "Checkpoint 2" << std::endl;

    for(unsigned long i = 0 ; i < edge_size ; i++) {

        if(h_source[i] == 1)
            h_csr_edges[index[h_source[i] - 1]++] = h_destination[i];        
        else
            h_csr_edges[h_csr_offset[h_source[i] - 1] + index[h_source[i] - 1]++] = h_destination[i];

    }

}

void generateCSRnew(unsigned long vertex_size, unsigned long edge_size, thrust::host_vector <unsigned long> &h_source, thrust::host_vector <unsigned long> &h_destination, thrust::host_vector <unsigned long> &h_csr_offset_new, thrust::host_vector <unsigned long> &h_csr_edges_new, thrust::host_vector <unsigned long> &h_source_degrees_new, thrust::host_vector <unsigned long> &h_edge_blocks_count, unsigned long batch_size, unsigned long total_batches) {

    unsigned long batch_offset = 0;
    unsigned long batch_number = 0;
    for(unsigned long i = 0 ; i < edge_size ; i++) {

        h_source_degrees_new[h_source[i] - 1 + batch_offset]++;
        // h_source_degrees_new[h_source[i] - 1]++;

        if((((i + 1) % batch_size ) == 0) && (i != 0)) {
            batch_offset = vertex_size * ++batch_number;
            // for(unsigned long i = 0 ; i < vertex_size * total_batches ; i++)
            //     std::cout << h_source_degrees_new[i] << " ";
            // std::cout << "Hit at CSR batch, new batch offset is " << batch_offset << std::endl;
        }

    }

    batch_number = 0;
    batch_offset = 0;
    unsigned long k = 0;
    for(unsigned long i = 0 ; i < total_batches ; i++) {


        h_csr_offset_new[batch_offset] = 0;
        h_csr_offset_new[batch_offset + 1] = h_source_degrees_new[batch_offset - k];
        // for(unsigned long i = 0 ; i < (vertex_size + 1) * total_batches ; i++) { 
        for(unsigned long j = 2 ; j < (vertex_size + 1) ; j++)
            h_csr_offset_new[j + batch_offset] += h_csr_offset_new[j - 1 + batch_offset] + h_source_degrees_new[j - 1 + batch_offset - k];

        batch_offset = (vertex_size + 1) * ++batch_number;
        k++;

    }
    thrust::host_vector <unsigned long> index(vertex_size * total_batches);
    // thrust::fill(index.begin(), index.end(), 0);

    // // // std::cout << "Checkpoint 2" << std::endl;

    batch_number = 0;
    batch_offset = 0;
    unsigned long batch_offset_index = 0;
    unsigned long batch_offset_csr = 0;
    for(unsigned long i = 0 ; i < edge_size ; i++) {

        if(h_source[i] == 1)
            h_csr_edges_new[index[h_source[i] - 1 + batch_offset_index]++ + batch_offset] = h_destination[i];        
        else
            h_csr_edges_new[h_csr_offset_new[h_source[i] - 1 + batch_offset_csr] + index[h_source[i] - 1 + batch_offset_index]++ + batch_offset] = h_destination[i];
        
        if((((i + 1) % batch_size ) == 0) && (i != 0)) {
            batch_offset = batch_size * ++batch_number;
            batch_offset_index = vertex_size * batch_number;
            batch_offset_csr = (vertex_size + 1) * batch_number;
            // for(unsigned long i = 0 ; i < vertex_size * total_batches ; i++)
            //     std::cout << h_source_degrees_new[i] << " ";
            std::cout << "Hit at CSR batch, new batch offset is " << batch_offset << std::endl;
        }

    }

    // for(unsigned long i = batch_number * batch_size; i < batch_number * batch_size + edge_size ; i++) {

    //     if(h_source[i] == 1)
    //         h_csr_edges_new[index[h_source[i] - 1]++] = h_destination[i];        
    //     else
    //         h_csr_edges_new[h_csr_offset_new[h_source[i] - 1] + index[h_source[i] - 1]++] = h_destination[i];

    // }

    thrust::host_vector <unsigned long> space_remaining(vertex_size);
    unsigned long total_edge_blocks_count_init = 0;

    std::cout << "Space Remaining" << std::endl;
    // std::cout << "Edge blocks calculation" << std::endl << "Source\tEdge block count\tGPU address" << std::endl;
    for(unsigned long i = 0 ; i < total_batches ; i++) {

        for(unsigned long j = 0 ; j < vertex_size ; j++) {

            if(h_source_degrees_new[j + (i * vertex_size)]) {

                unsigned long edge_blocks;
                if(i != 0)   
                    edge_blocks = ceil(double(h_source_degrees_new[j + (i * vertex_size)] - space_remaining[j]) / EDGE_BLOCK_SIZE);
                else
                    edge_blocks = ceil(double(h_source_degrees_new[j + (i * vertex_size)]) / EDGE_BLOCK_SIZE);


                h_edge_blocks_count[j + (i * vertex_size)] = edge_blocks;
                total_edge_blocks_count_init += edge_blocks;

                // if(h_source_degrees_new[j + (i * vertex_size)])
                space_remaining[j] = (h_source_degrees_new[j + (i * vertex_size)] + space_remaining[j]) % EDGE_BLOCK_SIZE;
                // else
                //     space_remaining[j] = 0;

            }

            else
                h_edge_blocks_count[j + (i * vertex_size)] = 0;


        }

        for(unsigned long j = 0 ; j < vertex_size ; j++)
            std::cout << space_remaining[j] << " ";
        std::cout << std::endl;

    }

    std::cout << std::endl << std::endl << "Printing batched CSR" << std::endl << "Source degrees\t\t" << std::endl;
    for(unsigned long i = 0 ; i < vertex_size * total_batches ; i++) {
        std::cout << h_source_degrees_new[i] << " ";
        if((i + 1) % vertex_size == 0)
            std::cout << std::endl;
    }
    std::cout << std::endl << "CSR offset\t\t" << std::endl;
    for(unsigned long i = 0 ; i < (vertex_size + 1) * total_batches ; i++) {
        std::cout << h_csr_offset_new[i] << " ";
        if(((i + 1) % (vertex_size + 1)) == 0)
            std::cout << std::endl;
    }
    std::cout << std::endl << "CSR edges\t\t" << std::endl;
    for(unsigned long i = 0 ; i < batch_size * total_batches ; i++) {
        std::cout << h_csr_edges_new[i] << " ";
        if((i + 1) % batch_size == 0)
            std::cout << std::endl;
    }
    std::cout << std::endl << "Edge blocks count\t\t" << std::endl;
    for(unsigned long i = 0 ; i < vertex_size * total_batches ; i++) {
        std::cout << h_edge_blocks_count[i] << " ";
        if((i + 1) % vertex_size == 0)
            std::cout << std::endl;
    }
    std::cout << std::endl << std::endl << std::endl;
}

void generate_random_batch(unsigned long vertex_size, unsigned long batch_size, thrust::host_vector <unsigned long> &h_source, thrust::host_vector <unsigned long> &h_destination, thrust::host_vector <unsigned long> &h_source_degrees_new) {

    // unsigned long batch_size = 10;
    // unsigned long vertex_size = 30;

    unsigned long seed = 0;
    unsigned long range = 0;
    unsigned long offset = 0;

    // unsigned long source_array[10];
    // unsigned long destination_array[10];

	srand(seed + 1);
	for (unsigned long i = 0; i < batch_size / 2; ++i)
	{
        // EdgeUpdateType edge_update_data;
        unsigned long intermediate = rand() % ((range && (range < vertex_size)) ? range : vertex_size);
        unsigned long source;
        if(offset + intermediate < vertex_size)
            source = offset + intermediate;
        else
            source = intermediate;
        h_source[i] = source + 1;
        h_destination[i] = (rand() % vertex_size) + 1;
        // edge_update->edge_update.push_back(edge_update_data);
	}

  for (unsigned long i = batch_size / 2; i < batch_size; ++i)
	{
        // EdgeUpdateType edge_update_data;
        unsigned long intermediate = rand() % (vertex_size);
        unsigned long source;
        if(offset + intermediate < vertex_size)
            source = offset + intermediate;
        else
            source = intermediate;
        h_source[i] = source + 1;
        h_destination[i] = (rand() % vertex_size) + 1;
        // edge_update->edge_update.push_back(edge_update_data);
	}

}

void generate_csr_batch(unsigned long vertex_size, unsigned long edge_size, thrust::host_vector <unsigned long> &h_source, thrust::host_vector <unsigned long> &h_destination, thrust::host_vector <unsigned long> &h_csr_offset_new, thrust::host_vector <unsigned long> &h_csr_edges_new, thrust::host_vector <unsigned long> &h_source_degrees_new, thrust::host_vector <unsigned long> &h_edge_blocks_count, thrust::host_vector <unsigned long> &h_prefix_sum_edge_blocks_new, unsigned long *h_batch_update_data, unsigned long batch_size, unsigned long total_batches, unsigned long batch_number,  thrust::host_vector <unsigned long> &space_remaining, unsigned long *total_edge_blocks_count_batch, clock_t *init_time) {

    thrust::host_vector <unsigned long> index(vertex_size);
    thrust::fill(h_source_degrees_new.begin(), h_source_degrees_new.end(), 0);

    // calculating start and end index of this batch, for use with h_source and h_destination
    unsigned long start_index = batch_number * batch_size;
    unsigned long end_index = start_index + batch_size;
    if(end_index > edge_size)
        end_index = edge_size;

    // thrust::host_vector <unsigned long> h_source_degrees_new_prev(vertex_size);
    // thrust::copy(h_source_degrees_new.begin(), h_source_degrees_new.end(), h_source_degrees_new_prev.begin());

    // unsigned long max = h_source_degrees_new[0];
    // unsigned long min = h_source_degrees_new[0];
    // unsigned long sum = h_source_degrees_new[0];
    // unsigned long non_zero_count = 0;

    // if(h_source_degrees_new[0])
    //     non_zero_count++;

    // calculating source degrees of this batch
    for(unsigned long i = start_index ; i < end_index ; i++) {

        h_source_degrees_new[h_source[i] - 1]++;

    }

    // std::cout << "Checkpoint 1" << std::endl;

    // for(unsigned long i = 1 ; i < vertex_size ; i++) {

    //     if(h_source_degrees_new[i] > max)
    //         max = h_source_degrees_new[i];

    //     if(h_source_degrees_new[i] < min)
    //         min = h_source_degrees_new[i];

    //     sum += h_source_degrees_new[i];

    //     if(h_source_degrees_new[i])
    //         non_zero_count++;
        
    // }

    // calculating csr offset of this batch
    h_csr_offset_new[0] = 0;
    h_csr_offset_new[1] = h_source_degrees_new[0];
    // h_batch_update_data[0] = 0;
    // h_batch_update_data[1] = h_source_degrees_new[0];
    for(unsigned long j = 2 ; j < (vertex_size + 1) ; j++) {
        h_csr_offset_new[j] = h_csr_offset_new[j - 1] + h_source_degrees_new[j - 1];
        // h_batch_update_data[j] = h_batch_update_data[j - 1] + h_source_degrees_new[j - 1];
    }

    // std::cout << "Checkpoint 2 , start_index is " << start_index << " and end_index is " << end_index << std::endl;


    // unsigned long offset = vertex_size + 1;
    unsigned long offset = 0;

    // calculating csr edges of this batch
    for(unsigned long i = start_index ; i < end_index ; i++) {

        if(h_source[i] == 1) {
            // h_csr_edges_new[index[h_source[i] - 1]++] = h_destination[i];  
            h_csr_edges_new[index[h_source[i] - 1]] = h_destination[i];  
            h_batch_update_data[offset + index[h_source[i] - 1]++] = h_destination[i];  
        }      
        else {
            // h_csr_edges_new[h_csr_offset_new[h_source[i] - 1] + index[h_source[i] - 1]++] = h_destination[i];
            h_csr_edges_new[h_csr_offset_new[h_source[i] - 1] + index[h_source[i] - 1]] = h_destination[i];
            h_batch_update_data[offset + h_csr_offset_new[h_source[i] - 1] + index[h_source[i] - 1]++] = h_destination[i];
        }
    }

    // std::cout << "Checkpoint 3" << std::endl;

    // calculating edge blocks required for this batch
    // unsigned long total_edge_blocks_count_init = 0;
    for(unsigned long j = 0 ; j < vertex_size ; j++) {

        if(h_source_degrees_new[j]) {

            unsigned long edge_blocks;
            if(batch_number != 0) {
                if(space_remaining[j] == 0) {
                    edge_blocks = ceil(double(h_source_degrees_new[j]) / EDGE_BLOCK_SIZE);
                    space_remaining[j] = (EDGE_BLOCK_SIZE - (h_source_degrees_new[j] % EDGE_BLOCK_SIZE)) % EDGE_BLOCK_SIZE;                    
                }
                else if(h_source_degrees_new[j] >= space_remaining[j]) {
                    edge_blocks = ceil(double(h_source_degrees_new[j] - space_remaining[j]) / EDGE_BLOCK_SIZE);
                    // space_remaining[j] = (h_source_degrees_new[j] - space_remaining[j]) % EDGE_BLOCK_SIZE;
                    space_remaining[j] = (EDGE_BLOCK_SIZE - ((h_source_degrees_new[j] - space_remaining[j]) % EDGE_BLOCK_SIZE)) % EDGE_BLOCK_SIZE;
                }
                else {
                    edge_blocks = 0;
                    space_remaining[j] = space_remaining[j] - h_source_degrees_new[j];
                }
                // h_prefix_sum_edge_blocks_new[j] = h_prefix_sum_edge_blocks_new[j - 1] + h_edge_blocks_count[j];
            }
            else {
                edge_blocks = ceil(double(h_source_degrees_new[j]) / EDGE_BLOCK_SIZE);
                space_remaining[j] = (EDGE_BLOCK_SIZE - (h_source_degrees_new[j] % EDGE_BLOCK_SIZE)) % EDGE_BLOCK_SIZE;
                // h_prefix_sum_edge_blocks_new[0] = edge_blocks;
            }

            // if((vertex_adjacency->edge_block_address != NULL) && (vertex_adjacency->last_insert_edge_offset != 0)) {

            //     space_remaining = EDGE_BLOCK_SIZE - device_vertex_dictionary->vertex_adjacency[id]->last_insert_edge_offset;
            //     // printf("id=%lu, last_insert_edge_offset is %lu\n", id, device_vertex_dictionary->vertex_adjacency[id]->last_insert_edge_offset);
            // }

            h_edge_blocks_count[j] = edge_blocks;
            *total_edge_blocks_count_batch += edge_blocks;

            // if(space_remaining[j] <= h_source_degrees_new[j])
            //     space_remaining[j] = (h_source_degrees_new[j] - space_remaining[j]) % EDGE_BLOCK_SIZE;
            // else
            //     space_remaining[j] = space_remaining[j] - h_source_degrees_new[j];

        }

        else
            h_edge_blocks_count[j] = 0;


    }

    // if((vertex_adjacency->edge_block_address != NULL) && (vertex_adjacency->last_insert_edge_offset != 0)) {

    //     space_remaining = EDGE_BLOCK_SIZE - device_vertex_dictionary->vertex_adjacency[id]->last_insert_edge_offset;
    //     // printf("id=%lu, last_insert_edge_offset is %lu\n", id, device_vertex_dictionary->vertex_adjacency[id]->last_insert_edge_offset);
    // }

    // offset += batch_size;

    // clock_t temp_time;
    // temp_time = clock();

    h_prefix_sum_edge_blocks_new[0] = h_edge_blocks_count[0];
    // h_batch_update_data[offset] = h_edge_blocks_count[0];
    // printf("Prefix sum array edge blocks\n%ld ", h_prefix_sum_edge_blocks[0]);
    for(unsigned long i = 1 ; i < vertex_size ; i++) {

        h_prefix_sum_edge_blocks_new[i] = h_prefix_sum_edge_blocks_new[i - 1] + h_edge_blocks_count[i];
        // h_batch_update_data[offset + i] = h_batch_update_data[offset + i - 1] + h_edge_blocks_count[i];
        // printf("%ld ", h_prefix_sum_edge_blocks[i]);

    }
    // temp_time = clock() - temp_time;
    // *init_time += temp_time;

    // printf("Max, Min, Average, and Non-zero Average degrees in this batch are %lu, %lu, %f, and %f respectively\n", max, min, float(sum) / vertex_size, float(sum) / non_zero_count);

    // std::cout << std::endl << std::endl << "Printing batched CSR" << std::endl << "Source degrees\t\t" << std::endl;
    // for(unsigned long i = 0 ; i < vertex_size ; i++) {
    //     std::cout << h_source_degrees_new[i] << " ";
    //     if((i + 1) % vertex_size == 0)
    //         std::cout << std::endl;
    // }
    // std::cout << std::endl << "CSR offset\t\t" << std::endl;
    // for(unsigned long i = 0 ; i < (vertex_size + 1) ; i++) {
    //     std::cout << h_csr_offset_new[i] << " ";
    //     if(((i + 1) % (vertex_size + 1)) == 0)
    //         std::cout << std::endl;
    // }
    // std::cout << std::endl << "CSR edges\t\t" << std::endl;
    // for(unsigned long i = 0 ; i < batch_size ; i++) {
    //     std::cout << h_csr_edges_new[i] << " ";
    //     if((i + 1) % batch_size == 0)
    //         std::cout << std::endl;
    // }
    // std::cout << std::endl << "Edge blocks count\t\t" << std::endl;
    // for(unsigned long i = 0 ; i < vertex_size ; i++) {
    //     std::cout << h_edge_blocks_count[i] << " ";
    //     if((i + 1) % vertex_size == 0)
    //         std::cout << std::endl;
    // }
    // std::cout << std::endl << "Prefix sum edge blocks\t\t" << std::endl;
    // for(unsigned long i = 0 ; i < vertex_size ; i++) {
    //     std::cout << h_prefix_sum_edge_blocks_new[i] << " ";
    //     if((i + 1) % vertex_size == 0)
    //         std::cout << std::endl;
    // }
    // std::cout << std::endl << "Space remaining\t\t" << std::endl;
    // for(unsigned long j = 0 ; j < vertex_size ; j++)
    //     std::cout << space_remaining[j] << " ";
    // std::cout << std::endl;
    // std::cout << std::endl << std::endl << std::endl;
}

void generateBatch(unsigned long vertex_size, unsigned long edge_size, thrust::host_vector <unsigned long> &h_csr_offset, thrust::host_vector <unsigned long> &h_csr_edges, unsigned long* h_source_degree, unsigned long* h_prefix_sum_vertex_degrees, unsigned long* h_source, unsigned long* h_destination, unsigned long start_index, unsigned long end_index, unsigned long batch_size, unsigned long current_batch) {

    unsigned long start_index_csr_offset = 0;
    unsigned long end_index_csr_offset = 0;

    for(unsigned long i = 0 ; i < vertex_size ; i++) {

        if(h_csr_offset[i] > start_index) {
            if(i != 0)
                start_index_csr_offset = i - 1;
            else
                start_index_csr_offset = -1;
            break;
        }

    }



    unsigned long current_vertex = start_index_csr_offset;
    unsigned long index = 0;

    for(unsigned long i = start_index ; i < end_index ; i++) {

        while(i >= h_csr_offset[current_vertex + 1]) {
            current_vertex++;
            h_prefix_sum_vertex_degrees[current_vertex + 1] += h_prefix_sum_vertex_degrees[current_vertex];
        }
        
        h_source[index] = current_vertex + 2;
        h_destination[index++] = h_csr_edges[i];

        h_source_degree[current_vertex + 1]++;

        h_prefix_sum_vertex_degrees[current_vertex + 1]++;

    }



}

void memory_usage() {

    // show memory usage of GPU

    size_t free_byte ;

    size_t total_byte ;

    cudaMemGetInfo( &free_byte, &total_byte ) ;
    // cuda_status = cudaMemGetInfo( &free_byte, &total_byte ) ;

    // if ( cudaSuccess != cuda_status ){

    //     printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status) );

    //     exit(1);

    // }

    double free_db = (double)free_byte ;

    double total_db = (double)total_byte ;

    double used_db = total_db - free_db ;

    printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",

        used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);

}

int main(void) {

    // char fileLoc[20] = "input.mtx";
    // char fileLoc[20] = "input1.mtx";
    // char fileLoc[30] = "chesapeake.mtx";
    // char fileLoc[30] = "inf-luxembourg_osm.mtx";
    // char fileLoc[30] = "rgg_n_2_16_s0.mtx";
    // char fileLoc[30] = "delaunay_n10.mtx";
    // char fileLoc[30] = "delaunay_n12.mtx";
    // char fileLoc[30] = "delaunay_n13.mtx";
    // char fileLoc[30] = "delaunay_n16.mtx";
    // char fileLoc[30] = "delaunay_n17.mtx";
    // char fileLoc[30] = "fe-ocean.mtx";
    // char fileLoc[30] = "co-papers-dblp.mtx";
    char fileLoc[30] = "co-papers-citeseer.mtx";
    // char fileLoc[30] = "hugetrace-00020.mtx";
    // char fileLoc[30] = "channel-500x100x100-b050.mtx";
    // char fileLoc[30] = "kron_g500-logn16.mtx";
    // char fileLoc[30] = "kron_g500-logn17.mtx";
    // char fileLoc[30] = "kron_g500-logn21.mtx";
    // char fileLoc[30] = "delaunay_n22.mtx";
    // char fileLoc[30] = "delaunay_n23.mtx";
    // char fileLoc[30] = "delaunay_n24.mtx";
    // char fileLoc[30] = "inf-europe_osm.mtx";
    // char fileLoc[30] = "rgg_n_2_23_s0.mtx";
    // char fileLoc[30] = "rgg_n_2_24_s0.mtx";

    memory_usage();


    // some random inits
    unsigned long choice = 1;
    // printf("Please enter structure of edge blocks\n1. Unsorted\n2. Sorted\n");
    // scanf("%lu", &choice);

    clock_t section1, section2, section2a, section3, search_times, al_time, vd_time, time_req, push_to_queues_time, temp_time, delete_time, init_time;

    struct graph_properties *h_graph_prop = (struct graph_properties*)malloc(sizeof(struct graph_properties));
    thrust::host_vector <unsigned long> h_source(1);
    thrust::host_vector <unsigned long> h_destination(1);
    // thrust::host_vector <unsigned long> h_source_degree(1);
    // thrust::host_vector <unsigned long> h_prefix_sum_vertex_degrees(1);
    thrust::host_vector <unsigned long> h_prefix_sum_edge_blocks(1);
    thrust::host_vector <unsigned long> h_edge_blocks_count_init(1);

    thrust::host_vector <unsigned long> h_source_degrees(1);
    thrust::host_vector <unsigned long> h_csr_offset(1);
    thrust::host_vector <unsigned long> h_csr_edges(1);

    thrust::host_vector <unsigned long> h_source_degrees_new(1);
    thrust::host_vector <unsigned long> h_csr_offset_new(1);
    thrust::host_vector <unsigned long> h_csr_edges_new(1);
    thrust::host_vector <unsigned long> h_edge_blocks_count(1);
    thrust::host_vector <unsigned long> h_prefix_sum_edge_blocks_new(1);

    // thrust::host_vector <unsigned long> h_batch_update_data(1);


    // reading file, after function call h_source has data on source vertex and h_destination on destination vertex
    // both represent the edge data on host
    section1 = clock();
    readFile(fileLoc, h_graph_prop, h_source, h_destination, h_source_degrees);
    section1 = clock() - section1;

    std::cout << "File read complete" << std::endl;

    // below device vectors are for correctness check of the data structure at the end
    // thrust::device_vector <unsigned long> d_c_source(h_graph_prop->total_edges);
    // thrust::device_vector <unsigned long> d_c_destination(h_graph_prop->total_edges);
    // thrust::copy(h_source.begin(), h_source.end(), d_c_source.begin());
    // thrust::copy(h_destination.begin(), h_destination.end(), d_c_destination.begin());
    // unsigned long* c_source = thrust::raw_pointer_cast(d_c_source.data());
    // unsigned long* c_destination = thrust::raw_pointer_cast(d_c_destination.data());

    section2 = clock();

    unsigned long vertex_size = h_graph_prop->xDim;
    unsigned long edge_size = h_graph_prop->total_edges;

    unsigned long batch_size = BATCH_SIZE;
    unsigned long total_batches = ceil(double(edge_size) / batch_size);
    // unsigned long batch_number = 0;

    std::cout << "Batches required is " << total_batches << std::endl;

    // h_source_degree.resize(vertex_size);
    // h_prefix_sum_vertex_degrees.resize(vertex_size);
    // h_prefix_sum_edge_blocks.resize(vertex_size);
    h_edge_blocks_count_init.resize(vertex_size);

    h_source_degrees.resize(vertex_size);
    // h_csr_offset.resize(vertex_size + 1);
    // h_csr_edges.resize(edge_size);

    h_source_degrees_new.resize(vertex_size);
    h_csr_offset_new.resize((vertex_size + 1));
    h_csr_edges_new.resize(batch_size);
    h_edge_blocks_count.resize(vertex_size);
    h_prefix_sum_edge_blocks_new.resize(vertex_size);

    // h_batch_update_data.resize(vertex_size + 1 + batch_size + vertex_size);
    unsigned long *h_batch_update_data = (unsigned long *)malloc((batch_size) * (sizeof(unsigned long)));

    // generateCSR(vertex_size, edge_size, h_source, h_destination, h_csr_offset, h_csr_edges, h_source_degrees);



    // generateCSRnew(vertex_size, edge_size, h_source, h_destination, h_csr_offset_new, h_csr_edges_new, h_source_degrees_new, h_edge_blocks_count, BATCH_SIZE, total_batches);


    section2 = clock() - section2;

    std::cout << "CSR Generation complete" << std::endl;

    // std::cout << "Generated CSR" << std::endl;
    // for(unsigned long i = 0 ; i < vertex_size + 1 ; i++)
    //     std::cout << h_csr_offset[i] << " ";
    // std::cout << std::endl;
    // for(unsigned long i = 0 ; i < edge_size ; i++)
    //     std::cout << h_csr_edges[i] << " ";
    // std::cout << std::endl;

    // thrust::fill(h_source_degree.begin(), h_source_degree.end(), 0);

    // std::cout << "Check, " << h_source.size() << " and " << h_destination.size() << std::endl;

    // sleep(5);

    section2a = clock();

    struct vertex_dictionary_structure *device_vertex_dictionary;
    cudaMalloc(&device_vertex_dictionary, sizeof(struct vertex_dictionary_structure));
    struct adjacency_sentinel_new *device_adjacency_sentinel;
    cudaMalloc((struct adjacency_sentinel_new**)&device_adjacency_sentinel, vertex_size * sizeof(struct adjacency_sentinel_new));
    cudaDeviceSynchronize();
    section2a = clock() - section2a;
    // init_time = section2a;

    // below till cudaMalloc is temp code


    unsigned long total_edge_blocks_count_init = 0;
    // std::cout << "Edge blocks calculation" << std::endl << "Source\tEdge block count\tGPU address" << std::endl;
    for(unsigned long i = 0 ; i < h_graph_prop->xDim ; i++) {

        unsigned long edge_blocks = ceil(double(h_source_degrees[i]) / EDGE_BLOCK_SIZE);
        h_edge_blocks_count_init[i] = edge_blocks;
        total_edge_blocks_count_init += edge_blocks;


    }

    printf("Total edge blocks needed = %lu\n", total_edge_blocks_count_init);

    // h_prefix_sum_edge_blocks[0] = h_edge_blocks_count_init[0];
    // // printf("Prefix sum array edge blocks\n%ld ", h_prefix_sum_edge_blocks[0]);
    // for(unsigned long i = 1 ; i < h_graph_prop->xDim ; i++) {

    //     h_prefix_sum_edge_blocks[i] += h_prefix_sum_edge_blocks[i-1] + h_edge_blocks_count_init[i];
    //     // printf("%ld ", h_prefix_sum_edge_blocks[i]);

    // }

    temp_time = clock();
    struct edge_block *device_edge_block;
    cudaMalloc((struct edge_block**)&device_edge_block, total_edge_blocks_count_init * sizeof(struct edge_block));
    cudaDeviceSynchronize();
    temp_time = clock() - temp_time;
    section2a += temp_time;
    init_time = section2a;

    thrust::device_vector <unsigned long> d_edge_blocks_count_init(vertex_size);
    thrust::copy(h_edge_blocks_count_init.begin(), h_edge_blocks_count_init.end(), d_edge_blocks_count_init.begin());
    unsigned long* ebci = thrust::raw_pointer_cast(d_edge_blocks_count_init.data());

    // thrust::device_vector <unsigned long> d_prefix_sum_edge_blocks(vertex_size);
    // thrust::copy(h_prefix_sum_edge_blocks.begin(), h_prefix_sum_edge_blocks.end(), d_prefix_sum_edge_blocks.begin());
    // unsigned long* pseb = thrust::raw_pointer_cast(d_prefix_sum_edge_blocks.data());

    // thrust::device_vector <unsigned long> d_source(batch_size);
    // thrust::device_vector <unsigned long> d_destination(batch_size);
    // thrust::device_vector <unsigned long> d_prefix_sum_vertex_degrees(vertex_size);

    // thrust::device_vector <unsigned long> d_csr_offset(vertex_size + 1);
    // thrust::device_vector <unsigned long> d_csr_edges(edge_size);

    thrust::device_vector <unsigned long> d_source_degrees_new(vertex_size);
    thrust::device_vector <unsigned long> d_csr_offset_new(vertex_size + 1);
    thrust::device_vector <unsigned long> d_csr_edges_new(batch_size);
    thrust::device_vector <unsigned long> d_edge_blocks_count(vertex_size);
    thrust::device_vector <unsigned long> d_prefix_sum_edge_blocks_new(vertex_size);
    // thrust::device_vector <unsigned long> d_source(BATCH_SIZE);
    // thrust::device_vector <unsigned long> d_destination(BATCH_SIZE);

    // thrust::device_vector <unsigned long> d_batch_update_data(vertex_size + 1 + batch_size + vertex_size);
    temp_time = clock();
    unsigned long *d_batch_update_data;
    cudaMalloc((unsigned long**)&d_batch_update_data, (batch_size) * sizeof(unsigned long));
    cudaDeviceSynchronize();
    temp_time = clock() - temp_time;
    init_time += temp_time;
    // section2a = clock() - section2a;

    unsigned long thread_blocks;

    // Parallelize this
    // push_preallocate_list_to_device_queue_kernel<<< 1, 1>>>(device_vertex_block, device_edge_block, dapl, vertex_blocks_count_init, ebci, total_edge_blocks_count_init);
    
    // thread_blocks = ceil(double(vertex_blocks_count_init) / THREADS_PER_BLOCK);    

    time_req = clock();
    push_to_queues_time = clock();

    data_structure_init<<< 1, 1>>>(device_vertex_dictionary);
    // parallel_push_vertex_preallocate_list_to_device_queue<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_block, dapl, vertex_blocks_count_init);

    thread_blocks = ceil(double(total_edge_blocks_count_init) / THREADS_PER_BLOCK);

    // parallel_push_edge_preallocate_list_to_device_queue<<< thread_blocks, THREADS_PER_BLOCK>>>(device_edge_block, dapl, ebci, total_edge_blocks_count_init);
    parallel_push_edge_preallocate_list_to_device_queue_v1<<< thread_blocks, THREADS_PER_BLOCK>>>(device_edge_block, total_edge_blocks_count_init);

    parallel_push_queue_update<<< 1, 1>>>(total_edge_blocks_count_init);
    cudaDeviceSynchronize();

    push_to_queues_time = clock() - push_to_queues_time;

    init_time += push_to_queues_time;

    // sleep(5);

    // Pass raw array and its size to kernel
    // thread_blocks = ceil(double(vertex_blocks_count_init) / THREADS_PER_BLOCK);
    // std::cout << "Thread blocks vertex init = " << thread_blocks << std::endl;
    // vertex_dictionary_init<<< thread_blocks, THREADS_PER_BLOCK>>>(dvpl, dapl, vertex_blocks_count_init, d_graph_prop, vertex_size, ebci);
    vd_time = clock();
    // vertex_dictionary_init<<< thread_blocks, THREADS_PER_BLOCK>>>(device_vertex_block, device_adjacency_sentinel, vertex_blocks_count_init, d_graph_prop, vertex_size, ebci);
    thread_blocks = ceil(double(vertex_size) / THREADS_PER_BLOCK);
    // std::cout << "Thread blocks vertex init = " << thread_blocks << std::endl;
    // parallel_vertex_dictionary_init<<< thread_blocks, THREADS_PER_BLOCK>>>(device_adjacency_sentinel, vertex_blocks_count_init, d_graph_prop, vertex_size, ebci, device_vertex_dictionary);
    parallel_vertex_dictionary_init_v1<<< thread_blocks, THREADS_PER_BLOCK>>>(device_adjacency_sentinel, vertex_size, ebci, device_vertex_dictionary);

    vd_time = clock() - vd_time;

    init_time += vd_time;

    // h_source.resize(batch_size);
    // h_destination.resize(batch_size);
    h_source.resize(1);
    h_destination.resize(1);

    // thrust::copy(h_csr_offset.begin(), h_csr_offset.end(), d_csr_offset.begin());
    // thrust::copy(h_csr_edges.begin(), h_csr_edges.end(), d_csr_edges.begin());
    // unsigned long* d_csr_offset_pointer = thrust::raw_pointer_cast(d_csr_offset.data());
    // unsigned long* d_csr_edges_pointer = thrust::raw_pointer_cast(d_csr_edges.data());

    unsigned long* d_source_degrees_new_pointer = thrust::raw_pointer_cast(d_source_degrees_new.data());
    unsigned long* d_csr_offset_new_pointer = thrust::raw_pointer_cast(d_csr_offset_new.data());
    unsigned long* d_csr_edges_new_pointer = thrust::raw_pointer_cast(d_csr_edges_new.data());
    unsigned long* d_edge_blocks_count_pointer = thrust::raw_pointer_cast(d_edge_blocks_count.data());
    unsigned long* d_prefix_sum_edge_blocks_new_pointer = thrust::raw_pointer_cast(d_prefix_sum_edge_blocks_new.data());
    // unsigned long* d_source_pointer = thrust::raw_pointer_cast(d_source.data());
    // unsigned long* d_destination_pointer = thrust::raw_pointer_cast(d_destination.data());

    // unsigned long* d_batch_update_data_pointer = thrust::raw_pointer_cast(d_batch_update_data.data());

    // struct batch_update_data batch_update;

    thrust::host_vector <unsigned long> space_remaining(vertex_size);
    unsigned long total_edge_blocks_count_batch;

    // al_time = clock();
    al_time = 0;

    std::cout << "Enter type of insertion required" << std::endl << "1. Regular batched insertion" << std::endl << "2. Insert and Delete performance benchmark" << std::endl;
    std::cin >> choice;


    if(choice == 1) {
        for(unsigned long i = 0 ; i < total_batches ; i++) {
        // for(unsigned long i = 0 ; i < 7 ; i++) {

            // section2 = clock();

            // for(unsigned long i = 0 ; i < h_graph_prop->total_edges ; i++) {
            
            //     // printf("%lu and %lu\n", h_source[i], h_destination[i]);
            //     h_source_degree[h_source[i] - 1]++;
            
            // }

            std::cout << std::endl << "Iteration " << i << std::endl;

            // thrust::copy(h_csr_offset_new.begin(), h_csr_offset_new.end(), d_csr_offset_new.begin());
            // thrust::copy(h_csr_edges_new.begin(), h_csr_edges_new.end(), d_csr_edges_new.begin());

            // std::cout << std::endl << "Iteration " << i << std::endl;

            total_edge_blocks_count_batch = 0;

            generate_csr_batch(vertex_size, edge_size, h_source, h_destination, h_csr_offset_new, h_csr_edges_new, h_source_degrees_new, h_edge_blocks_count, h_prefix_sum_edge_blocks_new, h_batch_update_data, BATCH_SIZE, total_batches, i, space_remaining, &total_edge_blocks_count_batch, &init_time);

            // std::cout << "Total edge blocks is " << total_edge_blocks_count_batch  << std::endl;

            // std::cout << std::endl << "Printing batched CSR" << std::endl << "Source degrees\t\t" << std::endl;
            // for(unsigned long j = 0 ; j < vertex_size ; j++) {
            //     std::cout << h_source_degrees_new[j] << " ";
            //     if((j + 1) % vertex_size == 0)
            //         std::cout << std::endl;
            // }
            // std::cout << std::endl << "CSR offset\t\t" << std::endl;
            // for(unsigned long j = 0 ; j < (vertex_size + 1) ; j++) {
            //     std::cout << h_csr_offset_new[j] << " ";
            //     if(((j + 1) % (vertex_size + 1)) == 0)
            //         std::cout << std::endl;
            // }
            // std::cout << std::endl << "CSR edges\t\t" << std::endl;
            // for(unsigned long j = 0 ; j < batch_size ; j++) {
            //     std::cout << h_csr_edges_new[j] << " ";
            //     if((j + 1) % batch_size == 0)
            //         std::cout << std::endl;
            // }
            // std::cout << std::endl << "Edge blocks count\t\t" << std::endl;
            // for(unsigned long j = 0 ; j < vertex_size ; j++) {
            //     std::cout << h_edge_blocks_count[j] << " ";
            //     if((j + 1) % vertex_size == 0)
            //         std::cout << std::endl;
            // }
            // std::cout << std::endl << "Prefix sum edge blocks\t\t" << std::endl;
            // for(unsigned long j = 0 ; j < vertex_size ; j++) {
            //     std::cout << h_prefix_sum_edge_blocks_new[j] << " ";
            //     if((j + 1) % vertex_size == 0)
            //         std::cout << std::endl;
            // }
            // std::cout << std::endl << "Space remaining\t\t" << std::endl;
            // for(unsigned long j = 0 ; j < vertex_size ; j++)
            //     std::cout << space_remaining[j] << " ";
            // std::cout << std::endl;
            // std::cout << std::endl << "Batch update vector\t\t" << std::endl;
            // for(unsigned long j = 0 ; j < vertex_size + 1 + batch_size + vertex_size ; j++)
            //     std::cout << h_batch_update_data[j] << " ";
            // std::cout << std::endl;
            // std::cout << std::endl << std::endl;

            // temp_time = clock();


            thrust::copy(h_source_degrees_new.begin(), h_source_degrees_new.end(), d_source_degrees_new.begin());
            thrust::copy(h_csr_offset_new.begin(), h_csr_offset_new.end(), d_csr_offset_new.begin());
            thrust::copy(h_csr_edges_new.begin(), h_csr_edges_new.end(), d_csr_edges_new.begin());
            thrust::copy(h_edge_blocks_count.begin(), h_edge_blocks_count.end(), d_edge_blocks_count.begin());
            thrust::copy(h_prefix_sum_edge_blocks_new.begin(), h_prefix_sum_edge_blocks_new.end(), d_prefix_sum_edge_blocks_new.begin());

            temp_time = clock();

            // thrust::copy(h_batch_update_data.begin(), h_batch_update_data.end(), d_batch_update_data.begin());
            // cudaMemcpy(d_batch_update_data, &h_batch_update_data, (vertex_size + 1 + batch_size + vertex_size) * sizeof(unsigned long), cudaMemcpyHostToDevice);
            cudaMemcpy(d_batch_update_data, h_batch_update_data, (batch_size) * sizeof(unsigned long), cudaMemcpyHostToDevice);

            // thrust::device_vector <unsigned long> d_source_degrees_new(vertex_size);
            // thrust::device_vector <unsigned long> d_csr_offset_new(vertex_size + 1);
            // thrust::device_vector <unsigned long> d_csr_edges_new(batch_size);
            // thrust::device_vector <unsigned long> d_edge_blocks_count_new(vertex_size);
            // thrust::device_vector <unsigned long> d_prefix_sum_edge_blocks_new(vertex_size);

            // temp_time = clock();

            unsigned long start_index = i * batch_size;
            unsigned long end_index;

            unsigned long remaining_edges = edge_size - start_index;

            if(remaining_edges <= batch_size)
                end_index = edge_size;
            else
                end_index = (i + 1) * batch_size; 

            unsigned long current_batch = end_index - start_index;

            // std::cout << "Current batch is " << current_batch << std::endl;

            // cudaDeviceSynchronize();
            // vd_time = clock() - vd_time;

            thread_blocks = ceil(double(h_graph_prop->xDim) / THREADS_PER_BLOCK);
            // std::cout << "Thread blocks edge init = " << thread_blocks << std::endl;

            // sleep(5);

            // adjacency_list_init<<< thread_blocks, THREADS_PER_BLOCK>>>(depl, ebci, d_graph_prop, source, destination, total_edge_blocks_count_init, vertex_size, edge_size);
            // al_time = clock();
            // adjacency_list_init<<< thread_blocks, THREADS_PER_BLOCK>>>(device_edge_block, ebci, d_graph_prop, source, destination, total_edge_blocks_count_init, vertex_size, edge_size, pseb, thread_blocks);
            // adjacency_list_init_modded<<< thread_blocks, THREADS_PER_BLOCK>>>(device_edge_block, ebci, d_graph_prop, source, destination, total_edge_blocks_count_init, vertex_size, edge_size, pseb, thread_blocks, device_vertex_dictionary);
            // adjacency_list_init_modded_v1<<< thread_blocks, THREADS_PER_BLOCK>>>(device_edge_block, ebci, d_graph_prop, source, destination, total_edge_blocks_count_init, vertex_size, edge_size, pseb, psvd, thread_blocks, device_vertex_dictionary);
            // adjacency_list_init_modded_v2<<< thread_blocks, THREADS_PER_BLOCK>>>(device_edge_block, ebci, source, destination, total_edge_blocks_count_init, vertex_size, edge_size, pseb, psvd, thread_blocks, device_vertex_dictionary, i, current_batch);
            // adjacency_list_init_modded_v3<<< thread_blocks, THREADS_PER_BLOCK>>>(device_edge_block, ebci, total_edge_blocks_count_init, vertex_size, edge_size, pseb, thread_blocks, device_vertex_dictionary, i, current_batch, start_index, end_index, d_csr_offset_pointer, d_csr_edges_pointer);

            // adjacency_list_init_modded_v4<<< thread_blocks, THREADS_PER_BLOCK>>>(device_edge_block, ebci, total_edge_blocks_count_init, vertex_size, edge_size, pseb, thread_blocks, device_vertex_dictionary, i, current_batch, start_index, end_index, d_csr_offset_pointer, d_csr_edges_pointer);

            // std::cout << "Checkpoint 1" << std::endl;
            // temp_time = clock();

            adjacency_list_init_modded_v5<<< thread_blocks, THREADS_PER_BLOCK>>>(device_edge_block, d_edge_blocks_count_pointer, total_edge_blocks_count_batch, vertex_size, edge_size, d_prefix_sum_edge_blocks_new_pointer, thread_blocks, device_vertex_dictionary, i, current_batch, start_index, end_index, d_csr_offset_new_pointer, d_csr_edges_new_pointer);
            update_edge_queue<<< 1, 1>>>(total_edge_blocks_count_batch);

            cudaDeviceSynchronize();
            temp_time = clock() - temp_time;

            // std::cout << "Checkpoint 2" << std::endl;

            // if(i != 7) {
            //     std::cout << "Hit here" << std::endl;
            // }
            // temp_time = clock() - temp_time;
            al_time += temp_time;
            // std::cout << "Batch #" << i << " took " << (float)temp_time/CLOCKS_PER_SEC << " seconds" << std::endl;
            // Seperate kernel for updating queues due to performance issues for global barriers
            // printKernelmodded_v1<<< 1, 1>>>(device_vertex_dictionary, vertex_size);
            // if(i < 10)
            // printKernelmodded_v2<<< 1, 1>>>(device_vertex_dictionary, vertex_size);
            cbt_stats<<< 1, 1>>>(device_vertex_dictionary, vertex_size);

            cudaDeviceSynchronize();
            // std::cout << "Outside checkpoint" << std::endl;


        }

        cudaDeviceSynchronize();

        // for(long i = 0 ; i < edge_size ; i++)
        //     std::cout << h_source[i] << " and " << h_destination[i] << std::endl;

        // printf("\nCorrectness check of data structure\n");
        // printf("*---------------*\n");
        // thread_blocks = ceil(double(vertex_size) / THREADS_PER_BLOCK);
        // // correctness_check_kernel<<< 1, 1>>>(device_vertex_dictionary, vertex_size, edge_size, c_source, c_destination);
        // unsigned long h_correctness_flag, *d_correctness_flag;
        // cudaMalloc(&d_correctness_flag, sizeof(unsigned long));
        // // correctness_check_kernel<<< thread_blocks, 1024>>>(device_vertex_dictionary, vertex_size, edge_size, d_csr_offset_pointer, d_csr_edges_pointer, d_correctness_flag);
        // correctness_check_kernel_v1<<< thread_blocks, 1024>>>(device_vertex_dictionary, vertex_size, edge_size, d_csr_offset_pointer, d_csr_edges_pointer, d_correctness_flag);
        // cudaMemcpy(&h_correctness_flag, d_correctness_flag, sizeof(unsigned long),cudaMemcpyDeviceToHost);
        // cudaDeviceSynchronize();
        // search_times = clock() - search_times;

        // if(h_correctness_flag)
        //     std::cout << "Data structure corrupted" << std::endl;
        // else
        //     std::cout << "Data structure uncorrupted" << std::endl;
        // printf("*---------------*\n\n");


        // sleep(5);

        // printKernel<<< 1, 1>>>(device_vertex_block, vertex_size);
        // printKernelmodded<<< 1, 1>>>(device_vertex_dictionary, vertex_size);
        // printKernelmodded_v1<<< 1, 1>>>(device_vertex_dictionary, vertex_size);

        cudaDeviceSynchronize();
        delete_time = 0;
    }

    else {
        for(unsigned long i = 0 ; i < 1 ; i++) {
        // for(unsigned long i = 0 ; i < 7 ; i++) {

            // section2 = clock();

            // for(unsigned long i = 0 ; i < h_graph_prop->total_edges ; i++) {
            
            //     // printf("%lu and %lu\n", h_source[i], h_destination[i]);
            //     h_source_degree[h_source[i] - 1]++;
            
            // }

            unsigned long graph_choice;
            std::cout << std::endl << "Enter input type" << std::endl << "1. Real Graph" << std::endl << "2. Random Graph" << std::endl;
            std::cin >> graph_choice;

            std::cout << std::endl << "Iteration " << i << std::endl;

            // thrust::copy(h_csr_offset_new.begin(), h_csr_offset_new.end(), d_csr_offset_new.begin());
            // thrust::copy(h_csr_edges_new.begin(), h_csr_edges_new.end(), d_csr_edges_new.begin());

            // std::cout << std::endl << "Iteration " << i << std::endl;

            total_edge_blocks_count_batch = 0;


            if(graph_choice == 1)
                generate_csr_batch(vertex_size, edge_size, h_source, h_destination, h_csr_offset_new, h_csr_edges_new, h_source_degrees_new, h_edge_blocks_count, h_prefix_sum_edge_blocks_new, h_batch_update_data, BATCH_SIZE, total_batches, i, space_remaining, &total_edge_blocks_count_batch, &init_time);
            else {

                generate_random_batch(vertex_size, BATCH_SIZE, h_source, h_destination, h_source_degrees_new);
                std::cout << "Generated random batch" << std::endl;
                generate_csr_batch(vertex_size, edge_size, h_source, h_destination, h_csr_offset_new, h_csr_edges_new, h_source_degrees_new, h_edge_blocks_count, h_prefix_sum_edge_blocks_new, h_batch_update_data, BATCH_SIZE, total_batches, i, space_remaining, &total_edge_blocks_count_batch, &init_time);
                std::cout << "Generated CSR batch" << std::endl;

            }
            // std::cout << "Total edge blocks is " << total_edge_blocks_count_batch  << std::endl;

            // std::cout << std::endl << "Printing batched CSR" << std::endl << "Source degrees\t\t" << std::endl;
            // for(unsigned long j = 0 ; j < vertex_size ; j++) {
            //     std::cout << h_source_degrees_new[j] << " ";
            //     if((j + 1) % vertex_size == 0)
            //         std::cout << std::endl;
            // }
            // std::cout << std::endl << "CSR offset\t\t" << std::endl;
            // for(unsigned long j = 0 ; j < (vertex_size + 1) ; j++) {
            //     std::cout << h_csr_offset_new[j] << " ";
            //     if(((j + 1) % (vertex_size + 1)) == 0)
            //         std::cout << std::endl;
            // }
            // std::cout << std::endl << "CSR edges\t\t" << std::endl;
            // for(unsigned long j = 0 ; j < batch_size ; j++) {
            //     std::cout << h_csr_edges_new[j] << " ";
            //     if((j + 1) % batch_size == 0)
            //         std::cout << std::endl;
            // }
            // std::cout << std::endl << "Edge blocks count\t\t" << std::endl;
            // for(unsigned long j = 0 ; j < vertex_size ; j++) {
            //     std::cout << h_edge_blocks_count[j] << " ";
            //     if((j + 1) % vertex_size == 0)
            //         std::cout << std::endl;
            // }
            // std::cout << std::endl << "Prefix sum edge blocks\t\t" << std::endl;
            // for(unsigned long j = 0 ; j < vertex_size ; j++) {
            //     std::cout << h_prefix_sum_edge_blocks_new[j] << " ";
            //     if((j + 1) % vertex_size == 0)
            //         std::cout << std::endl;
            // }
            // std::cout << std::endl << "Space remaining\t\t" << std::endl;
            // for(unsigned long j = 0 ; j < vertex_size ; j++)
            //     std::cout << space_remaining[j] << " ";
            // std::cout << std::endl;
            // std::cout << std::endl << "Batch update vector\t\t" << std::endl;
            // for(unsigned long j = 0 ; j < vertex_size + 1 + batch_size + vertex_size ; j++)
            //     std::cout << h_batch_update_data[j] << " ";
            // std::cout << std::endl;
            // std::cout << std::endl << std::endl;

            // temp_time = clock();


            thrust::copy(h_source_degrees_new.begin(), h_source_degrees_new.end(), d_source_degrees_new.begin());
            thrust::copy(h_csr_offset_new.begin(), h_csr_offset_new.end(), d_csr_offset_new.begin());
            thrust::copy(h_csr_edges_new.begin(), h_csr_edges_new.end(), d_csr_edges_new.begin());
            thrust::copy(h_edge_blocks_count.begin(), h_edge_blocks_count.end(), d_edge_blocks_count.begin());
            thrust::copy(h_prefix_sum_edge_blocks_new.begin(), h_prefix_sum_edge_blocks_new.end(), d_prefix_sum_edge_blocks_new.begin());
            // thrust::copy(h_source.begin(), h_source.begin() + BATCH_SIZE, d_source.begin());
            // thrust::copy(h_destination.begin(), h_destination.begin() + BATCH_SIZE, d_destination.begin());

            temp_time = clock();

            // thrust::copy(h_batch_update_data.begin(), h_batch_update_data.end(), d_batch_update_data.begin());
            // cudaMemcpy(d_batch_update_data, &h_batch_update_data, (vertex_size + 1 + batch_size + vertex_size) * sizeof(unsigned long), cudaMemcpyHostToDevice);
            cudaMemcpy(d_batch_update_data, h_batch_update_data, (batch_size) * sizeof(unsigned long), cudaMemcpyHostToDevice);

            // thrust::device_vector <unsigned long> d_source_degrees_new(vertex_size);
            // thrust::device_vector <unsigned long> d_csr_offset_new(vertex_size + 1);
            // thrust::device_vector <unsigned long> d_csr_edges_new(batch_size);
            // thrust::device_vector <unsigned long> d_edge_blocks_count_new(vertex_size);
            // thrust::device_vector <unsigned long> d_prefix_sum_edge_blocks_new(vertex_size);

            // temp_time = clock();

            unsigned long start_index = i * batch_size;
            unsigned long end_index;

            unsigned long remaining_edges = edge_size - start_index;

            if(remaining_edges <= batch_size)
                end_index = edge_size;
            else
                end_index = (i + 1) * batch_size; 

            unsigned long current_batch = end_index - start_index;

            // std::cout << "Current batch is " << current_batch << std::endl;

            // cudaDeviceSynchronize();
            // vd_time = clock() - vd_time;

            thread_blocks = ceil(double(h_graph_prop->xDim) / THREADS_PER_BLOCK);
            // std::cout << "Thread blocks edge init = " << thread_blocks << std::endl;

            // sleep(5);

            // adjacency_list_init<<< thread_blocks, THREADS_PER_BLOCK>>>(depl, ebci, d_graph_prop, source, destination, total_edge_blocks_count_init, vertex_size, edge_size);
            // al_time = clock();
            // adjacency_list_init<<< thread_blocks, THREADS_PER_BLOCK>>>(device_edge_block, ebci, d_graph_prop, source, destination, total_edge_blocks_count_init, vertex_size, edge_size, pseb, thread_blocks);
            // adjacency_list_init_modded<<< thread_blocks, THREADS_PER_BLOCK>>>(device_edge_block, ebci, d_graph_prop, source, destination, total_edge_blocks_count_init, vertex_size, edge_size, pseb, thread_blocks, device_vertex_dictionary);
            // adjacency_list_init_modded_v1<<< thread_blocks, THREADS_PER_BLOCK>>>(device_edge_block, ebci, d_graph_prop, source, destination, total_edge_blocks_count_init, vertex_size, edge_size, pseb, psvd, thread_blocks, device_vertex_dictionary);
            // adjacency_list_init_modded_v2<<< thread_blocks, THREADS_PER_BLOCK>>>(device_edge_block, ebci, source, destination, total_edge_blocks_count_init, vertex_size, edge_size, pseb, psvd, thread_blocks, device_vertex_dictionary, i, current_batch);
            // adjacency_list_init_modded_v3<<< thread_blocks, THREADS_PER_BLOCK>>>(device_edge_block, ebci, total_edge_blocks_count_init, vertex_size, edge_size, pseb, thread_blocks, device_vertex_dictionary, i, current_batch, start_index, end_index, d_csr_offset_pointer, d_csr_edges_pointer);

            // adjacency_list_init_modded_v4<<< thread_blocks, THREADS_PER_BLOCK>>>(device_edge_block, ebci, total_edge_blocks_count_init, vertex_size, edge_size, pseb, thread_blocks, device_vertex_dictionary, i, current_batch, start_index, end_index, d_csr_offset_pointer, d_csr_edges_pointer);

            // std::cout << "Checkpoint 1" << std::endl;
            // temp_time = clock();

            adjacency_list_init_modded_v5<<< thread_blocks, THREADS_PER_BLOCK>>>(device_edge_block, d_edge_blocks_count_pointer, total_edge_blocks_count_batch, vertex_size, edge_size, d_prefix_sum_edge_blocks_new_pointer, thread_blocks, device_vertex_dictionary, i, current_batch, start_index, end_index, d_csr_offset_new_pointer, d_csr_edges_new_pointer);
            update_edge_queue<<< 1, 1>>>(total_edge_blocks_count_batch);
            // printKernelmodded_v2<<< 1, 1>>>(device_vertex_dictionary, vertex_size);

            cudaDeviceSynchronize();
            memory_usage();
            temp_time = clock() - temp_time;

            std::cout << "Insert done" << std::endl;

            // cbt_stats<<< 1, 1>>>(device_vertex_dictionary, vertex_size);

            // if(i != 7) {
            //     std::cout << "Hit here" << std::endl;
            // }
            // temp_time = clock() - temp_time;
            al_time += temp_time;

            d_csr_edges_new.resize(1);
            d_csr_offset_new.resize(1);

            // d_csr_offset_new.clear();
            // device_vector<T>().swap(d_csr_offset_new);
            // d_csr_edges_new.clear();
            // device_vector<T>().swap(d_csr_edges_new);
            // d_csr_offset_new.shrink_to_fit();
            // d_csr_edges_new.shrink_to_fit();

            cudaDeviceSynchronize();

            thrust::device_vector <unsigned long> d_source(BATCH_SIZE);
            thrust::device_vector <unsigned long> d_destination(BATCH_SIZE);
            unsigned long* d_source_pointer = thrust::raw_pointer_cast(d_source.data());
            unsigned long* d_destination_pointer = thrust::raw_pointer_cast(d_destination.data());
            thrust::copy(h_source.begin(), h_source.begin() + BATCH_SIZE, d_source.begin());
            thrust::copy(h_destination.begin(), h_destination.begin() + BATCH_SIZE, d_destination.begin());

            cudaDeviceSynchronize();
            delete_time = clock();
            // thread_blocks = ceil(double(vertex_size) / THREADS_PER_BLOCK);
            // batched_delete_preprocessing<<< thread_blocks, 1024>>>(device_vertex_dictionary, vertex_size, d_csr_offset_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_degrees_new_pointer);
            // thread_blocks = ceil(double(total_edge_blocks_count_batch) / THREADS_PER_BLOCK);
            // batched_delete_kernel_v1<<< thread_blocks, 1024>>>(device_vertex_dictionary, vertex_size, total_edge_blocks_count_batch, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_prefix_sum_edge_blocks_new_pointer);

            // Below is the code for edge-centric batch deletes
            thread_blocks = ceil(double(vertex_size) / THREADS_PER_BLOCK);
            batched_delete_preprocessing_edge_centric<<< thread_blocks, 1024>>>(device_vertex_dictionary, vertex_size, d_csr_offset_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_degrees_new_pointer);
            thread_blocks = ceil(double(batch_size) / THREADS_PER_BLOCK);
            batched_delete_kernel_edge_centric<<< thread_blocks, 1024>>>(device_vertex_dictionary, vertex_size, batch_size, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_pointer, d_destination_pointer);

            // Below is the test code for parallelized edge-centric batch deletes
            // thread_blocks = ceil(double(vertex_size) / THREADS_PER_BLOCK);
            // batched_delete_preprocessing<<< thread_blocks, 1024>>>(device_vertex_dictionary, vertex_size, d_csr_offset_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_degrees_new_pointer);
            // thread_blocks = ceil(double(batch_size * EDGE_BLOCK_SIZE) / THREADS_PER_BLOCK);
            // batched_delete_kernel_edge_centric_parallelized<<< thread_blocks, 1024>>>(device_vertex_dictionary, vertex_size, batch_size * EDGE_BLOCK_SIZE, batch_size, d_csr_offset_new_pointer, d_csr_edges_new_pointer, d_prefix_sum_edge_blocks_new_pointer, d_source_pointer, d_destination_pointer);


            // thread_blocks = ceil(double(vertex_size) / THREADS_PER_BLOCK);
            // batched_delete_kernel<<< thread_blocks, 1024>>>(device_vertex_dictionary, vertex_size, batch_size, d_csr_offset_new_pointer, d_csr_edges_new_pointer);
            cudaDeviceSynchronize();
            delete_time = clock() - delete_time;
            // std::cout << "Batch #" << i << " took " << (float)temp_time/CLOCKS_PER_SEC << " seconds" << std::endl;
            // Seperate kernel for updating queues due to performance issues for global barriers
            // printKernelmodded_v1<<< 1, 1>>>(device_vertex_dictionary, vertex_size);
            // if(i < 10)
            // printKernelmodded_v2<<< 1, 1>>>(device_vertex_dictionary, vertex_size);

            // std::cout << "Outside checkpoint" << std::endl;


        }
    }

    // update_edge_queue<<< 1, 1>>>(total_edge_blocks_count_init);

    // al_time = clock() - al_time;
    time_req = clock() - time_req;

    // memory_usage();

    // printKernelmodded_v2<<< 1, 1>>>(device_vertex_dictionary, vertex_size);

    // cudaDeviceSynchronize();

    // // for(long i = 0 ; i < edge_size ; i++)
    // //     std::cout << h_source[i] << " and " << h_destination[i] << std::endl;

    // printf("\nCorrectness check of data structure\n");
    // printf("*---------------*\n");
    // thread_blocks = ceil(double(vertex_size) / THREADS_PER_BLOCK);
    // // correctness_check_kernel<<< 1, 1>>>(device_vertex_dictionary, vertex_size, edge_size, c_source, c_destination);
    // unsigned long h_correctness_flag, *d_correctness_flag;
    // cudaMalloc(&d_correctness_flag, sizeof(unsigned long));
    // // correctness_check_kernel<<< thread_blocks, 1024>>>(device_vertex_dictionary, vertex_size, edge_size, d_csr_offset_pointer, d_csr_edges_pointer, d_correctness_flag);
    // correctness_check_kernel_v1<<< thread_blocks, 1024>>>(device_vertex_dictionary, vertex_size, edge_size, d_csr_offset_pointer, d_csr_edges_pointer, d_correctness_flag);
    // cudaMemcpy(&h_correctness_flag, d_correctness_flag, sizeof(unsigned long),cudaMemcpyDeviceToHost);
    // cudaDeviceSynchronize();
    // // search_times = clock() - search_times;

    // if(h_correctness_flag)
    //     std::cout << "Data structure corrupted" << std::endl;
    // else
    //     std::cout << "Data structure uncorrupted" << std::endl;
    // printf("*---------------*\n\n");


    // // sleep(5);

    // // printKernel<<< 1, 1>>>(device_vertex_block, vertex_size);
    // // printKernelmodded<<< 1, 1>>>(device_vertex_dictionary, vertex_size);
    // // printKernelmodded_v1<<< 1, 1>>>(device_vertex_dictionary, vertex_size);

    // cudaDeviceSynchronize();

    unsigned long exitFlag = 1;
    unsigned long menuChoice;
    unsigned long h_search_flag, *d_search_flag;
    unsigned long search_source, search_destination, total_search_threads;
    cudaMalloc(&d_search_flag, sizeof(unsigned long));

    while(exitFlag) {

        std::cout << std::endl << "Please enter any of the below options" << std::endl << "1. Search for and edge" << std::endl << "2. Delete an edge" << std::endl << "3. Print Adjacency" << std::endl << "4. Exit" << std::endl;
        scanf("%lu", &menuChoice);

        switch(menuChoice) {

            case 1  :
                        std::cout << "Enter the source and destination vertices respectively" << std::endl;
                        // unsigned long search_source, search_destination, total_search_threads;
                        scanf("%lu %lu", &search_source, &search_destination);
                        std::cout << "Edge blocks count for " << search_source << " is " << h_edge_blocks_count_init[search_source - 1] << std::endl;
                        
                        search_times = clock();

                        total_search_threads = h_edge_blocks_count_init[search_source - 1] * EDGE_BLOCK_SIZE;
                        thread_blocks = ceil(double(total_search_threads) / THREADS_PER_BLOCK);

                        // search_edge_kernel<<< thread_blocks, 1024>>>(device_vertex_dictionary, vertex_size, total_search_threads, search_source, search_destination, d_search_flag);
                        search_pre_processing<<< 1, 1>>>(device_vertex_dictionary, search_source);
                        search_edge_kernel_v1<<< thread_blocks, 1024>>>(device_vertex_dictionary, vertex_size, total_search_threads, search_source, search_destination, d_search_flag);
                        
                        cudaMemcpy(&h_search_flag, d_search_flag, sizeof(unsigned long),cudaMemcpyDeviceToHost);
                        cudaDeviceSynchronize();
                        search_times = clock() - search_times;

                        std::cout << "Search result was " << h_search_flag << " with time " << (float)search_times/CLOCKS_PER_SEC << " seconds" << std::endl;
                        h_search_flag = 0;
                        cudaMemcpy(d_search_flag, &h_search_flag, sizeof(unsigned long), cudaMemcpyHostToDevice);
                        cudaDeviceSynchronize();
                        break;

            case 2  :
                        std::cout << "Enter the source and destination vertices respectively" << std::endl;
                        // unsigned long search_source, search_destination, total_search_threads;
                        scanf("%lu %lu", &search_source, &search_destination);
                        // std::cout << "Edge blocks count for " << search_source << " is " << h_edge_blocks_count_init[search_source - 1] << std::endl;
                        
                        total_search_threads = h_edge_blocks_count_init[search_source - 1] * EDGE_BLOCK_SIZE;
                        thread_blocks = ceil(double(total_search_threads) / THREADS_PER_BLOCK);

                        // search_edge_kernel<<< thread_blocks, 1024>>>(device_vertex_dictionary, vertex_size, total_search_threads, search_source, search_destination, d_search_flag);
                        search_pre_processing<<< 1, 1>>>(device_vertex_dictionary, search_source);
                        delete_edge_kernel<<< thread_blocks, 1024>>>(device_vertex_dictionary, vertex_size, total_search_threads, search_source, search_destination, d_search_flag);

                        search_pre_processing<<< 1, 1>>>(device_vertex_dictionary, search_destination);
                        delete_edge_kernel<<< thread_blocks, 1024>>>(device_vertex_dictionary, vertex_size, total_search_threads, search_destination, search_source, d_search_flag);

                        cudaDeviceSynchronize();
                        
                        break;

            case 3  :

                        printKernelmodded_v2<<< 1, 1>>>(device_vertex_dictionary, vertex_size);
                        cudaDeviceSynchronize();
                        break;

            case 4  :   
                        exitFlag = 0;
                        break;

            default :;

        }

    }

    printf("xDim = %lu, yDim = %lu, Total Edges = %lu\n", h_graph_prop -> xDim, h_graph_prop -> yDim, h_graph_prop -> total_edges);
    std::cout << "Queues: " << (float)push_to_queues_time/CLOCKS_PER_SEC << " seconds" << std::endl;
    std::cout << "Vertex Dictionary: " << (float)vd_time/CLOCKS_PER_SEC << " seconds" << std::endl;
    std::cout << "Initialization   : " << (float)init_time/CLOCKS_PER_SEC << " seconds" << std::endl;
    std::cout << "Adjacency List   : " << (float)al_time/CLOCKS_PER_SEC << " seconds" << std::endl;
    std::cout << "Delete Batch     : " << (float)delete_time/CLOCKS_PER_SEC << " seconds" << std::endl;

    std::cout << "Time taken: " << (float)time_req/CLOCKS_PER_SEC << " seconds" << std::endl;
    // std::cout << "Added time: " << (float)total_time/CLOCKS_PER_SEC << " seconds" << std::endl;
    std::cout << "Read file : " << (float)section1/CLOCKS_PER_SEC << " seconds" << std::endl;
    std::cout << "CSR gen   : " << (float)section2/CLOCKS_PER_SEC << " seconds" << std::endl;
    std::cout << "Prefix sum, cudaMalloc, and cudaMemcpy: " << (float)section2a/CLOCKS_PER_SEC << " seconds" << std::endl;
    // std::cout << "Prefix sum, cudaMalloc, and cudaMemcpy: " << (float)section3/CLOCKS_PER_SEC << " seconds" << std::endl;      
	// // Cleanup
	// cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
	// printf("%d\n", c);
	return 0;
}