#!/usr/bin/env python3
"""
Test script for CompilerAgent with OpenMP examples
"""

import os
import sys
from agents.compiler_agent import CompilerAgent

def test_openmp_basic():
    """Test basic OpenMP parallel for loop"""
    print("\n=== TESTING OPENMP PARALLEL FOR ===")
    
    code = """
    #include <stdio.h>
    #include <omp.h>
    
    #define N 100
    
    int main(int argc, char* argv[]) {
        int i;
        int a[N];
        
        // Initialize array
        for (i = 0; i < N; i++) {
            a[i] = i;
        }
        
        // Parallel computation
        #pragma omp parallel for
        for (i = 0; i < N; i++) {
            a[i] = a[i] * 2;
            printf("Thread %d processing element %d\\n", 
                   omp_get_thread_num(), i);
        }
        
        // Verify results
        printf("\\nResults verification:\\n");
        for (i = 0; i < N; i += 10) {
            printf("a[%d] = %d\\n", i, a[i]);
        }
        
        return 0;
    }
    """
    
    agent = CompilerAgent(working_dir="./compiler_temp")
    result = agent.compile_and_run(code, "OPENMP", timeout=5)
    
    print("\nCompilation Success:", result["success"])
    print("Execution Time:", result.get("execution_time", "N/A"), "seconds")
    
    return result["success"]

def test_openmp_reduction():
    """Test OpenMP reduction operation"""
    print("\n=== TESTING OPENMP REDUCTION ===")
    
    code = """
    #include <stdio.h>
    #include <omp.h>
    
    #define N 1000000
    
    int main(int argc, char* argv[]) {
        int i;
        double sum = 0.0;
        double start_time, end_time;
        
        start_time = omp_get_wtime();
        
        // Parallel sum with reduction
        #pragma omp parallel for reduction(+:sum)
        for (i = 0; i < N; i++) {
            sum += i;
        }
        
        end_time = omp_get_wtime();
        
        printf("Sum of numbers from 0 to %d is: %.0f\\n", N-1, sum);
        printf("Computed in %.6f seconds\\n", end_time - start_time);
        printf("Using %d threads\\n", omp_get_max_threads());
        
        return 0;
    }
    """
    
    agent = CompilerAgent(working_dir="./compiler_temp")
    result = agent.compile_and_run(code, "OPENMP", timeout=5)
    
    print("\nCompilation Success:", result["success"])
    print("Execution Time:", result.get("execution_time", "N/A"), "seconds")
    
    return result["success"]

def test_openmp_sections():
    """Test OpenMP sections for task parallelism"""
    print("\n=== TESTING OPENMP SECTIONS ===")
    
    code = """
    #include <stdio.h>
    #include <omp.h>
    #include <unistd.h>  // For sleep function
    
    void task_a() {
        printf("Task A started by thread %d\\n", omp_get_thread_num());
        // Simulate work
        #ifdef _WIN32
        Sleep(1000);  // Windows sleep in milliseconds
        #else
        sleep(1);     // Unix sleep in seconds
        #endif
        printf("Task A completed by thread %d\\n", omp_get_thread_num());
    }
    
    void task_b() {
        printf("Task B started by thread %d\\n", omp_get_thread_num());
        // Simulate work
        #ifdef _WIN32
        Sleep(2000);
        #else
        sleep(2);
        #endif
        printf("Task B completed by thread %d\\n", omp_get_thread_num());
    }
    
    void task_c() {
        printf("Task C started by thread %d\\n", omp_get_thread_num());
        // Simulate work
        #ifdef _WIN32
        Sleep(1500);
        #else
        sleep(1);
        #endif
        printf("Task C completed by thread %d\\n", omp_get_thread_num());
    }
    
    int main(int argc, char* argv[]) {
        double start_time, end_time;
        
        printf("Starting parallel sections with %d threads\\n", omp_get_max_threads());
        
        start_time = omp_get_wtime();
        
        #pragma omp parallel sections
        {
            #pragma omp section
            {
                task_a();
            }
            
            #pragma omp section
            {
                task_b();
            }
            
            #pragma omp section
            {
                task_c();
            }
        }
        
        end_time = omp_get_wtime();
        
        printf("All tasks completed in %.6f seconds\\n", end_time - start_time);
        
        return 0;
    }
    """
    
    agent = CompilerAgent(working_dir="./compiler_temp")
    result = agent.compile_and_run(code, "OPENMP", timeout=10)
    
    print("\nCompilation Success:", result["success"])
    print("Execution Time:", result.get("execution_time", "N/A"), "seconds")
    
    return result["success"]

def test_openmp_nested_parallelism():
    """Test OpenMP nested parallelism with parallel regions inside parallel regions"""
    print("\n=== TESTING OPENMP NESTED PARALLELISM ===")
    
    code = """
    #include <stdio.h>
    #include <omp.h>
    
    #define N 10
    
    int main(int argc, char* argv[]) {
        int i, j;
        int matrix[N][N];
        
        // Enable nested parallelism
        omp_set_nested(1);
        
        // Get the maximum number of threads
        int max_threads = omp_get_max_threads();
        printf("Maximum number of threads: %d\\n", max_threads);
        
        // Initialize matrix
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                matrix[i][j] = i * N + j;
            }
        }
        
        // Use nested parallelism to process the matrix
        #pragma omp parallel num_threads(2)
        {
            int outer_thread = omp_get_thread_num();
            int outer_num_threads = omp_get_num_threads();
            
            printf("Outer thread %d of %d started\\n", outer_thread, outer_num_threads);
            
            // Each outer thread processes half of the rows
            #pragma omp parallel num_threads(2)
            {
                int inner_thread = omp_get_thread_num();
                int inner_num_threads = omp_get_num_threads();
                
                printf("  Inner thread %d of %d started within outer thread %d\\n", 
                       inner_thread, inner_num_threads, outer_thread);
                
                // Calculate the range of rows this thread should process
                int start_row = outer_thread * (N/2);
                int end_row = (outer_thread + 1) * (N/2);
                
                // Calculate the range of columns this thread should process
                int start_col = inner_thread * (N/2);
                int end_col = (inner_thread + 1) * (N/2);
                
                // Process the assigned block of the matrix
                for (i = start_row; i < end_row; i++) {
                    for (j = start_col; j < end_col; j++) {
                        // Double each element
                        matrix[i][j] *= 2;
                        printf("    Thread [%d,%d] processed matrix[%d][%d] = %d\\n", 
                               outer_thread, inner_thread, i, j, matrix[i][j]);
                    }
                }
                
                printf("  Inner thread %d within outer thread %d finished\\n", 
                       inner_thread, outer_thread);
            }
            
            printf("Outer thread %d finished\\n", outer_thread);
        }
        
        // Verify results
        printf("\\nMatrix verification (showing corners):\\n");
        printf("Top-left: matrix[0][0] = %d\\n", matrix[0][0]);
        printf("Top-right: matrix[0][%d] = %d\\n", N-1, matrix[0][N-1]);
        printf("Bottom-left: matrix[%d][0] = %d\\n", N-1, matrix[N-1][0]);
        printf("Bottom-right: matrix[%d][%d] = %d\\n", N-1, N-1, matrix[N-1][N-1]);
        
        return 0;
    }
    """
    
    agent = CompilerAgent(working_dir="./compiler_temp")
    result = agent.compile_and_run(code, "OPENMP", timeout=10)
    
    print("\nCompilation Success:", result["success"])
    print("Execution Time:", result.get("execution_time", "N/A"), "seconds")
    
    return result["success"]

def test_openmp_task_parallelism():
    """Test OpenMP task-based parallelism with dependencies"""
    print("\n=== TESTING OPENMP TASK PARALLELISM ===")
    
    code = """
    #include <stdio.h>
    #include <stdlib.h>
    #include <omp.h>
    
    // Fibonacci calculation using task parallelism
    int fib(int n) {
        int i, j;
        
        if (n < 2)
            return n;
            
        #pragma omp task shared(i)
        i = fib(n-1);
        
        #pragma omp task shared(j)
        j = fib(n-2);
        
        #pragma omp taskwait
        return i + j;
    }
    
    // Function to simulate a complex task with dependencies
    void process_data(int* data, int size, int task_id) {
        printf("Task %d started processing data of size %d\\n", task_id, size);
        
        // Simulate processing time
        double start = omp_get_wtime();
        double end;
        do {
            end = omp_get_wtime();
        } while (end - start < 0.1); // Process for 0.1 seconds
        
        printf("Task %d completed processing\\n", task_id);
    }
    
    int main(int argc, char* argv[]) {
        int n = 10; // Fibonacci number to calculate
        int result;
        double start_time, end_time;
        
        // Allocate some data for our tasks
        int* data1 = (int*)malloc(1000 * sizeof(int));
        int* data2 = (int*)malloc(2000 * sizeof(int));
        int* data3 = (int*)malloc(1500 * sizeof(int));
        
        printf("Starting OpenMP task parallelism test\\n");
        
        start_time = omp_get_wtime();
        
        // Create a parallel region
        #pragma omp parallel
        {
            #pragma omp single
            {
                printf("Thread %d starting task creation\\n", omp_get_thread_num());
                
                // Create tasks with dependencies
                #pragma omp task depend(out: data1)
                {
                    process_data(data1, 1000, 1);
                }
                
                #pragma omp task depend(out: data2)
                {
                    process_data(data2, 2000, 2);
                }
                
                #pragma omp task depend(in: data1, data2) depend(out: data3)
                {
                    printf("Task 3 started after Tasks 1 and 2 completed\\n");
                    process_data(data3, 1500, 3);
                }
                
                #pragma omp task depend(in: data3)
                {
                    printf("Task 4 started after Task 3 completed\\n");
                    process_data(NULL, 0, 4);
                }
                
                // Calculate Fibonacci in parallel
                #pragma omp task shared(result)
                {
                    printf("Starting Fibonacci calculation for n=%d\\n", n);
                    result = fib(n);
                    printf("Fibonacci(%d) = %d\\n", n, result);
                }
                
                printf("All tasks have been created\\n");
            }
        }
        
        end_time = omp_get_wtime();
        
        printf("All tasks completed in %.6f seconds\\n", end_time - start_time);
        
        // Clean up
        free(data1);
        free(data2);
        free(data3);
        
        return 0;
    }
    """
    
    agent = CompilerAgent(working_dir="./compiler_temp")
    result = agent.compile_and_run(code, "OPENMP", timeout=15)
    
    print("\nCompilation Success:", result["success"])
    print("Execution Time:", result.get("execution_time", "N/A"), "seconds")
    
    return result["success"]

def test_openmp_simd_vectorization():
    """Test OpenMP SIMD vectorization directives"""
    print("\n=== TESTING OPENMP SIMD VECTORIZATION ===")
    
    code = """
    #include <stdio.h>
    #include <stdlib.h>
    #include <omp.h>
    #include <math.h>
    
    #define N 10000000
    
    // Function to be vectorized
    void vector_add(double* a, double* b, double* c, int n) {
        #pragma omp simd
        for (int i = 0; i < n; i++) {
            c[i] = a[i] + b[i];
        }
    }
    
    // Function with more complex operations
    void vector_complex(double* a, double* b, double* c, int n) {
        #pragma omp simd
        for (int i = 0; i < n; i++) {
            c[i] = sqrt(a[i] * a[i] + b[i] * b[i]);
        }
    }
    
    // Function using both parallelization and vectorization
    void vector_parallel_simd(double* a, double* b, double* c, int n) {
        #pragma omp parallel for simd
        for (int i = 0; i < n; i++) {
            c[i] = a[i] * b[i] + sin(a[i]);
        }
    }
    
    int main(int argc, char* argv[]) {
        double *a, *b, *c;
        double start_time, end_time;
        
        // Allocate memory
        a = (double*)malloc(N * sizeof(double));
        b = (double*)malloc(N * sizeof(double));
        c = (double*)malloc(N * sizeof(double));
        
        // Initialize arrays
        for (int i = 0; i < N; i++) {
            a[i] = (double)i / N;
            b[i] = (double)(N - i) / N;
        }
        
        printf("Testing OpenMP SIMD vectorization with array size %d\\n", N);
        
        // Test 1: Simple vector addition with SIMD
        start_time = omp_get_wtime();
        vector_add(a, b, c, N);
        end_time = omp_get_wtime();
        
        printf("Vector addition (SIMD): %.6f seconds\\n", end_time - start_time);
        printf("Sample results: c[0]=%f, c[%d]=%f\\n", c[0], N-1, c[N-1]);
        
        // Test 2: Complex vector operation with SIMD
        start_time = omp_get_wtime();
        vector_complex(a, b, c, N);
        end_time = omp_get_wtime();
        
        printf("Vector complex (SIMD): %.6f seconds\\n", end_time - start_time);
        printf("Sample results: c[0]=%f, c[%d]=%f\\n", c[0], N-1, c[N-1]);
        
        // Test 3: Combined parallel for and SIMD
        start_time = omp_get_wtime();
        vector_parallel_simd(a, b, c, N);
        end_time = omp_get_wtime();
        
        printf("Vector parallel SIMD: %.6f seconds\\n", end_time - start_time);
        printf("Sample results: c[0]=%f, c[%d]=%f\\n", c[0], N-1, c[N-1]);
        
        // Clean up
        free(a);
        free(b);
        free(c);
        
        return 0;
    }
    """
    
    agent = CompilerAgent(working_dir="./compiler_temp")
    result = agent.compile_and_run(code, "OPENMP", timeout=15)
    
    print("\nCompilation Success:", result["success"])
    print("Execution Time:", result.get("execution_time", "N/A"), "seconds")
    
    return result["success"]

def main():
    """Run all tests"""
    # Create temp directory if it doesn't exist
    os.makedirs("./compiler_temp", exist_ok=True)
    
    success_count = 0
    total_tests = 6  # Updated count (removed Fortran tests)
    
    # Run basic tests
    if test_openmp_basic():
        success_count += 1
    
    if test_openmp_reduction():
        success_count += 1
    
    if test_openmp_sections():
        success_count += 1
    
    # Run advanced tests
    if test_openmp_nested_parallelism():
        success_count += 1
    
    if test_openmp_task_parallelism():
        success_count += 1
    
    if test_openmp_simd_vectorization():
        success_count += 1
    
    # Print summary
    print("\n=== OPENMP TEST SUMMARY ===")
    print(f"Passed: {success_count}/{total_tests}")
    
    return 0 if success_count == total_tests else 1

if __name__ == "__main__":
    sys.exit(main())
