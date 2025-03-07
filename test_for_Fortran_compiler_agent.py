#!/usr/bin/env python3
"""
Test script for CompilerAgent with Fortran examples
"""

import os
import sys
from agents.compiler_agent import CompilerAgent

def test_fortran_monte_carlo_pi():
    """Test Fortran Monte Carlo Pi calculation with OpenMP"""
    print("\n=== TESTING FORTRAN MONTE CARLO PI ===")
    
    code = """
    program monte_carlo_pi
    implicit none
    integer, parameter :: n = 10000000 
    integer :: i, count
    real(8) :: x, y, pi_estimate
    real(8) :: start_time, end_time, elapsed_time

    count = 0
    call random_seed()  

    call cpu_time(start_time)

    !$omp parallel do private(x,y) reduction(+:count)
    do i = 1, n
        call random_number(x)
        call random_number(y)
        if ((x*x + y*y) <= 1.0d0) then
        count = count + 1
        end if
    end do
    !$omp end parallel do

    call cpu_time(end_time)
    elapsed_time = end_time - start_time

    pi_estimate = 4.0d0 * count / n

    print *, 'Estimated Pi = ', pi_estimate
    print *, 'Elapsed Time (seconds) = ', elapsed_time

    end program monte_carlo_pi
    """
    
    agent = CompilerAgent(working_dir="./compiler_temp")
    result = agent.compile_and_run(code, "FORTRAN", timeout=15)
    
    print("\nCompilation Success:", result["success"])
    print("Execution Time:", result.get("execution_time", "N/A"), "seconds")
    
    return result["success"]

def test_fortran_parallel_loop():
    """Test Fortran parallel loop with potential data race"""
    print("\n=== TESTING FORTRAN PARALLEL LOOP ===")
    
    code = """
    program DRB063_outeronly1_orig_no
     use omp_lib
     implicit none

     call foo()
    contains
     subroutine foo()
     integer :: i, j, n, m, len
     real, dimension(:,:), allocatable :: b

     len = 100
     allocate (b(len,len))
     n = len
     m = len
     
     ! Initialize array to avoid undefined values
     do i = 1, n
       do j = 1, m
         b(i,j) = i + j
       end do
     end do
     
     print *, "Starting parallel computation with", omp_get_max_threads(), "threads"
     
     !$omp parallel do private(j)
     do i = 1, n
       do j = 1, m-1
         b(i,j) = b(i,j+1)
       end do
     end do
     !$omp end parallel do
     
     ! Print some results for verification
     print *, "Computation complete. Sample results:"
     print *, "b(1,1) =", b(1,1)
     print *, "b(50,50) =", b(50,50)
     print *, "b(100,100) =", b(100,100)

     deallocate(b)
     end subroutine foo
    end program
    """
    
    agent = CompilerAgent(working_dir="./compiler_temp")
    result = agent.compile_and_run(code, "FORTRAN", timeout=10)
    
    print("\nCompilation Success:", result["success"])
    print("Execution Time:", result.get("execution_time", "N/A"), "seconds")
    
    return result["success"]

def test_fortran_matrix_multiplication():
    """Test Fortran matrix multiplication with OpenMP"""
    print("\n=== TESTING FORTRAN MATRIX MULTIPLICATION ===")
    
    code = """
    program matrix_multiplication
      use omp_lib
      implicit none
      
      integer, parameter :: N = 500
      real, dimension(:,:), allocatable :: A, B, C 
      integer :: i, j, k
      real :: start_time, end_time
      
      allocate(A(N,N), B(N,N), C(N,N))
      
      ! Initialize matrices
      !$omp parallel do private(j)
      do i = 1, N
        do j = 1, N
          A(i,j) = i * 0.1 + j * 0.2
          B(i,j) = i * 0.3 - j * 0.1
          C(i,j) = 0.0
        end do
      end do
      !$omp end parallel do
      
      print *, "Starting matrix multiplication with size", N, "x", N
      print *, "Using", omp_get_max_threads(), "threads"
      
      start_time = omp_get_wtime()
      
      ! Matrix multiplication with OpenMP
      !$omp parallel do private(j,k)
      do i = 1, N
        do j = 1, N
          do k = 1, N
            C(i,j) = C(i,j) + A(i,k) * B(k,j)
          end do
        end do
      end do
      !$omp end parallel do
      
      end_time = omp_get_wtime()
      
      print *, "Matrix multiplication completed in", end_time - start_time, "seconds"
      print *, "Sample results:"
      print *, "C(1,1) =", C(1,1)
      print *, "C(N/2,N/2) =", C(N/2,N/2)
      print *, "C(N,N) =", C(N,N)
      
      deallocate(A, B, C)
      
    end program matrix_multiplication
    """
    
    agent = CompilerAgent(working_dir="./compiler_temp")
    result = agent.compile_and_run(code, "FORTRAN", timeout=30)
    
    print("\nCompilation Success:", result["success"])
    print("Execution Time:", result.get("execution_time", "N/A"), "seconds")
    
    return result["success"]

def main():
    """Run all Fortran tests"""
    # Create temp directory if it doesn't exist
    os.makedirs("./compiler_temp", exist_ok=True)
    
    success_count = 0
    total_tests = 3
    
    # Run Fortran tests
    if test_fortran_monte_carlo_pi():
        success_count += 1
    
    if test_fortran_parallel_loop():
        success_count += 1
    
    if test_fortran_matrix_multiplication():
        success_count += 1
    
    # Print summary
    print("\n=== FORTRAN TEST SUMMARY ===")
    print(f"Passed: {success_count}/{total_tests}")
    
    return 0 if success_count == total_tests else 1

if __name__ == "__main__":
    sys.exit(main())
