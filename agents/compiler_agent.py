#!/usr/bin/env python3
"""
Compiler Agent
Responsible for compiling and executing translated code to validate correctness and measure performance
"""

import os
import re
import subprocess
import tempfile
import time
from typing import Dict, Any, Optional, Tuple
import platform
import shutil
from pathlib import Path

class CompilerAgent:
    """Agent responsible for compiling and executing translated code"""
    
    def __init__(self, working_dir: Optional[str] = None):
        """Initialize the compiler agent"""
        self.working_dir = working_dir or tempfile.mkdtemp()
        Path(self.working_dir).mkdir(parents=True, exist_ok=True)
        
        # Detect available compilers and tools
        self.compilers = self._detect_compilers()
        self.runtime_environments = self._detect_runtime_environments()
    
    def _detect_compilers(self) -> Dict[str, str]:
        """Detect available compilers in the system"""
        compilers = {}
        
        # C/C++ compilers
        for compiler in ["gcc", "g++", "clang", "clang++"]:
            if shutil.which(compiler):
                compilers[compiler] = shutil.which(compiler)
        
        # Fortran compilers
        for compiler in ["gfortran", "ifort", "flang", "ifx"]:
            if shutil.which(compiler):
                compilers[compiler] = shutil.which(compiler)
        
        # CUDA compiler
        if shutil.which("nvcc"):
            compilers["nvcc"] = shutil.which("nvcc")
        
        # Print detected compilers
        print("Detected compilers:")
        for compiler, path in compilers.items():
            print(f"  {compiler}: {path}")
        
        return compilers
    
    def _detect_runtime_environments(self) -> Dict[str, bool]:
        """Detect available runtime environments for different language executions"""
        environments = {
            "cuda": False,
            "openmp": False,
            "mpi": False,
            "fortran": False
        }
        
        # Check for CUDA
        try:
            output = subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT, text=True)
            environments["cuda"] = "NVIDIA" in output
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
        
        # Check for OpenMP
        if "gcc" in self.compilers or "g++" in self.compilers:
            environments["openmp"] = True
        
        # Check for MPI
        for mpi_cmd in ["mpirun", "mpiexec"]:
            if shutil.which(mpi_cmd):
                environments["mpi"] = True
                break
        
        # Check for Fortran
        if "gfortran" in self.compilers or "ifort" in self.compilers:
            environments["fortran"] = True
        
        return environments
    
    def compile_and_run(self, code: str, language: str, timeout: int = 10) -> Dict[str, Any]:
        """
        Compile and run the translated code
        
        Args:
            code: Source code to compile and run
            language: Target language (C++, CUDA, FORTRAN, etc.)
            timeout: Maximum execution time in seconds
            
        Returns:
            Dictionary containing compilation and execution results, errors, and performance metrics
        """
        result = {
            "success": False,
            "compiler_output": "",
            "execution_output": "",
            "errors": [],
            "execution_time": None,
            "performance_metrics": {}
        }
        
        # Normalize language name
        language = language.upper()
        print(f"\n=== COMPILING {language} CODE ===")
        
        # Create temporary files
        temp_file_path = self._create_temp_file(code, language)
        if not temp_file_path:
            error_msg = "Failed to create temporary file"
            result["errors"].append(error_msg)
            print(f"ERROR: {error_msg}")
            return result
        
        print(f"Created temporary file: {temp_file_path}")
        print(f"Code length: {len(code)} characters")
        
        # Compile the code
        print("\n--- COMPILATION PHASE ---")
        compile_result = self._compile_code(temp_file_path, language)
        result.update(compile_result)
        
        if not compile_result["success"]:
            print("Compilation failed. Skipping execution phase.")
            return result
        
        # Run the compiled code
        print("\n--- EXECUTION PHASE ---")
        run_result = self._run_code(compile_result.get("executable"), language, timeout)
        result.update(run_result)
        
        # Clean up temporary files (optional)
        self._cleanup_files(temp_file_path, compile_result.get("executable"))
        print("Temporary files cleaned up")
        
        print(f"=== COMPILATION AND EXECUTION COMPLETE ===\n")
        return result
    
    def _create_temp_file(self, code: str, language: str) -> Optional[str]:
        """Create a temporary file with the provided code"""
        # Determine file extension based on language
        extension = self._get_file_extension(language)
        
        try:
            # Create a temporary file with the appropriate extension
            fd, temp_path = tempfile.mkstemp(suffix=extension, dir=self.working_dir)
            with os.fdopen(fd, 'w') as f:
                f.write(code)
            return temp_path
        except Exception as e:
            print(f"Error creating temporary file: {e}")
            return None
    
    def _get_file_extension(self, language: str) -> str:
        """Get the appropriate file extension for the language"""
        extensions = {
            "C": ".c",
            "C++": ".cpp",
            "CUDA": ".cu",
            "FORTRAN": ".f90",  # Modern Fortran uses .f90 extension
            "OPENMP": ".cpp",   # OpenMP is typically used with C/C++
            "MPI": ".cpp"       # MPI is typically used with C/C++
        }
        return extensions.get(language, ".txt")
    
    def _compile_code(self, source_file: str, language: str) -> Dict[str, Any]:
        """Compile the code based on the language"""
        result = {
            "success": False,
            "compiler_output": "",
            "executable": None,
            "errors": []
        }
        
        # Generate output executable path
        output_file = os.path.splitext(source_file)[0]
        if platform.system() == "Windows":
            output_file += ".exe"
        
        # Build compilation command based on language
        compile_cmd = self._build_compile_command(source_file, output_file, language)
        
        if not compile_cmd:
            result["errors"].append(f"No compiler available for {language}")
            print(f"ERROR: No compiler available for {language}")
            return result
        
        # Execute the compilation command
        try:
            print(f"Executing compilation command: {' '.join(compile_cmd)}")
            
            process = subprocess.run(
                compile_cmd,
                text=True,
                capture_output=True,
                check=False
            )
            
            result["compiler_output"] = process.stdout + process.stderr
            
            # Print detailed compiler output for debugging
            print("\n--- COMPILER OUTPUT START ---")
            print(result["compiler_output"])
            print("--- COMPILER OUTPUT END ---\n")
            
            if process.returncode == 0:
                result["success"] = True
                result["executable"] = output_file
                print(f"Compilation successful. Executable: {output_file}")
                
                # For Fortran with OpenMP on Windows, copy the OpenMP DLL to the executable directory
                if platform.system() == "Windows" and language == "FORTRAN":
                    for compiler_name, compiler_path in self.compilers.items():
                        if compiler_name in ["gfortran"]:
                            compiler_dir = os.path.dirname(compiler_path)
                            dll_path = os.path.join(compiler_dir, "libgomp-1.dll")
                            if os.path.exists(dll_path):
                                output_dir = os.path.dirname(output_file)
                                target_dll_path = os.path.join(output_dir, "libgomp-1.dll")
                                try:
                                    shutil.copy2(dll_path, target_dll_path)
                                    print(f"Copied {dll_path} to {target_dll_path}")
                                except Exception as e:
                                    print(f"Warning: Failed to copy OpenMP DLL: {str(e)}")
                            break
            else:
                result["errors"].append("Compilation failed")
                # Extract specific error messages from compiler output
                error_lines = [line for line in result["compiler_output"].split('\n') 
                              if 'error:' in line.lower() or 'fatal error:' in line.lower()]
                
                if error_lines:
                    print("Compilation errors detected:")
                    for i, error in enumerate(error_lines[:5]):  # Show first 5 errors
                        clean_error = error.strip()
                        result["errors"].append(clean_error)
                        print(f"  {i+1}. {clean_error}")
                    
                    if len(error_lines) > 5:
                        print(f"  ... and {len(error_lines) - 5} more errors")
                else:
                    print("Compilation failed with unknown errors")
        
        except Exception as e:
            result["errors"].append(f"Error during compilation: {str(e)}")
            print(f"ERROR during compilation: {str(e)}")
        
        return result
    
    def _build_compile_command(self, source_file: str, output_file: str, language: str) -> Optional[list]:
        """Build the compilation command based on the language"""
        print(f"Building compilation command for {language} source file: {source_file}")
        
        if language == "C":
            if "gcc" in self.compilers:
                print(f"Using GCC compiler: {self.compilers['gcc']}")
                return [self.compilers["gcc"], "-o", output_file, source_file, "-Wall"]
            elif "clang" in self.compilers:
                print(f"Using Clang compiler: {self.compilers['clang']}")
                return [self.compilers["clang"], "-o", output_file, source_file, "-Wall"]
        
        elif language == "C++":
            if "g++" in self.compilers:
                print(f"Using G++ compiler: {self.compilers['g++']}")
                return [self.compilers["g++"], "-o", output_file, source_file, "-Wall", "-std=c++11"]
            elif "clang++" in self.compilers:
                print(f"Using Clang++ compiler: {self.compilers['clang++']}")
                return [self.compilers["clang++"], "-o", output_file, source_file, "-Wall", "-std=c++11"]
        
        elif language == "CUDA":
            if "nvcc" in self.compilers:
                print(f"Using NVCC compiler: {self.compilers['nvcc']}")
                # Try to find Visual Studio compiler path
                vs_paths = [
                    "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.36.32532\\bin\\Hostx64\\x64",
                    "C:\\Program Files\\Microsoft Visual Studio\\2022\\Professional\\VC\\Tools\\MSVC\\14.36.32532\\bin\\Hostx64\\x64",
                    "C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\VC\\Tools\\MSVC\\14.36.32532\\bin\\Hostx64\\x64",
                    "C:\\Program Files\\Microsoft Visual Studio\\2019\\Community\\VC\\Tools\\MSVC\\14.29.30133\\bin\\Hostx64\\x64",
                    "C:\\Program Files\\Microsoft Visual Studio\\2019\\Professional\\VC\\Tools\\MSVC\\14.29.30133\\bin\\Hostx64\\x64",
                    "C:\\Program Files\\Microsoft Visual Studio\\2019\\Enterprise\\VC\\Tools\\MSVC\\14.29.30133\\bin\\Hostx64\\x64",
                    "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC\\Tools\\MSVC\\14.29.30133\\bin\\Hostx64\\x64",
                    "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Professional\\VC\\Tools\\MSVC\\14.29.30133\\bin\\Hostx64\\x64",
                    "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Enterprise\\VC\\Tools\\MSVC\\14.29.30133\\bin\\Hostx64\\x64"
                ]
                
                vs_path = None
                for path in vs_paths:
                    if os.path.exists(path):
                        vs_path = path
                        break
                    
                if vs_path:
                    print(f"Found Visual Studio compiler at: {vs_path}")
                    return [self.compilers["nvcc"], "-ccbin", vs_path, "-o", output_file, source_file]
                else:
                    print("WARNING: Visual Studio compiler not found. Trying without -ccbin option.")
                    return [self.compilers["nvcc"], "-o", output_file, source_file]
            else:
                print("ERROR: NVCC compiler not found for CUDA compilation")
        
        elif language == "FORTRAN":
            if "gfortran" in self.compilers:
                print(f"Using GFortran compiler: {self.compilers['gfortran']}")
                # 使用静态链接OpenMP库，并增加栈大小或使用堆分配
                if platform.system() == "Windows":
                    return [self.compilers["gfortran"], "-o", output_file, source_file, "-fopenmp", "-static", "-fstack-arrays"]
                else:
                    return [self.compilers["gfortran"], "-o", output_file, source_file, "-fopenmp", "-fstack-arrays"]
            elif "ifort" in self.compilers:
                print(f"Using Intel Fortran compiler: {self.compilers['ifort']}")
                return [self.compilers["ifort"], "-o", output_file, source_file, "-qopenmp"]
            else:
                # Try to find any available Fortran compiler
                fortran_compilers = ["gfortran", "ifort", "flang", "ifx"]
                for compiler in fortran_compilers:
                    compiler_path = shutil.which(compiler)
                    if compiler_path:
                        print(f"Found Fortran compiler: {compiler_path}")
                        self.compilers[compiler] = compiler_path
                        if compiler in ["gfortran", "flang"]:
                            return [compiler_path, "-o", output_file, source_file, "-fopenmp"]
                        else:  # Intel compilers
                            return [compiler_path, "-o", output_file, source_file, "-qopenmp"]
                print("ERROR: No Fortran compiler found")
        
        elif language == "OPENMP":
            # For OpenMP, we need to add the -fopenmp flag
            if "gcc" in self.compilers and source_file.endswith(".c"):
                print(f"Using GCC compiler with OpenMP: {self.compilers['gcc']}")
                return [self.compilers["gcc"], "-fopenmp", "-o", output_file, source_file, "-Wall"]
            elif "g++" in self.compilers:
                print(f"Using G++ compiler with OpenMP: {self.compilers['g++']}") 
                return [self.compilers["g++"], "-fopenmp", "-o", output_file, source_file, "-Wall", "-std=c++11"]
            elif "clang" in self.compilers and source_file.endswith(".c"):
                print(f"Using Clang compiler with OpenMP: {self.compilers['clang']}")
                return [self.compilers["clang"], "-fopenmp", "-o", output_file, source_file, "-Wall"]
            elif "clang++" in self.compilers:
                print(f"Using Clang++ compiler with OpenMP: {self.compilers['clang++']}")
                return [self.compilers["clang++"], "-fopenmp", "-o", output_file, source_file, "-Wall", "-std=c++11"]
            else:
                # Try to find any available compiler that supports OpenMP
                for compiler_name in ["gcc", "g++", "clang", "clang++"]:
                    if compiler_name in self.compilers:
                        print(f"Using {compiler_name} compiler with OpenMP: {self.compilers[compiler_name]}")
                        return [self.compilers[compiler_name], "-fopenmp", "-o", output_file, source_file]
                print("ERROR: No compiler with OpenMP support found")
        
        elif language == "MPI":
            mpicc = shutil.which("mpicc")
            mpicxx = shutil.which("mpicxx")
            
            if mpicxx:
                print(f"Using MPI C++ compiler: {mpicxx}")
                return [mpicxx, "-o", output_file, source_file]
            elif mpicc:
                print(f"Using MPI C compiler: {mpicc}")
                return [mpicc, "-o", output_file, source_file]
        
        print(f"ERROR: No suitable compiler found for {language}")
        return None
    
    def _run_code(self, executable: Optional[str], language: str, timeout: int) -> Dict[str, Any]:
        """Run the compiled executable and measure performance"""
        result = {
            "success": False,
            "execution_output": "",
            "execution_time": None,
            "errors": [],
            "performance_metrics": {}
        }
        
        if not executable or not os.path.exists(executable):
            result["errors"].append("Executable not found")
            return result
        
        # Build execution command based on language
        run_cmd = self._build_run_command(executable, language)
        
        if not run_cmd:
            result["errors"].append(f"Cannot execute {language} code")
            return result
        
        # Set up environment for execution
        env = os.environ.copy()
        
        # For OpenMP, set the number of threads
        if language == "OPENMP" or language == "FORTRAN":
            num_threads = os.cpu_count() or 4
            env["OMP_NUM_THREADS"] = str(num_threads)
            print(f"Setting OMP_NUM_THREADS={num_threads}")
            
            # For Windows, ensure the OpenMP DLL can be found
            if platform.system() == "Windows":
                # Add MinGW bin directory to PATH for libgomp-1.dll
                for compiler_name, compiler_path in self.compilers.items():
                    if compiler_name in ["gcc", "g++", "gfortran"]:
                        compiler_dir = os.path.dirname(compiler_path)
                        if compiler_dir not in env["PATH"]:
                            env["PATH"] = compiler_dir + os.pathsep + env["PATH"]
                            print(f"Added {compiler_dir} to PATH for OpenMP DLL")
                        break
        
        # Execute the compiled code
        try:
            print(f"Executing command: {' '.join(run_cmd)}")
            start_time = time.time()
            
            process = subprocess.run(
                run_cmd,
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
                env=env  # Pass the environment variables
            )
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            result["execution_time"] = execution_time
            result["execution_output"] = process.stdout + process.stderr
            
            # Print execution output for debugging
            print("\n--- EXECUTION OUTPUT START ---")
            # print(result["execution_output"])
            print("--- EXECUTION OUTPUT END ---\n")
            
            if process.returncode == 0:
                result["success"] = True
                result["performance_metrics"]["execution_time_seconds"] = execution_time
                print(f"Execution successful. Time: {execution_time:.6f} seconds")
            else:
                result["errors"].append(f"Execution failed with return code: {process.returncode} (0x{process.returncode:08X})")
                print(f"Execution failed with return code: {process.returncode} (0x{process.returncode:08X})")
                
                # For Windows DLL errors, provide more information
                if platform.system() == "Windows" and process.returncode == 0xC0000139:
                    print("This error (0xC0000139) typically indicates a missing DLL.")
                    print("Make sure libgomp-1.dll is in the PATH or in the same directory as the executable.")
        
        except subprocess.TimeoutExpired:
            result["errors"].append(f"Execution timed out after {timeout} seconds")
            print(f"ERROR: Execution timed out after {timeout} seconds")
        except Exception as e:
            result["errors"].append(f"Error during execution: {str(e)}")
            print(f"ERROR during execution: {str(e)}")
        
        return result
    
    def _build_run_command(self, executable: str, language: str) -> Optional[list]:
        """Build the execution command based on the language"""
        print(f"Building run command for {language} executable: {executable}")
        
        if language in ["C", "C++", "FORTRAN", "OPENMP"]:
            return [executable]
        
        elif language == "CUDA":
            return [executable]
        
        elif language == "MPI":
            if shutil.which("mpirun"):
                num_processes = os.cpu_count() or 4
                return ["mpirun", "-np", str(num_processes), executable]
            elif shutil.which("mpiexec"):
                num_processes = os.cpu_count() or 4
                return ["mpiexec", "-n", str(num_processes), executable]
        
        print(f"ERROR: No suitable run command found for {language}")
        return None
    
    def _cleanup_files(self, source_file: Optional[str], executable: Optional[str]) -> None:
        """Clean up temporary files"""
        if source_file and os.path.exists(source_file):
            try:
                os.remove(source_file)
            except Exception:
                pass
        
        if executable and os.path.exists(executable):
            try:
                os.remove(executable)
            except Exception:
                pass
    
    def analyze_compilation_errors(self, compiler_output: str) -> Dict[str, Any]:
        """Analyze compilation error messages and suggest fixes"""
        analysis = {
            "error_count": 0,
            "warning_count": 0,
            "error_types": [],
            "common_issues": [],
            "suggested_fixes": []
        }
        
        if not compiler_output:
            return analysis
        
        # Count errors and warnings
        errors = re.findall(r'error:', compiler_output, re.IGNORECASE)
        warnings = re.findall(r'warning:', compiler_output, re.IGNORECASE)
        
        analysis["error_count"] = len(errors)
        analysis["warning_count"] = len(warnings)
        
        # Analyze common error types
        if "undefined reference" in compiler_output:
            analysis["error_types"].append("undefined_reference")
            analysis["common_issues"].append("Missing library or undefined function")
            analysis["suggested_fixes"].append("Check function declarations and library linking")
        
        if "expected ';'" in compiler_output:
            analysis["error_types"].append("syntax_error")
            analysis["common_issues"].append("Missing semicolon")
            analysis["suggested_fixes"].append("Add semicolon at the end of statements")
        
        if "undeclared identifier" in compiler_output:
            analysis["error_types"].append("undeclared_identifier")
            analysis["common_issues"].append("Using variable or function before declaration")
            analysis["suggested_fixes"].append("Declare variables/functions before using them")
        
        # Add CUDA-specific error analysis
        if "kernel launch" in compiler_output and "failed" in compiler_output:
            analysis["error_types"].append("cuda_kernel_launch_failure")
            analysis["common_issues"].append("CUDA kernel launch configuration issue")
            analysis["suggested_fixes"].append("Check grid/block dimensions and memory allocation")
        
        return analysis
    
    def analyze_execution_output(self, output: str, language: str) -> Dict[str, Any]:
        """Analyze execution output for common issues"""
        analysis = {
            "contains_error": False,
            "error_types": [],
            "performance_issues": [],
            "correctness_issues": []
        }
        
        if not output:
            return analysis
        
        # Look for common runtime errors
        if "segmentation fault" in output.lower():
            analysis["contains_error"] = True
            analysis["error_types"].append("segmentation_fault")
            analysis["correctness_issues"].append("Memory access violation")
        
        if "memory allocation failed" in output.lower():
            analysis["contains_error"] = True
            analysis["error_types"].append("memory_allocation_failure")
            analysis["correctness_issues"].append("Failed to allocate memory")
        
        # CUDA-specific errors
        if language == "CUDA":
            if "out of memory" in output.lower():
                analysis["contains_error"] = True
                analysis["error_types"].append("cuda_out_of_memory")
                analysis["performance_issues"].append("GPU memory allocation exceeded available memory")
            
            if "invalid device function" in output.lower():
                analysis["contains_error"] = True
                analysis["error_types"].append("cuda_invalid_device_function")
                analysis["correctness_issues"].append("Calling a host function from device code")
        
        # OpenMP specific
        if language == "OPENMP":
            if "data race" in output.lower():
                analysis["contains_error"] = True
                analysis["error_types"].append("openmp_data_race")
                analysis["correctness_issues"].append("Data race detected in parallel region")
        
        return analysis
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get the status of the compiler agent, including available compilers and environments"""
        return {
            "available_compilers": list(self.compilers.keys()),
            "runtime_environments": self.runtime_environments,
            "working_directory": self.working_dir
        }
