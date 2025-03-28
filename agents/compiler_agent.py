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
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
import sys

class CompilerAgent:
    """Agent responsible for compiling and executing translated code"""
    
    def __init__(self, llm, working_dir: Optional[str] = None):
        """Initialize the compiler agent"""
        self.llm = llm
        self.working_dir = working_dir or tempfile.mkdtemp()
        Path(self.working_dir).mkdir(parents=True, exist_ok=True)
        
        # Detect available compilers and tools
        self.compilers = self._detect_compilers()
        self.runtime_environments = self._detect_runtime_environments()
    
    def _select_compiler_with_llm(self, code: str, language: str, state: Dict = None) -> Dict[str, Any]:
        prompt_template = """
        As a compiler expert, please analyze the following code and select the most appropriate compiler.

        Code language: {{language}}
        
        Code content:
        ```
        {{code}}
        ```
        
        {% if state %}
        Additional context information:
        - Source language: {{state.get('source_language', 'Unknown')}}
        - Optimization level: {{state.get('optimization_level', 'O2')}}
        - Current phase: {{state.get('current_phase', 'Unknown')}}
        {% endif %}

        Available compilers:
        {% for compiler, path in available_compilers.items() %}
        - {{compiler}}: {{path}}
        {% endfor %}

        Please analyze the following aspects:
        1. Code features (OpenMP, SIMD, CUDA, etc.)
        2. Performance requirements
        3. Platform compatibility
        4. Compiler optimization capabilities

        Please answer in the following format:
        Selected compiler: [compiler name]
        Compiler path: [path]
        Reasoning: [detailed explanation]
        Recommended compiler options: [list of options]
        """

        try:
            # prepare template variables
            template_vars = {
                "language": language,
                "code": code,
                "state": state if state else {},
                "available_compilers": self.compilers
            }

            # create prompt and call LLM
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["language", "code"],
                partial_variables={
                    "available_compilers": self.compilers,
                    "state": state if state else {}
                },
                template_format="jinja2"
            )
            
            chain = prompt | self.llm
            result = chain.invoke(template_vars)
            
            # parse LLM's response
            response = self._clean_thinking_process(result.content)
            
            # extract key information
            compiler_name = None
            compiler_path = None
            compile_options = []
            
            for line in response.split('\n'):
                if line.startswith('Selected compiler:'):
                    compiler_name = line.split(':')[1].strip()
                elif line.startswith('Compiler path:'):
                    compiler_path = line.split(':')[1].strip()
                elif line.startswith('Recommended compiler options:'):
                    options_str = line.split(':')[1].strip()
                    compile_options = [opt.strip() for opt in options_str.split() if opt.strip()]
            
            # verify if the selected compiler is available
            if compiler_name and compiler_name in self.compilers:
                return {
                    "compiler_name": compiler_name,
                    "compiler_path": self.compilers[compiler_name],
                    "compile_options": compile_options,
                    "success": True
                }
            else:
                print(f"Warning: LLM selected compiler {compiler_name} is not available, using default compiler")
                return {
                    "success": False,
                    "error": f"Selected compiler {compiler_name} is not available"
                }
            
        except Exception as e:
            print(f"Compiler selection process error: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
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
            "fortran": False,
            "jax": False 
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
        
        # Check for JAX
        try:
            import sys
            import importlib.util
            jax_spec = importlib.util.find_spec("jax")
            jaxlib_spec = importlib.util.find_spec("jaxlib")
            if jax_spec and jaxlib_spec:
                # Check JAX version
                try:
                    import jax
                    print(f"Detected JAX version: {jax.__version__}")
                    environments["jax"] = True
                except ImportError:
                    pass
        except Exception as e:
            print(f"Error checking JAX installation: {str(e)}")
        
        return environments
    
    def compile_and_run(self, code: str, language: str, timeout: int = 10) -> Dict[str, Any]:
        result = {
            "success": False,
            "compiler_output": "",
            "execution_output": "",
            "errors": [],
            "execution_time": None,
            "performance_metrics": {}
        }
        
        # Standardize language name
        language = language.upper()
        print(f"\n=== COMPILING/RUNNING {language} CODE ===")
        
        # Add necessary import statements for JAX code
        if language == "JAX" and "import jax" not in code:
            # Check if code already has import statements
            print("Adding necessary import statements for JAX code...")
            imports = "import jax\nimport jax.numpy as jnp\nfrom jax import grad, jit, vmap\nimport numpy as np\n\n"
            code = imports + code
        
        # for windows, changethe special characters
        if platform.system() == "Windows":
            # replace special characters in test
            code = code.replace("✓", "[OK]")
            code = code.replace("✗", "[ERROR]")
            code = code.replace("²", "^2")
            # replace other special characters
            code = code.replace("…", "...")
        
        # Process code cleanup
        code = self._clean_code_for_compilation(code)
        
        print("\n=== SOURCE CODE TO COMPILE/RUN ===")
        for i, line in enumerate(code.split('\n')):
            print(f"{i+1:4d}: {line}")
        print("=== END OF SOURCE CODE ===\n")
        
        # use utf-8 encoding
        try:
            # Create temporary file with explicit UTF-8 encoding
            if language in ["JAX", "PYTHON"]:
                # For Python files, explicitly add encoding declaration
                if not code.startswith("# -*- coding:"):
                    code = "# -*- coding: utf-8 -*-\n" + code
                
            # Create temporary file
            extension = self._get_file_extension(language)
            fd, temp_file_path = tempfile.mkstemp(suffix=extension, dir=self.working_dir)
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                f.write(code)
            
            if not os.path.exists(temp_file_path):
                error_msg = "Failed to create temporary file"
                result["errors"].append(error_msg)
                print(f"ERROR: {error_msg}")
                return result
            
            print(f"Created temporary file: {temp_file_path}")
            print(f"Code length: {len(code)} characters")
            
            if language in ["JAX", "PYTHON"]:
                print("\n--- Skipping compilation phase (interpreted language) ---")
                result["success"] = True
                result["executable"] = temp_file_path
                
                print("\n--- Execution phase ---")
                run_result = self._run_code(temp_file_path, language, timeout)
                result.update(run_result)
                
                # Clean up temporary files
                self._cleanup_files(temp_file_path, None)
                print("Temporary files cleaned up")
                
                print(f"=== Execution complete ===\n")
                return result
        except Exception as e:
            error_msg = f"Error creating temporary file: {str(e)}"
            result["errors"].append(error_msg)
            print(f"ERROR: {error_msg}")
            return result
        
        # Normalize language name
        language = language.upper()
        print(f"\n=== COMPILING {language} CODE ===")
        
        # emergency fix printf statement cross-line problem
        # this step is the last safety net, ensuring that even if previous cleanup steps fail, it can catch and fix the problem
        code = self._emergency_fix_printf(code)
        
        # Print source code with line numbers for debugging
        print("\n=== SOURCE CODE TO COMPILE ===")
        for i, line in enumerate(code.split('\n')):
            print(f"{i+1:4d}: {line}")
        print("=== END OF SOURCE CODE ===\n")
        
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
        
        # fix the function call here - _compile_code method needs to be properly defined or called correctly
        output_file = os.path.splitext(temp_file_path)[0]
        if platform.system() == "Windows":
            output_file += ".exe"
            
        # build compile command
        compile_cmd = self._build_compile_command(temp_file_path, output_file, language)
        if not compile_cmd:
            error_msg = f"Cannot compile {language} code - no suitable compiler found"
            result["errors"].append(error_msg)
            print(f"ERROR: {error_msg}")
            return result
            
        # execute compilation
        try:
            print(f"Executing compilation command: {' '.join(compile_cmd)}")
            process = subprocess.run(
                compile_cmd,
                text=True,
                capture_output=True,
                check=False
            )
            
            compiler_output = process.stdout + process.stderr
            result["compiler_output"] = compiler_output
            
            # print compile output (for debugging)
            if compiler_output:
                print("\n--- COMPILER OUTPUT START ---")
                print(compiler_output)
                print("--- COMPILER OUTPUT END ---\n")
            
            if process.returncode == 0 and os.path.exists(output_file):
                result["success"] = True
                result["executable"] = output_file
                print("Compilation successful.")
            else:
                result["success"] = False
                if compiler_output:
                    error_lines = compiler_output.split('\n')
                    errors = [line for line in error_lines if "error:" in line.lower()]
                    result["errors"] = errors if errors else ["Compilation failed with unknown error"]
                    print("Compilation errors detected:")
                    for i, error in enumerate(errors, 1):
                        print(f"  {i}. {error}")
                else:
                    result["errors"].append(f"Compilation failed with return code: {process.returncode}")
                    print(f"Compilation failed with return code: {process.returncode}")
                print("Compilation failed. Skipping execution phase.")
                return result
                
        except Exception as e:
            result["success"] = False
            error_msg = f"Error during compilation: {str(e)}"
            result["errors"].append(error_msg)
            print(f"ERROR: {error_msg}")
            return result
        
        # Run the compiled code
        print("\n--- EXECUTION PHASE ---")
        run_result = self._run_code(output_file, language, timeout)
        result.update(run_result)
        
        # Clean up temporary files (optional)
        self._cleanup_files(temp_file_path, output_file)
        print("Temporary files cleaned up")
        
        print(f"=== COMPILATION AND EXECUTION COMPLETE ===\n")
        return result
    
    def _create_temp_file(self, code: str, language: str) -> Optional[str]:
        """Create a temporary file with the provided code"""
        # Determine file extension based on language
        extension = self._get_file_extension(language)
        
        try:
            # Replace problematic Unicode characters in the code if on Windows
            if platform.system() == "Windows":
                # Replace check mark with "[OK]"
                code = code.replace("✓", "[OK]")
                # Replace "×" with "[ERROR]"
                code = code.replace("✗", "[ERROR]")
                # Replace "²" and other superscripts
                code = code.replace("²", "^2")
                code = code.replace("³", "^3")
                # Replace other potential problematic characters
                code = code.replace("…", "...")
            
            # Create a temporary file with the appropriate extension and proper encoding
            fd, temp_path = tempfile.mkstemp(suffix=extension, dir=self.working_dir)
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
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
            "MPI": ".cpp",      # MPI is typically used with C/C++
            "JAX": ".py",       # JAX uses Python file extension
            "PYTHON": ".py"     # Add support for normal Python
        }
        return extensions.get(language, ".txt")
    
    def _compile_code(self, state: Dict) -> Dict:
        """Compile translated code and measure performance"""
        print("===========================================")
        print("Start Compilation")
        
        # Extract code and target language
        translated_code = str(state.get("translated_code", "")) if state.get("translated_code") is not None else ""
        target_language = str(state.get("target_language", "")) if state.get("target_language") is not None else ""
        
        if not translated_code or not target_language:
            return {
                "compilation_success": False,
                "compilation_errors": ["Missing code or target language"]
            }
            
        # Clean code for compilation
        cleaned_code = self._clean_code_for_compilation(translated_code)
        
        # Print cleaned code for debugging (with line numbers)
        print("\n=== CLEANED CODE FOR COMPILATION ===")
        for i, line in enumerate(cleaned_code.split('\n')):
            print(f"{i+1:4d}: {line}")
        print("=== END OF CLEANED CODE ===\n")
        
        # Skip compilation for languages that don't need it
        if not self._is_language_compilable(target_language):
            return {
                "compilation_success": True,
                "compilation_message": f"{target_language} does not require compilation",
                "execution_output": "Execution not supported for this language"
            }
            
        # Compile and run
        compilation_result = self.compiler_agent.compile_and_run(
            code=cleaned_code,
            language=target_language
        )
        
        # Extract relevant fields
        success = compilation_result.get("success", False)
        errors = compilation_result.get("errors", [])
        compiler_output = compilation_result.get("compiler_output", "")
        execution_output = compilation_result.get("execution_output", "")
        execution_time = compilation_result.get("execution_time")
        
        # Analyze compiler output for errors if compilation failed
        if not success and compiler_output:
            error_analysis = self.compiler_agent.analyze_compilation_errors(compiler_output)
            
            # Log detailed error analysis
            if error_analysis:
                common_issues = error_analysis.get("common_issues", [])
                if common_issues:
                    print("Common Issues:")
                    for issue in common_issues:
                        print(f"- {issue}")
                
                suggested_fixes = error_analysis.get("suggested_fixes", [])
                if suggested_fixes:
                    print("Suggested Fixes:")
                    for fix in suggested_fixes:
                        print(f"- {fix}")
        else:
            error_analysis = {}
            
        # Update state
        state["compilation_success"] = success
        state["compilation_errors"] = errors
        state["compilation_output"] = compiler_output
        state["execution_output"] = execution_output if success else ""
        state["execution_time_seconds"] = execution_time
        state["compilation_error_analysis"] = error_analysis
            
        return state
    
    def _build_compile_command(self, source_file: str, output_file: str, language: str, state: Dict = None) -> Optional[list]:
        """Build the compilation command based on the language"""
        print(f"Building compilation command for {language} source file: {source_file}")
        
        try:
            with open(source_file, 'r') as f:
                code_content = f.read()
                
            # use LLM to select compiler
            compiler_selection = self._select_compiler_with_llm(code_content, language, state)
            
            if compiler_selection["success"]:
                compiler_path = compiler_selection["compiler_path"]
                compile_options = compiler_selection["compile_options"]
                
                # build basic compile command
                cmd = [compiler_path, "-o", output_file, source_file]
                
                # add LLM recommended compile options
                cmd.extend(compile_options)
                
                print(f"Using compiler: {compiler_selection['compiler_name']}")
                print(f"Compile command: {' '.join(cmd)}")
                
                return cmd
                
        except Exception as e:
            print(f"Error reading source file or building compile command: {str(e)}")
        
        print("Fallback to default compiler selection logic")
        
        # backup original compiler selection logic
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
                # use static link OpenMP library, and increase stack size or use heap allocation
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
        
        # For Python/JAX, ensure proper encoding
        if language in ["JAX", "PYTHON"]:
            # Set environment variable for UTF-8 on Windows
            if platform.system() == "Windows":
                env["PYTHONIOENCODING"] = "utf-8"
        
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
            print(result["execution_output"])
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
        
        # Standardize language name
        language = language.upper()
        
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
        
        elif language in ["PYTHON", "JAX"]:
            # For Python/JAX code, use the Python interpreter to run
            python_executable = sys.executable
            return [python_executable, executable]
        
        print(f"Error: No suitable run command found for {language}")
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
        
        # JAX specific
        if language == "JAX":
            if "tracer error" in output.lower():
                analysis["contains_error"] = True
                analysis["error_types"].append("jax_tracer_error")
                analysis["correctness_issues"].append("JAX tracer error, possibly using Python control flow in JIT compiled functions")
            
            if "no gpu/tpu found" in output.lower():
                analysis["contains_error"] = False  # This is not an error, just a warning
                analysis["performance_issues"].append("JAX did not detect GPU/TPU, using CPU")
            
            if "illegal memory access" in output.lower():
                analysis["contains_error"] = True
                analysis["error_types"].append("jax_memory_error")
                analysis["correctness_issues"].append("JAX memory access error, possibly due to insufficient GPU memory or array out of bounds")
        
        return analysis
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get the status of the compiler agent, including available compilers and environments"""
        return {
            "available_compilers": list(self.compilers.keys()),
            "runtime_environments": self.runtime_environments,
            "working_directory": self.working_dir
        }
    
    def _clean_code_for_compilation(self, code: str) -> str:
        """Clean code for compilation, fixing common issues"""
        if not code:
            return code
            
        print("\n=== CLEANING CODE FOR COMPILATION ===")
        print("Original code length:", len(code))
        
        # Fix 1: Normalize line endings to system standard
        code = code.replace('\r\n', '\n').replace('\r', '\n')
        
        # Fix 2: Fix multi-line string literals in printf statements
        print("Fixing multi-line string literals in printf...")
        pattern = r'(printf\s*\(\s*"[^"]*?)(\n)([^"]*"\s*(?:,|\)))'
        fixed_count = 0
        while re.search(pattern, code):
            code = re.sub(pattern, r'\1\\n\3', code, count=1)
            fixed_count += 1
        if fixed_count > 0:
            print(f"  Fixed {fixed_count} multi-line printf statements")
        
        # Fix 3: Fix broken string literals without proper line continuation
        print("Fixing broken string literals...")
        pattern = r'("[^"]*?)(\n)([^"]*")'
        fixed_count = 0
        while re.search(pattern, code):
            code = re.sub(pattern, r'\1 \3', code, count=1)
            fixed_count += 1
        if fixed_count > 0:
            print(f"  Fixed {fixed_count} broken string literals")
        
        # Fix 4: Handle more complex multi-line strings with quotes in the middle
        print("Fixing complex multi-line strings...")
        lines = code.split('\n')
        fixed_lines = []
        open_quote = False
        fixed_count = 0
        
        for i, line in enumerate(lines):
            # Count quotes in the line (ignoring escaped quotes)
            quotes = [m.start() for m in re.finditer(r'(?<!\\)"', line)]
            quote_count = len(quotes)
            
            # If we have odd number of quotes, toggle the open_quote state
            if quote_count % 2 == 1:
                if open_quote:
                    # This line closes a quote
                    open_quote = False
                else:
                    # This line opens a quote that continues to next line
                    open_quote = True
                    # Check if next line continues the string
                    if i < len(lines) - 1 and '"' in lines[i+1]:
                        # Convert to proper line continuation with string concatenation
                        if not line.endswith('"'):
                            line = line + '" "'  # Close string and start new one
                            fixed_count += 1
            
            # If line ends with an open quote, fix it
            if open_quote and quotes and quotes[-1] == len(line) - 1:
                line = line + '\\'  # Add line continuation
                fixed_count += 1
                
            fixed_lines.append(line)
        
        if fixed_count > 0:
            print(f"  Fixed {fixed_count} complex multi-line string issues")
            
        code = '\n'.join(fixed_lines)
        
        # Fix 5: Fix common error of missing newline at the end of file
        if not code.endswith('\n'):
            code += '\n'
            print("  Added missing newline at end of file")
        
        print("Cleaned code length:", len(code))
        print("=== CODE CLEANING COMPLETE ===\n")
        return code
    
    def _emergency_fix_printf(self, code: str) -> str:
        print("\n=== EMERGENCY PRINTF FIX ===")
        fixed_code = code
        
        # find potentially problematic printf statements
        printf_pattern = re.compile(r'(printf\s*\(\s*"[^"]*?)(\n)([^"]*")', re.DOTALL)
        if printf_pattern.search(fixed_code):
            print("Detected potentially problematic printf statements, fixing...")
            fixed_code = printf_pattern.sub(r'\1 \3', fixed_code)
            
        # check for quote mismatches
        lines = fixed_code.split('\n')
        open_quotes = False
        fixed_lines = []
        
        for i, line in enumerate(lines):
            # if this line contains an odd number of quotes, it indicates a change in quote state
            if line.count('"') % 2 == 1:
                if open_quotes:
                    # if there is an unclosed quote, this line should close it
                    open_quotes = False
                else:
                    # if there is no unclosed quote, this line starts a new string
                    if i < len(lines) - 1:
                        # check if the next line might be part of the string
                        next_line = lines[i+1]
                        if '"' in next_line and next_line.count('"') % 2 == 1:
                            # it is likely a cross-line string, try to merge
                            print(f"Merging line {i+1} and line {i+2} of cross-line string")
                            combined_line = line + " " + next_line
                            fixed_lines.append(combined_line)
                            i += 1  # skip the next line
                            continue
                    open_quotes = True
            
            # handle normal lines
            fixed_lines.append(line)
        
        # if there is an unclosed quote, try to fix it
        if open_quotes:
            print("Warning: code contains unclosed quotes! Adding closing quote to avoid compilation errors")
            fixed_lines[-1] = fixed_lines[-1] + '"'
        
        fixed_code = '\n'.join(fixed_lines)
        print("=== EMERGENCY FIX COMPLETE ===\n")
        
        return fixed_code
    
    def _clean_thinking_process(self, text: str) -> str:
        """
        Remove thinking process markers like <think>...</think>
        
        Args:
            text: text to clean
            
        Returns:
            cleaned text
        """
        if not text:
            return ""
            
        # remove <think>...</think>
        think_pattern = re.compile(r'<think>.*?</think>', re.DOTALL)
        text = think_pattern.sub('', text).strip()
        
        # remove common thinking patterns
        common_patterns = [
            r'(?i)Let me think\s*:.*?\n\s*\n',      # "Let me think: ..." 
            r'(?i)I\'ll analyze this.*?\n\s*\n',    # "I'll analyze this..."
            r'(?i)Let\'s analyze.*?\n\s*\n',        # "Let's analyze..."
            r'(?i)First, I need to.*?\n\s*\n',      # "First, I need to..."
            r'(?i)Step \d+:.*?\n\s*\n',             # "Step 1: ..."
            r'(?i)My thinking:.*?\n\s*\n',          # "My thinking:..."
            r'(?i)Hmm, .*?\n\s*\n',                 # "Hmm, ..."
            r'(?i)Okay, .*?\n\s*\n'                 # "Okay, ..."
        ]
        
        for pattern in common_patterns:
            text = re.sub(pattern, '', text, flags=re.DOTALL)
            
        return text.strip()

    
