#!/usr/bin/env python3
"""
Analysis Agent
Responsible for code analysis, planning, and requirements detection
"""

import re
from typing import Dict, List, Any, Tuple

from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser

class AnalysisAgent:
    """Agent responsible for advanced code analysis and planning"""
    
    def __init__(self, llm, knowledge_base):
        """Initialize the analysis agent"""
        self.llm = llm
        self.knowledge_base = knowledge_base
        
        # List of supported languages
        self.supported_languages = [
            "C", "C++", "FORTRAN", "CUDA", "OpenMP", "JAX", 
            "Python", "MPI", "OpenCL", "OpenACC"
        ]
    
    def analyze_code(self, user_input: str) -> Dict[str, Any]:
        """
        Enhanced analysis of user input to extract source language, 
        target language and code content with improved robustness
        """
        # First, use a specialized prompt to extract key information
        analysis_template = """You are a senior code analysis expert specializing in High Performance Computing.
        Analyze the following user input and extract key information:
        
        User input: {{user_input}}
        
        Perform these specific tasks in order:
        1. Source Language Identification (must be a programming language, preferably C/C++/FORTRAN/CUDA/OpenMP/JAX)
        2. Target Language Identification (must be a programming language, preferably C/C++/FORTRAN/CUDA/OpenMP/JAX)
        3. Code Segmentation: Extract ONLY the code block needing conversion
        4. Analyze potential conversion challenges for HPC (parallelism, memory, optimization)
        5. Generate a structured task description
        
        {% if analysis_rules %}
        ADDITIONAL ANALYSIS RULES:
        {{ analysis_rules }}
        {% endif %}
        
        Respond ONLY in this strict format:
        Source Language: [detected source language]
        Target Language: [detected target language]
        Code Content: 
        ```[source language]
        [extracted code block]
        ```
        Potential Issues: 
        - [Issue1 description]
        - [Issue2 description]
        Task Description: "Convert the following [source] code to [target] with HPC optimizations"
        """
        
        prompt = PromptTemplate(
            template=analysis_template,
            input_variables=["user_input"],
            partial_variables={"analysis_rules": self._get_analysis_rules("")}, 
            template_format="jinja2"
        )
        
        chain = prompt | self.llm | StrOutputParser() | {
            # Add required field validation
            "analysis": lambda x: x,
            "is_validated": lambda x: all(
                kw in x.lower() 
                for kw in ["target language", "source language", "code content:"]
            )
        }
        
        result = chain.invoke({"user_input": user_input})
        
        # Parse the analysis text
        try:
            parsed_data = self._parse_analysis_text(result["analysis"])
            
            # Validate languages are supported
            self._validate_languages(parsed_data)
            
            # Return structured result
            return {
                "analysis": result["analysis"],
                "is_validated": result["is_validated"],
                "parsed_data": {
                    "source_language": parsed_data["source_lang"],
                    "target_language": parsed_data["target_lang"],
                    "code_content": parsed_data["code_content"],
                    "potential_issues": parsed_data["potential_issues"]
                }
            }
        except Exception as e:
            return {
                "analysis": result["analysis"],
                "is_validated": False,
                "error": str(e)
            }
    
    def _validate_languages(self, parsed_data):
        """Validate that the languages are in our supported list or provide fallbacks"""
        if not parsed_data["source_lang"] or parsed_data["source_lang"].lower() == "unknown":
            # Try to detect source language from code
            parsed_data["source_lang"] = self._detect_language_from_code(parsed_data["code_content"])
        
        if not parsed_data["target_lang"] or parsed_data["target_lang"].lower() == "unknown":
            # Default to a common HPC language
            parsed_data["target_lang"] = "CUDA"  # Default to CUDA if not specified
    
    def _detect_language_from_code(self, code):
        """Use code patterns to detect the programming language"""
        if not code:
            return "Unknown"
            
        # Common language patterns
        patterns = {
            "CUDA": [r"__global__", r"__device__", r"cudaMalloc", r"blockIdx", r"threadIdx"],
            "OpenMP": [r"#pragma\s+omp", r"omp_get_thread_num"],
            "MPI": [r"MPI_Init", r"MPI_Comm_size", r"MPI_Finalize"],
            "FORTRAN": [r"\bSUBROUTINE\b", r"\bPROGRAM\b", r"\bMODULE\b", r"\bENTRY\b", r"IMPLICIT"],
            "C++": [r"#include\s+<(iostream|vector|string)", r"std::", r"template", r"class.*{"],
            "C": [r"#include\s+<(stdio\.h|stdlib\.h)", r"\bmalloc\b", r"\bfree\b"],
            "JAX": [r"import\s+jax", r"jax\.numpy", r"jax\.jit", r"jax\.grad"]
        }
        
        lang_scores = {}
        for lang, expressions in patterns.items():
            lang_scores[lang] = sum(1 for expr in expressions if re.search(expr, code))
        
        # Return the language with the highest score, if any patterns matched
        if max(lang_scores.values(), default=0) > 0:
            return max(lang_scores.items(), key=lambda x: x[1])[0]
        
        return "Unknown"
    
    def generate_plan(self, source_language: str, target_language: str, 
                      code_content: str, potential_issues: List[str]) -> str:
        """Generate an enhanced conversion plan based on analysis"""

        source_language = str(source_language) if source_language is not None else ""
        target_language = str(target_language) if target_language is not None else ""
        code_content = str(code_content) if code_content is not None else ""

        safe_issues = []
        if potential_issues:
            for issue in potential_issues:
                if issue is not None:
                    safe_issues.append(str(issue))
        
        planning_template = """Develop a detailed HPC code conversion plan based on the following analysis:
        Source Language: {{source_language}}
        Target Language: {{target_language}}
        Code Content: 
        ```
        {{code_content}}
        ```
        Known Issues: {{potential_issues}}
        
        Please create a comprehensive conversion plan covering:
        1. Key Architectural Differences: Identify fundamental differences between {{source_language}} and {{target_language}} relevant to HPC
        2. Parallelism Strategy: How parallel constructs will be translated
        3. Memory Management: How memory operations will be handled
        4. Performance Considerations: Key optimizations to apply
        5. Validation Criteria: How to verify correctness post-conversion
        
        Output Format:
        Conversion Plan:
        - [Phase 1: Foundation]: Convert basic syntax and structure
        - [Phase 2: Parallelism]: Map parallel constructs to {{target_language}} equivalents
        - [Phase 3: Memory Optimization]: Optimize memory access patterns
        - [Phase 4: Performance Tuning]: Apply {{target_language}}-specific optimizations
        
        Current Phase: [Phase 1]
        """
        
        chain = PromptTemplate.from_template(planning_template, template_format="jinja2") | self.llm
        
        plan = chain.invoke({
            "source_language": source_language,
            "target_language": target_language,
            "code_content": code_content,
            "potential_issues": safe_issues
        })
        
        print(plan.content)
        return plan.content
    
    def _get_analysis_rules(self, target_lang: str) -> str:
        """Get enhanced analysis rules from knowledge base"""
        if not self.knowledge_base:
            return ""
            
        # Get general analysis rules
        general_rules = self.knowledge_base.get("analysis_rules", [])
        
        # Get target language specific rules if available
        target_rules = self.knowledge_base.get(target_lang, {}).get("analysis_rules", [])
        
        all_rules = general_rules + target_rules
        return "Special Analysis Rules:\n" + "\n".join([f"- {rule}" for rule in all_rules])
    
    def _parse_analysis_text(self, analysis_text: str) -> Dict:
        """Structured parsing of analysis text with improved robustness"""
        parsed_data = {
            "source_lang": None,
            "target_lang": None,
            "code_content": None,
            "potential_issues": [],
            "task_description": None
        }
        
        current_section = None
        code_content_started = False
        code_block_marker = False
        
        for line in analysis_text.split('\n'):
            raw_line = line.rstrip()  # Keep original format
            clean_line = raw_line.strip()
            
            # Source language detection
            if re.match(r"^source[ *]*lang(uage)?\s*:", clean_line, re.I):
                parsed_data["source_lang"] = re.split(r":\s*", clean_line, 1)[-1].strip()
                code_content_started = False
                
            # Target language detection
            elif re.match(r"^target[ *]*lang(uage)?\s*:", clean_line, re.I):
                parsed_data["target_lang"] = re.split(r":\s*", clean_line, 1)[-1].strip()
                code_content_started = False
                
            # Code content detection - handle both with and without code blocks
            elif re.match(r"^code[ *]*content\s*:", clean_line, re.I):
                parsed_data["code_content"] = ""
                code_content_started = True
                # Check if there's code on the same line
                if ":" in clean_line:
                    code_part = clean_line.split(":", 1)[1].strip()
                    if code_part and not code_part.startswith("```"):
                        parsed_data["code_content"] = code_part
                
            # Handle code blocks with triple backticks
            elif code_content_started and clean_line.startswith("```"):
                if not code_block_marker:
                    # Start of code block - skip the line with language identifier
                    code_block_marker = True
                else:
                    # End of code block
                    code_block_marker = False
                    code_content_started = False
                    
            # Process potential issues section
            elif re.match(r"^potential[ *]*issues?\s*:", clean_line, re.I):
                current_section = "potential_issues"
                code_content_started = False
                
            # Capture task description
            elif re.match(r"^task[ *]*description?\s*:", clean_line, re.I):
                current_section = "task_description"
                code_content_started = False
                parsed_data["task_description"] = clean_line.split(":", 1)[1].strip() if ":" in clean_line else ""
                
            # Process issues list items
            elif current_section == "potential_issues" and clean_line.startswith(('-', '*')):
                parsed_data["potential_issues"].append(clean_line[1:].strip())
                
            # Append to task description
            elif current_section == "task_description" and parsed_data["task_description"]:
                parsed_data["task_description"] += " " + clean_line
                
            # Handle code content lines
            elif code_content_started and not (code_block_marker and clean_line.startswith("```")):
                if parsed_data["code_content"]:
                    parsed_data["code_content"] += "\n" + raw_line  # Keep original indentation
                else:
                    parsed_data["code_content"] = raw_line
        
        # Process code content
        if parsed_data["code_content"]:
            parsed_data["code_content"] = self._process_code_content(parsed_data["code_content"])
        
        # Validation - ensure we have the mandatory fields
        if not parsed_data["source_lang"] or not parsed_data["target_lang"]:
            raise ValueError("Missing required language fields")
        
        return parsed_data
    
    def _process_code_content(self, code_block: str) -> str:
        """Enhanced cleanup of code block formatting"""
        # Remove potential code language declarations (like ```cuda)
        if re.search(r"^```[\w+-]*\s*\n", code_block):
            code_block = re.sub(r"^```[\w+-]*\s*\n", "", code_block, count=1, flags=re.IGNORECASE)
        if code_block.endswith("```"):
            code_block = code_block.rsplit("```", 1)[0]
            
        # Remove leading/trailing whitespace while preserving internal indentation
        code_lines = code_block.split('\n')
        while code_lines and not code_lines[0].strip():
            code_lines.pop(0)
        while code_lines and not code_lines[-1].strip():
            code_lines.pop()
            
        return "\n".join(code_lines)
        
    def extract_code_features(self, code_content: str) -> str:
        """Extract key structural features from the code for better HPC analysis"""
        feature_set = {
            "parallel_constructs": set(),
            "memory_operations": set(),
            "numerical_calculations": set(),
            "control_flow": set(),
            "hardware_interaction": set(),
            "external_dependencies": set()
        }
        
        # Enhanced pattern detection - add more languages and patterns
        patterns = {
            # CUDA patterns
            "cuda_kernel": re.compile(r"__global__\s+void"),
            "cuda_memory": re.compile(r"cudaMalloc|cudaMemcpy|cudaFree"),
            "cuda_sync": re.compile(r"__syncthreads|cudaDeviceSynchronize"),
            
            # OpenMP patterns
            "openmp": re.compile(r"#pragma\s+omp"),
            "omp_parallel": re.compile(r"#pragma\s+omp\s+parallel"),
            "omp_for": re.compile(r"#pragma\s+omp\s+for"),
            
            # MPI patterns
            "mpi": re.compile(r"MPI_\w+"),
            "mpi_comm": re.compile(r"MPI_Comm_\w+"),
            "mpi_collective": re.compile(r"MPI_Allreduce|MPI_Bcast|MPI_Scatter|MPI_Gather"),
            
            # SIMD patterns
            "simd": re.compile(r"#pragma\s+(simd|ivdep|vector)"),
            "vectorization": re.compile(r"__m(128|256|512)|_mm(256|512)_\w+"),
            
            # Memory operations
            "memory_ops": re.compile(r"\b(malloc|free|new|delete|cudaMalloc|cudaFree)\b"),
            "aligned_memory": re.compile(r"aligned_alloc|posix_memalign|__align__"),
            
            # Numerical patterns
            "num_calcs": re.compile(r"exp[fl]?|log[fl]?|pow[fl]?|sin[fl]?|sqrt[fl]?"),
            "complex_math": re.compile(r"cblas_\w+|lapack_\w+"),
            
            # Synchronization
            "critical_sec": re.compile(r"\b#pragma\s+critical\b|pthread_mutex_\w+"),
            "atomic_ops": re.compile(r"\b(atomicAdd|__sync_fetch_and_add|std::atomic)\b"),
            
            # Complex types
            "complex_types": re.compile(r"struct|class|template|typedef")
        }
        
        # Language-specific patterns
        fortran_patterns = {
            "coarray": re.compile(r"\[\s*:\s*\]", re.IGNORECASE),
            "parallel_do": re.compile(r"!\$omp\s+parallel\s+do", re.IGNORECASE),
            "blas_calls": re.compile(r"call\s+(dgemm|sgemm|daxpy)", re.IGNORECASE),
            "do_concurrent": re.compile(r"do\s+concurrent", re.IGNORECASE)
        }
        
        jax_patterns = {
            "jit_decorator": re.compile(r"@jax.jit"),
            "vmap_usage": re.compile(r"jax.vmap"),
            "pmap_usage": re.compile(r"jax.pmap"),
            "random_key": re.compile(r"jax.random.PRNGKey"),
            "grad_func": re.compile(r"jax.grad")
        }
        
        # Enhanced multi-pattern scanning
        for line in code_content.split('\n'):
            # CUDA detection
            if patterns["cuda_kernel"].search(line):
                feature_set["parallel_constructs"].add("CUDA Kernel")
                feature_set["hardware_interaction"].add("GPU Execution")
            
            if patterns["cuda_memory"].search(line):
                feature_set["memory_operations"].add("CUDA Memory Management")
                
            if patterns["cuda_sync"].search(line):
                feature_set["control_flow"].add("CUDA Synchronization")
            
            # OpenMP detection
            if patterns["openmp"].search(line):
                if "parallel for" in line:
                    feature_set["parallel_constructs"].add("OpenMP Parallel Loop")
                elif "critical" in line:
                    feature_set["control_flow"].add("Critical Section")
                elif "simd" in line:
                    feature_set["parallel_constructs"].add("SIMD Vectorization")
            
            # MPI detection
            if patterns["mpi"].search(line):
                feature_set["parallel_constructs"].add("MPI Communication")
                if patterns["mpi_collective"].search(line):
                    feature_set["parallel_constructs"].add("MPI Collective Operations")
                    
            # Numerical calculation features
            if patterns["num_calcs"].search(line):
                if "_double" in line or "d" + patterns["num_calcs"].pattern[0] in line:
                    feature_set["numerical_calculations"].add("Double Precision Calculation")
                else:
                    feature_set["numerical_calculations"].add("Floating Point Operation")
                    
            # Memory access pattern detection
            if "stride" in line.lower() or "align" in line.lower():
                feature_set["memory_operations"].add("Memory Access Pattern Optimization")
                
            # Check for complex math operations
            if patterns["complex_math"].search(line):
                feature_set["numerical_calculations"].add("Linear Algebra Operations")
                    
            # Fortran feature scanning
            for pattern_name, pattern in fortran_patterns.items():
                if pattern.search(line):
                    if pattern_name == "coarray":
                        feature_set["parallel_constructs"].add("Fortran Coarray Parallelism")
                    elif pattern_name == "parallel_do":
                        feature_set["parallel_constructs"].add("Fortran OpenMP Parallel Loop")
                    elif pattern_name == "blas_calls":
                        feature_set["numerical_calculations"].add("BLAS Function Calls")
                    elif pattern_name == "do_concurrent":
                        feature_set["parallel_constructs"].add("Fortran DO CONCURRENT")
                
            # JAX feature scanning
            for pattern_name, pattern in jax_patterns.items():
                if pattern.search(line):
                    if pattern_name == "jit_decorator":
                        feature_set["parallel_constructs"].add("JAX JIT Compilation")
                    elif pattern_name == "vmap_usage":
                        feature_set["parallel_constructs"].add("JAX Vectorization (vmap)")
                    elif pattern_name == "pmap_usage":
                        feature_set["parallel_constructs"].add("JAX Parallel Execution (pmap)")
                    elif pattern_name == "grad_func":
                        feature_set["numerical_calculations"].add("JAX Automatic Differentiation")
        
        # Generate feature report
        report = []
        for category, features in feature_set.items():
            if features:
                feat_list = "\n".join([f"  - {f}" for f in sorted(features)])
                report.append(f"{category.upper()}:\n{feat_list}")
                
        return "\n\n".join(report) if report else "No significant HPC features detected"