#!/usr/bin/env python3
"""
Analysis Agent
Responsible for code analysis, planning, and requirements detection
"""

import re
from typing import Dict, List, Any, Tuple
from datetime import datetime
from pathlib import Path

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
        analysis_template = """
        You are a senior code analysis expert specializing in High Performance Computing.
        Analyze the following user input and extract key information:
        
        User input: {{user_input}}
        
        Perform these specific tasks in order:
        1. Source Language Identification (must be a programming language, preferably C/C++/FORTRAN/CUDA/OpenMP/JAX/Pytorch)
        2. Target Language Identification (must be a programming language, preferably C/C++/FORTRAN/CUDA/OpenMP/JAX/Pytorch)
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
        
        # clean thinking process in analysis result
        if "analysis" in result:
            result["analysis"] = self._clean_thinking_process(result["analysis"])
        
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
        # ensure all inputs are string types
        source_language = str(source_language) if source_language is not None else ""
        target_language = str(target_language) if target_language is not None else ""
        code_content = str(code_content) if code_content is not None else ""
        
        # ensure potential_issues is a list of strings
        safe_issues = []
        if potential_issues:
            for issue in potential_issues:
                if issue is not None:
                    safe_issues.append(str(issue))
        
        # extract code features, for generating more specific plan
        code_features = self.extract_code_features(code_content)
        
        # create a more detailed plan template, including HPC-specific considerations
        planning_template = """
        Develop a comprehensive HPC code conversion plan based on the following analysis:
        Source Language: {{source_language}}
        Target Language: {{target_language}}
        
        Code Features:
        {{code_features}}
        
        Known Issues: 
        {% for issue in potential_issues %}
        - {{issue}}
        {% endfor %}
        
        Please create a detailed conversion plan covering:
        
        1. KEY ARCHITECTURAL DIFFERENCES:
           - Compare memory models between {{source_language}} and {{target_language}}
           - Identify thread/process model differences
           - Analyze synchronization mechanism differences
           - Evaluate data layout compatibility issues
        
        2. PARALLELISM STRATEGY:
           - Map parallel constructs from {{source_language}} to {{target_language}}
           - Define work distribution patterns for optimal load balancing
           - Develop synchronization strategy for parallel regions
           - Determine optimal granularity for parallelization
        
        3. MEMORY MANAGEMENT:
           - Design data transfer strategy for heterogeneous memory
           - Optimize array access patterns for spatial/temporal locality
           - Address memory alignment requirements
           - Implement efficient memory allocation/deallocation patterns
        
        4. PERFORMANCE OPTIMIZATION TARGETS:
           - Identify critical hotspots for optimization
           - List vectorization opportunities
           - Develop loop transformation strategy (fusion, tiling, unrolling)
           - Define instruction-level parallelism approach
        
        5. TRANSLATION WORKFLOW:
           - Identify code regions requiring special attention
           - Define incremental testing milestones
           - Specify performance validation criteria
           - Set correctness validation methodology
        
        6. HPC-SPECIFIC CONSIDERATIONS:
           - Address scalability requirements
           - Implement performance portability strategies
           - Plan for resilience/fault tolerance if necessary
           - Consider energy efficiency optimizations
        
        Output Format:
        CONVERSION PLAN:
        
        [Phase 1: Foundation] - ETA: X hours
        - Task 1.1: [Specific task with approach]
        - Task 1.2: [Specific task with approach]
        ...
        
        [Phase 2: Parallelism] - ETA: X hours
        - Task 2.1: [Specific task with approach]
        - Task 2.2: [Specific task with approach]
        ...
        
        [Phase 3: Memory Optimization] - ETA: X hours
        - Task 3.1: [Specific task with approach]
        - Task 3.2: [Specific task with approach]
        ...
        
        [Phase 4: Performance Tuning] - ETA: X hours
        - Task 4.1: [Specific task with approach]
        - Task 4.2: [Specific task with approach]
        ...
        
        VALIDATION STRATEGY:
        - Correctness checks: [Specific methodology]
        - Performance checks: [Specific metrics and methodology]
        
        POTENTIAL CHALLENGES:
        - [Challenge 1]: [Mitigation strategy]
        - [Challenge 2]: [Mitigation strategy]
        
        CURRENT PHASE: [Phase 1]
        """
        
        chain = PromptTemplate.from_template(planning_template, template_format="jinja2") | self.llm
        
        # call LLM to generate plan
        plan = chain.invoke({
            "source_language": source_language,
            "target_language": target_language,
            "code_features": code_features,
            "potential_issues": safe_issues
        })
        
        # save plan for later use
        self._save_plan_to_log(source_language, target_language, plan.content)
        
        print("Generated conversion plan:")
        print("=" * 50)
        print(plan.content)
        print("=" * 50)
        
        return plan.content
        
    def _save_plan_to_log(self, source_language: str, target_language: str, plan: str) -> None:
        """Save the generated plan to a log file for later analysis and review"""
        try:
            # create logs directory
            log_dir = Path("logs/plans")
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # create unique filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"plan_{source_language}_to_{target_language}_{timestamp}.txt"
            filepath = log_dir / filename
            
            # save plan
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"CONVERSION PLAN: {source_language} TO {target_language}\n")
                f.write(f"Generated: {datetime.now().isoformat()}\n\n")
                f.write(plan)
                
            print(f"Plan saved to: {filepath}")
        except Exception as e:
            print(f"Error saving plan to log: {e}")

    def extract_plan_for_phase(self, plan: str, phase_number: int) -> Dict[str, Any]:
        """Extract tasks and strategies for a specific phase from the plan"""
        phase_map = {
            1: "Foundation",
            2: "Parallelism", 
            3: "Memory Optimization",
            4: "Performance Tuning"
        }
        phase_name = phase_map.get(phase_number, f"Phase {phase_number}")
        
        # find phase start and end positions
        phase_start = plan.find(f"[Phase {phase_number}:")
        if phase_start == -1:
            phase_start = plan.find(f"Phase {phase_number}:")
        
        if phase_start == -1:
            return {"tasks": [], "phase_name": phase_name, "found": False}
            
        # find start of next phase or end of plan
        next_phase_start = plan.find(f"[Phase {phase_number+1}:", phase_start)
        if next_phase_start == -1:
            next_phase_start = plan.find(f"Phase {phase_number+1}:", phase_start)
            
        if next_phase_start == -1:
            # if there is no next phase, find VALIDATION or POTENTIAL CHALLENGES sections
            next_sections = ["VALIDATION", "POTENTIAL CHALLENGES", "CURRENT PHASE"]
            for section in next_sections:
                pos = plan.find(section, phase_start)
                if pos != -1 and (next_phase_start == -1 or pos < next_phase_start):
                    next_phase_start = pos
        
        # if still no end position found, use the whole remaining text
        if next_phase_start == -1:
            next_phase_start = len(plan)
            
        # extract content of current phase
        phase_content = plan[phase_start:next_phase_start].strip()
        
        # extract task list
        tasks = []
        task_pattern = re.compile(r'- Task \d+\.\d+: (.*?)(?=\n- Task \d+\.\d+:|$)', re.DOTALL)
        task_matches = task_pattern.findall(phase_content)
        
        if not task_matches:
            # try more general pattern
            task_pattern = re.compile(r'- (.*?)(?=\n-|$)', re.DOTALL)
            task_matches = task_pattern.findall(phase_content)
        
        # handle tasks
        for task in task_matches:
            tasks.append(task.strip())
            
        return {
            "phase_name": phase_name,
            "tasks": tasks,
            "found": True,
            "phase_content": phase_content
        }
    
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
        
        # clean thinking process first
        analysis_text = self._clean_thinking_process(analysis_text)
        
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
            # try some fallback mechanisms to extract from text
            try:
                if not parsed_data["source_lang"]:
                    source_match = re.search(r"(?:from|in)\s+(\w+)(?:\s+code)?(?:\s+to|\s+into)", analysis_text, re.I)
                    if source_match:
                        parsed_data["source_lang"] = source_match.group(1)
                
                if not parsed_data["target_lang"]:
                    target_match = re.search(r"(?:to|into)\s+(\w+)(?:\s+code)?", analysis_text, re.I)
                    if target_match:
                        parsed_data["target_lang"] = target_match.group(1)
            except:
                pass
                
            # if still not found, provide default values
            if not parsed_data["source_lang"]:
                parsed_data["source_lang"] = "C"  # default assume C
            if not parsed_data["target_lang"]:
                parsed_data["target_lang"] = "OpenMP"  # default assume OpenMP
        
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
            
        # remove <think>...</think> blocks
        think_pattern = re.compile(r'<think>.*?</think>', re.DOTALL)
        text = think_pattern.sub('', text).strip()
        
        # remove other common thinking markers
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