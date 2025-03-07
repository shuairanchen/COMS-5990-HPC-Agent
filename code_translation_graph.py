#!/usr/bin/env python3
"""
Enhanced Code Translation Graph
Main workflow coordinator for code translation with improved user input handling
and compiler validation
"""

import os
import re
import json
import yaml
import hashlib
import difflib
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
from threading import Lock
from pathlib import Path
import tempfile

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langgraph.graph import StateGraph, END

# Import the enhanced analysis agent
from agents.analysis_agent import AnalysisAgent
from agents.translation_agent import TranslationAgent
from agents.verification_agent import VerificationAgent
from agents.compiler_agent import CompilerAgent
# from utils.safety_checker import SafetyChecker

class CodeTranslationGraph:
    """
    Enhanced code translation workflow using a state graph
    with improved error handling and performance monitoring
    """
    
    def __init__(self, kb_path: Optional[str] = None, working_dir: Optional[str] = None):
        """Initialize the code translation graph"""
        # Set up knowledge base
        self.kb_path = kb_path
        self.knowledge_base = self.load_knowledge_base(kb_path) if kb_path else {}
        
        # Set up working directory for compilation
        self.working_dir = working_dir or tempfile.mkdtemp()
        Path(self.working_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize agents
        self.llm = ChatOpenAI(temperature=0.9, model="gpt-4o-mini")
        self.analysis_agent = AnalysisAgent(self.llm, self.knowledge_base)
        self.translation_agent = TranslationAgent(self.llm, self.knowledge_base)
        self.verification_agent = VerificationAgent(self.llm, self.knowledge_base)
        self.compiler_agent = CompilerAgent(working_dir=self.working_dir)
        
        # Set up caching
        self.translation_cache = {}
        self.rule_cache = {}
        self.cache_lock = Lock()
        
        # Set up logging
        self.execution_log = []
        self.error_log = []
        
        # Create logs directory
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Set maximum iterations
        self.max_iterations = 3
        
        # Build workflow
        self.workflow = self._build_workflow()
    
    def _generate_cache_key(self, code: str, target_lang: str) -> str:
        """Generate a unique hash key for code snippet"""
        adjusted_code = code.strip().replace('\r\n', '\n')  # Normalize line endings
        algorithms = {
            'sha256': hashlib.sha256,
            'md5': hashlib.md5 
        }
        hasher = algorithms.get(os.getenv("HASH_ALGO", "sha256"))  # Configurable algorithm
        return hasher(f"{adjusted_code}::{target_lang}".encode()).hexdigest()
    
    def load_knowledge_base(self, file_path: str) -> Dict:
        """Load knowledge base from YAML file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self._log_step("load_kb_error", {"path": file_path}, str(e))
            return {}
    
    def _get_code_rules(self, target_lang: str) -> str:
        """Get the target language code rules with improved formatting"""
        rules = self.knowledge_base.get(target_lang, {})
        return "\n".join(
            [f"# {cat.upper()}\n" + "\n".join(f"- {item}" for item in items)
             for cat, items in rules.items() if cat != "analysis_rules"]
        )
    
    def _get_analysis_rules(self, target_lang: str) -> str:
        """Get enhanced analysis rules"""
        if not self.knowledge_base:
            return ""
            
        # Get both general rules and language-specific rules
        general_rules = self.knowledge_base.get("analysis_rules", [])
        target_rules = self.knowledge_base.get(target_lang, {}).get("analysis_rules", [])
        
        all_rules = general_rules + target_rules
        return "Special Rules:\n" + "\n".join([f"- {rule}" for rule in all_rules])
    
    def _log_step(self, step_name: str, input_data: dict, output_data: Any):
        """Log execution step with enhanced data visualization"""
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # Create a more detailed log entry
        log_entry = {
            "step": step_name,
            "timestamp": timestamp,
            "input": input_data,
            "output": output_data
        }
        
        # Add to execution log
        self.execution_log.append(log_entry)
        
        # Print step information with better formatting
        print(f"\n[{step_name.upper()}] - {timestamp}")
        
        # Print step-specific metrics for better visibility
        if step_name == "initial_translation_complete":
            if isinstance(output_data, dict) and "char_count" in output_data:
                print(f"  Code size: {output_data['char_count']} chars, {output_data['line_count']} lines")
        
        elif step_name == "compilation_complete":
            if isinstance(output_data, dict):
                success = output_data.get("success", False)
                status = "SUCCESS" if success else "FAILED"
                print(f"  Compilation: {status}")
                if not success and "errors" in output_data:
                    print(f"  Errors: {len(output_data['errors'])}")
        
        elif step_name == "validation_complete":
            if isinstance(output_data, dict) and "metadata" in output_data:
                metadata = output_data["metadata"]
                print(f"  Classification: {metadata.get('classification', 'unknown')}")
                print(f"  Severity: {metadata.get('severity', 'unknown')}")
                print(f"  Priority: {metadata.get('priority', 'unknown')}")
                if "violated_rules" in metadata:
                    print(f"  Violated rules: {len(metadata['violated_rules'])}")
        
        elif step_name == "code_improvement_complete":
            if isinstance(output_data, dict) and "improvement_metrics" in output_data:
                metrics = output_data["improvement_metrics"]
                print(f"  Changes: {metrics.get('changes_count', 0)}")
                print(f"  Improvement rate: {metrics.get('improvement_rate', 0):.2f}")
        
        # Save execution log periodically
        if len(self.execution_log) % 10 == 0:
            self._save_execution_log()
    
    def _log_error(self, error_msg: str, state: Dict = None, details: Any = None):
        """Enhanced error logging with better visualization"""
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # Create detailed error log
        error_log = {
            "error": error_msg,
            "timestamp": timestamp,
            "state": state,
            "details": str(details) if details else None
        }
        
        # Add to error log
        self.error_log.append(error_log)
        
        # Print error with better formatting
        print(f"\n[ERROR] - {timestamp}")
        print(f"  Message: {error_msg}")
        
        if state and "current_phase" in state:
            print(f"  Phase: {state['current_phase']}")
        
        if state and "iteration" in state:
            print(f"  Iteration: {state['iteration']}")
        
        if details:
            print(f"  Details: {str(details)}")
        
        # Save error log immediately
        self._save_error_log()
    
    def _generate_plan_with_fallback(self, state):
        """Generate a detailed plan with fallback mechanism"""
        try:
            return self._generate_plan(state)
        except Exception as e:
            self._log_step("fallback_plan", state, str(e))
            # Design default conversion plan template with more structure
            fallback_plan = f"""Conversion Plan:
            - [Phase 1: Foundation]: Basic conversion from {state.get('source_language','')} to {state.get('target_language','')}
            - [Phase 2: Parallelism]: Map parallel constructs to {state.get('target_language','')} equivalents
            - [Phase 3: Memory Optimization]: Optimize memory access patterns for {state.get('target_language','')}
            - [Phase 4: Performance Tuning]: Apply {state.get('target_language','')}-specific optimizations
            
            Current Phase: [Phase 1]
            """
            return {"conversion_plan": fallback_plan}
    
    def _build_workflow(self) -> StateGraph:
        """Building enhanced LangGraph Workflow with compiler validation"""
        workflow = StateGraph(state_schema=dict)
        
        # Define nodes with improved analysis and translation
        workflow.add_node("analyze_user_input", self._analyze_user_input)
        workflow.add_node("analyze_requirements", self._analyze_requirements)
        workflow.add_node("generate_plan", self._generate_plan_with_fallback)
        workflow.add_node("initial_translation", self._initial_translation)
        workflow.add_node("compile_code", self._compile_code)  # New node for compilation
        workflow.add_node("validate_code", self._validate_code)
        workflow.add_node("improve_code", self._improve_code)
        workflow.add_node("finalize_output", self._finalize_output)
        workflow.add_node("error_handling", self._handle_error)
        
        # Enhanced workflow edges
        workflow.set_entry_point("analyze_user_input")
        workflow.add_edge("analyze_user_input", "analyze_requirements")
        workflow.add_edge("analyze_requirements", "generate_plan")
        workflow.add_edge("generate_plan", "initial_translation")
        workflow.add_edge("initial_translation", "compile_code")  # Add compilation step
        workflow.add_edge("compile_code", "validate_code")
        
        # Set up the validation loop
        workflow.add_conditional_edges(
            "validate_code",
            self._should_improve,
            {
                "improve": "improve_code",
                "final": "finalize_output",
                "error": "error_handling"
            }
        )
        workflow.add_edge("improve_code", "compile_code")  # Test improvements with compiler
        
        # Set the final nodes
        workflow.add_edge("finalize_output", END)
        workflow.add_edge("error_handling", END)
        workflow.set_finish_point(["finalize_output", "error_handling"])
        
        return workflow.compile()
    
    def _analyze_user_input(self, state: Dict) -> Dict:
        """Node to pre-process and structure user input using LLM instead of regex"""
        print("===========================================")
        print("Start User Input Analysis")
        
        user_input = state.get("user_input", "")
        
        # Check if input is too short or empty
        if len(user_input.strip()) < 10:
            self._log_error("User input is too short or empty", state)
            state["error"] = "Please provide a more detailed request with code to translate"
            return state
        
        # Use LLM to extract information instead of regex
        analysis_template = """
        You are an expert code translator assistant. Analyze the following user request and extract key information:
        
        User request: {{user_input}}
        
        Extract the following information:
        1. Source programming language (what language is the original code in)
        2. Target programming language (what language the user wants the code translated to)
        3. The code that needs to be translated (extract the exact code snippet)
        
        Respond in this exact format:
        Source Language: [language name or "Unknown" if unclear]
        Target Language: [language name or "Unknown" if unclear]
        Code Content:
        ```
        [extracted code here, or "No code found" if no code is present]
        ```
        """
        
        prompt = PromptTemplate(
            template=analysis_template,
            input_variables=["user_input"],
            template_format="jinja2"
        )
        
        chain = prompt | self.llm
        
        try:
            # Get LLM analysis
            result = chain.invoke({"user_input": user_input})
            analysis_text = result.content
            
            # Parse the LLM output
            source_lang = None
            target_lang = None
            extracted_code = None
            
            for line in analysis_text.split('\n'):
                line = line.strip()
                if line.startswith("Source Language:"):
                    source_lang = line[len("Source Language:"):].strip()
                    if source_lang.lower() == "unknown":
                        source_lang = None
                elif line.startswith("Target Language:"):
                    target_lang = line[len("Target Language:"):].strip()
                    if target_lang.lower() == "unknown":
                        target_lang = None
            
            # Extract code between triple backticks
            code_match = re.search(r"```(?:\w+)?\n(.*?)```", analysis_text, re.DOTALL)
            if code_match:
                extracted_code = code_match.group(1).strip()
                if extracted_code.lower() == "no code found":
                    extracted_code = None
            
            # Update state with extracted information
            if source_lang:
                state["detected_source"] = source_lang
            if target_lang:
                state["detected_target"] = target_lang
            if extracted_code:
                state["extracted_code"] = extracted_code
            
            # Log the extraction results
            self._log_step("user_request_analyzed_by_llm", state, {
                "source": source_lang,
                "target": target_lang,
                "code_extracted": extracted_code is not None,
                "code_length": len(extracted_code) if extracted_code else 0
            })
            
        except Exception as e:
            self._log_error(f"Error during LLM analysis: {str(e)}", state)
            # Fallback to regex extraction if LLM fails
            self._fallback_regex_extraction(state, user_input)
        
        print("===========================================")
        return state

    def _fallback_regex_extraction(self, state: Dict, user_input: str) -> None:
        """Fallback method using regex if LLM extraction fails"""
        # Extract language translation request
        translation_pattern = re.compile(
            r"(?:translate|convert)\s+(?:this|from)\s+(\w+)(?:\s+code)?\s+(?:to|into)\s+(\w+)",
            re.IGNORECASE
        )
        match = translation_pattern.search(user_input)
        
        if match:
            state["detected_source"] = match.group(1)
            state["detected_target"] = match.group(2)
            self._log_step("user_request_detected_by_regex", state, {
                "source": match.group(1),
                "target": match.group(2)
            })
        
        # Extract code blocks if present
        code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", user_input, re.DOTALL)
        if code_blocks:
            # Use the largest code block if multiple are found
            largest_block = max(code_blocks, key=len)
            state["extracted_code"] = largest_block.strip()
            self._log_step("code_block_extracted_by_regex", state, {
                "code_length": len(state["extracted_code"])
            })
    
    def _analyze_requirements(self, state: Dict) -> Dict:
        """Enhanced requirement analysis node using the improved analysis agent"""
        print("===========================================")
        print("Start Analysis Requirements")
        
        # Delegate to enhanced analysis agent
        result = self.analysis_agent.analyze_code(state["user_input"])
        
        if not result["is_validated"]:
            error_msg = "Analysis result format validation failed. Missing required fields."
            state.update({
                "error": error_msg,
                "raw_analysis": result["analysis"]
            })
            self._log_step("invalid_analysis", state, result["analysis"])
            raise ValueError("Analysis results are missing key fields")
            
        # Update state with parsed data
        state.update(result["parsed_data"])
        
        # Add code features analysis
        if state.get("code_content"):
            code_features = self.analysis_agent.extract_code_features(state["code_content"])
            state["code_features"] = code_features
        
        # Check if the target language is compilable
        state["is_compilable"] = self._is_language_compilable(state.get("target_language", ""))
        
        self._log_step("requirement_analyzed", state, {
            "char_count": len(result["analysis"]),
            "validation_status": True,
            "features_extracted": "code_features" in state,
            "is_compilable": state["is_compilable"]
        })
        
        print("Analysis Requirements Parsed Data:", result["parsed_data"])
        print("===========================================")
        return state
    
    def _is_language_compilable(self, language: str) -> bool:
        """Check if the target language is compilable by our compiler agent"""
        if not language:
            return False
            
        compilable_languages = ["C", "C++", "CUDA", "FORTRAN", "OPENMP"]
        return language.upper() in compilable_languages
    
    def _generate_plan(self, state: Dict) -> Dict:
        """Generate enhanced conversion plan with code features consideration"""
        print("===========================================")
        print("Start Generate Plan")
        
        # Delegate to analysis agent for planning
        plan = self.analysis_agent.generate_plan(
            source_language=state.get('source_language'),
            target_language=state.get('target_language'),
            code_content=state.get('code_content'),
            potential_issues=state.get('potential_issues', [])
        )
        
        state["conversion_plan"] = plan
        
        # Add feature-specific plan elements if we have code features
        if "code_features" in state:
            features = state["code_features"]
            if "PARALLEL_CONSTRUCTS" in features:
                state["parallelism_strategy"] = "Detected parallel constructs - will optimize for parallel execution"
            if "MEMORY_OPERATIONS" in features:
                state["memory_strategy"] = "Detected memory operations - will implement efficient memory patterns"
        
        # Add compilation strategy if the language is compilable
        if state.get("is_compilable", False):
            compiler_status = self.compiler_agent.get_agent_status()
            state["compiler_status"] = compiler_status
            state["compilation_strategy"] = f"Will compile and verify code using {', '.join(compiler_status.get('available_compilers', ['N/A']))}"
        
        self._log_step("plan_generated", state, {
            "plan_length": len(plan),
            "has_parallelism_strategy": "parallelism_strategy" in state,
            "has_memory_strategy": "memory_strategy" in state,
            "has_compilation_strategy": "compilation_strategy" in state
        })
        
        print("Plan Generated:", plan)
        print("===========================================")
        return state
    
    def _initial_translation(self, state: Dict) -> Dict:
        """Node to perform initial code translation"""
        print("===========================================")
        print("Start Initial Translation")
        
        # Extract required fields
        source_language = state.get("source_language", "")
        target_language = state.get("target_language", "")
        code_content = state.get("code_content", "")
        
        # Validate inputs
        if not source_language or not target_language or not code_content:
            error_msg = "Missing required fields for translation"
            self._log_error(error_msg, state)
            state["error"] = error_msg
            return state
        
        # Check cache for existing translation
        cache_key = self._generate_cache_key(code_content, target_language)
        with self.cache_lock: 
            if cache_key in self.translation_cache:
                cached_result = self.translation_cache[cache_key]
                state["translated_code"] = cached_result["code"]
                state["cache_hit"] = True
                self._log_step("translation_cache_hit", state, cached_result["metadata"])
                print("Translation found in cache")
                return state
        
        # Perform translation
        try:
            translated_code = self.translation_agent.translate_code(
                source_language, target_language, code_content
            )
            
            # Clean up the translated code - remove markdown code block markers
            translated_code = self._clean_code_for_compilation(translated_code)
            
            # Update state
            state["translated_code"] = translated_code
            state["cache_hit"] = False
            
            # Cache the result
            with self.cache_lock:
                self.translation_cache[cache_key] = {
                        "code": translated_code,
                    "metadata": {
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "source_language": source_language,
                            "target_language": target_language
                        }
                    }
            
            self._log_step("initial_translation_complete", state, {
                "char_count": len(translated_code),
                "line_count": translated_code.count('\n') + 1
            })
            
        except Exception as e:
            error_msg = f"Translation failed: {str(e)}"
            self._log_error(error_msg, state, e)
            state["error"] = error_msg
        
        print("===========================================")
        return state

    def _clean_code_for_compilation(self, code: str) -> str:
        """Clean code by removing markdown formatting and other non-compilable elements"""
        # Remove markdown code block markers at the beginning
        code = re.sub(r'^```\w*\s*\n', '', code, flags=re.MULTILINE)
        
        # Remove markdown code block markers at the end
        code = re.sub(r'\n```\s*$', '', code, flags=re.MULTILINE)
        
        # Remove any remaining markdown code block markers
        code = re.sub(r'```\w*', '', code)
        
        # Remove any HTML-like comments that might have been added by the LLM
        code = re.sub(r'<!--.*?-->', '', code, flags=re.DOTALL)
        
        # Trim leading/trailing whitespace while preserving internal indentation
        code = code.strip()
        
        return code

    def _compile_code(self, state: Dict) -> Dict:
        """Node to compile and test the translated code"""
        print("===========================================")
        print("Start Code Compilation")
        
        # Skip if not compilable or no code
        if not state.get("is_compilable", False):
            print("Target language is not compilable, skipping compilation")
            state["compilation_success"] = None
            return state
        
        if "translated_code" not in state:
            error_msg = "No translated code to compile"
            self._log_error(error_msg, state)
            state["error"] = error_msg
            return state
        
        # Ensure code is clean before compilation
        code_to_compile = self._clean_code_for_compilation(state["translated_code"])
        
        # Compile and run the code
        target_language = state.get("target_language", "")
        try:
            compile_result = self.compiler_agent.compile_and_run(
                code_to_compile, target_language, timeout=10
            )
            
            # Update state with compilation results
            state["compilation_success"] = compile_result.get("success", False)
            state["compilation_output"] = compile_result.get("compiler_output", "")
            state["execution_output"] = compile_result.get("execution_output", "")
            state["execution_time_seconds"] = compile_result.get("execution_time")
            state["compilation_errors"] = compile_result.get("errors", [])
            
            # Add performance metrics if available
            if "performance_metrics" in compile_result:
                state["performance_metrics"] = compile_result["performance_metrics"]
            
            # Log compilation results
            self._log_step("code_compilation", state, {
                "success": state["compilation_success"],
                "error_count": len(state.get("compilation_errors", [])),
                "execution_time": state.get("execution_time_seconds")
            })
            
            # Analyze compilation errors if any
            if not state["compilation_success"] and state.get("compilation_output"):
                error_analysis = self.compiler_agent.analyze_compilation_errors(
                    state["compilation_output"]
                )
                state["compilation_error_analysis"] = error_analysis
                
                # Log error analysis
                self._log_step("compilation_error_analysis", state, error_analysis)
            
        except Exception as e:
            error_msg = f"Compilation process failed: {str(e)}"
            self._log_error(error_msg, state, e)
            state["error"] = error_msg
            state["compilation_success"] = False
            state["compilation_errors"] = [str(e)]
        
        print("Compilation Result:", state.get("compilation_success"))
        if not state.get("compilation_success", False):
            print("Compilation Errors:", state.get("compilation_errors", [])[:5])
        print("===========================================")
        return state
    
    def _validate_code(self, state: Dict) -> Dict:
        """Enhanced code validation with HPC-specific checks and compiler feedback"""
        print("===========================================")
        print("Start Validation Code")
        
        new_state = state.copy()
        
        # Pre-validation checks
        required_keys = ["target_language", "translated_code", "conversion_plan"]
        for key in required_keys:
            if key not in new_state:
                self._log_error(f"VALIDATION_PARAM_MISSING: {key}", new_state)
                raise KeyError(f"Validation requires: {key}")
        
        # Add HPC-specific validation context
        validation_context = {}
        if "code_features" in state:
            features = state["code_features"]
            if "PARALLEL_CONSTRUCTS" in features:
                validation_context["check_parallelism"] = True
            if "MEMORY_OPERATIONS" in features:
                validation_context["check_memory_patterns"] = True
        
        # Extract current phase from conversion plan
        current_phase = self._get_current_phase(new_state)
        
        # Include compiler feedback in validation if available
        validation_issues = state.get("potential_issues", []).copy()
        if "compilation_result" in state:
            comp_result = state["compilation_result"]
            
            # Add compilation errors to potential issues
            if not comp_result.get("success", False):
                for error in comp_result.get("errors", []):
                    validation_issues.append(f"Compilation error: {error}")
                
                # Add compiler analysis suggestions
                if "compiler_analysis" in state:
                    analysis = state["compiler_analysis"]
                    for issue in analysis.get("common_issues", []):
                        validation_issues.append(f"Compiler issue: {issue}")
                    for fix in analysis.get("suggested_fixes", []):
                        validation_issues.append(f"Suggested fix: {fix}")
            
            # Add execution issues if compilation succeeded but execution had issues
            elif "execution_analysis" in state:
                exec_analysis = state["execution_analysis"]
                if exec_analysis.get("contains_error", False):
                    for issue in exec_analysis.get("correctness_issues", []):
                        validation_issues.append(f"Runtime issue: {issue}")
                    for issue in exec_analysis.get("performance_issues", []):
                        validation_issues.append(f"Performance issue: {issue}")
        
        # Delegate to verification agent with enhanced context and compiler feedback
        result = self.verification_agent.validate_code(
            code=new_state["translated_code"],
            target_language=new_state["target_language"],
            current_phase=current_phase,
            potential_issues=validation_issues,
            iteration=new_state.get("iteration", 0)
        )
        
        # Update state with validation metadata
        new_state.update({
            "validation_result": result["analysis"],
            "validation_metadata": result["metadata"],
            "validation_context": validation_context
        })
        
        # Add compilation performance metrics if available
        if "compilation_result" in state and state["compilation_result"].get("success", False):
            perf_metrics = {
                "execution_time": state["compilation_result"].get("execution_time"),
                "performance_metrics": state["compilation_result"].get("performance_metrics", {})
            }
            new_state["validation_metadata"]["performance_metrics"] = perf_metrics
        
        self._log_step("validate_code", state, {
            "validation_phase": current_phase,
            "validation_context": validation_context,
            "metadata": new_state["validation_metadata"],
            "includes_compiler_feedback": "compilation_result" in state
        })
        
        print("Validation Result:", new_state["validation_result"])
        print("===========================================")
        return new_state
    
    def _improve_code(self, state: Dict) -> Dict:
        """Node to improve code based on validation results"""
        print("===========================================")
        print("Start Code Improvement")
        
        # Extract required fields
        validation_result = state.get("validation_result", "")
        translated_code = state.get("translated_code", "")
        target_language = state.get("target_language", "")
        
        # Validate inputs
        if not validation_result or not translated_code:
            error_msg = "Missing validation result or code for improvement"
            self._log_error(error_msg, state)
            state["error"] = error_msg
            return state
        
        # Extract metadata from validation
        validation_metadata = state.get("validation_metadata", {})
        classification = validation_metadata.get("classification", "unknown")
        severity = validation_metadata.get("severity", "medium")
        priority = validation_metadata.get("priority", "deferred")
        violated_rules = validation_metadata.get("violated_rules", [])
        
        # Get current phase
        current_phase = self._get_current_phase(state)
        
        # Get relevant rules for violated rules
        relevant_rules = self._retrieve_relevant_rules(violated_rules, target_language)
        
        # Generate code diff if we have previous versions
        code_diff = ""
        if state.get("previous_versions"):
            previous_code = state["previous_versions"][-1]
            code_diff = self._generate_code_diff(previous_code, translated_code)
        
        # Get compiler feedback if available
        compiler_feedback = ""
        if not state.get("compilation_success", True) and state.get("compilation_output"):
            compiler_feedback = state["compilation_output"]
            if state.get("compilation_error_analysis"):
                compiler_feedback += "\n\nError Analysis:\n"
                for issue in state.get("compilation_error_analysis", {}).get("common_issues", []):
                    compiler_feedback += f"- {issue}\n"
                for fix in state.get("compilation_error_analysis", {}).get("suggested_fixes", []):
                    compiler_feedback += f"- Suggestion: {fix}\n"
        
        # Track previous version
        if "previous_versions" not in state:
            state["previous_versions"] = []
        state["previous_versions"].append(translated_code)
        
        # Increment iteration counter
        state["iteration"] = state.get("iteration", 0) + 1
        
        # Check if we've reached max iterations
        if state["iteration"] >= self.max_iterations:
            state["max_iterations_reached"] = True
            self._log_step("max_iterations_reached", state, {
                "iterations": state["iteration"],
                "max_allowed": self.max_iterations
            })
            return state
        
        # Improve the code
        try:
            improved_code = self.translation_agent.improve_code(
                translated_code,
                validation_result,
                current_phase,
                target_language,
                priority,
                severity,
                relevant_rules,
                code_diff,
                compiler_feedback
            )
            
            # Clean up the improved code for compilation
            improved_code = self._clean_code_for_compilation(improved_code)
            
            # Update state
            state["translated_code"] = improved_code
            
            # Log improvement
            self._log_step("code_improved", state, {
                "iteration": state["iteration"],
                "char_count": len(improved_code),
                "line_count": improved_code.count('\n') + 1,
                "diff_size": len(code_diff)
            })
            
        except Exception as e:
            error_msg = f"Code improvement failed: {str(e)}"
            self._log_error(error_msg, state, e)
            state["error"] = error_msg
        
        print("===========================================")
        return state
    
    def _finalize_output(self, state: Dict) -> Dict:
        """Finalize output processing and organize results"""
        print("===========================================")
        print("Start Finalizing Output")
        
        final_state = state.copy()
        
        # Ensure necessary output fields exist
        final_state.update({
            "source_language": state.get("source_language", "Unknown"),
            "target_language": state.get("target_language", "Unknown"),
            "translated_code": state.get("translated_code", ""),
            "error_log": state.get("error", "No errors")
        })
        
        # If compilation results exist, add to final output
        if "compilation_result" in state:
            comp_result = state["compilation_result"]
            final_state.update({
                "compilation_success": comp_result.get("success", False),
                "compilation_errors": comp_result.get("errors", []),
                "execution_output": comp_result.get("execution_output", ""),
                "execution_time_seconds": comp_result.get("execution_time", 0)
            })
        
        # If code feature analysis exists, add HPC-related information
        if "code_features" in state:
            features = state["code_features"]
            hpc_analysis = []
            if "PARALLEL_CONSTRUCTS" in features:
                hpc_analysis.append("Parallel structures detected")
            if "MEMORY_OPERATIONS" in features:
                hpc_analysis.append("Memory operation optimizations included")
            if "NUMERICAL_CALCULATIONS" in features:
                hpc_analysis.append("Numerical calculation optimizations included")
            if hpc_analysis:
                final_state["hpc_analysis"] = "\n".join(hpc_analysis)
        
        # Log final step
        self._log_step("finalize_output", state, {
            "output_fields": list(final_state.keys()),
            "has_compilation_info": "compilation_result" in state,
            "has_hpc_analysis": "hpc_analysis" in final_state
        })
        
        print("Output finalized")
        print("===========================================")
        return final_state

    def _handle_error(self, state: Dict) -> Dict:
        """Handle errors in the workflow"""
        print("===========================================")
        print("Start Error Handling")
        
        error_state = state.copy()
        
        # Organize error information
        error_info = {
            "error_message": state.get("error", "Unknown error"),
            "error_type": state.get("error_type", "general_error"),
            "error_details": state.get("error_details", {}),
            "failed_step": state.get("failed_step", "unknown"),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # If compilation errors exist, add related information
        if "compilation_result" in state and not state["compilation_result"].get("success", False):
            error_info.update({
                "compilation_errors": state["compilation_result"].get("errors", []),
                "compiler_output": state["compilation_result"].get("compiler_output", "")
            })
        
        # If validation errors exist, add related information
        if "validation_metadata" in state:
            error_info.update({
                "validation_issues": state["validation_metadata"].get("violated_rules", []),
                "validation_severity": state["validation_metadata"].get("severity", "unknown")
            })
        
        # Log error
        self._log_step("error_handling", state, error_info)
        
        # Update state
        error_state.update({
            "error_log": error_info,
            "status": "failed",
            "translated_code": state.get("translated_code", ""),  # Keep last translation result (if any)
            "source_language": state.get("source_language", "Unknown"),
            "target_language": state.get("target_language", "Unknown")
        })
        
        print(f"Error handling complete: {error_info['error_message']}")
        print("===========================================")
        return error_state
    
    def _should_improve(self, state: Dict) -> str:
        """Determine if code needs further improvement"""
        print("===========================================")
        print("Evaluating need for further improvement")
        
        # Check if maximum iterations reached
        current_iteration = state.get("iteration", 0)
        if current_iteration >= self.max_iterations:
            print(f"Maximum iterations reached ({self.max_iterations})")
            return "final"
            
        # Check for serious errors
        if state.get("error"):
            print(f"Error detected: {state['error']}")
            return "error"
        
        # Check compilation results (if any)
        if "compilation_result" in state:
            if not state["compilation_result"].get("success", False):
                print("Compilation failed, improvement needed")
                return "improve"
        
        # Check validation results
        if "validation_metadata" in state:
            validation = state["validation_metadata"]
            
            # Check severity
            severity = validation.get("severity", "low")
            if severity in ["high", "critical"]:
                print(f"Detected {severity} level issues, improvement needed")
                return "improve"
            
            # Check violated rules
            violated_rules = validation.get("violated_rules", [])
            if violated_rules:
                print(f"Detected {len(violated_rules)} rules requiring improvement")
                return "improve"
        
        # Check performance metrics (if any)
        if "performance_metrics" in state:
            metrics = state["performance_metrics"]
            if metrics.get("needs_improvement", False):
                print("Performance metrics indicate improvement needed")
            return "improve"
            
        print("Code quality meets requirements, no further improvement needed")
        return "final"
    
    def _get_current_phase(self, state: Dict) -> str:
        """Extract current phase from conversion plan"""
        if "conversion_plan" not in state:
            return "Phase 1: Foundation"  # Default phase
        
        plan = state["conversion_plan"]
        
        # Try to extract current phase from plan
        current_phase_match = re.search(r"Current Phase:\s*\[(Phase \d+[^]]*)\]", plan)
        if current_phase_match:
            return current_phase_match.group(1)
        
        # If no explicit current phase, infer from iteration count
        iteration = state.get("iteration", 0)
        if iteration == 0:
            return "Phase 1: Foundation"
        elif iteration == 1:
            return "Phase 2: Parallelism"
        elif iteration == 2:
            return "Phase 3: Memory Optimization"
        else:
            return "Phase 4: Performance Tuning"
    
    def _retrieve_relevant_rules(self, violated_rules: List[str], target_lang: str) -> str:
        """Retrieve relevant rules based on violated rules"""
        if not violated_rules or not self.knowledge_base:
            return ""
        
        # Get all rules for target language
        lang_rules = self.knowledge_base.get(target_lang, {})
        
        # Initialize results
        relevant_rules = []
        
        # Iterate through all rule categories
        for category, rules_list in lang_rules.items():
            if category == "analysis_rules":
                continue  # Skip analysis rules
            
            # Iterate through all rules in this category
            for rule in rules_list:
                # Check if rule ID is in violated list
                rule_id_match = re.search(r"([A-Z]{2}-\d{3})", rule)
                if rule_id_match:
                    rule_id = rule_id_match.group(1)
                    if any(rule_id in violated for violated in violated_rules):
                        relevant_rules.append(f"[{category.upper()}] {rule}")
            
            # Also check keyword matching
            for violated in violated_rules:
                if not re.match(r"[A-Z]{2}-\d{3}", violated) and violated.lower() in rule.lower():
                    relevant_rules.append(f"[{category.upper()}] {rule}")
        
        # If no relevant rules found, return some general rules
        if not relevant_rules:
            general_rules = []
            for category, rules_list in lang_rules.items():
                if category != "analysis_rules":
                    # Add first two rules from each category as general rules
                    general_rules.extend([f"[{category.upper()}] {rule}" for rule in rules_list[:2]])
            
            if general_rules:
                return "General Rules:\n" + "\n".join(general_rules)
            return ""
        
        # Update rule cache
        with self.cache_lock:
            for rule_id in violated_rules:
                if re.match(r"[A-Z]{2}-\d{3}", rule_id):
                    self.rule_cache[rule_id] = datetime.now(timezone.utc).isoformat()
        
        return "Relevant Rules:\n" + "\n".join(relevant_rules)

    def _generate_code_diff(self, old_code: str, new_code: str) -> str:
        """Generate code difference report"""
        if not old_code or not new_code:
            return "Cannot generate diff report: code is empty"
        
        # Ensure inputs are string type
        if not isinstance(old_code, str) or not isinstance(new_code, str):
            old_code = str(old_code) if old_code is not None else ""
            new_code = str(new_code) if new_code is not None else ""
        
        # Normalize line endings
        old_code = old_code.replace('\r\n', '\n')
        new_code = new_code.replace('\r\n', '\n')
        
        # Generate diff
        old_lines = old_code.split('\n')
        new_lines = new_code.split('\n')
        
        # Use difflib to generate diff
        diff = difflib.unified_diff(
            old_lines, 
            new_lines,
            fromfile='previous_version',
            tofile='current_version',
            lineterm=''
        )
        
        # Convert diff to string
        diff_text = '\n'.join(list(diff))
        
        # If diff is too long, only return summary
        if len(diff_text) > 2000:
            # Extract first 10 lines and last 10 lines
            diff_lines = diff_text.split('\n')
            if len(diff_lines) > 20:
                diff_text = '\n'.join(diff_lines[:10] + ['...'] + diff_lines[-10:])
        
        return diff_text if diff_text else "No changes in code"
    
    def process_request(self, user_input: str) -> Dict:
        """Process user translation request with enhanced logging"""
        print("=" * 50)
        print(" STARTING TRANSLATION REQUEST ".center(50, "="))
        print("=" * 50)
        
        start_time = datetime.now(timezone.utc)
        
        try:
            # Initialize state
            initial_state = {
                "user_input": user_input,
                "iteration": 0,
                "start_time": start_time.isoformat()
            }
            
            # Create logs directory if it doesn't exist
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            
            # Run workflow using invoke method
            final_state = self.workflow.invoke(initial_state)
            
            # Calculate processing time
            end_time = datetime.now(timezone.utc)
            processing_time = (end_time - start_time).total_seconds()
            
            # Add processing time to state
            final_state["processing_time"] = processing_time
            
            # Print summary statistics
            print("\n" + "=" * 50)
            print(" TRANSLATION REQUEST COMPLETE ".center(50, "="))
            print("=" * 50)
            print(f"Processing time: {processing_time:.2f} seconds")
            print(f"Iterations: {final_state.get('iteration', 0)}")
            print(f"Status: {final_state.get('status', 'unknown')}")
            
            # Save final execution log
            self._save_execution_log()
            
            return final_state
            
        except Exception as e:
            # Calculate processing time even for errors
            end_time = datetime.now(timezone.utc)
            processing_time = (end_time - start_time).total_seconds()
            
            error_state = {
                "error": str(e),
                "error_type": type(e).__name__,
                "error_details": {
                    "traceback": str(e.__traceback__),
                    "step": "process_request"
                },
                "status": "failed",
                "processing_time": processing_time
            }
            
            self._log_error("Error processing request", error_state, e)
            
            print("\n" + "=" * 50)
            print(" ERROR PROCESSING REQUEST ".center(50, "="))
            print("=" * 50)
            print(f"Error: {str(e)}")
            print(f"Processing time: {processing_time:.2f} seconds")
            
            # Save final error log
            self._save_error_log()
            
            return error_state

    def _save_execution_log(self):
        """Save execution log to file for analysis"""
        try:
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"execution_log_{timestamp}.json"
            
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(self.execution_log, f, indent=2, default=str)
            
            print(f"\nExecution log saved to: {log_file}")
        except Exception as e:
            print(f"Error saving execution log: {e}")

    def _save_error_log(self):
        """Save error log to file for analysis"""
        try:
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"error_log_{timestamp}.json"
            
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(self.error_log, f, indent=2, default=str)
            
            print(f"\nError log saved to: {log_file}")
        except Exception as e:
            print(f"Error saving error log: {e}")