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
from langchain_groq import ChatGroq
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
        # self.llm = ChatOpenAI(model="o1-mini")

        # if not os.getenv("GROQ_API_KEY"):
        #     print("Warning: GROQ_API_KEY is not set. Using default model.")
        
        # self.llm = ChatGroq(
        #     model="deepseek-r1-distill-llama-70b",
        #     temperature=0.9,
        #     api_key=os.getenv("GROQ_API_KEY")
        # )
        self.analysis_agent = AnalysisAgent(self.llm, self.knowledge_base)
        self.translation_agent = TranslationAgent(self.llm, self.knowledge_base)
        self.verification_agent = VerificationAgent(self.llm, self.knowledge_base)
        self.compiler_agent = CompilerAgent(self.llm, working_dir=self.working_dir)
        
        # Set up caching
        self.translation_cache = {}
        self.rule_cache = {}
        self.cache_lock = Lock()
        
        # Set up logging
        self.execution_log = []
        self.error_log = []
        
        # 跟踪编译成功的翻译版本
        self.successful_translations = []
        
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
        workflow.add_node("compile_code", self._compile_code)
        workflow.add_node("validate_code", self._validate_code)
        workflow.add_node("improve_code", self._improve_code)
        workflow.add_node("finalize_output", self._finalize_output)
        workflow.add_node("error_handling", self._handle_error)
        
        # Enhanced workflow edges
        workflow.set_entry_point("analyze_user_input")
        workflow.add_edge("analyze_user_input", "analyze_requirements")
        workflow.add_edge("analyze_requirements", "generate_plan")
        workflow.add_edge("generate_plan", "initial_translation")
        workflow.add_edge("initial_translation", "compile_code")
        workflow.add_edge("compile_code", "validate_code")
        
        # Set up the validation loop
        workflow.add_conditional_edges(
            "validate_code",
            self._should_improve,
            {
                "improve": "improve_code",
                "final": "finalize_output",
                "complete": "finalize_output",
                "max_iterations_reached": "finalize_output",  
                "tasks_remain": "improve_code",
                "critical_issues": "improve_code",
                "performance_issues": "improve_code",
                "phase_incomplete": "improve_code",
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
        """Node to analyze user input and extract source language, target language and code"""
        print("===========================================")
        print("Start Analyze User Input")
        
        try:
            # Extract user input from state
            user_input = state.get("user_input", "")
            if not user_input:
                self._log_error("Empty user input", state)
                return self._handle_error(state)
                
            # Use analysis agent to analyze user input
            result = self.analysis_agent.analyze_code(user_input)
            
            # Check if analysis is validated
            is_validated = result.get("is_validated", False)
            
            if is_validated:
                # Extract parsed data
                parsed_data = result.get("parsed_data", {})
                
                # remove <think>...</think>
                for key, value in parsed_data.items():
                    if isinstance(value, str):
                        parsed_data[key] = self._clean_thinking_process(value)
                
                # Update state with parsed data
                state.update(parsed_data)
                
                # Add potential issues to state
                if "potential_issues" in parsed_data:
                    state["potential_issues"] = parsed_data["potential_issues"]
                
                # Ensure target language is correct
                if state.get("target_language") == "Unknown" and "jax" in user_input.lower():
                    print("Detected JAX keyword, correcting target language")
                    state["target_language"] = "JAX"
                
                # Log step
                self._log_step("analyze_user_input", 
                              {"input_length": len(user_input)},
                              {"result": "Analysis successful", "parsed_data": parsed_data})
                              
                print(f"Analysis successful:")
                print(f"  Source language: {state.get('source_language', 'Unknown')}")
                print(f"  Target language: {state.get('target_language', 'Unknown')}")
                print(f"  Code length: {len(state.get('code_content', ''))}")
                
                return state
            else:
                # If validation fails, try regex extraction
                print("Analysis validation failed, trying regex extraction")
                
                # Try regex extraction
                self._fallback_regex_extraction(state, user_input)
                
                # Ensure target language is correct
                if (state.get("target_language") == "Unknown" or not state.get("target_language")) and "jax" in user_input.lower():
                    print("After regex extraction, detected JAX keyword, correcting target language")
                    state["target_language"] = "JAX"
                    if "pytorch" in user_input.lower() or "torch" in user_input.lower():
                        state["source_language"] = "PYTORCH"
                
                if state.get("source_language") and state.get("target_language") and state.get("code_content"):
                    # Regex extraction successful
                    print("Regex extraction successful")
                    self._log_step("analyze_user_input", 
                                  {"input_length": len(user_input), "method": "regex"},
                                  {"result": "Regex extraction successful"})
                    return state
                else:
                    # Both analysis and regex extraction failed
                    self._log_error("Failed to extract required information from user input", state)
                    return self._handle_error(state)
                    
        except Exception as e:
            self._log_error(f"Error in user input analysis: {str(e)}", state)
            return self._handle_error(state)

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
        
        # ensure all parameters are string types
        source_language = str(state.get('source_language', '')) if state.get('source_language') is not None else ''
        target_language = str(state.get('target_language', '')) if state.get('target_language') is not None else ''
        code_content = str(state.get('code_content', '')) if state.get('code_content') is not None else ''
        
        # ensure potential_issues is a list of strings
        potential_issues = []
        if 'potential_issues' in state and state['potential_issues'] is not None:
            for issue in state['potential_issues']:
                if issue is not None:
                    potential_issues.append(str(issue))
        
        # check if code_features exists, if not, try to extract
        if 'code_features' not in state and code_content:
            try:
                code_features = self.analysis_agent.extract_code_features(code_content)
                state['code_features'] = code_features
                print("Extracted code features for planning:")
                print(code_features)
            except Exception as e:
                print(f"Warning: Failed to extract code features: {e}")
        
        # Delegate to analysis agent for planning
        plan = self.analysis_agent.generate_plan(
            source_language=source_language,
            target_language=target_language,
            code_content=code_content,
            potential_issues=potential_issues
        )
        
        # store full plan
        state["conversion_plan"] = plan
        
        # parse plan for challenges and tasks
        try:
            # extract tasks for current phase
            current_phase_num = state.get("iteration", 0) + 1  # default start from phase 1
            phase_data = self.analysis_agent.extract_plan_for_phase(plan, current_phase_num)
            
            if phase_data["found"]:
                state["current_phase_name"] = phase_data["phase_name"]
                state["current_phase_tasks"] = phase_data["tasks"]
                print(f"Current phase: {phase_data['phase_name']}")
                print(f"Tasks for this phase: {len(phase_data['tasks'])}")
                for i, task in enumerate(phase_data["tasks"]):
                    print(f"  {i+1}. {task}")
            
            # extract potential challenges
            challenges_pattern = r"POTENTIAL CHALLENGES:(.*?)(?=CURRENT PHASE:|$)"
            challenges_match = re.search(challenges_pattern, plan, re.DOTALL)
            if challenges_match:
                challenges_text = challenges_match.group(1).strip()
                challenge_items = re.findall(r"- \[(.*?)\]:(.*?)(?=\n- \[|$)", challenges_text, re.DOTALL)
                if challenge_items:
                    challenges = {}
                    for challenge, mitigation in challenge_items:
                        challenges[challenge.strip()] = mitigation.strip()
                    state["potential_challenges"] = challenges
                    print(f"Extracted {len(challenges)} potential challenges from plan")
        except Exception as e:
            print(f"Warning: Error parsing plan details: {e}")
        
        # record time of plan generation
        state["plan_generated_time"] = datetime.now(timezone.utc).isoformat()
        
        # record plan generation step
        self._log_step("plan_generated", state, {
            "plan_length": len(plan),
            "source_language": source_language,
            "target_language": target_language
        })
        
        print("===========================================")
        return state
    
    def _initial_translation(self, state: Dict) -> Dict:
        """Perform initial code translation"""
        print("===========================================")
        print("Start Initial Translation")
        
        try:
            # Extract required fields
            source_language = state.get("source_language", "")
            target_language = state.get("target_language", "")
            code_content = state.get("code_content", "")
            
            # Ensure we have required fields
            if not source_language or not target_language or not code_content:
                self._log_error("Missing required translation fields", state)
                return self._handle_error(state)
            
            # Generate a cache key
            cache_key = self._generate_cache_key(code_content, target_language)
            
            # Check if translation exists in cache
            with self.cache_lock:
                if cache_key in self.translation_cache:
                    print("Using cached translation result")
                    translation_result = self.translation_cache[cache_key]
                    
                    # Update state
                    state["translated_code"] = translation_result
                    state["iteration"] = 1  # First iteration from cache
                    state["previous_versions"] = [{"code": translation_result, "iteration": 1}]
                    
                    # Log step
                    self._log_step("initial_translation", 
                                {"source": source_language, "target": target_language, "cache_hit": True},
                                {"result": "Used cached translation"})
                    
                    return state
            
            # Perform translation
            print(f"Translating from {source_language} to {target_language}")
            try:
                translated_code = self.translation_agent.translate_code(
                    source_language=source_language,
                    target_language=target_language,
                    code_content=code_content
                )
                
                # call _extract_code_from_result again to ensure clean code
                if hasattr(self.translation_agent, '_extract_code_from_result'):
                    translated_code = self.translation_agent._extract_code_from_result(translated_code)
                else:
                    # clean thinking process directly here
                    translated_code = self._clean_thinking_process(translated_code)
                
                # ensure translation result is not empty
                if not translated_code or len(translated_code.strip()) < 10:
                    self._log_error("Translation returned empty or too short result", state)
                    translated_code = f"// Failed to translate code\n// Source language: {source_language}\n// Target language: {target_language}\n\n{code_content}"
                
                # save cleaned code to cache
                with self.cache_lock:
                    self.translation_cache[cache_key] = translated_code
                
                # update state
                state["translated_code"] = translated_code
                state["iteration"] = 1
                state["previous_versions"] = [{"code": translated_code, "iteration": 1}]
                
                # ensure conversion_plan field exists
                if "conversion_plan" not in state or not state["conversion_plan"]:
                    state["conversion_plan"] = f"Convert {source_language} code to {target_language}"
                
                # save translation version
                self._save_translation_version(translated_code, state)
                
                # record step
                self._log_step("initial_translation", 
                            {"source": source_language, "target": target_language},
                            {"result": "Translation successful"})
                
                print("Initial translation completed successfully")
                
            except Exception as e:
                self._log_error(f"Error during translation: {str(e)}", state)
                # provide a default translation result, so the workflow can continue
                state["translated_code"] = f"// Error during translation: {str(e)}\n// Source language: {source_language}\n// Target language: {target_language}\n\n{code_content}"
                state["iteration"] = 1
                state["previous_versions"] = [{"code": state["translated_code"], "iteration": 1}]
                
                # ensure conversion_plan field exists
                if "conversion_plan" not in state or not state["conversion_plan"]:
                    state["conversion_plan"] = f"Convert {source_language} code to {target_language}"
            
            return state
            
        except Exception as e:
            self._log_error(f"Unexpected error in initial translation: {str(e)}", state)
            return self._handle_error(state)

    def _clean_code_for_compilation(self, code: str) -> str:
        """Clean code by removing markdown formatting and fixing compilation issues"""
        # Remove markdown code block markers at the beginning
        code = re.sub(r'^```\w*\s*\n', '', code, flags=re.MULTILINE)
        
        # Remove markdown code block markers at the end
        code = re.sub(r'\n```\s*$', '', code, flags=re.MULTILINE)
        
        # Remove any remaining markdown code block markers
        code = re.sub(r'```\w*', '', code)
        
        # Remove any HTML-like comments that might have been added by the LLM
        code = re.sub(r'<!--.*?-->', '', code, flags=re.DOTALL)
        
        # fix special printf statement cross-line problem
        # 1. find lines that start with printf, containing quotes but no closing quotes
        lines = code.split('\n')
        fixed_lines = []
        i = 0
        
        print("\n=== FIXING PRINTF STATEMENTS ===")
        while i < len(lines):
            line = lines[i]
            
            # check if it is a printf statement, and contains odd number of quotes
            if ("printf" in line or "cout" in line) and line.count('"') % 2 == 1:
                print(f"Found potentially broken printf at line {i+1}: {line}")
                
                # merge next line
                if i + 1 < len(lines):
                    next_line = lines[i+1]
                    # if the next line contains quotes
                    if '"' in next_line:
                        # merge two lines, using space to connect
                        combined = line.rstrip() + " " + next_line.lstrip()
                        fixed_lines.append(combined)
                        print(f"  Fixed by combining with next line: {combined}")
                        i += 2  # skip next line, because it has been merged
                        continue
            
            fixed_lines.append(line)
            i += 1
        
        code = '\n'.join(fixed_lines)
        
        # use stronger regex pattern to fix string literals cross-line problem
        print("\n=== FIXING STRING LITERALS ===")

        # fix printf statements with parameters (e.g. "format", var)
        printf_pattern = r'(printf\s*\(\s*"[^"]*?)(\n)([^"]*"\s*,)'
        code = re.sub(printf_pattern, r'\1 \3', code)
        
        # fix general printf statements (e.g. "message")
        printf_pattern2 = r'(printf\s*\(\s*"[^"]*?)(\n)([^"]*"\s*\))'
        code = re.sub(printf_pattern2, r'\1 \3', code)
        
        # fix string multi-line problem
        string_pattern = r'("[^"]*?)(\n)([^"]*")'
        code = re.sub(string_pattern, r'\1 \3', code)
        
        # ensure file ends with a newline
        if not code.endswith('\n'):
            code += '\n'
        
        # print cleaned code for debugging
        print("\n=== CLEANED CODE ===")
        for i, line in enumerate(code.split('\n')):
            print(f"{i+1:4d}: {line}")
        print("=== END OF CLEANED CODE ===\n")
        
        return code

    def _compile_code(self, state: Dict) -> Dict:
        """Node to compile translated code and measure performance"""
        print("===========================================")
        print("Start Compilation")
        
        result_state = state.copy()
        
        # Extract code and target language
        translated_code = str(state.get("translated_code", "")) if state.get("translated_code") is not None else ""
        target_language = str(state.get("target_language", "")) if state.get("target_language") is not None else ""
        
        print(f"Target language: {target_language}")
        print(f"Translated code: {translated_code}")

        if not translated_code or not target_language:
            result_state.update({
                "compilation_success": False,
                "compilation_errors": ["Missing code or target language"]
            })
            print("Warning: missing code or target language, continue validation process")
            return result_state
        
        # Clean code for compilation
        cleaned_code = self._clean_code_for_compilation(translated_code)
        
        # Print code to be compiled (with line numbers) for debugging
        print("\n=== CODE TO BE SENT TO COMPILER ===")
        for i, line in enumerate(cleaned_code.split('\n')):
            print(f"{i+1:4d}: {line}")
        print("=== END OF CODE TO COMPILER ===\n")
        
        # Skip compilation for languages that don't need it
        if not self._is_language_compilable(target_language):
            result_state.update({
                "compilation_success": True,
                "compilation_message": f"{target_language} does not require compilation",
                "execution_output": "Execution not supported for this language"
            })
            return result_state
            
        # Compile and run
        # directly use the cleaned code, prevent compiler_agent from processing again
        try:
            compilation_result = self.compiler_agent.compile_and_run(
                code=cleaned_code,
                language=target_language
            )
        except Exception as e:
            print(f"Compilation error: {str(e)}")
            return {
                "compilation_success": False,
                "compilation_errors": [f"Exception during compilation: {str(e)}"],
                "compilation_output": str(e)
            }
        
        # Extract relevant fields
        success = compilation_result.get("success", False)
        errors = compilation_result.get("errors", [])
        compiler_output = compilation_result.get("compiler_output", "")
        execution_output = compilation_result.get("execution_output", "")
        execution_time = compilation_result.get("execution_time")
        
        # if compilation succeeds, save current translation version
        if success:
            # create an entry containing code and related metadata
            successful_entry = {
                "code": cleaned_code,
                "iteration": state.get("iteration", 0),
                "source_language": state.get("source_language", "Unknown"),
                "target_language": target_language,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "execution_output": execution_output,
                "execution_time": execution_time
            }
            self.successful_translations.append(successful_entry)
            print(f"Compilation successful! Saved as successful translation #{len(self.successful_translations)}")
        
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
        result_state["compilation_success"] = success
        result_state["compilation_errors"] = errors
        result_state["compilation_output"] = compiler_output
        result_state["execution_output"] = execution_output if success else ""
        result_state["execution_time_seconds"] = execution_time
        result_state["compilation_error_analysis"] = error_analysis
            
        return result_state
    
    def _validate_code(self, state: Dict) -> Dict:
        """Validate translated code against quality standards"""
        print("===========================================")
        print("Start Validation Code")
        
        try:
            result_state = state.copy()
            # print(f"Result state: {result_state}")
            
            target_language = str(result_state.get("target_language", "Unknown")) if result_state.get("target_language") is not None else "Unknown"
            translated_code = str(result_state.get("translated_code", "")) if result_state.get("translated_code") is not None else ""
            current_iteration = int(result_state.get("iteration", 1)) if result_state.get("iteration") is not None else 1
            
            print(f"Current iteration: {current_iteration}")
            print(f"Target language: {target_language}")
            print(f"Translated code: {translated_code}")
            
            # clean translated code, ensure no thinking process
            if translated_code:
                cleaned_code = self._clean_thinking_process(translated_code)
                if cleaned_code != translated_code:
                    print("Detected thinking process and cleaned it")
                    result_state["translated_code"] = cleaned_code
                    translated_code = cleaned_code
            
            # ensure conversion_plan field exists
            if "conversion_plan" not in state or not state["conversion_plan"]:
                source_language = state.get("source_language", "Unknown")
                result_state["conversion_plan"] = f"Convert {source_language} code to {target_language}"
            
            # check missing parameters and emit warnings, but allow workflow to continue
            missing_params = []
            if not target_language or target_language == "Unknown":
                missing_params.append("target_language")
            if not translated_code:
                missing_params.append("translated_code")
            if "conversion_plan" not in state or not state["conversion_plan"]:
                missing_params.append("conversion_plan")
                
            if missing_params:
                error_msg = f"VALIDATION_PARAM_MISSING: {', '.join(missing_params)}"
                self._log_error(error_msg, state)
                print(f"Warning: validation missing necessary parameters, but workflow will continue: {missing_params}")
            
            # get current phase
            current_phase = self._get_current_phase(state)
            
            # safely get validation result
            try:
                validation_result = self.verification_agent.validate_code(
                    code=translated_code,
                    target_language=target_language,
                    current_phase=current_phase,
                    potential_issues=state.get("potential_issues", []),
                    iteration=current_iteration
                )
                
                # clean validation result, if it contains thinking process
                if isinstance(validation_result, dict) and "analysis" in validation_result:
                    validation_result["analysis"] = self._clean_thinking_process(validation_result["analysis"])
                
                # update state
                result_state["validation_result"] = validation_result["analysis"]
                result_state["validation_metadata"] = validation_result["metadata"]
                
                # record step
                self._log_step("validate_code", 
                               {"code_length": len(translated_code), "target_language": target_language},
                               {"result": "Validation completed", "metadata": validation_result["metadata"]})
                
                print("Validation completed successfully")
                
            except Exception as e:
                error_msg = f"Error during code validation: {str(e)}"
                self._log_error(error_msg, state)
                print(f"Warning: {error_msg}")
                
                # set default validation result, so the workflow can continue
                result_state["validation_result"] = "Validation failed due to technical issues"
                result_state["validation_metadata"] = {
                    "classification": "unknown",
                    "severity": "medium",
                    "priority": "deferred",
                    "violated_rules": [],
                    "solution_approach": "Address technical issues and try again"
                }
            
            return result_state
            
        except Exception as e:
            self._log_error(f"Unexpected error in validation: {str(e)}", state)
            return self._handle_error(state)
    
    def _improve_code(self, state: Dict) -> Dict:
        """Improve the translated code based on validation feedback"""
        print("===========================================")
        print("Start Code Improvement")
        
        try:
            # Extract required information
            translated_code = state.get("translated_code", "")
            target_language = state.get("target_language", "")
            validation_result = state.get("validation_result", "")
            validation_metadata = state.get("validation_metadata", {})
            compilation_output = state.get("compilation_output", "")
            compilation_errors = state.get("compilation_errors", [])
            current_iteration = state.get("iteration", 1)
            
            # Get priority and severity
            priority = validation_metadata.get("priority", "deferred")
            severity = validation_metadata.get("severity", "medium")
            
            # Get relevant rules
            violated_rules = validation_metadata.get("violated_rules", [])
            relevant_rules = self._retrieve_relevant_rules(violated_rules, target_language)
            
            # Generate code diff if available
            previous_versions = state.get("previous_versions", [])
            code_diff = ""
            if len(previous_versions) >= 2:
                previous_code = previous_versions[-2]["code"]
                code_diff = self._generate_code_diff(previous_code, translated_code)
            
            # Get compiler feedback
            compiler_feedback = ""
            if compilation_errors:
                compiler_feedback = "\n".join(compilation_errors)
            
            # Get current phase
            current_phase = self._get_current_phase(state)
            
            # Improve code
            try:
                improved_code = self.translation_agent.improve_code(
                    code=translated_code,
                    validation_result=validation_result,
                    current_phase=current_phase,
                    target_language=target_language,
                    priority=priority,
                    severity=severity,
                    relevant_rules=relevant_rules,
                    code_diff=code_diff,
                    compiler_feedback=compiler_feedback
                )
                
                # clean possible thinking process
                improved_code = self._clean_thinking_process(improved_code)
                
                # Update state
                state["translated_code"] = improved_code
                state["iteration"] = current_iteration + 1
                
                # Add to previous versions
                previous_versions.append({
                    "code": improved_code,
                    "iteration": current_iteration + 1
                })
                state["previous_versions"] = previous_versions
                
                # Save new version
                self._save_translation_version(improved_code, state)
                
                # Log step
                self._log_step("improve_code", 
                              {"iteration": current_iteration, "code_length": len(translated_code)},
                              {"result": "Improvement successful", "new_length": len(improved_code)})
                
                print(f"Code improvement iteration {current_iteration} completed successfully")
                
            except Exception as e:
                self._log_error(f"Error during code improvement: {str(e)}", state)
                
                # Increment iteration but keep the same code
                state["iteration"] = current_iteration + 1
                
                # Add to previous versions (same code, but marked as new iteration)
                previous_versions.append({
                    "code": translated_code,
                    "iteration": current_iteration + 1
                })
                state["previous_versions"] = previous_versions
                
                print(f"Code improvement failed, continuing with unchanged code")
            
            return state
            
        except Exception as e:
            self._log_error(f"Unexpected error in code improvement: {str(e)}", state)
            return self._handle_error(state)
    
    def _save_translation_version(self, code: str, state: Dict) -> None:
        """Save each version of translated code to a file"""
        if not code:
            return
            
        try:
            # create versions directory
            versions_dir = Path("logs/code_versions")
            versions_dir.mkdir(parents=True, exist_ok=True)
            
            # create unique filename with iteration number and timestamp
            iteration = state.get("iteration", 0)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            target_language = state.get("target_language", "unknown").lower()
            
            # set different file extensions for different target languages
            file_extensions = {
                "c": ".c",
                "c++": ".cpp", 
                "cuda": ".cu",
                "openmp": ".c",
                "fortran": ".f90",
                "python": ".py",
                "jax": ".py",
                "mpi": ".c"
            }
            
            # get appropriate file extension, default is .txt
            file_ext = file_extensions.get(target_language.lower(), ".txt")
            
            # create filename
            filename = f"iteration_{iteration}_{timestamp}{file_ext}"
            file_path = versions_dir / filename
            
            # save code
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(code)
                
            # add metadata file
            metadata_file = versions_dir / f"iteration_{iteration}_{timestamp}_meta.json"
            
            # safely get execution time and compilation status
            exec_time = state.get("execution_time_seconds")
            compilation_success = state.get("compilation_success", False)
            
            metadata = {
                "iteration": iteration,
                "timestamp": timestamp,
                "source_language": state.get("source_language", "Unknown"),
                "target_language": target_language,
                "compilation_success": compilation_success if compilation_success is not None else False,
                "execution_time": exec_time if exec_time is not None else "N/A"
            }
            
            # if there are compilation errors, record them
            if "compilation_errors" in state and state["compilation_errors"]:
                metadata["compilation_errors"] = state["compilation_errors"]
                
            # if there is execution output, record it
            if "execution_output" in state and state["execution_output"]:
                # limit output length, avoid file too large
                output = state["execution_output"]
                if len(output) > 1000:
                    metadata["execution_output"] = output[:1000] + "... [truncated]"
                else:
                    metadata["execution_output"] = output
            
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, default=str)
                
            print(f"Saved translation version {iteration} to: {file_path}")
            
        except Exception as e:
            print(f"Error saving translation version: {e}")

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
        """Enhanced decision logic to determine if code should be improved"""
        # check if forced to continue iteration
        if state.get("force_improvement", False):
            return "continue"
        
        # check if maximum iterations reached
        if state.get("iteration", 0) >= self.max_iterations:
            return "max_iterations_reached"
            
        # check if compilation is successful - if compilation failed, continue improving code instead of terminating workflow
        if not state.get("compilation_success", True):
            return "improve"  # modified: return "improve" instead of "compilation_failed", continue improving code
            
        # get current phase tasks
        current_phase_num = state.get("iteration", 0) + 1
        if "conversion_plan" in state:
            plan = state["conversion_plan"]
            phase_data = self.analysis_agent.extract_plan_for_phase(plan, current_phase_num)
            
            # if there are tasks for next phase, continue iteration
            if phase_data["found"] and phase_data["tasks"]:
                return "tasks_remain"
                
        # based on validation result, decide if continue improving code
        severity = "unknown"
        if "validation_metadata" in state:
            metadata = state["validation_metadata"]
            severity = metadata.get("severity", "unknown").lower()
            classification = metadata.get("classification", "unknown").lower()
            
            # if there are critical or high priority issues, continue improving code
            if severity in ["critical", "high"]:
                return "critical_issues"
                
            # if there are performance issues and we are in performance tuning phase, continue improving code
            if classification == "performance" and current_phase_num >= 3:
                return "performance_issues"
                
        # if there are no critical issues, check if all tasks for current phase are completed
        if state.get("iteration", 0) < current_phase_num:
            return "phase_incomplete"
            
        return "complete"
    
    def _get_current_phase(self, state: Dict) -> str:
        """Get the current phase from the conversion plan based on iteration"""
        # if there is already parsed phase_name, use it directly
        if "current_phase_name" in state:
            return f"Phase {state.get('iteration', 0) + 1}: {state['current_phase_name']}"
            
        # otherwise, decide current phase based on iteration number
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
        """Retrieve relevant rules from knowledge base based on violated rules"""
        if not violated_rules or not target_lang:
            return ""
            
        safe_rules = []
        for rule in violated_rules:
            if rule is not None:
                safe_rules.append(str(rule))
        
        print("===========================================")
        print("Debug Information in _retrieve_relevant_rules:")
        print(f"Violated Rules: {safe_rules}")
        print(f"Target Language: {target_lang}")
        print("===========================================")
        
        # Get rules from knowledge base
        kb_rules = self.knowledge_base.get(target_lang, {})
        
        # Extract relevant rules
        relevant_rules = []
        
        for rule_id in safe_rules:
            # Try to find rule in knowledge base
            for category, rules in kb_rules.items():
                if category == "analysis_rules":
                    continue
                    
                for rule in rules:
                    if rule_id in rule:
                        relevant_rules.append(f"[{rule_id}] {rule}")
                        break
        
        # If no specific rules found, return general rules
        if not relevant_rules and "general_rules" in kb_rules:
            relevant_rules = kb_rules["general_rules"]
            
        return "\n".join(relevant_rules)

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
            
            # reset successful translations list
            self.successful_translations = []
            
            # Run workflow using invoke method
            final_state = self.workflow.invoke(initial_state)
            
            # Calculate processing time
            end_time = datetime.now(timezone.utc)
            processing_time = (end_time - start_time).total_seconds()
            
            # Add processing time to state
            final_state["processing_time"] = processing_time
            
            # ensure even if there are errors, basic fields exist
            if "translated_code" not in final_state or not final_state.get("translated_code"):
                final_state["translated_code"] = ""
                
            if "source_language" not in final_state:
                final_state["source_language"] = "Unknown"
                
            if "target_language" not in final_state:
                final_state["target_language"] = "Unknown"
                
            if "status" not in final_state:
                if "error" in final_state:
                    final_state["status"] = "failed"
                else:
                    final_state["status"] = "unknown"
            
            # select best translation version
            # if there are successful translations, select the latest one
            if self.successful_translations:
                best_translation = self.successful_translations[-1]  # latest successful version
                print(f"\nUsing successful compilation (iteration {best_translation['iteration']}) for final output")
                final_state["translated_code"] = best_translation["code"]
                final_state["compilation_success"] = True
                final_state["execution_output"] = best_translation["execution_output"]
                final_state["execution_time_seconds"] = best_translation["execution_time"]
                final_state["status"] = "success"
                final_state["successful_translations_count"] = len(self.successful_translations)
                final_state["selected_translation_iteration"] = best_translation["iteration"]
            else:
                print("\nNo successful compilations found, using last translation attempt")
                final_state["successful_translations_count"] = 0
            
            # add all translations to final state
            if hasattr(self, 'successful_translations'):
                final_state["all_successful_translations"] = self.successful_translations
            
            # Print summary statistics
            print("\n" + "=" * 50)
            print(" TRANSLATION REQUEST COMPLETE ".center(50, "="))
            print("=" * 50)
            print(f"Processing time: {processing_time:.2f} seconds")
            print(f"Iterations: {final_state.get('iteration', 0)}")
            print(f"Status: {final_state.get('status', 'unknown')}")
            print(f"Successful compilations: {len(self.successful_translations)}")
            
            # if processing failed, try to generate error analysis
            if final_state.get("status") == "failed" or "error" in final_state:
                final_state = self._generate_error_analysis(final_state)
            
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
                "processing_time": processing_time,
                "source_language": "Unknown",
                "target_language": "Unknown",
                "translated_code": "",
                "user_input": user_input
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

    def _generate_error_analysis(self, state: Dict) -> Dict:
        """Generate error analysis to help user understand failure reasons"""
        print("===========================================")
        print("Generating Error Analysis")
        
        error_msg = state.get("error", "Unknown error")
        error_type = state.get("error_type", "unspecified_error")
        
        try:
            # create error analysis template
            error_analysis_template = """Analyze the following error that occurred during code translation:
            Error Type: {{error_type}}
            Error Message: {{error_message}}
            
            Source Language: {{source_language}}
            Target Language: {{target_language}}
            
            {% if code_content %}
            User Code Sample (first 500 chars):
            ```
            {{code_sample}}
            ```
            {% endif %}
            
            Please provide:
            1. A brief explanation of what likely went wrong
            2. Possible causes of this error
            3. Suggestions for how the user could fix this issue
            4. Alternative approaches that might work better
            
            Format your response as:
            
            ERROR ANALYSIS:
            [Brief explanation]
            
            POSSIBLE CAUSES:
            - [Cause 1]
            - [Cause 2]
            
            SUGGESTED FIXES:
            - [Fix 1]
            - [Fix 2]
            
            ALTERNATIVE APPROACHES:
            - [Approach 1]
            - [Approach 2]
            """
            
            # prepare parameters
            code_content = state.get("code_content", "")
            code_sample = code_content[:500] + "..." if len(code_content) > 500 else code_content
            
            # call LLM to generate error analysis
            chain = PromptTemplate.from_template(error_analysis_template, template_format="jinja2") | self.llm
            
            error_analysis = chain.invoke({
                "error_type": error_type,
                "error_message": error_msg,
                "source_language": state.get("source_language", "Unknown"),
                "target_language": state.get("target_language", "Unknown"),
                "code_sample": code_sample,
                "code_content": bool(code_content)
            })
            
            # add analysis result to state
            state["error_analysis"] = error_analysis.content
            print("Error analysis generated")
            
        except Exception as e:
            print(f"Failed to generate error analysis: {e}")
            state["error_analysis"] = f"Error analysis generation failed: {str(e)}"
            
        print("===========================================")
        return state

    def _print_translation_results(self, state: Dict) -> None:
        """Enhanced results display with detailed report generation"""
        print("\n" + "=" * 50)
        print(" TRANSLATION RESULTS ".center(50, "="))
        print("=" * 50)
        
        # basic information
        source_language = state.get("source_language", "Unknown")
        target_language = state.get("target_language", "Unknown")
        processing_time = state.get("processing_time", 0)
        iteration = state.get("iteration", 0)
        status = state.get("status", "unknown")
        
        print("\nTranslation Summary")
        print("-" * 50)
        print(f"Source Language   : {source_language}")
        print(f"Target Language   : {target_language}")
        print(f"Processing Time   : {processing_time:.2f} seconds")
        print(f"Iterations        : {iteration}")
        print(f"Status            : {status}")
        
        # show successful compilation information
        if "successful_translations_count" in state:
            successful_count = state.get("successful_translations_count", 0)
            print(f"Successful Builds  : {successful_count}")
            
            if successful_count > 0:
                selected_iteration = state.get("selected_translation_iteration", 0)
                print(f"Selected Version  : Iteration #{selected_iteration}")
        
        # if there are errors, show error information
        if "error" in state:
            print("\nError Information")
            print("-" * 50)
            print(f"Error Type       : {state.get('error_type', 'Unknown')}")
            print(f"Error Message    : {state.get('error', 'Unknown error')}")
            
            # if there is error analysis, show summary
            if "error_analysis" in state and state["error_analysis"]:
                print("\nError Analysis Summary")
                print("-" * 50)
                error_analysis = state["error_analysis"]
                # show first 300 characters of error analysis
                print(error_analysis[:300] + "..." if len(error_analysis) > 300 else error_analysis)
                print("\nSee full error analysis in the detailed report")
        
        # show code result
        print("\n" + "=" * 50)
        print(" TRANSLATED CODE ".center(50, "="))
        print("=" * 50)
        
        translated_code = state.get("translated_code", "")
        if translated_code:
            # if code is too large, only show summary
            code_lines = translated_code.splitlines()
            if len(code_lines) > 500 or len(translated_code) > 10000:
                print(f"Code output is too large to display ({len(code_lines)} lines, {len(translated_code)} chars)")
                print("First 20 lines:")
                print("\n".join(code_lines[:20]))
                print("...\n[truncated]")
                print("See full code in the detailed report")
            else:
                print(translated_code)
        else:
            print("Error: No translated code")
            
        # generate detailed report
        report_filename = self._generate_detailed_report(state)
        print(f"\nDetailed translation report saved to: {report_filename} ")
        
    def _generate_detailed_report(self, state: Dict) -> str:
        """Generate detailed translation report"""
        # create logs directory
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # create unique report filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"logs/translation_report_{timestamp}.txt"
        
        try:
            with open(filename, "w", encoding="utf-8") as f:
                # title
                f.write("=" * 80 + "\n")
                f.write("HPC CODE TRANSLATION DETAILED REPORT\n")
                f.write("=" * 80 + "\n\n")
                
                # basic information
                f.write("BASIC INFORMATION\n")
                f.write("-" * 80 + "\n")
                f.write(f"Report Generated    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Source Language     : {state.get('source_language', 'Unknown')}\n")
                f.write(f"Target Language     : {state.get('target_language', 'Unknown')}\n")
                f.write(f"Processing Time     : {state.get('processing_time', 0):.2f} seconds\n")
                f.write(f"Iterations          : {state.get('iteration', 0)}\n")
                f.write(f"Status              : {state.get('status', 'unknown')}\n")
                
                # add successful compilation information
                if "successful_translations_count" in state:
                    successful_count = state.get("successful_translations_count", 0)
                    f.write(f"Successful Builds    : {successful_count}\n")
                    
                    if successful_count > 0:
                        selected_iteration = state.get("selected_translation_iteration", 0)
                        f.write(f"Selected Version    : Iteration #{selected_iteration}\n")
                
                f.write("\n")
                
                # input code
                f.write("INPUT CODE\n")
                f.write("-" * 80 + "\n")
                code_content = state.get("code_content", "")
                if code_content:
                    f.write(code_content + "\n\n")
                else:
                    f.write("No input code provided or extraction failed\n\n")
                
                # translated code
                f.write("TRANSLATED CODE\n")
                f.write("-" * 80 + "\n")
                translated_code = state.get("translated_code", "")
                if translated_code:
                    f.write(translated_code + "\n\n")
                else:
                    f.write("No translated code generated\n\n")
                
                # if there are multiple successful compilations, add summary information
                if "all_successful_translations" in state and state["all_successful_translations"]:
                    f.write("SUCCESSFUL TRANSLATIONS SUMMARY\n")
                    f.write("-" * 80 + "\n")
                    for idx, trans in enumerate(state["all_successful_translations"]):
                        f.write(f"Version {idx+1} (Iteration {trans['iteration']}):\n")
                        f.write(f"  Timestamp: {trans['timestamp']}\n")
                        
                        # safely get execution time - avoid None value formatting error
                        exec_time = trans.get('execution_time')
                        if exec_time is not None:
                            f.write(f"  Execution Time: {exec_time:.4f} seconds\n")
                        else:
                            f.write(f"  Execution Time: N/A\n")
                        f.write("\n")
                
                # if there are errors, show error information
                if "error" in state:
                    f.write("ERROR INFORMATION\n")
                    f.write("-" * 80 + "\n")
                    f.write(f"Error Type          : {state.get('error_type', 'Unknown')}\n")
                    f.write(f"Error Message       : {state.get('error', 'Unknown error')}\n")
                    
                    # error details
                    error_details = state.get("error_details", {})
                    if error_details:
                        f.write("\nError Details:\n")
                        for key, value in error_details.items():
                            f.write(f"- {key}: {value}\n")
                    
                    # full error analysis
                    if "error_analysis" in state and state["error_analysis"]:
                        f.write("\nERROR ANALYSIS\n")
                        f.write("-" * 80 + "\n")
                        f.write(state["error_analysis"] + "\n\n")
                
                # compilation results
                if "compilation_success" in state:
                    f.write("COMPILATION RESULTS\n")
                    f.write("-" * 80 + "\n")
                    f.write(f"Compilation Success : {state.get('compilation_success', False)}\n")
                    
                    # if there are compilation errors
                    compilation_errors = state.get("compilation_errors", [])
                    if compilation_errors:
                        f.write("\nCompilation Errors:\n")
                        for i, error in enumerate(compilation_errors):
                            f.write(f"{i+1}. {error}\n")
                    
                    # compiler output
                    compiler_output = state.get("compilation_output", "")
                    if compiler_output:
                        f.write("\nCompiler Output:\n")
                        f.write(compiler_output + "\n")
                    
                    # execution output
                    execution_output = state.get("execution_output", "")
                    if execution_output:
                        f.write("\nExecution Output:\n")
                        f.write(execution_output + "\n")
                        
                    # execution time
                    execution_time = state.get("execution_time_seconds")
                    if execution_time is not None:
                        f.write(f"\nExecution Time: {execution_time:.4f} seconds\n")
                    else:
                        f.write("\nExecution Time: N/A\n")
                
                # conversion plan
                conversion_plan = state.get("conversion_plan", "")
                if conversion_plan:
                    f.write("\nCONVERSION PLAN\n")
                    f.write("-" * 80 + "\n")
                    f.write(conversion_plan + "\n\n")
                
                # performance analysis
                if "performance_metrics" in state:
                    f.write("PERFORMANCE ANALYSIS\n")
                    f.write("-" * 80 + "\n")
                    metrics = state["performance_metrics"]
                    for key, value in metrics.items():
                        f.write(f"{key}: {value}\n")
                
                # add version information
                if "previous_versions" in state and state["previous_versions"]:
                    f.write("\nVERSION HISTORY\n")
                    f.write("-" * 80 + "\n")
                    f.write(f"Total Versions: {len(state['previous_versions'])}\n")
                    f.write("All versions saved in logs/code_versions directory\n")
                
            return filename
        except Exception as e:
            print(f"Error generating detailed report: {e}")
            # try to save basic information
            try:
                with open(filename, "w", encoding="utf-8") as f:
                    f.write("ERROR GENERATING DETAILED REPORT\n")
                    f.write(f"Error: {str(e)}\n\n")
                    f.write("BASIC INFORMATION\n")
                    f.write(f"Source Language: {state.get('source_language', 'Unknown')}\n")
                    f.write(f"Target Language: {state.get('target_language', 'Unknown')}\n\n")
                    f.write("TRANSLATED CODE\n")
                    f.write(state.get("translated_code", "No code available"))
                return filename
            except:
                return "Failed to generate report"

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