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
                "complete": "finalize_output",
                "max_iterations_reached": "finalize_output",  # 最大迭代次数到达时，转到finalize_output而不是error_handling
                "tasks_remain": "improve_code",
                "critical_issues": "improve_code",
                "performance_issues": "improve_code",
                "phase_incomplete": "improve_code",
                "error": "error_handling"  # 只有真正的错误才转到error_handling
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
        
        user_input = state.get("user_input", "")
        if not user_input:
            error_msg = "EMPTY_USER_INPUT: No user input provided"
            self._log_error(error_msg, state)
            state["error"] = error_msg
            state["source_language"] = "Unknown"
            state["target_language"] = "Unknown"
            state["code_content"] = ""
            state["error_type"] = "input_error"
            return state
            
        print(f"User input length: {len(user_input)} characters")
        
        try:
            # Advanced prompt-based analysis using Analysis Agent
            result = self.analysis_agent.analyze_code(user_input)
            
            # 调试输出分析结果
            print("\nAnalysis Result:")
            print(f"Validation Status: {result.get('is_validated', False)}")
            
            if result.get("is_validated", False):
                parsed_data = result.get("parsed_data", {})
                source_language = parsed_data.get("source_language", "Unknown")
                target_language = parsed_data.get("target_language", "Unknown")
                code_content = parsed_data.get("code_content", "")
                potential_issues = parsed_data.get("potential_issues", [])
                
                # 确保语言信息不为空
                if source_language == "Unknown" or target_language == "Unknown":
                    # 尝试使用回退策略
                    print("Warning: Language information is incomplete. Using fallback extraction.")
                    self._fallback_regex_extraction(state, user_input)
                else:
                    # 添加到状态中
                    state["source_language"] = source_language
                    state["target_language"] = target_language
                    state["code_content"] = code_content
                    state["potential_issues"] = potential_issues
                    
                    print(f"Extracted Source Language: {source_language}")
                    print(f"Extracted Target Language: {target_language}")
                    print(f"Extracted Code Length: {len(code_content)} characters")
                    print(f"Identified Potential Issues: {len(potential_issues)}")
                
                # 即使分析成功，仍然执行回退策略以确保不会漏掉重要信息
                if not code_content or not source_language or not target_language:
                    print("Warning: Some critical information is missing. Using fallback extraction.")
                    self._fallback_regex_extraction(state, user_input)
            else:
                # 分析失败，使用正则表达式提取
                print("Warning: Prompt-based analysis failed. Using regex extraction.")
                self._fallback_regex_extraction(state, user_input)
                
                # 记录错误
                if "error" in result:
                    error_msg = f"ANALYSIS_FAILED: {result.get('error', 'Unknown error')}"
                    self._log_error(error_msg, state)
                    state["error_details"] = result.get("error")
                
            # 在这一步确保关键字段已赋予默认值
            if "source_language" not in state or not state["source_language"]:
                state["source_language"] = "Unknown"
            if "target_language" not in state or not state["target_language"]:
                state["target_language"] = "Unknown"
            if "code_content" not in state:
                state["code_content"] = ""
                
            # 记录分析结果
            self._log_step("input_analyzed", state, {
                "source_language": state.get("source_language", "Unknown"),
                "target_language": state.get("target_language", "Unknown"),
                "code_length": len(state.get("code_content", ""))
            })
            
        except Exception as e:
            error_msg = f"INPUT_ANALYSIS_FAILED: {str(e)}"
            self._log_error(error_msg, state, e)
            state["error"] = error_msg
            state["error_type"] = "analysis_error"
            
            # 确保关键字段已赋予默认值
            state["source_language"] = "Unknown"
            state["target_language"] = "Unknown"
            state["code_content"] = ""
        
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
        
        # 确保所有参数都是字符串类型
        source_language = str(state.get('source_language', '')) if state.get('source_language') is not None else ''
        target_language = str(state.get('target_language', '')) if state.get('target_language') is not None else ''
        code_content = str(state.get('code_content', '')) if state.get('code_content') is not None else ''
        
        # 确保potential_issues是字符串列表
        potential_issues = []
        if 'potential_issues' in state and state['potential_issues'] is not None:
            for issue in state['potential_issues']:
                if issue is not None:
                    potential_issues.append(str(issue))
        
        # 检查是否有code_features，如果没有，尝试提取
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
        
        # 存储完整的计划
        state["conversion_plan"] = plan
        
        # 解析计划中的挑战和任务
        try:
            # 提取当前阶段的任务
            current_phase_num = state.get("iteration", 0) + 1  # 默认从第1阶段开始
            phase_data = self.analysis_agent.extract_plan_for_phase(plan, current_phase_num)
            
            if phase_data["found"]:
                state["current_phase_name"] = phase_data["phase_name"]
                state["current_phase_tasks"] = phase_data["tasks"]
                print(f"Current phase: {phase_data['phase_name']}")
                print(f"Tasks for this phase: {len(phase_data['tasks'])}")
                for i, task in enumerate(phase_data["tasks"]):
                    print(f"  {i+1}. {task}")
            
            # 提取潜在挑战
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
        
        # 记录生成计划的时间
        state["plan_generated_time"] = datetime.now(timezone.utc).isoformat()
        
        # 记录规划步骤
        self._log_step("plan_generated", state, {
            "plan_length": len(plan),
            "source_language": source_language,
            "target_language": target_language
        })
        
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
                # 即使是缓存命中的代码，也保存起来以保持完整记录
                self._save_translation_version(cached_result["code"], state)
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
            
            # 保存初始翻译版本
            self._save_translation_version(translated_code, state)
            
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
        """Clean code by removing markdown formatting and fixing compilation issues"""
        # Remove markdown code block markers at the beginning
        code = re.sub(r'^```\w*\s*\n', '', code, flags=re.MULTILINE)
        
        # Remove markdown code block markers at the end
        code = re.sub(r'\n```\s*$', '', code, flags=re.MULTILINE)
        
        # Remove any remaining markdown code block markers
        code = re.sub(r'```\w*', '', code)
        
        # Remove any HTML-like comments that might have been added by the LLM
        code = re.sub(r'<!--.*?-->', '', code, flags=re.DOTALL)
        
        # 修复特殊的printf语句跨行问题
        # 1. 找到以printf开头，包含引号但没有结束引号的行
        lines = code.split('\n')
        fixed_lines = []
        i = 0
        
        print("\n=== FIXING PRINTF STATEMENTS ===")
        while i < len(lines):
            line = lines[i]
            
            # 检查是否是printf语句，并且包含奇数个引号（表示字符串未关闭）
            if ("printf" in line or "cout" in line) and line.count('"') % 2 == 1:
                print(f"Found potentially broken printf at line {i+1}: {line}")
                
                # 合并下一行
                if i + 1 < len(lines):
                    next_line = lines[i+1]
                    # 如果下一行包含引号（可能是字符串的结束）
                    if '"' in next_line:
                        # 合并两行，使用空格连接
                        combined = line.rstrip() + " " + next_line.lstrip()
                        fixed_lines.append(combined)
                        print(f"  Fixed by combining with next line: {combined}")
                        i += 2  # 跳过下一行，因为已经合并了
                        continue
            
            fixed_lines.append(line)
            i += 1
        
        code = '\n'.join(fixed_lines)
        
        # 使用更强大的正则表达式模式来修复换行的字符串
        print("\n=== FIXING STRING LITERALS ===")
        
        # 修复带参数的printf语句（如"format", var）
        printf_pattern = r'(printf\s*\(\s*"[^"]*?)(\n)([^"]*"\s*,)'
        code = re.sub(printf_pattern, r'\1 \3', code)
        
        # 修复一般的printf语句（如"message"）
        printf_pattern2 = r'(printf\s*\(\s*"[^"]*?)(\n)([^"]*"\s*\))'
        code = re.sub(printf_pattern2, r'\1 \3', code)
        
        # 修复字符串多行问题
        string_pattern = r'("[^"]*?)(\n)([^"]*")'
        code = re.sub(string_pattern, r'\1 \3', code)
        
        # 确保文件末尾有换行符
        if not code.endswith('\n'):
            code += '\n'
        
        # 打印清理后的代码以便调试
        print("\n=== CLEANED CODE ===")
        for i, line in enumerate(code.split('\n')):
            print(f"{i+1:4d}: {line}")
        print("=== END OF CLEANED CODE ===\n")
        
        return code

    def _compile_code(self, state: Dict) -> Dict:
        """Node to compile translated code and measure performance"""
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
        
        # Print code to be compiled (with line numbers) for debugging
        print("\n=== CODE TO BE SENT TO COMPILER ===")
        for i, line in enumerate(cleaned_code.split('\n')):
            print(f"{i+1:4d}: {line}")
        print("=== END OF CODE TO COMPILER ===\n")
        
        # Skip compilation for languages that don't need it
        if not self._is_language_compilable(target_language):
            return {
                "compilation_success": True,
                "compilation_message": f"{target_language} does not require compilation",
                "execution_output": "Execution not supported for this language"
            }
            
        # Compile and run
        # 直接使用我们已经清理过的代码，防止compiler_agent再次处理
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
        
        # 如果编译成功，保存当前翻译版本
        if success:
            # 创建一个包含代码和相关元数据的条目
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
        state["compilation_success"] = success
        state["compilation_errors"] = errors
        state["compilation_output"] = compiler_output
        state["execution_output"] = execution_output if success else ""
        state["execution_time_seconds"] = execution_time
        state["compilation_error_analysis"] = error_analysis
            
        return state
    
    def _validate_code(self, state: Dict) -> Dict:
        """Enhanced code validation with HPC-specific checks and compiler feedback"""
        print("===========================================")
        print("Start Validation Code")
        
        new_state = state.copy()
        
        # Pre-validation checks
        required_keys = ["target_language", "translated_code", "conversion_plan"]
        missing_keys = []
        for key in required_keys:
            if key not in new_state:
                missing_keys.append(key)
                
        if missing_keys:
            error_msg = f"VALIDATION_PARAM_MISSING: {', '.join(missing_keys)}"
            self._log_error(error_msg, new_state)
            new_state["error"] = error_msg
            new_state["validation_result"] = f"验证失败: 缺少必要参数 - {', '.join(missing_keys)}"
            new_state["validation_metadata"] = {
                "classification": "error",
                "severity": "high",
                "priority": "immediate",
                "violated_rules": ["MISSING_PARAMS"]
            }
            print(f"警告：验证缺少必要参数，但会继续流程: {missing_keys}")
            return new_state
        
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
        try:
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
        except Exception as e:
            error_msg = f"验证代码失败: {str(e)}"
            self._log_error(error_msg, new_state, e)
            
            # 即使验证失败也提供默认的验证结果和元数据
            new_state.update({
                "validation_result": f"验证时出错: {str(e)}。错误类型: {type(e).__name__}",
                "validation_metadata": {
                    "classification": "error",
                    "severity": "high",
                    "priority": "immediate",
                    "violated_rules": ["VALIDATION_ERROR"]
                },
                "validation_context": validation_context,
                "error": error_msg
            })
            print("验证出错，但会继续流程并尝试改进代码")
        
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
        
        try:
            # Extract and convert required fields
            validation_result = str(state.get("validation_result", "")) if state.get("validation_result") is not None else ""
            translated_code = str(state.get("translated_code", "")) if state.get("translated_code") is not None else ""
            target_language = str(state.get("target_language", "")) if state.get("target_language") is not None else ""
            
            # Debug information
            print("===========================================")
            print("Debug Information:")
            print(f"Validation Result Type: {type(state.get('validation_result'))}")
            print(f"Translated Code Type: {type(state.get('translated_code'))}")
            print(f"Target Language Type: {type(state.get('target_language'))}")
            print(f"Validation Metadata Type: {type(state.get('validation_metadata', {}))}")
            print("===========================================")
            
            # Validate inputs
            if not validation_result or not translated_code:
                error_msg = "Missing validation result or code for improvement"
                self._log_error(error_msg, state)
                state["error"] = error_msg
                return state
            
            # Extract metadata from validation
            validation_metadata = state.get("validation_metadata", {})
            if not isinstance(validation_metadata, dict):
                validation_metadata = {}
                
            classification = str(validation_metadata.get("classification", "unknown"))
            severity = str(validation_metadata.get("severity", "medium"))
            priority = str(validation_metadata.get("priority", "deferred"))
            
            violated_rules = []
            if "violated_rules" in validation_metadata:
                raw_rules = validation_metadata["violated_rules"]
                if isinstance(raw_rules, list):
                    for rule in raw_rules:
                        if rule is not None:
                            violated_rules.append(str(rule))
                elif raw_rules is not None:
                    violated_rules.append(str(raw_rules))
            
            # 打印调试信息
            print("Validation Metadata Details:")
            print(f"  Classification: {classification}")
            print(f"  Severity: {severity}")
            print(f"  Priority: {priority}")
            print(f"  Violated Rules: {violated_rules}")
            
            # Get current phase
            current_phase = self._get_current_phase(state)
            
            # Get relevant rules for violated rules
            relevant_rules = self._retrieve_relevant_rules(violated_rules, target_language)
            
            # Generate code diff if we have previous versions
            code_diff = ""
            if state.get("previous_versions"):
                previous_code = str(state["previous_versions"][-1]) if state["previous_versions"][-1] is not None else ""
                code_diff = self._generate_code_diff(previous_code, translated_code)
            
            # Get compiler feedback if available
            compiler_feedback = ""
            if not state.get("compilation_success", True) and state.get("compilation_output"):
                compiler_feedback = str(state.get("compilation_output", ""))
                if state.get("compilation_error_analysis"):
                    compiler_feedback += "\n\nError Analysis:\n"
                    for issue in state.get("compilation_error_analysis", {}).get("common_issues", []):
                        compiler_feedback += f"- {str(issue)}\n"
                    for fix in state.get("compilation_error_analysis", {}).get("suggested_fixes", []):
                        compiler_feedback += f"- Suggestion: {str(fix)}\n"
            
            # 保存当前版本的代码到文件
            self._save_translation_version(translated_code, state)
            
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
            
            # Clean up the improved code for compilation
            improved_code = self._clean_code_for_compilation(str(improved_code) if improved_code is not None else "")
        
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
            print(f"Error details: {str(e)}")
            import traceback
            print("Stack trace:")
            print(traceback.format_exc())
        
        print("===========================================")
        return state
    
    def _save_translation_version(self, code: str, state: Dict) -> None:
        """保存翻译代码的每个版本到文件"""
        if not code:
            return
            
        try:
            # 创建版本目录
            versions_dir = Path("logs/code_versions")
            versions_dir.mkdir(parents=True, exist_ok=True)
            
            # 使用迭代号和时间戳创建唯一的文件名
            iteration = state.get("iteration", 0)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            target_language = state.get("target_language", "unknown").lower()
            
            # 为不同的目标语言设置不同的文件扩展名
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
            
            # 获取适当的文件扩展名，默认为.txt
            file_ext = file_extensions.get(target_language.lower(), ".txt")
            
            # 创建文件名
            filename = f"iteration_{iteration}_{timestamp}{file_ext}"
            file_path = versions_dir / filename
            
            # 保存代码
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(code)
                
            # 添加元数据文件
            metadata_file = versions_dir / f"iteration_{iteration}_{timestamp}_meta.json"
            
            # 安全获取执行时间和编译状态
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
            
            # 如果有编译错误，记录下来
            if "compilation_errors" in state and state["compilation_errors"]:
                metadata["compilation_errors"] = state["compilation_errors"]
                
            # 如果有执行输出，记录下来
            if "execution_output" in state and state["execution_output"]:
                # 限制输出长度，避免文件过大
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
        # 检查是否强制指定了继续迭代
        if state.get("force_improvement", False):
            return "continue"
        
        # 检查是否已达到最大迭代次数
        if state.get("iteration", 0) >= self.max_iterations:
            return "max_iterations_reached"
            
        # 检查编译是否成功 - 如果编译失败，继续改进代码而不是终止流程
        if not state.get("compilation_success", True):
            return "improve"  # 修改: 返回"improve"而不是"compilation_failed"，继续改进代码
            
        # 获取当前阶段的任务
        current_phase_num = state.get("iteration", 0) + 1
        if "conversion_plan" in state:
            plan = state["conversion_plan"]
            phase_data = self.analysis_agent.extract_plan_for_phase(plan, current_phase_num)
            
            # 如果存在下一阶段的任务，继续迭代
            if phase_data["found"] and phase_data["tasks"]:
                return "tasks_remain"
                
        # 基于验证结果做决定
        severity = "unknown"
        if "validation_metadata" in state:
            metadata = state["validation_metadata"]
            severity = metadata.get("severity", "unknown").lower()
            classification = metadata.get("classification", "unknown").lower()
            
            # 如果是严重或高优先级问题，继续改进
            if severity in ["critical", "high"]:
                return "critical_issues"
                
            # 如果是性能问题且我们正处于性能调优阶段，继续改进
            if classification == "performance" and current_phase_num >= 3:
                return "performance_issues"
                
        # 如果没有严重问题，检查是否已完成当前阶段的所有任务
        if state.get("iteration", 0) < current_phase_num:
            return "phase_incomplete"
            
        return "complete"
    
    def _get_current_phase(self, state: Dict) -> str:
        """Get the current phase from the conversion plan based on iteration"""
        # 如果已经有解析好的phase_name，直接使用
        if "current_phase_name" in state:
            return f"Phase {state.get('iteration', 0) + 1}: {state['current_phase_name']}"
            
        # 否则基于迭代次数决定当前阶段
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
            
            # 重置成功翻译列表
            self.successful_translations = []
            
            # Run workflow using invoke method
            final_state = self.workflow.invoke(initial_state)
            
            # Calculate processing time
            end_time = datetime.now(timezone.utc)
            processing_time = (end_time - start_time).total_seconds()
            
            # Add processing time to state
            final_state["processing_time"] = processing_time
            
            # 确保即使有错误，基本字段也存在
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
            
            # 选择最佳翻译版本
            # 如果有编译成功的版本，选择最新的一个
            if self.successful_translations:
                best_translation = self.successful_translations[-1]  # 最新的成功版本
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
            
            # 将所有翻译版本添加到最终状态
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
            
            # 如果处理失败，尝试生成错误分析
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
        """生成错误分析，帮助用户理解失败原因"""
        print("===========================================")
        print("Generating Error Analysis")
        
        error_msg = state.get("error", "Unknown error")
        error_type = state.get("error_type", "unspecified_error")
        
        try:
            # 创建错误分析模板
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
            
            # 准备参数
            code_content = state.get("code_content", "")
            code_sample = code_content[:500] + "..." if len(code_content) > 500 else code_content
            
            # 调用LLM生成错误分析
            chain = PromptTemplate.from_template(error_analysis_template, template_format="jinja2") | self.llm
            
            error_analysis = chain.invoke({
                "error_type": error_type,
                "error_message": error_msg,
                "source_language": state.get("source_language", "Unknown"),
                "target_language": state.get("target_language", "Unknown"),
                "code_sample": code_sample,
                "code_content": bool(code_content)
            })
            
            # 将分析结果添加到状态
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
        
        # 基本信息
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
        
        # 显示成功编译信息
        if "successful_translations_count" in state:
            successful_count = state.get("successful_translations_count", 0)
            print(f"Successful Builds  : {successful_count}")
            
            if successful_count > 0:
                selected_iteration = state.get("selected_translation_iteration", 0)
                print(f"Selected Version  : Iteration #{selected_iteration}")
        
        # 如果有错误，显示错误信息
        if "error" in state:
            print("\nError Information")
            print("-" * 50)
            print(f"Error Type       : {state.get('error_type', 'Unknown')}")
            print(f"Error Message    : {state.get('error', 'Unknown error')}")
            
            # 如果有错误分析，显示摘要
            if "error_analysis" in state and state["error_analysis"]:
                print("\nError Analysis Summary")
                print("-" * 50)
                error_analysis = state["error_analysis"]
                # 显示错误分析的前300个字符
                print(error_analysis[:300] + "..." if len(error_analysis) > 300 else error_analysis)
                print("\nSee full error analysis in the detailed report")
        
        # 显示代码结果
        print("\n" + "=" * 50)
        print(" TRANSLATED CODE ".center(50, "="))
        print("=" * 50)
        
        translated_code = state.get("translated_code", "")
        if translated_code:
            # 如果代码超过500行或10000个字符，只显示摘要
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
            
        # 生成详细报告
        report_filename = self._generate_detailed_report(state)
        print(f"\nDetailed translation report saved to: {report_filename} ")
        
    def _generate_detailed_report(self, state: Dict) -> str:
        """生成详细的翻译报告"""
        # 创建logs目录
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # 使用时间戳创建唯一的报告文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"logs/translation_report_{timestamp}.txt"
        
        try:
            with open(filename, "w", encoding="utf-8") as f:
                # 标题
                f.write("=" * 80 + "\n")
                f.write("HPC CODE TRANSLATION DETAILED REPORT\n")
                f.write("=" * 80 + "\n\n")
                
                # 基本信息
                f.write("BASIC INFORMATION\n")
                f.write("-" * 80 + "\n")
                f.write(f"Report Generated    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Source Language     : {state.get('source_language', 'Unknown')}\n")
                f.write(f"Target Language     : {state.get('target_language', 'Unknown')}\n")
                f.write(f"Processing Time     : {state.get('processing_time', 0):.2f} seconds\n")
                f.write(f"Iterations          : {state.get('iteration', 0)}\n")
                f.write(f"Status              : {state.get('status', 'unknown')}\n")
                
                # 添加成功编译信息
                if "successful_translations_count" in state:
                    successful_count = state.get("successful_translations_count", 0)
                    f.write(f"Successful Builds    : {successful_count}\n")
                    
                    if successful_count > 0:
                        selected_iteration = state.get("selected_translation_iteration", 0)
                        f.write(f"Selected Version    : Iteration #{selected_iteration}\n")
                
                f.write("\n")
                
                # 输入代码
                f.write("INPUT CODE\n")
                f.write("-" * 80 + "\n")
                code_content = state.get("code_content", "")
                if code_content:
                    f.write(code_content + "\n\n")
                else:
                    f.write("No input code provided or extraction failed\n\n")
                
                # 翻译代码
                f.write("TRANSLATED CODE\n")
                f.write("-" * 80 + "\n")
                translated_code = state.get("translated_code", "")
                if translated_code:
                    f.write(translated_code + "\n\n")
                else:
                    f.write("No translated code generated\n\n")
                
                # 如果有多个成功编译的版本，添加摘要信息
                if "all_successful_translations" in state and state["all_successful_translations"]:
                    f.write("SUCCESSFUL TRANSLATIONS SUMMARY\n")
                    f.write("-" * 80 + "\n")
                    for idx, trans in enumerate(state["all_successful_translations"]):
                        f.write(f"Version {idx+1} (Iteration {trans['iteration']}):\n")
                        f.write(f"  Timestamp: {trans['timestamp']}\n")
                        
                        # 安全获取执行时间 - 避免None值导致的格式化错误
                        exec_time = trans.get('execution_time')
                        if exec_time is not None:
                            f.write(f"  Execution Time: {exec_time:.4f} seconds\n")
                        else:
                            f.write(f"  Execution Time: N/A\n")
                        f.write("\n")
                
                # 如果有错误，显示错误信息
                if "error" in state:
                    f.write("ERROR INFORMATION\n")
                    f.write("-" * 80 + "\n")
                    f.write(f"Error Type          : {state.get('error_type', 'Unknown')}\n")
                    f.write(f"Error Message       : {state.get('error', 'Unknown error')}\n")
                    
                    # 错误详情
                    error_details = state.get("error_details", {})
                    if error_details:
                        f.write("\nError Details:\n")
                        for key, value in error_details.items():
                            f.write(f"- {key}: {value}\n")
                    
                    # 完整的错误分析
                    if "error_analysis" in state and state["error_analysis"]:
                        f.write("\nERROR ANALYSIS\n")
                        f.write("-" * 80 + "\n")
                        f.write(state["error_analysis"] + "\n\n")
                
                # 编译结果
                if "compilation_success" in state:
                    f.write("COMPILATION RESULTS\n")
                    f.write("-" * 80 + "\n")
                    f.write(f"Compilation Success : {state.get('compilation_success', False)}\n")
                    
                    # 如果有编译错误
                    compilation_errors = state.get("compilation_errors", [])
                    if compilation_errors:
                        f.write("\nCompilation Errors:\n")
                        for i, error in enumerate(compilation_errors):
                            f.write(f"{i+1}. {error}\n")
                    
                    # 编译输出
                    compiler_output = state.get("compilation_output", "")
                    if compiler_output:
                        f.write("\nCompiler Output:\n")
                        f.write(compiler_output + "\n")
                    
                    # 执行输出
                    execution_output = state.get("execution_output", "")
                    if execution_output:
                        f.write("\nExecution Output:\n")
                        f.write(execution_output + "\n")
                        
                    # 执行时间
                    execution_time = state.get("execution_time_seconds")
                    if execution_time is not None:
                        f.write(f"\nExecution Time: {execution_time:.4f} seconds\n")
                    else:
                        f.write("\nExecution Time: N/A\n")
                
                # 转换计划
                conversion_plan = state.get("conversion_plan", "")
                if conversion_plan:
                    f.write("\nCONVERSION PLAN\n")
                    f.write("-" * 80 + "\n")
                    f.write(conversion_plan + "\n\n")
                
                # 性能分析
                if "performance_metrics" in state:
                    f.write("PERFORMANCE ANALYSIS\n")
                    f.write("-" * 80 + "\n")
                    metrics = state["performance_metrics"]
                    for key, value in metrics.items():
                        f.write(f"{key}: {value}\n")
                
                # 添加版本信息
                if "previous_versions" in state and state["previous_versions"]:
                    f.write("\nVERSION HISTORY\n")
                    f.write("-" * 80 + "\n")
                    f.write(f"Total Versions: {len(state['previous_versions'])}\n")
                    f.write("All versions saved in logs/code_versions directory\n")
                
            return filename
        except Exception as e:
            print(f"Error generating detailed report: {e}")
            # 尝试保存最基本的信息
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