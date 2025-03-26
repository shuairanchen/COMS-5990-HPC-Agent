#!/usr/bin/env python3
"""
Verification Agent
Responsible for code validation and analysis
"""

import re
from typing import Dict, List, Any

from langchain.prompts import PromptTemplate

class VerificationAgent:
    """Agent responsible for validating code quality and correctness"""
    
    def __init__(self, llm, knowledge_base):
        """Initialize the verification agent"""
        self.llm = llm
        self.knowledge_base = knowledge_base
    
    def validate_code(self, code: str, target_language: str, current_phase: str, 
                     potential_issues: List[str], iteration: int) -> Dict[str, Any]:
        """Validate code against target language rules and standards"""
        
        code = str(code) if code is not None else ""
        target_language = str(target_language) if target_language is not None else ""
        current_phase = str(current_phase) if current_phase is not None else ""
        iteration = int(iteration) if iteration is not None else 0
        
        # clean thinking process in code
        code = self._clean_thinking_process(code)
        
        safe_issues = []
        if potential_issues:
            for issue in potential_issues:
                if issue is not None:
                    safe_issues.append(str(issue))
        
        # Get contextual information
        code_rules = self._get_code_rules(target_language)
        
        # React-style analysis template
        react_template = """Perform structured code validation with context-aware analysis:
        **Context**
        - Current Phase: {{current_phase}}
        - Iteration: {{iteration}}
        **Code to Validate**
        {{code}}
        **Validation Rules**
        {{code_rules}}
        **Required Analysis**
        1. Problem Classification: [syntax/logic/performance/style]
        2. Severity Assessment: [critical/high/medium/low]
        3. Repair Priority: [immediate/deferred]
        4. Rule Violations: List specific rule IDs
        5. Suggested Fix Approach: Brief technical solution
        **Response Format**
        Analysis Result:
        - Classification: <classification>
        - Severity: <severity>
        - Priority: <priority>
        - Violated Rules: <rule_ids>
        - Solution Approach: <solution_notes>
        Validation Report:
        {% if potential_issues %}Issues Found: Yes
        Detailed Findings:
        - <rule1>: <description> (line X)
        - <rule2>: <description> (line Y)
        {% else %}Issues Found: No{% endif %}"""
        
        # Prepare prompt
        prompt = PromptTemplate(
            template=react_template,
            input_variables=["code", "current_phase", "potential_issues", "iteration"],
            partial_variables={
                "code_rules": code_rules
            },
            template_format="jinja2"
        )
        
        # Build chain
        analysis_chain = prompt | self.llm
        
        # Execute analysis
        react_result = analysis_chain.invoke({
            "code": code,
            "current_phase": current_phase,
            "potential_issues": safe_issues,
            "iteration": iteration
        })
        
        # get result content
        result_content = ""
        if hasattr(react_result, 'content'):
            result_content = react_result.content
        elif isinstance(react_result, str):
            result_content = react_result
        elif isinstance(react_result, dict) and 'content' in react_result:
            result_content = react_result['content']
        else:
            # try to convert to string
            result_content = str(react_result)
            
        # clean thinking process
        cleaned_result = self._clean_thinking_process(result_content)
        
        # Parse structured output
        parsed_analysis = self._parse_react_analysis(cleaned_result)
        
        return {
            "analysis": cleaned_result,
            "metadata": {
                "classification": parsed_analysis.get("classification", "unknown"),
                "severity": parsed_analysis.get("severity", "medium"),
                "priority": parsed_analysis.get("priority", "deferred"),
                "violated_rules": parsed_analysis.get("violated_rules", []),
                "solution_approach": parsed_analysis.get("solution_approach", "")
            }
        }
    
    def _get_code_rules(self, target_lang: str) -> str:
        """Get the target language code rules"""
        rules = self.knowledge_base.get(target_lang, {})
        return "\n".join(
            [f"# {cat.upper()}\n" + "\n".join(f"- {item}" for item in items)
             for cat, items in rules.items() if cat != "analysis_rules"]
        )
    
    def _parse_react_analysis(self, analysis_text: str) -> Dict:
        """Parse structured React analysis output"""
        parsed = {
            "classification": "unknown",
            "severity": "medium",
            "priority": "deferred",
            "violated_rules": [],
            "solution_approach": ""
        }
        
        # Extract classification
        class_match = re.search(r"Classification:\s*(\w+)", analysis_text, re.I)
        if class_match:
            parsed["classification"] = class_match.group(1).lower()
            
        # Extract severity
        severity_match = re.search(r"Severity:\s*(\w+)", analysis_text, re.I)
        if severity_match:
            parsed["severity"] = severity_match.group(1).lower()
            
        # Extract priority
        priority_match = re.search(r"Priority:\s*(\w+)", analysis_text, re.I)
        if priority_match:
            parsed["priority"] = priority_match.group(1).lower()
            
        # Extract violated rules
        rule_matches = re.findall(r"([A-Z]{2}-\d{3}):", analysis_text)
        parsed["violated_rules"] = list(set(rule_matches))
        
        # Extract solution approach
        solution_match = re.search(r"Solution Approach:\s*(.+?)(?=\n\S+:|$)", analysis_text, re.S)
        if solution_match:
            parsed["solution_approach"] = solution_match.group(1).strip()
            
        return parsed

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
