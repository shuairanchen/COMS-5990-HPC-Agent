#!/usr/bin/env python3
"""
Translation Agent
Responsible for code translation and improvement with compiler feedback integration
"""

from typing import Dict, List, Optional
import re

from langchain.prompts import PromptTemplate

class TranslationAgent:
    """Agent responsible for translating code between languages with compiler feedback"""
    
    def __init__(self, llm, knowledge_base):
        """Initialize the translation agent"""
        self.llm = llm
        self.knowledge_base = knowledge_base
    
    def translate_code(self, source_language: str, target_language: str, code_content: str) -> str:
        """Translate code from source language to target language with compiler awareness"""
        translation_template = """You are an HPC code conversion expert. Convert this {{source_language}} code to {{target_language}}:
        Requirements:
        1. Maintain identical algorithmic logic
        2. Follow target language's performance best practices
        3. Add necessary comments explaining modifications
        4. Ensure syntactic correctness
        5. Prioritize compilation success and runtime performance
        6. IMPORTANT: Return a COMPLETE, COMPILABLE program with all necessary:
           - Include statements/imports
           - Main function or entry point
           - Variable declarations
           - Initialization code
           - Error handling
        
        // Original {{source_language}} code:
        {{code_input}}
        
        If the original code is just a fragment (e.g., just a loop or function), wrap it in a complete program structure with all necessary declarations and initializations.
        
        Return ONLY the converted {{target_language}} code without explanations.
        """
        
        prompt = PromptTemplate(
            template=translation_template,
            input_variables=["source_language", "target_language", "code_input"],
            template_format="jinja2"
        )
        
        chain = prompt | self.llm
        
        result = chain.invoke({
            "source_language": source_language,
            "target_language": target_language,
            "code_input": code_content
        })
        
        return result.content
    
    def improve_code(self, code: str, validation_result: str, current_phase: str,
                    target_language: str, priority: str, severity: str, 
                    relevant_rules: str, code_diff: str, compiler_feedback: str = "") -> str:
        """Improve code based on validation findings and compiler feedback"""
        # Ensure all inputs are string type
        validation_result = str(validation_result) if validation_result is not None else ""
        current_phase = str(current_phase) if current_phase is not None else ""
        target_language = str(target_language) if target_language is not None else ""
        priority = str(priority) if priority is not None else "deferred"
        severity = str(severity) if severity is not None else "medium"
        relevant_rules = str(relevant_rules) if relevant_rules is not None else ""
        code_diff = str(code_diff) if code_diff is not None else ""
        compiler_feedback = str(compiler_feedback) if compiler_feedback is not None else ""
        
        improvement_template = """Improve the code based on validation findings and compiler feedback while maintaining original functionality.
        
        **Context**
        - Current Phase: {{current_phase}}
        - Severity Level: {{severity | upper}}
        - Priority Level: {{priority}}
        - Related Rules: {% for rule in violated_rules %}{{ rule }}{% if not loop.last %}，{% endif %}{% endfor %}
        
        **Required Modifications**
        {{validation_summary}}
        
        {% if compiler_feedback %}
        **Compiler Feedback**
        {{compiler_feedback}}
        {% endif %}
        
        **Relevant Rules**
        {{relevant_rules}}
        
        **Code Diff Analysis**
        {{code_diff}}
        
        **Implementation Requirements**
        1. Modify only necessary content (add // MODIFIED comment)
        2. Keep the original code structure
        3. Give priority to {{ target_language }} recommended style
        4. Ensure code compiles successfully and executes correctly
        5. Focus on both correctness and performance
        6. IMPORTANT: Return a COMPLETE, COMPILABLE program with all necessary:
           - Include statements/imports
           - Main function or entry point
           - Variable declarations
           - Initialization code
           - Error handling
        
        Return FULL IMPLEMENTATION with changes clearly visible.
        """
        
        # Extract violated rules from validation result
        violated_rules = []
        for line in validation_result.split("\n"):
            if "Rule" in line and ":" in line:
                rule_id = line.split(":")[0].strip()
                if rule_id and any(c.isdigit() for c in rule_id):
                    violated_rules.append(rule_id)
        
        prompt = PromptTemplate(
            template=improvement_template,
            input_variables=["validation_summary", "current_phase", "priority", "severity", "code_diff", "compiler_feedback"],
            partial_variables={
                "violated_rules": violated_rules,
                "relevant_rules": relevant_rules,
                "target_language": target_language
            },
            template_format="jinja2"
        )
        
        chain = prompt | self.llm
        
        result = chain.invoke({
            "validation_summary": validation_result,
            "current_phase": current_phase,
            "priority": priority,
            "severity": severity,
            "code_diff": code_diff,
            "compiler_feedback": compiler_feedback
        })
        
        # Extract code from result if it contains surrounding markdown or text
        code = self._extract_code_from_result(result.content)
        
        return code
    
    def _extract_code_from_result(self, result: str) -> str:
        """Extract code from LLM result that might contain explanations"""
        # Check if the result is wrapped in code blocks
        if "```" in result:
            # Get content between first and last code block markers
            code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", result, re.DOTALL)
            if code_blocks:
                # Return the largest code block (most likely the complete program)
                return max(code_blocks, key=len).strip()
        
        # If no code blocks, assume entire result is code
        return result.strip()