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

        source_language = str(source_language) if source_language is not None else ""
        target_language = str(target_language) if target_language is not None else ""
        code_content = str(code_content) if code_content is not None else ""
        
        translation_template = """
        You are an HPC code conversion expert. Convert this {{source_language}} code to {{target_language}}:
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
        
        try:
            chain = prompt | self.llm
            
            result = chain.invoke({
                "source_language": source_language,
                "target_language": target_language,
                "code_input": code_content
            })
            
            translated_code = None
            
            # try to extract content from result
            if hasattr(result, 'content'):
                translated_code = result.content
            elif isinstance(result, str):
                translated_code = result
            elif isinstance(result, dict) and 'content' in result:
                translated_code = result['content']
            else:
                # if cannot extract content directly, try to convert to string
                translated_code = str(result)
            
            # clean possible thinking process (sometimes model returns <think>...</think> blocks)
            translated_code = self._extract_code_from_result(translated_code)
            
            # ensure the returned code contains necessary elements
            if not translated_code or len(translated_code.strip()) < 10:
                print("Warning: model returned empty or too short translation result")
                # provide a basic template
                translated_code = f"""// Failed to translate {source_language} to {target_language}
                // Here is a template for the target language:

                // Target language: {target_language}
                {code_content}
                """
            
            print(f"Translation completed: {len(translated_code)} characters")
            return translated_code
            
        except Exception as e:
            print(f"Error in translate_code: {str(e)}")
            # return original code, so the system can continue running
            return f"// Error during translation: {str(e)}\n\n{code_content}"
    
    def improve_code(self, code: str, validation_result: str, current_phase: str,
                    target_language: str, priority: str, severity: str, 
                    relevant_rules: str, code_diff: str, compiler_feedback: str = "") -> str:
        """Improve code based on validation findings and compiler feedback"""
        try:
            # Ensure all inputs are string type
            code = str(code) if code is not None else ""
            validation_result = str(validation_result) if validation_result is not None else ""
            current_phase = str(current_phase) if current_phase is not None else ""
            target_language = str(target_language) if target_language is not None else ""
            priority = str(priority) if priority is not None else "deferred"
            severity = str(severity) if severity is not None else "medium"
            relevant_rules = str(relevant_rules) if relevant_rules is not None else ""
            code_diff = str(code_diff) if code_diff is not None else ""
            compiler_feedback = str(compiler_feedback) if compiler_feedback is not None else ""
            
            # Debug information
            print("===========================================")
            print("Debug Information in improve_code:")
            print(f"Code Type: {type(code)}")
            print(f"Validation Result Type: {type(validation_result)}")
            print(f"Current Phase Type: {type(current_phase)}")
            print("===========================================")
            
            improvement_template = """
            Improve the code based on validation findings and compiler feedback while maintaining original functionality.
            
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
                        violated_rules.append(str(rule_id))
            
            safe_violated_rules = []
            for rule in violated_rules:
                if rule is not None:
                    safe_violated_rules.append(str(rule))
            
            # print debug information
            print("Violated Rules Types:")
            for i, rule in enumerate(safe_violated_rules):
                print(f"  Rule {i}: {type(rule)} - {rule}")
            
            prompt = PromptTemplate(
                template=improvement_template,
                input_variables=["validation_summary", "current_phase", "priority", "severity", "code_diff", "compiler_feedback"],
                partial_variables={
                    "violated_rules": safe_violated_rules,
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
            
            print(result.content)
            # Extract code from result if it contains surrounding markdown or text
            code = self._extract_code_from_result(result.content)
            print("Extract code from result")
            print(code)
            return code
        except Exception as e:
            print(f"Error in improve_code: {e}")
            import traceback
            print("Stack trace:")
            print(traceback.format_exc())
            return ""
    
    def _extract_code_from_result(self, result: str) -> str:
        """Extract code from LLM result that might contain explanations or thinking process"""
        if not result:
            return ""
            
        # first, remove <think>...</think>
        think_pattern = re.compile(r'<think>.*?</think>', re.DOTALL)
        result = think_pattern.sub('', result).strip()
        
        # check if there are other thinking formats (e.g. "Let me think...")
        thinking_patterns = [
            r'(?i)Let me think\s*:.*?\n\s*\n',  # "Let me think: ..."
            r'(?i)I need to analyze.*?\n\s*\n',  # "I need to analyze..."
            r'(?i)First, I\'ll.*?\n\s*\n',      # "First, I'll..."
            r'(?i)Step \d+:.*?\n\s*\n'          # "Step 1: ..."
        ]
        
        for pattern in thinking_patterns:
            result = re.sub(pattern, '', result, flags=re.DOTALL)
        
        # remove thinking/analysis content until finding code indicator
        common_intros = [
            r'(?i)Here\'s the .* code:',
            r'(?i)Here is the .* code:',
            r'(?i)The translated code is:',
            r'(?i)Here\'s my solution:',
            r'(?i)Here is the solution:'
        ]
        
        for intro in common_intros:
            intro_match = re.search(intro, result)
            if intro_match:
                intro_end = intro_match.end()
                result = result[intro_end:].strip()
                break
        
        # check if there are code blocks
        if "```" in result:
            # extract all code blocks
            code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", result, re.DOTALL)
            if code_blocks:
                # return the longest code block, usually a complete program
                return max(code_blocks, key=len).strip()
        
        # if there are no code block markers, try to find specific language markers
        language_markers = ["#include", "import ", "package ", "using namespace", "public class"]
        lines = result.split('\n')
        start_line = 0
        
        for i, line in enumerate(lines):
            if any(marker in line for marker in language_markers):
                start_line = i
                break
        
        # return the code part starting from the marked line
        final_code = '\n'.join(lines[start_line:])
        
        # if still no code found, return the whole result content (removed thinking process)
        return final_code.strip()