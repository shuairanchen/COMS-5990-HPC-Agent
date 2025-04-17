#!/usr/bin/env python3
"""
Translation Agent
Responsible for code translation and improvement with compiler feedback integration
"""

from typing import Dict, List, Optional, Tuple
import re
from difflib import SequenceMatcher

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
        7. If original code has a complete model implementation (e.g., neural network, transformer, etc.), preserve it fully
        8. Maintain full functionality - do not simplify or remove features
        
        // Original {{source_language}} code:
        {{code_input}}
        
        If the original code is just a fragment (e.g., just a loop or function), wrap it in a complete program structure with all necessary declarations and initializations.
        
        Return ONLY the converted {{target_language}} code without explanations. The converted code must be a complete, runnable program that preserves all functionality of the original code.
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
                    relevant_rules: str, code_diff: str, compiler_feedback: str = "") -> Tuple[str, List[Dict]]:
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
            
            # Check if it's a PyTorch to JAX conversion, add specific knowledge
            source_language = ""
            is_pytorch_to_jax = False
            pytorch_jax_knowledge = ""
            
            if hasattr(self, 'knowledge_base'):
                # Try to get the source language of the current translation
                if isinstance(self.knowledge_base, dict) and 'current_translation' in self.knowledge_base:
                    source_language = self.knowledge_base.get('current_translation', {}).get('source_language', '')
                
                # Check if it's a PyTorch to JAX translation
                is_pytorch_to_jax = source_language.lower() in ['pytorch', 'torch'] and target_language.lower() == 'jax'
                
                # Get specific knowledge for PyTorch to JAX conversion
                if is_pytorch_to_jax and 'pytorch_to_jax' in self.knowledge_base:
                    pytorch_jax_knowledge = self._format_pytorch_jax_knowledge()
            
            # Track applied rules
            applied_rules = []
            
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
            2. IMPORTANT: Keep the original code structure - do not completely rewrite the program
            3. Give priority to {{ target_language }} recommended style
            4. Ensure code compiles successfully and executes correctly
            5. Focus on both correctness and performance 
            6. CRITICAL: Return a COMPLETE, COMPILABLE program with all necessary:
               - Include statements/imports
               - Main function or entry point
               - Variable declarations
               - Initialization code
               - Error handling
            7. PRESERVE ALL EXISTING FUNCTIONALITY - do not remove or simplify working code
            8. If the original code has a complete transformer model, training loop, or data processing, keep it
            
            Original {{ target_language }} code:
            ```
            {{code}}
            ```
            
            Return FULL IMPLEMENTATION with changes clearly visible. Return the entire program, not just fragments.
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
            
            # Add PyTorch-JAX knowledge to the prompt
            if is_pytorch_to_jax and pytorch_jax_knowledge:
                improvement_template += """
                
                **PyTorch to JAX conversion specific knowledge**
                {{pytorch_jax_knowledge}}
                """
            
            prompt = PromptTemplate(
                template=improvement_template,
                input_variables=["validation_summary", "current_phase", "priority", "severity", 
                                 "code_diff", "compiler_feedback", "code"],
                partial_variables={
                    "violated_rules": safe_violated_rules,
                    "relevant_rules": relevant_rules,
                    "target_language": target_language,
                    "pytorch_jax_knowledge": pytorch_jax_knowledge
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
                "compiler_feedback": compiler_feedback,
                "code": code
            })
            
            print("LLM Response received for code improvement.")
            # Extract code from result if it contains surrounding markdown or text
            improved_code = self._extract_code_from_result(result.content)
            
            # # Check if improved code appears to be a valid program
            # if not self._is_valid_improvement(code, improved_code, target_language):
            #     print("Warning: Improved code may not be valid or complete. Using original code.")
            #     improved_code = code
            
            # Detect which PyTorch-JAX rules are applied
            if is_pytorch_to_jax and pytorch_jax_knowledge:
                applied_rules = self._detect_applied_rules(improved_code, result.content, 
                                                          self.knowledge_base.get('pytorch_to_jax', {}))
            
            print(f"Improved Code: {improved_code}")
            print("Code improvement completed. Length of improved code:", len(improved_code))
            return improved_code, applied_rules
        except Exception as e:
            print(f"Error in improve_code: {e}")
            import traceback
            print("Stack trace:")
            print(traceback.format_exc())
            return "", []
    
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
        
        # First try to find code blocks with proper markdown
        if "```" in result:
            # Find all code blocks
            code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", result, re.DOTALL)
            if code_blocks:
                # Get the longest code block, which is likely the complete program
                return max(code_blocks, key=len).strip()
        
        # If no code blocks found, try to find the code by looking for intro markers
        common_intros = [
            r'(?i)Here\'s the .* code:',
            r'(?i)Here is the .* code:',
            r'(?i)The translated code is:',
            r'(?i)Here\'s my solution:',
            r'(?i)Here is the solution:',
            r'(?i)The improved code:',
            r'(?i)Improved code:',
            r'(?i)Final code:'
        ]
        
        for intro in common_intros:
            intro_match = re.search(intro, result)
            if intro_match:
                intro_end = intro_match.end()
                # Extract content after intro marker
                code_after_intro = result[intro_end:].strip()
                
                # Check if there's another markdown code block after intro
                if "```" in code_after_intro:
                    # Extract code between first pair of markdown fences
                    code_match = re.search(r"```(?:\w+)?\n(.*?)```", code_after_intro, re.DOTALL)
                    if code_match:
                        return code_match.group(1).strip()
                
                # If no markdown block found but intro exists, consider rest as code
                return code_after_intro
        
        # If no intro markers, look for common language-specific indicators
        language_markers = [
            "#include", "import ", "package ", "using namespace", "public class",
            "def ", "class ", "function", "module", "@jit", "if __name__"
        ]
        
        lines = result.split('\n')
        start_line = 0
        has_markers = False
        
        for i, line in enumerate(lines):
            if any(marker in line for marker in language_markers):
                start_line = i
                has_markers = True
                break
        
        if has_markers:
            # Return code from first identified marker
            return '\n'.join(lines[start_line:]).strip()
        
        # Last resort: Just strip MODIFIED marker comments and return all content
        result_without_explanation = re.sub(r'^.*?(?=import|#include|package|using)', '', result, flags=re.DOTALL)
        result_without_explanation = result_without_explanation.strip()
        
        # If we have a substantial result after cleaning, return it
        if len(result_without_explanation) > 50:
            return result_without_explanation
        
        # If still no code found, return the whole result content (with thinking process removed)
        return result.strip()
    
    def _format_pytorch_jax_knowledge(self) -> str:
        # Format specific knowledge for PyTorch to JAX conversion
        if not hasattr(self, 'knowledge_base') or 'pytorch_to_jax' not in self.knowledge_base:
            return ""
        
        pytorch_jax_kb = self.knowledge_base.get('pytorch_to_jax', {})
        formatted_knowledge = ["Common PyTorch to JAX conversion patterns:"]
        
        # Format knowledge base content
        for error_key, error_data in pytorch_jax_kb.items():
            if isinstance(error_data, dict):
                pytorch_code = error_data.get('pytorch_code', '')
                jax_code = error_data.get('jax_code', '')
                error_pattern = error_data.get('error_pattern', '')
                solution = error_data.get('solution', '')
                
                # Use more specific markers
                formatted_knowledge.append(f"\n## Conversion pattern: [{error_key}]")
                if error_pattern:
                    formatted_knowledge.append(f"Problem characteristics: ```{error_pattern}```")
                if solution:
                    formatted_knowledge.append(f"Solution: ```{solution}```")
                if pytorch_code and jax_code:
                    formatted_knowledge.append("Reference code comparison:")
                    formatted_knowledge.append("PyTorch code:")
                    formatted_knowledge.append(f"```python\n{pytorch_code}\n```")
                    formatted_knowledge.append("JAX equivalent code:")
                    formatted_knowledge.append(f"```python\n{jax_code}\n```")
        
        return "\n".join(formatted_knowledge)

    def _calculate_similarity(self, a: str, b: str) -> float:

        if not a or not b:
            return 0.0
        
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    def _detect_applied_rules(self, code: str, improvement_text: str, rule_base: Dict) -> List[Dict]:
        """Detect which rules are applied"""
        applied_rules = []
        similarity_threshold = 0.6  # set 60% similarity threshold
        
        if not rule_base:
            return applied_rules
        
        # Check if each rule is applied in the code improvement process
        for rule_id, rule_data in rule_base.items():
            if isinstance(rule_data, dict):
                # Get PyTorch code and JAX code snippets
                pytorch_code = rule_data.get('pytorch_code', '')
                jax_code = rule_data.get('jax_code', '')
                error_pattern = rule_data.get('error_pattern', '')
                solution = rule_data.get('solution', '')
                
                # Check if the rule is applied
                was_applied = False
                max_similarity = 0.0
                match_reason = ""
                
                # Check the similarity between error pattern and improvement text
                if error_pattern:
                    # Split improvement text into paragraphs for comparison
                    for paragraph in improvement_text.split('\n\n'):
                        similarity = self._calculate_similarity(error_pattern, paragraph)
                        if similarity > max_similarity:
                            max_similarity = similarity
                            match_reason = "Error pattern match"
                
                # Check the similarity between solution and improvement text
                if solution and not was_applied:
                    for paragraph in improvement_text.split('\n\n'):
                        similarity = self._calculate_similarity(solution, paragraph)
                        if similarity > max_similarity:
                            max_similarity = similarity
                            match_reason = "Solution match"
                
                # Check if JAX code snippet is applied
                if jax_code and not was_applied:
                    # Split JAX code into lines and find if any key line is in the generated code
                    jax_lines = [line.strip() for line in jax_code.split('\n') if line.strip()]
                    matched_lines = 0
                    total_significant_lines = len([line for line in jax_lines if len(line) > 10])
                    
                    for jax_line in jax_lines:
                        if len(jax_line) > 10:  # Only check meaningful code lines
                            # Calculate the maximum similarity between each line and generated code
                            for code_line in code.split('\n'):
                                similarity = self._calculate_similarity(jax_line, code_line)
                                if similarity >= 0.7:  # Use higher similarity threshold for line level
                                    matched_lines += 1
                                    break
                
                    # Calculate the percentage of matching code lines
                    if total_significant_lines > 0:
                        code_similarity = matched_lines / total_significant_lines
                        if code_similarity > max_similarity:
                            max_similarity = code_similarity
                            match_reason = "Code implementation match"
                
                # If rule ID appears directly in the text, also give higher similarity
                if rule_id.lower() in improvement_text.lower():
                    direct_match_similarity = 0.8
                    if direct_match_similarity > max_similarity:
                        max_similarity = direct_match_similarity
                        match_reason = "Rule ID direct match"
                
                # If similarity exceeds threshold, consider the rule applied
                if max_similarity >= similarity_threshold:
                    print(f"Applied rule: {rule_id} (similarity: {max_similarity:.2f}, match type: {match_reason})")
                    applied_rules.append({
                        "rule_id": rule_id,
                        "error_pattern": error_pattern,
                        "solution": solution,
                        "similarity": max_similarity,
                        "match_type": match_reason
                    })
        
        return applied_rules