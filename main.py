#!/usr/bin/env python3
"""
Enhanced Code Translation Framework
Main entry point with improved user input analysis
"""

import os
import argparse
import re
import tempfile
import sys
import time
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from code_translation_graph import CodeTranslationGraph

# Add logging functionality to capture terminal output
class Logger:
    """Class to capture terminal output to both console and file"""
    def __init__(self, log_file_path):
        self.terminal = sys.stdout
        self.log_file = open(log_file_path, 'w', encoding='utf-8')
        
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()
        
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
        
    def close(self):
        self.log_file.close()

def analyze_user_input(user_input):
    """
    Analyze user input to extract source language, target language and code
    Returns a structured request format for the translation system
    """
    # Check if there's a clear translation request pattern
    translation_pattern = re.compile(
        r"(?:translate|convert)\s+(?:this|from)\s+(\w+)(?:\s+code)?\s+(?:to|into)\s+(\w+)",
        re.IGNORECASE
    )
    match = translation_pattern.search(user_input)
    
    if match:
        source_lang = match.group(1)
        target_lang = match.group(2)
        
        # Now try to extract code blocks
        code_block = extract_code_block(user_input)
        if code_block:
            return f"Translate this code from {source_lang} to {target_lang}:\n\n{code_block}"
    
    # If no clear pattern is found, return the original input
    # The analysis agent will handle the unstructured input
    return user_input

def extract_code_block(text):
    """
    Extract code blocks from user input text
    Supports both markdown code blocks and indented code
    """
    # Try to extract markdown code blocks first
    code_block_pattern = re.compile(r"```(?:\w+)?\n(.*?)```", re.DOTALL)
    code_blocks = code_block_pattern.findall(text)
    
    if code_blocks:
        return code_blocks[0].strip()
    
    # If no markdown blocks, look for indented code (4+ spaces or tabs)
    lines = text.split('\n')
    indented_lines = []
    in_code_block = False
    
    for line in lines:
        if line.startswith(('    ', '\t')):
            indented_lines.append(line.lstrip())
            in_code_block = True
        elif in_code_block and line.strip() == '':
            # Keep empty lines in indented blocks
            indented_lines.append('')
        elif in_code_block:
            in_code_block = False
    
    if indented_lines:
        return '\n'.join(indented_lines)
    
    # If we can't identify a clear code block, look for the largest content
    # between language mentions
    if "```" not in text and len(text.split()) > 30:
        return text
    
    return None

def print_section_header(title):
    """Print a formatted section header for better readability"""
    print("\n" + "=" * 50)
    print(f" {title} ".center(50, "="))
    print("=" * 50)

def print_data_table(data, title=None):
    """Print data in a formatted table for better visualization"""
    if title:
        print(f"\n{title}")
        print("-" * 50)
    
    if isinstance(data, dict):
        max_key_len = max([len(str(k)) for k in data.keys()]) if data else 0
        for key, value in data.items():
            print(f"{str(key):<{max_key_len + 2}} : {value}")
    elif isinstance(data, list):
        for i, item in enumerate(data):
            print(f"{i+1}. {item}")
    else:
        print(data)

def save_translation_report(result, output_file=None):
    """Save a detailed translation report to a file"""
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    if not output_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = logs_dir / f"translation_report_{timestamp}.txt"
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("CODE TRANSLATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("TRANSLATION DETAILS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Source Language: {result.get('source_language', 'Unknown')}\n")
            f.write(f"Target Language: {result.get('target_language', 'Unknown')}\n")
            f.write(f"Processing Time: {result.get('processing_time', 0):.2f} seconds\n")
            f.write(f"Iterations: {result.get('iteration', 0)}\n")
            
            # add successful compilation information
            successful_translations_count = result.get('successful_translations_count', 0)
            f.write(f"Successful Compilations: {successful_translations_count}\n")
            
            if successful_translations_count > 0:
                selected_iteration = result.get('selected_translation_iteration', 0)
                f.write(f"Selected Translation: Iteration #{selected_iteration}\n")
            
            f.write("\n")
            
            f.write("TRANSLATED CODE\n")
            f.write("-" * 80 + "\n")
            translated_code = result.get("translated_code", "No code generated")
            if translated_code:
                f.write(translated_code + "\n\n")
            else:
                f.write("No code generated\n\n")
            
            # if there are multiple successful compilations, add summary information
            if "all_successful_translations" in result and result["all_successful_translations"]:
                f.write("SUCCESSFUL TRANSLATIONS SUMMARY\n")
                f.write("-" * 80 + "\n")
                for idx, trans in enumerate(result["all_successful_translations"]):
                    f.write(f"Version {idx+1} (Iteration {trans['iteration']}):\n")
                    f.write(f"  Timestamp: {trans['timestamp']}\n")
                    
                    # safely get execution time - avoid None value formatting error
                    exec_time = trans.get('execution_time')
                    if exec_time is not None:
                        f.write(f"  Execution Time: {exec_time:.4f} seconds\n")
                    else:
                        f.write(f"  Execution Time: N/A\n")
                    f.write("\n")
            
            if "compilation_success" in result:
                f.write("COMPILATION RESULTS\n")
                f.write("-" * 80 + "\n")
                f.write(f"Compilation Success: {result.get('compilation_success', False)}\n")
                
                if result.get("compilation_success", False):
                    f.write("\nEXECUTION OUTPUT\n")
                    f.write("-" * 80 + "\n")
                    execution_output = result.get("execution_output", "No output")
                    f.write(execution_output if execution_output else "No output" + "\n\n")
                    
                    # safely get execution time
                    exec_time = result.get('execution_time_seconds')
                    if exec_time is not None:
                        f.write(f"Execution Time: {exec_time:.4f} seconds\n\n")
                    else:
                        f.write("Execution Time: N/A\n\n")
                else:
                    f.write("\nCOMPILATION ERRORS\n")
                    f.write("-" * 80 + "\n")
                    compilation_errors = result.get("compilation_errors", ["Unknown error"])
                    if compilation_errors:
                        f.write("\n".join(compilation_errors) + "\n\n")
                    else:
                        f.write("No specific error information available\n\n")
            
            if "hpc_analysis" in result:
                f.write("HPC ANALYSIS\n")
                f.write("-" * 80 + "\n")
                f.write(result["hpc_analysis"] + "\n\n")
            
            if "error_log" in result:
                f.write("ERROR LOG\n")
                f.write("-" * 80 + "\n")
                error_log = result["error_log"]
                if isinstance(error_log, dict):
                    for key, value in error_log.items():
                        f.write(f"{key}: {value}\n")
                else:
                    f.write(str(error_log) + "\n")
                    
            # add complete translation history
            if "previous_versions" in result and result["previous_versions"]:
                f.write("\nTRANSLATION HISTORY\n")
                f.write("-" * 80 + "\n")
                f.write(f"Total Iterations: {len(result['previous_versions'])}\n\n")
                
            # save complete session information
            f.write("\nSESSION INFORMATION\n")
            f.write("-" * 80 + "\n")
            f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Processing Time: {result.get('processing_time', 0):.2f} seconds\n")
            
        print(f"\nDetailed translation report saved to: {output_file}")
        return output_file
        
    except Exception as e:
        print(f"Error generating report: {str(e)}")
        # try to save basic information in case of error
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("ERROR IN REPORT GENERATION\n")
                f.write(f"Error: {str(e)}\n\n")
                f.write("BASIC TRANSLATION INFO\n")
                f.write(f"Source Language: {result.get('source_language', 'Unknown')}\n")
                f.write(f"Target Language: {result.get('target_language', 'Unknown')}\n")
                f.write("TRANSLATED CODE\n")
                f.write(result.get("translated_code", "No code generated"))
            return output_file
        except:
            print("Failed to save even basic report information")
            return "Failed to generate report"

def main():
    """Main entry point for the enhanced code translation system"""
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Set up logging to capture terminal output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"translation_log_{timestamp}.txt"
    sys.stdout = Logger(log_file)
    
    print_section_header("HPC CODE TRANSLATION SYSTEM")
    print(f"Session started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Log file: {log_file}")
    
    # Load environment variables
    load_dotenv()
    
    # Parse command line arguments
    # parser = argparse.ArgumentParser(description="Code Translation System")
    # parser.add_argument("--input", "-i", type=str, help="Input code file path")
    # parser.add_argument("--target", "-t", type=str, help="Target language")
    # parser.add_argument("--kb", "-k", type=str, default="code_rules.yaml", help="Knowledge base path")
    # parser.add_argument("--output", "-o", type=str, help="Output file path")
    # parser.add_argument("--iterations", type=int, default=3, help="Maximum iterations")
    # parser.add_argument("--user-input", "-u", type=str, help="Direct user input text")
    # args = parser.parse_args()
    
    # # Read input code
    # code_content = ""
    # if args.input:
    #     try:
    #         with open(args.input, 'r', encoding='utf-8') as f:
    #             code_content = f.read()
    #     except Exception as e:
    #         print(f"Error reading input file: {e}")
    #         return 1

    print_section_header("USER INPUT")
    user_input = """
    Please help me convert the following C code into OpenMP code, and ensure the code functionality is consistent:
    #include <stdio.h>
    #include <stdlib.h>

    int dummyMethod1();
    int dummyMethod2();
    int dummyMethod3();
    int dummyMethod4();

    int main(int argc,char *argv[])
    {
    int i;
    int x;
    int len = 10000;
    if (argc > 1) 
        len = atoi(argv[1]);
    dummyMethod1();
    
    //#pragma rose_outline
    for (i = 0; i <= len - 1; i += 1) {
        x = i;
    }
    dummyMethod2();
    printf("x=%d",x);
    return 0;
    }

    int dummyMethod1()
    {
    return 0;
    }

    int dummyMethod2()
    {
    return 0;
    }

    int dummyMethod3()
    {
    return 0;
    }

    int dummyMethod4()
    {
    return 0;
    }
    """
    print(user_input)
    
    # Process user input if provided
    # user_input = args.user_input or ""
    # if user_input and not code_content:
    #     # Pre-analyze user input
    #     structured_input = analyze_user_input(user_input)
    # elif code_content and args.target:
    #     structured_input = f"Translate this code to {args.target}:\n\n{code_content}"
    # else:
    #     print("Error: Either provide a file with --input and target with --target OR")
    #     print("provide user input with --user-input")
    #     return 1
    
    # Initialize translation graph
    print_section_header("INITIALIZING TRANSLATION SYSTEM")
    kb_path="KB/code_rules.yaml"
    working_dir = "./compiler_temp"
    translator = CodeTranslationGraph(kb_path=kb_path, working_dir=working_dir)
    translator.max_iterations = 3
    print(f"Knowledge base: {kb_path}")
    print(f"Working directory: {working_dir}")
    print(f"Maximum iterations: {translator.max_iterations}")
    
    # Process request using the analyzer and translator
    print_section_header("PROCESSING TRANSLATION REQUEST")
    start_time = time.time()
    result = translator.process_request(user_input)
    total_time = time.time() - start_time
    
    # Display results in a structured format
    print_section_header("TRANSLATION RESULTS")
    print_data_table({
        "Source Language": result.get("source_language", "Unknown"),
        "Target Language": result.get("target_language", "Unknown"),
        "Processing Time": f"{total_time:.2f} seconds",
        "Iterations": result.get("iteration", 0),
        "Successful Compilations": result.get("successful_translations_count", 0)
    }, "Translation Summary")
    
    # show successful compilation information
    if result.get("successful_translations_count", 0) > 0:
        selected_iteration = result.get("selected_translation_iteration", 0)
        print(f"\nSelected translation from iteration #{selected_iteration} for final report")
        if "all_successful_translations" in result:
            print("\nSuccessful Compilations:")
            for idx, trans in enumerate(result["all_successful_translations"]):
                exec_time = trans.get("execution_time", "N/A")
                exec_time_str = f"{exec_time:.4f} seconds" if isinstance(exec_time, (int, float)) else "N/A"
                print(f"  Version {idx+1} (Iteration {trans['iteration']}): {exec_time_str}")

    print_section_header("TRANSLATED CODE")
    print(result.get("translated_code", "Error: No translated code"))
    
    if "error_log" in result:
        print_section_header("ERROR LOG")
        error_log = result.get("error_log", "No error log")
        if isinstance(error_log, dict):
            print_data_table(error_log)
        else:
            print(error_log)
    
    if "compilation_success" in result:
        print_section_header("COMPILATION RESULT")
        print(f"Compilation Success: {result.get('compilation_success', False)}")
            
        if result.get("compilation_success", False):
            print_section_header("EXECUTION OUTPUT")
            print(result.get("execution_output", "No output"))
                
            print_section_header("EXECUTION TIME")
            print(f"{result.get('execution_time_seconds', 0):.4f} seconds")
        else:
            print_section_header("COMPILATION ERRORS")
            print("\n".join(result.get("compilation_errors", ["Unknown error"])))
        
    # Print HPC analysis if available
    if "hpc_analysis" in result:
        print_section_header("HPC ANALYSIS")
        print(result["hpc_analysis"])
    
    # Save detailed report
    report_file = save_translation_report(result)
    
    # Restore stdout
    if isinstance(sys.stdout, Logger):
        sys.stdout.close()
        sys.stdout = sys.stdout.terminal
    
    print(f"\nTranslation process completed in {total_time:.2f} seconds")
    print(f"Terminal output log saved to: {log_file}")
    print(f"Detailed report saved to: {report_file}")
    
    return 0

if __name__ == "__main__":
    exit(main())