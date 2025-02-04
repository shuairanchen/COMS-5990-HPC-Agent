import os 
from langchain.chains import LLMChain, TransformChain, SequentialChain
from transformers import OPTForCausalLM, AutoTokenizer, EncoderDecoderModel
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool
from typing import Dict, Any

# Environment setup
os.environ["OPENAI_API_KEY"] = "api-key"

# Code Analysis Module
def create_analysis_chain():
    """Create an LLM Chain for analyzing code requirements"""
    analysis_prompt = PromptTemplate(
        input_variables=["code"],
        template="""Analyze the following HPC code and determine the conversion requirements:
        [Code]
        {code}
        
        Please return the analysis results in JSON format, including:
        - source_language: Original programming language
        - target_framework: Target parallel framework
        - conversion_type: Type of conversion (e.g., cpp_to_cuda)
        - special_requirements: Special requirements (e.g., MPI handling, memory optimization)
        """
    )
    return LLMChain(
        llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
        prompt=analysis_prompt,
        output_key="analysis"
    )

# Conversion Routing Module
def conversion_router(inputs: Dict[str, Any]) -> Dict[str, Any]:
    analysis = eval(inputs["analysis"])
    conversion_tools = {
        "cpp_to_cuda": CodeRosettaConverter().cpp_to_cuda,
        "fortran_to_cpp": CodeRosettaConverter().fortran_to_cpp,
        "cpp_to_fortran": CodeRosettaConverter().cpp_to_fortran, 
        # "openmp_optimize": OpenMPOptimizer().optimize,
        "default": GeneralConverter().convert
    }
    
    # Select conversion tool
    tool = conversion_tools.get(
        analysis["conversion_type"], 
        conversion_tools["default"]
    )
    
    # Perform conversion
    converted_code = tool(inputs["code"])
    return {"converted_code": converted_code}

# Code Conversion Toolset
class CodeRosettaConverter:
    """A code converter integrating multiple models"""
    def __init__(self):
        # Preload all required models
        self.models = {
            'cpp_to_fortran': {
                'model': OPTForCausalLM.from_pretrained("facebook/opt-2.7b"), # need to change the finetuned model path
                'tokenizer': AutoTokenizer.from_pretrained("facebook/opt-2.7b"),
                'config': {
                    'max_length': 768,
                    'num_beams': 4,
                    'temperature': 0.7
                }
            },
            'fortran_to_cpp': {
                'model': EncoderDecoderModel.from_pretrained('CodeRosetta/CodeRosetta_fortran2cpp_ft'),
                'tokenizer': AutoTokenizer.from_pretrained('CodeRosetta/CodeRosetta_fortran2cpp_ft'),
                'start_token': "<CPP>",
                'config': {
                    'max_length': 512,
                    'num_beams': 5
                }
            },
            'cpp_to_cuda': {
                'model': EncoderDecoderModel.from_pretrained('CodeRosetta/CodeRosetta_cpp2cuda_ft'),
                'tokenizer': AutoTokenizer.from_pretrained('CodeRosetta/CodeRosetta_cpp2cuda_ft'),
                'start_token': "<CUDA>",
                'config': {
                    'max_length': 1024, 
                    'num_beams': 3
                }
            }
        }

    def cpp_to_cuda(self, code: str) -> str:
        """Convert C++ to CUDA using a pretrained model"""
        return self._convert_code(code, 'cpp_to_cuda')

    def fortran_to_cpp(self, code: str) -> str:
        """Convert Fortran to C++ using a pretrained model"""
        return self._convert_code(code, 'fortran_to_cpp')
    
    def cpp_to_fortran(self, code: str) -> str:
        """Convert C++ to Fortran using a fine-tuned model"""
        return self._convert_code(code, 'cpp_to_fortran')

    def _convert_code(self, code: str, conversion_type: str) -> str:
        """General conversion process"""
        config = self.models.get(conversion_type)
        if not config:
            raise ValueError(f"Unsupported conversion type: {conversion_type}")

        try:
            if 'EncoderDecoder' in str(type(config['model'])):
                # Encoder-Decoder architecture processing
                inputs = config['tokenizer'](
                    code, 
                    return_tensors="pt",
                    truncation=True,
                    max_length=config['config']['max_length']
                )
                outputs = config['model'].generate(
                    inputs.input_ids,
                    decoder_start_token_id=config['tokenizer'].convert_tokens_to_ids(config['start_token']),
                    **config['config']
                )
            else:
                # Causal language model processing
                prompt = self._build_prompt(code, conversion_type)
                inputs = config['tokenizer'](
                    prompt,
                    return_tensors="pt",
                    max_length=config['config']['max_length'],
                    truncation=True
                )
                outputs = config['model'].generate(
                    inputs.input_ids,
                    **config['config']
                )
            
            return self._postprocess(outputs, conversion_type, config)

        except Exception as e:
            print(f"{conversion_type} conversion failed: {str(e)}")
            return self._fallback_conversion(code, conversion_type)
        
    def _build_prompt(self, code: str, conversion_type: str) -> str:
        """Construct prompt templates for generative models"""
        prompt_templates = {
            'cpp_to_fortran': (
                "Convert the following C++ code to modern Fortran while preserving performance:\n"
                "Pay attention to:\n"
                "- Array indexing (1-based)\n"
                "- Memory layout (column-major)\n"
                "- Explicit interface definitions\n\n"
                "C++ code:\n{code}\n\n"
                "Fortran code:"
            )
        }
        return prompt_templates[conversion_type].format(code=code)

    def _postprocess(self, outputs, conversion_type: str, config: dict) -> str:
        """Unified post-processing procedure"""
        decoded = config['tokenizer'].decode(
            outputs[0], 
            skip_special_tokens=True
        )

        if conversion_type == 'cpp_to_cuda':
            headers = [
                "#include <cuda_runtime.h>",
                "#include <device_launch_parameters.h>\n"
            ]
            return '\n'.join(headers) + decoded
        
        if conversion_type == 'cpp_to_fortran':
            # Remove potential duplicate prompts
            decoded = decoded.split("Fortran code:")[-1].strip()
            # Add Fortran module header
            if "subroutine" in decoded and "module" not in decoded:
                decoded = f"module generated_code\nimplicit none\ncontains\n{decoded}\nend module"
        
        return decoded

    def _fallback_conversion(self, code: str, conversion_type: str) -> str:
        """Fallback conversion strategy"""
        print(f"Using GPT-3.5 for fallback {conversion_type} conversion")
        prompt_template = {
            'cpp_to_cuda': "Convert the following C++ code into a high-performance CUDA kernel:\n{code}",
            'fortran_to_cpp': "Convert the following Fortran code into modern C++:\n{code}"
        }
        prompt = PromptTemplate(
            input_variables=["code"],
            template=prompt_template.get(conversion_type, "Convert code:\n{code}")
        )
        return LLMChain(
            llm=ChatOpenAI(temperature=0.3, model="gpt-3.5-turbo"),
            prompt=prompt
        ).run(code)

class GeneralConverter:
    def convert(self, code: str) -> str:
        """General converter (using GPT-3.5)"""
        llm = ChatOpenAI(temperature=0.3, model="gpt-3.5-turbo")
        prompt = PromptTemplate(
            input_variables=["code"],
            template="Convert the following code into the target framework while maintaining high-performance characteristics:\n{code}"
        )
        return LLMChain(llm=llm, prompt=prompt).run(code)

# Code Quality Evaluation Module
def create_evaluation_chain():
    """Create a Chain for code quality evaluation"""
    eval_prompt = PromptTemplate(
        input_variables=["original", "converted"],
        template="""Evaluate the quality of the code conversion:
        [Original Code]
        {original}
        
        [Converted Code] 
        {converted}
        
        Please evaluate the following aspects (1-5 points):
        - Functional correctness
        - Performance retention
        - Readability
        - Parallel efficiency
        Finally, provide optimization suggestions. Display the evaluation results in a Markdown table."""
    )
    return LLMChain(
        llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
        prompt=eval_prompt,
        output_key="evaluation"
    )

# Pipeline
def build_pipeline():
    analysis_chain = create_analysis_chain()
    
    routing_chain = TransformChain(
        input_variables=["code", "analysis"],
        output_variables=["converted_code"],
        transform=conversion_router
    )
    
    evaluation_chain = create_evaluation_chain()
    
    return SequentialChain(
        chains=[analysis_chain, routing_chain, evaluation_chain],
        input_variables=["code"],
        output_variables=["analysis", "converted_code", "evaluation"],
        verbose=True
    )