class LLMTestConfig:
    def __init__(self, model_name: str, max_iterations: int, patience: int, context_type: str,
                 stop_on_repeat: bool = True, data_url_or_file_path: str = None, 
                 system_prompt: str = None, embedding_model_name: str = None, 
                 context_file: str = None, reasoning_factor: str = "medium", 
                 is_reasoning: bool = True, temperature: float = 1):
        self.model_name = model_name.lower()  # Name of the LLM (e.g., 'GPT-4')
        self.max_iterations = max_iterations  # Maximum number of iterations the LLM can take
        self.patience = patience  # Patience level (how many times the output should repeat before stopping)
        self.stop_on_repeat = stop_on_repeat  # Whether to stop if the output repeats
        self.data_url_or_file_path = data_url_or_file_path  # Data URL or file path for RAG
        self.embedding_model_name = embedding_model_name  # Optional, custom embedding model name
        self.reasoning_factor = reasoning_factor.lower()  # Normalize input to lowercase
        self.context_type = context_type.lower()
        self.is_reasoning = is_reasoning  # Boolean flag for reasoning mode or not
        self.temperature = temperature  # Temperature value for response creativity

        # Validate reasoning factor
        if self.reasoning_factor not in ["high", "medium", "low"]:
            raise ValueError("reasoning_factor must be 'high', 'medium', or 'low'.")

        # Validate temperature (should be between 0 and 1)
        if not (0 <= self.temperature <= 1):
            raise ValueError("Temperature must be between 0 and 1.")

        # Store context file name without opening
        self.context_file = context_file  

        # Adjust system prompt based on reasoning factor
        reasoning_prompts = {
            "high": "Provide a detailed, step-by-step explanation with rigorous justification.",
            "medium": "Give a balanced explanation with key reasoning points.",
            "low": "Provide a brief, high-level response with minimal details."
        }
        
        self.system_prompt = system_prompt or (
            f"You are an expert in molecular simulations in chemistry and energy calculations for molecules. "
            f"Use the retrieved context (if available) to answer the question.\n\n"
            f"{reasoning_prompts[self.reasoning_factor]}"
        )  # Default system prompt if not provided

# Example usage
llm_config = LLMTestConfig(
    model_name = "o1",           # Model name
    max_iterations=3,              # Maximum number of iterations
    patience=2,                     # Patience level for stopping condition
    stop_on_repeat=True,            # Whether to stop if the output repeats
    data_url_or_file_path="https://www.nist.gov/mml/csd/chemical-informatics-group/spce-water-reference-calculations-non-cuboid-cell-10a-cutoff",  # Data URL or file path for RAG
    embedding_model_name="text-embedding-3-large",  # Custom embedding model (if desired)
    reasoning_factor="medium",        # Reasoning level: "high", "medium", or "low"
    context_type="short",           # Context mode: "short", "medium", or "large"
    context_file="../context/shorter_context.txt",  # Context file (only storing name, not opening)
    is_reasoning=True,  # Boolean flag indicating reasoning mode
)
