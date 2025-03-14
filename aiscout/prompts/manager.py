class PromptManager:
    """Manages prompts for the Scout system, allowing customization and extension."""
    
    def __init__(self):
        self._custom_prompts = {}
        self._append_prompts = {}
        
    def set_prompt(self, prompt_type: str, prompt: str):
        """Replace the default prompt with a custom one.
        
        Args:
            prompt_type: Type of prompt to replace ('identify_objects', 'analyze_targets', 'refine_detections')
            prompt: The new prompt to use
        """
        self._custom_prompts[prompt_type] = prompt
        
    def append_to_prompt(self, prompt_type: str, additional_prompt: str):
        """Append additional instructions to the default prompt.
        
        Args:
            prompt_type: Type of prompt to append to ('identify_objects', 'analyze_targets', 'refine_detections')
            additional_prompt: Additional instructions to append
        """
        if prompt_type not in self._append_prompts:
            self._append_prompts[prompt_type] = []
        self._append_prompts[prompt_type].append(additional_prompt)
        
    def get_prompt(self, prompt_type: str, default_prompt: str) -> str:
        """Get the final prompt, either custom or default with any appended instructions.
        
        Args:
            prompt_type: Type of prompt to get
            default_prompt: The default prompt to use if no custom prompt is set
            
        Returns:
            The final prompt string
        """
        # If there's a custom prompt, use it instead of default
        base_prompt = self._custom_prompts.get(prompt_type, default_prompt)
        
        # If there are appended prompts, add them
        if prompt_type in self._append_prompts:
            appended = "\n\nAdditional Instructions:\n" + "\n".join(self._append_prompts[prompt_type])
            return base_prompt + appended
            
        return base_prompt
        
    def reset_prompt(self, prompt_type: str):
        """Reset a prompt type to its default by removing customizations.
        
        Args:
            prompt_type: Type of prompt to reset
        """
        self._custom_prompts.pop(prompt_type, None)
        self._append_prompts.pop(prompt_type, None)
        
    def reset_all(self):
        """Reset all prompts to their defaults."""
        self._custom_prompts.clear()
        self._append_prompts.clear()
