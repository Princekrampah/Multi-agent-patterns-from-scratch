import os
from openai import OpenAI
from dotenv import load_dotenv
from colorama import Fore
from rich.markdown import Markdown
from rich.console import Console


class ReflectionAgent:
    def __init__(
        self,
        generator_prompts,
        reflection_prompts,
        api_key=None,
        num_steps=1,
        model="gpt-4o"
    ):
        """
        generator_prompts: list of dicts, e.g. [{"role": "system", "content": "..."}]
        reflection_prompts: list of dicts, e.g. [{"role": "system", "content": "..."}]
        """
        load_dotenv()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.generator_prompts = list(generator_prompts)
        self.reflection_prompts = list(reflection_prompts)
        self.generated_code = None
        self.reflection_feedback = None
        self.improved_code = None
        self.num_steps = num_steps

    def generate_code(self, user_prompt):
        self.generator_prompts.append({"role": "user", "content": user_prompt})
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.generator_prompts,
        )
        self.generated_code = response.choices[0].message.content
        return self.generated_code

    def reflect_on_code(self):
        self.reflection_prompts.append({
            "role": "user",
            "content": f"Here is the code generated by the generator block: {self.generated_code}"
        })
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.reflection_prompts,
        )
        self.reflection_feedback = response.choices[0].message.content
        return self.reflection_feedback

    def improve_code(self):
        self.generator_prompts.append({
            "role": "user",
            "content": f"Here is the feedback from the reflection block: {self.reflection_feedback}"
        })
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.generator_prompts,
        )
        self.improved_code = response.choices[0].message.content
        return self.improved_code

    def display_markdown(self, content):
        console = Console()
        console.print(Markdown(content))

    def run(self, user_prompt, display_steps=True):
        """
        Runs the agent for a specified number of improvement steps.

        Args:
            user_prompt (str): The initial user prompt.
            display_steps (bool): Whether to display each step.
            num_steps (int): Number of improvement iterations to perform.
        """
        code = self.generate_code(user_prompt)
        if display_steps:
            print(Fore.CYAN + "Generated Code:")
            print(Fore.RESET + code)
        for step in range(self.num_steps):
            feedback = self.reflect_on_code()
            if display_steps:
                print(Fore.YELLOW + f"Reflection Feedback (Step {step+1}):")
                print(Fore.RESET + feedback)
            improved = self.improve_code()
            if display_steps:
                print(Fore.GREEN + f"Improved Code (Step {step+1}):")
                print(Fore.RESET + improved)
            # Prepare for next iteration
            self.generated_code = self.improved_code
        return self.improved_code
