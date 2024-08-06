from inspect_ai.dataset import csv_dataset, FieldSpec
from inspect_ai import Task, task
from inspect_ai.solver import generate, prompt_template
from inspect_ai.scorer import model_graded_fact
from inspect_ai.model import GenerateConfig

CUSTOM_GRADER_INSTRUCTIONS = f"""
After assessing the submitted answer, reply with 'GRADE: $NUMBER' (without quotes) where NUMBER is one of (1,2,3,4,5).  Please choose ONE option for the grade: The grade scale is a Likert scale, where 1 Is a terrible answer, and 5 is a near perfect answer. If the answer is not a number between 1 and 5, the grade will be considered invalid.

For example, after reviewing an amazing answer you might write 'GRADE: 5' or after reviewing a good answer that is missing some key points you might write 'GRADE: 3'.

First, write out in a step by step manner your reasoning about the criterion to be sure that your conclusion is correct. Avoid simply stating the correct answers at the outset. Then, end with your answer formatted as 'GRADE: $NUMBER' (without quotes) where NUMBER is one of (1,2,3,4,5).
"""
CUSTOM_GRADER_REGEX = r"(?i)GRADE\s*:\s*([1-5])"


@task
def job_summary_task(
    shuffle=True,
    prompt_directory=r"./default_prompt.txt",
    dataset=r"./dataset.csv",
    grader_model=None,
):
    with open(prompt_directory, "r") as f:
        PROMPT_TEMPLATE = f.read()
    return Task(
        dataset=csv_dataset(  # Inspect has an inbuilt function to format csvs in the required format
            dataset,
            FieldSpec(
                input="job_details"
            ),  # datasets natively expect an 'input' column as a minimum
            # we can use FieldSpec to re-define this, along with others
            # (i.e target, metadata etc)
        ),
        shuffle=shuffle,  # if True, data is randomly shuffled - good during development, or when testing prompts.
        plan=[
            prompt_template(PROMPT_TEMPLATE),
            generate(),
        ],  # order of operations, they have a selection
        # of inbuilts, but its easy to add custom ones.
        scorer=model_graded_fact(
            model="openai/gpt-4",
            instructions=CUSTOM_GRADER_INSTRUCTIONS,
            grade_pattern=CUSTOM_GRADER_REGEX,
        ),  # the evaulation function, 'model_graded_fact()' is llm-as-judge'.
        config=GenerateConfig(
            temperature=0
        ),  # model config, we could pass this as a parameter, of course.
    )
