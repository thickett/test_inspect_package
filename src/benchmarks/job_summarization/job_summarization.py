from inspect_ai.dataset import csv_dataset, FieldSpec
from inspect_ai import Task, task
from inspect_ai.solver import generate, prompt_template
from inspect_ai.scorer import model_graded_fact
from inspect_ai.model import GenerateConfig


@task
def job_summary_task(
    shuffle=True,
    prompt_directory=r"./default_prompt.txt",
    dataset=r"./dataset.csv",
):
    with open(prompt_directory, "r") as f:
        PROMPT_TEMPLATE = f.read()
    return Task(
        dataset=csv_dataset(  # Inspect have an inbuilt function to format csvs in the required format
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
        scorer=model_graded_fact(),  # the evaulation function, 'model_graded_fact()' is llm-as-judge'.
        config=GenerateConfig(
            temperature=0
        ),  # model config, we could pass this as a parameter, of course.
    )
