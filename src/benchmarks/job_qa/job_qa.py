from inspect_ai.dataset import csv_dataset, FieldSpec
from inspect_ai import Task, task
from inspect_ai.solver import generate, prompt_template
from inspect_ai.scorer import model_graded_fact
from inspect_ai.model import GenerateConfig


# In this one, we will add CoT as an optional parameter that we can add to the plan.


@task
def job_qa_task(
    shuffle=True,
    CoT=False,
    prompt_directory=r"./default_prompt.txt",
    dataset=r"./dataset.csv",
):
    with open(prompt_directory, "r") as f:
        PROMPT_TEMPLATE = f.read()

    plan = [
        prompt_template(
            PROMPT_TEMPLATE,
        ),
        generate(),
    ]

    if CoT:
        middle_index = len(plan) // 2
        plan = plan[:middle_index] + [CoT] + plan[middle_index:]
    return Task(
        dataset=csv_dataset(
            dataset,
            FieldSpec(input="JobDescription", metadata=["Question"]),
        ),
        shuffle=shuffle,
        plan=plan,
        scorer=model_graded_fact(),
        config=GenerateConfig(temperature=0),
    )
