import logging
import re

import boto3
import pandas as pd
from dotenv import load_dotenv
from llmbo import ModelInput, StructuredBatchInferer
from pydantic import BaseModel, Field

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[
        # logging.FileHandler("app.log"),  # Save logs to a file
        logging.StreamHandler(),  # Display logs in the terminal
    ],
)


SYSTEM_PROMPT = """
You are an expert in extracting skills from job descriptions and assessing their potential automation with GPT technology.
"""


def prompt(job_description):
    return f"""
    Extract a list of tasks from the job advert in the provided format. Do not include tasks which concern the recruitment process, the onboarding process, working hours, or working conditions.
    <job_title>{job_description}</job_title>
    """


class Task(BaseModel):
    task_number: int = Field(..., description="A unique task number")
    task_details: str = Field(
        ..., description="the text extract which describes the task"
    )
    exposure_score: float = Field(
        ...,
        description="a score of range 0-1 of potential automation of the task with GPT technology",
    )  # float, or literal/enum categories


class TaskOutput(BaseModel):
    tasks: list[Task] = Field(
        ..., description="A list of tasks extracted from job_advert"
    )


def clean_column_names(df):
    df.columns = [
        re.sub(r"[^a-zA-Z0-9_]+", "_", col)
        .lower()
        .strip("_")
        .replace("vacancy_form_", "")
        for col in df.columns
    ]

    return df


def load_department_jobs(department: str) -> pd.DataFrame:
    jobs_data = pd.read_excel("data/export (27).xlsx")
    jobs_data = clean_column_names(jobs_data)
    department_data = jobs_data.loc[jobs_data["department"] == department].copy()
    department_data["full_job_description"] = (
        department_data["job_summary"] + department_data["job_description"]
    )

    return department_data


def skewer_department(department_name: str) -> str:
    """
    turn a department kebabcase

    Args:
        department_name (str): name of a department

    Returns:
        str: name of the department kebabcase
    """
    return re.sub(r"[^a-zA-Z0-9_]+", "-", department_name).lower().strip("-")


def create_job(
    department_data: pd.DataFrame,
    department_name: str,
    job_id: str,
    session: boto3.Session,
) -> str:
    """
    create a task extraction batch job

    Args:
        department_data (pd.DataFrame): a dataframe container vacancy_id and job_description
        department_name (str): name of the department (must be kebabcase)
        job_id (str): an id to associate with the job
        session (boto3.Session): a session to use

    Returns:
        str: created job arn
    """
    inputs = {
        f"vacancy_id={row.vacancy_id}": ModelInput(
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt(row.job_description)}],
        )
        for row in department_data.itertuples()
    }

    sbi = StructuredBatchInferer(
        model_name="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        job_name=f"{department_name}-task-extraction-{job_id}",
        region="us-east-1",
        bucket_name="cddo-af-bedrock-batch-inference-us-east-1",
        role_arn="arn:aws:iam::992382722318:role/BatchInferenceRole",
        output_model=TaskOutput,  # type: ignore
        session=session,
    )

    sbi.prepare_requests(inputs)
    sbi.push_requests_to_s3()
    sbi.create()
    return sbi.job_arn


# def analyse_job():
#     sbi = StructuredBatchInferer.recover_structured_job(
#         job_arn="arn:aws:bedrock:us-east-1:992382722318:model-invocation-job/7whskxnh15zc",
#         region="us-east-1",
#         output_model=TaskOutput,
#     )

#     sbi.download_results()
#     sbi.load_results()

#     print(len(sbi.instances))


def convert_tasks_output_to_dataframe(tasks):
    tasks_dict = {}
    for item in tasks:
        if item:
            # Extract vacancy_id from the recordId string
            vacancy_id = int(item["recordId"].split("=")[1])
            # Get the tasks from the outputModel
            if item["outputModel"]:
                tasks = item["outputModel"].model_dump()["tasks"]
                tasks_dict[vacancy_id] = tasks

    return pd.DataFrame(
        [
            {"vacancy_id": key, **item}
            for key, tasks in tasks_dict.items()
            for item in tasks
        ]
    ).set_index("vacancy_id")


if __name__ == "__main__":
    pass
    # analyse_job()
