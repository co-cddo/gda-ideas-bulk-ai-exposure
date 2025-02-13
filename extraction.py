import logging
import re
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from llmbo import BatchInferer, ModelInput, StructuredBatchInferer
from pydantic import BaseModel, Field

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[
        # logging.FileHandler("app.log"),  # Save logs to a file
        logging.StreamHandler(),  # Display logs in the terminal
    ],
)

load_dotenv()

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


def create_job() -> None:
    hmrc_data = load_department_jobs("HM Revenue and Customs")

    inputs = {
        f"vacancy_id={row.vacancy_id}": ModelInput(
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt(row.job_description)}],
        )
        for row in hmrc_data.itertuples()
    }

    sbi = StructuredBatchInferer(
        model_name="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        job_name=f"hmrc-task-extraction-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        region="us-east-1",
        bucket_name="cddo-af-bedrock-batch-inference-us-east-1",
        role_arn="arn:aws:iam::992382722318:role/BatchInferenceRole",
        output_model=TaskOutput,  # type: ignore
    )

    sbi.prepare_requests(inputs)
    sbi.push_requests_to_s3()
    sbi.create()
    print(sbi.job_arn)


def analyse_job():
    sbi = StructuredBatchInferer.recover_structured_job(
        job_arn="arn:aws:bedrock:us-east-1:992382722318:model-invocation-job/7whskxnh15zc",
        region="us-east-1",
        output_model=TaskOutput,
    )

    sbi.download_results()
    sbi.load_results()

    print(len(sbi.instances))


if __name__ == "__main__":
    analyse_job()
