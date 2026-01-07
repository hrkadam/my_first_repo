import os

print("Hello Python")

def add(a,b,c):
    v=a+b+c
    return v

def rem(a,b,c):
    v=a-b-c
    return v

print("line added")

def mul(a,b,c):
    v=a*b*c
    return v

print("multiplication function")

print("list added")
x=[4,5,6,3,2,1]
print(len(x))

def add_two(a,b):
    return a+b

ls=[4,5,6,3,2,1]
print("last line")

list_vals = [4,5,6,3,2,1]
print("final last line")

new_list = [6,2,1,0,9]
print("really fixed?")


def tiny(a, b):
    return a + b

hw_list = [4,5,6,3,2,1]
print("testing end")

temp_list=[9,6,3,2]
print("finish it!!")

def multi123(x,y):
    t=x*y
    return t

print("new date")
ls_new=[1,2,4]

def win_add(x,y,z):
    print("making addition")
    return x+y+z

print("End of script")

def rem_fuc(x,y,z):
    print("making multiplication")
    return x*y*z

print("Finally script ending")

def mul_three(x,y,z):
    print("making multiplication")
    return x*y*z

print("line 70 end")

def mul_next(x,y,z):
    print("making multiplication")
    return x*y*z

print("line 80 end")

prime_nums = []
def prime_finder(max_num):
    for num in range(2,max_num):
        is_prime = True
        for i in range(2, int(num**0.5) + 1):
            if num % i == 0:
                is_prime = False
                break
        if is_prime:
            prime_nums.append(num)
    return prime_nums

prime_nums_cnt = prime_finder(50)
print('Total prime numbers count = ',len(prime_nums_cnt))


def check_even_odd(num):
    if num % 2 == 0:
        return "Even"
    else:
        return "Odd"

# Example
print(check_even_odd(7))

import json
import logging
import random
import time
import traceback
import uuid
from decimal import Decimal
from functools import reduce
from logging import Logger
from typing import List, Iterator, Tuple, cast, Callable, TypeVar

from google import genai
from google.genai.types import GenerateContentConfigDict, Part, Content

import api_utils
import auditing
import custom_rbac
from api_validation import validate_request_body
from code_profiler import CodeProfiler
from config import (
    CLOUD_PROVIDERS,
    WHITESPACE_RATIO,
    DOMAINS,
    SAFETY_SETTINGS,
    SAFETY_SETTINGS_STRICT,
    IS_PROD,
    AI_NAME,
    LONG_CONTEXT_THRESHOLD,
    REGEX_MULTI_NL,
    EMPTY_LIST,
    DEFAULT_MODEL_CONFIG,
    DEFAULT_MODEL,
    DEFAULT_QUERY_TYPE,
    MILLION,
)
from custom_system_instructions import (
    get_system_instructions_potential_questions,
    get_system_instructions_suggest_reports,
    get_system_instructions_api_payload,
    get_system_instructions_casual_interactions,
    get_system_instructions_analyse_query,
    get_system_instructions_analyse_recommend,
)
from my_types import (
    QueryType,
    LLModel,
    ModelConfig,
    QueryDomain,
    CloudProvider,
)
from pricing import (
    PRICE_PER_INPUT_SMALL_CONTEXT,
    PRICE_PER_OUTPUT_SMALL_CONTEXT,
    PRICE_PER_OUTPUT_LARGE_CONTEXT,
    PRICE_PER_INPUT_LARGE_CONTEXT,
    CHARS_PER_TOKEN,
)
from utility_functions import (
    strip_invalid_links,
    parse_llm_json_responses,
    get_valid_links,
    get_examples,
)

try:
    from flask import Request
except ModuleNotFoundError:
    from unittest.mock import MagicMock

    Request = MagicMock()

from api_utils import HttpStatus, set_content_type, validate_jwt
import cors_config
import custom_logger
from custom_types import HandlerResponse


T = TypeVar("T")


def with_retry(request_logger: logging.Logger, fn: Callable[[], T], n=5) -> T:
    error = Exception(f"Failed to execute query, exceeded {n} tries")
    for i in range(1, n + 1):
        try:
            return fn()
        except Exception as e:
            error = e
            request_logger.error("%s %s", str(type(e)), str(e))
            duration = 2 ** ((i + random.random()) / 2)
            request_logger.warning(
                "sleeping for %f seconds (retry %d/%d)", duration, i, n
            )
            time.sleep(duration)
    raise error


@api_utils.time_cache(max_age_seconds=300)
def get_client(location: str, project_id: str):
    return genai.Client(
        vertexai=True,
        project=project_id,
        location=location,
    )


def get_random_region() -> str:
    """
    We load balance the requests across multiple regions to avoid hitting quota limits.

    There is a "global" endpoint but this endpoint does not provide us the same data protection assurances. When using the global endpoint, the data might be sent to the US.
    """
    return random.choice(
        sorted(
            {
                "europe-west1",
                "europe-central2",
                "europe-north1",
                "europe-west4",
                "europe-west8",
                "europe-west9",
            }
        )
    )


def parse_into_contents_v2(strings: List[str]) -> List[Content]:
    return [
        Content(
            parts=[Part(text=s)],
            role="user" if s.lstrip().startswith("User>") else "model",
        )
        for s in strings
    ]


def generate(
    tenant: str,
    prompt: str,
    history: List[str],
    model: LLModel,
    query_type: QueryType,
    model_config: ModelConfig,
    cloud_providers: Tuple[CloudProvider, ...],
    domains: Tuple[QueryDomain, ...],
    logger: Logger,
    project_id: str,
    cp: CodeProfiler,
) -> Iterator[Tuple[str, Decimal, List[str]]]:
    json_mode: bool

    if query_type == "analyse_query":
        system_instruction = cp.run(
            lambda: get_system_instructions_analyse_query(tenant, prompt.lower()),
            "get system instructions: analyse query",
        )
        json_mode = True

    elif query_type == "potential_questions":
        system_instruction = cp.run(
            lambda: get_system_instructions_potential_questions(
                tenant, prompt.lower(), cloud_providers
            ),
            "get system instructions: potential questions",
        )
        json_mode = True

    elif query_type == "suggest_reports":
        system_instruction = cp.run(
            lambda: get_system_instructions_suggest_reports(
                tenant,
                prompt.lower(),
                cloud_providers,
            ),
            "get system instructions: suggest reports",
        )
        json_mode = True

    elif query_type == "casual_interactions":
        system_instruction = cp.run(
            lambda: get_system_instructions_casual_interactions(
                tenant,
                prompt.lower(),
                cloud_providers,
            ),
            "get system instructions: casual interactions",
        )
        json_mode = False

    elif query_type == "analyse_recommend":
        system_instruction = cp.run(
            lambda: get_system_instructions_analyse_recommend(
                tenant,
                prompt.lower(),
                cloud_providers,
            ),
            "get system instructions: analyse recommend",
        )
        json_mode = False

    elif query_type == "api_payload":
        system_instruction = cp.run(
            lambda: get_system_instructions_api_payload(
                tenant, prompt.lower(), cloud_providers, domains
            ),
            "get system instructions: api payload",
        )
        json_mode = True

    else:
        raise Exception("Unrecognised query type: " + str(query_type))

    system_instruction: List[str] = [
        REGEX_MULTI_NL.sub("\n\n", s) for s in system_instruction
    ]

    # there are 2 prices, for context window above 128K and for context below 128K
    # above the threshold we get charged more
    price_per_input: Decimal = PRICE_PER_INPUT_SMALL_CONTEXT.get(model) or Decimal(0)
    price_per_output: Decimal = PRICE_PER_OUTPUT_SMALL_CONTEXT.get(model) or Decimal(0)

    # white space is not part of the context count, we can do some simple math to remove the whitespace from the total
    # count by multiplying by 0.85
    system_instruction_len: Decimal = (
        reduce(lambda x, y: x + y, (Decimal(len(i)) for i in system_instruction))
        * WHITESPACE_RATIO
    ).to_integral()

    if system_instruction_len > LONG_CONTEXT_THRESHOLD:
        price_per_input = PRICE_PER_INPUT_LARGE_CONTEXT.get(model) or price_per_input

    yield "", (
        (system_instruction_len / CHARS_PER_TOKEN / MILLION)
    ) * price_per_input, system_instruction

    generation_config: GenerateContentConfigDict = (
        dict(**DEFAULT_MODEL_CONFIG) | model_config
    )

    if json_mode:
        generation_config["response_mime_type"] = "application/json"

    generation_config["system_instruction"] = system_instruction
    generation_config["safety_settings"] = (
        SAFETY_SETTINGS if json_mode else SAFETY_SETTINGS_STRICT
    )

    text: str = cp.run(
        lambda: with_retry(
            logger,
            lambda: cp.run(
                lambda: get_client(get_random_region(), project_id), "get client"
            )
            .models.generate_content(
                contents=cp.run(
                    lambda: parse_into_contents_v2(
                        history + [f"User> {prompt}", f"{AI_NAME}> "]
                    ),
                    "parse content into contents",
                ),
                config=generation_config,
                model=model,
            )
            .text,
            n=3,
        ),
        "generate text with LLM",
    )

    # white space is not part of the context count, we can do some simple math to remove the whitespace from the total
    # count by multiplying by 0.85
    # this logic only makes sense for text responses
    output_len = (
        Decimal(len(text or ""))
        * (Decimal(1) if json_mode else WHITESPACE_RATIO).to_integral()
    )

    if output_len > LONG_CONTEXT_THRESHOLD:
        price_per_output = PRICE_PER_OUTPUT_LARGE_CONTEXT.get(model) or price_per_output

    yield text, (output_len / CHARS_PER_TOKEN / MILLION) * price_per_output, EMPTY_LIST


def process_candidate(
    is_main: bool,
    log: Logger,
    tenant: str,
    history: List[str],
    query_type: QueryType,
    prompt: str,
    cloud_providers: Tuple[CloudProvider, ...],
    domains: Tuple[QueryDomain, ...],
    model: LLModel,
    model_config: ModelConfig,
    project_id: str,
    cp: CodeProfiler,
):
    total_cost = Decimal(0)
    response = ""
    system_instructions_total: List[str] = []

    log.info("query type: %s", query_type)
    log.debug("user prompt: %s", prompt)
    log.info("model: %s", model)

    for txt, cost, system_instructions in generate(
        tenant,
        prompt,
        history,
        model,
        query_type,
        model_config,
        cloud_providers,
        domains,
        log,
        project_id,
        cp,
    ):
        total_cost += cost
        response += txt
        system_instructions_total.extend(system_instructions)

    log.info("cost of query: %s", str(total_cost))
    log.info("generated text: %s", response)

    if query_type in ("casual_interactions", "analyse_recommend"):
        response = cp.run(
            lambda: strip_invalid_links(log, response), "strip invalid links"
        )

    elif query_type == "suggest_reports":
        valid_links = get_valid_links()
        response: str = cp.run(
            lambda: json.dumps(
                [
                    l
                    for l in cast(
                        List[str], parse_llm_json_responses(response, "[", "]")
                    )
                    if l in valid_links
                ]
            ),
            "JSONify suggest reports response",
        )

    response_body = {"Response": response}

    if not IS_PROD:
        response_body["Cost"] = str(total_cost)
        if is_main:
            response_body["SystemInstructions"] = system_instructions_total

    return response_body


class SpecialHandler(logging.Handler):

    def __init__(self) -> None:
        self.logs: List[logging.LogRecord] = []
        logging.Handler.__init__(self=self)

    def emit(self, record: logging.LogRecord) -> None:
        self.logs.append(record)


@auditing.audit_request("/queries/llm")
@set_content_type()
@cors_config.set_cors_headers
@validate_request_body("/queries/llm")
@validate_jwt
@custom_logger.add_logging
def main(request: Request) -> HandlerResponse:
    """
    Processes all types of LLM queries from the frontend and returns the response.

    1. potential_questions - suggests potential questions based on the prompt
    2. suggest_reports - suggests reports based on the prompt
    3. casual_interactions - general chat with the LLM
    4. analyse_query - analyses the query and provides insights
    5. analyse_recommend - analyses and provides recommendations
    6. api_payload - returns structured data based on the prompt

    """
    try:
        request_logger = custom_logger.get_logger(request)
        if not IS_PROD:
            log_handler = SpecialHandler()
            request_logger.addHandler(log_handler)

        cp = CodeProfiler(request_logger)

        payload = request.json

        tenant = request.headers.get("X-Vf-Api-Tenant", "vodafone")
        request_logger.info("tenant: %s", tenant)

        # used to limit the amount of information injected into the prompt (system instructions)
        cloud_providers: Tuple[CloudProvider, ...] = tuple(
            sorted(
                [
                    csp_id
                    for csp_id in (
                        (
                            payload.get("CloudProviders")
                            if payload.get("CloudProviders") is not None
                            else list(CLOUD_PROVIDERS.keys())
                        )
                    )
                    if csp_id in CLOUD_PROVIDERS
                ]
            )
        )
        request_logger.info("cloud providers: %s", ", ".join(cloud_providers))

        # used to limit the amount of information injected into the prompt (system instructions)
        domains: Tuple[QueryDomain, ...] = tuple(
            sorted(
                [
                    d
                    for d in (
                        payload.get("Domains")
                        if payload.get("Domains") is not None
                        else list(DOMAINS.keys())
                    )
                    if d in DOMAINS
                ]
            )
        )
        request_logger.info("domains: %s", ", ".join(domains))

        prompt = str((payload or {}).get("Prompt"))
        model = payload.get("Model") or DEFAULT_MODEL
        query_type = payload.get("QueryType") or DEFAULT_QUERY_TYPE
        model_config: ModelConfig = payload.get("ModelConfig") or {}
        history = payload.get("History") or []

        user_details: custom_rbac.UserDetails = cp.run(
            lambda: with_retry(
                request_logger,
                lambda: custom_rbac.get_user(
                    request.headers.get("X-Forwarded-Authorization"),
                    tuple(),
                    "/",
                    request.headers.get("X-Vf-Api-Tenant", None),
                    ("tenant_details",),
                ),
                n=2,
            ),
            "get tenant details",
        )

        response_body = process_candidate(
            True,
            request_logger,
            tenant,
            history,
            query_type,
            prompt,
            cloud_providers,
            domains,
            model,
            model_config,
            (user_details.get("tenant_details") or {}).get("GcpProjectId"),
            cp,
        )

        response_body["MessageId"] = str(uuid.uuid4())

        if not IS_PROD:
            response_body["Logs"] = cp.run(
                lambda: [
                    "{level} {message}".format(level=r.levelname, message=r.message)
                    for r in log_handler.logs
                ],
                "format logs",
            )
            response_body["TotalCost"] = cp.run(
                lambda: str(
                    reduce(
                        lambda x, y: x + y,
                        [Decimal(n) for n in [response_body.get("Cost", "0")]],
                        Decimal("0"),
                    )
                ),
                "compute total cost",
            )

        return (
            cp.run(lambda: json.dumps(response_body), "JSONify response body"),
            HttpStatus.OK.value,
            {},
        )
    except Exception as e:
        print(traceback.format_exc())
        return (
            json.dumps({"Response": str(type(e)) + ": " + str(e)}),
            HttpStatus.INTERNAL_SERVER_ERROR.value,
            {},
        )


if __name__ == "xxx":
    from unittest.mock import MagicMock
    from os import environ

    environ["DEBUGGING_LOCALLY"] = "1"

    class MyMock:
        headers = {
            "X-Forwarded-Authorization": "Bearer xxx",
        }
        method = "POST"
        query_string = ""
        is_json = True
        json = {
            "Prompt": "Show me cost per azure subscription",
            "History": [
                "VCloudSmart UI> The date is: 29/07/2025. ISO timestamp: 2025-07-29.",
                "VCloudSmart UI> The user has permissions in the following 12 tenancies: Anomaly-Detection, CCE, PCS API Integrations, VCI, VF AL, VGSL, brm, evo, looker, nucleus, pcs, vams. The user has Read permissions to all accounts in aws. The user has Read permissions to all projects in gcp. The user has Read permissions to all subscriptions in azure. The user has Read permissions to all compartments in drcc. The user has Read permissions to all compartments in oci",
                "TOBi> Hi there 👋 ! I am TOBi, your friendly AI assistant\n\n here to help you understand and optimize your cloud ☁️ usage in:\n\n- Amazon Web Services (AWS)\n\n- Google Cloud Platform (GCP)\n\n- Microsoft Azure\n\n- Oracle Cloud Infrastructure Dedicated Customer Region (DRCC)\n\nI can help you to visualize 📈 and understand your:\n\n- Cloud costs 💵\n- Resources\n\n 📦\n- Potential savings 💡\n\nI also explain concepts and best practices in cloud, DevOps\n\n, and FinOps\n\n, and compare services between cloud providers. You can ask me which pages (reports) have the information you need and you can also ask about the current VCloudSmart report and the data displayed on the screen. \n\n Think of me as personal guide to the world of cloud computing\n\n 😊!",
                "User> Show me cost per azure subscription",
            ],
            "Model": "gemini-2.0-flash-lite-001",
            "ModelConfig": {
                "max_output_tokens": 1000,
                "top_p": 0.93,
                "temperature": 1.35,
            },
            "CloudProviders": ["azure"],
            "Domains": ["billing"],
            "QueryType": "api_payload",
        }

    m = MyMock()

    # response = main(m)
    #
    # response_body = json.loads(response[0])
    #
    # print(response_body["SystemInstructions"][0])
    # print(response_body["Response"])
    # print(response_body["TotalCost"])

    examples = get_examples()
    print(
        json.dumps(
            [
                e
                for e in examples
                if (e.get("Query") or {}).get("CloudProvider") == "azure"
                and (e.get("Query") or {}).get("DataStore") == "RecommendationsV2"
            ],
            indent=2,
        )
    )

    print(22)
 