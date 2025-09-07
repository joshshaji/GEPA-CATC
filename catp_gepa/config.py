import json
from pathlib import Path
import pickle
from pydantic import BaseModel
from openai import OpenAI
from src.catpllm.data.plan_dataset import PlanDataset
from catp_gepa.dataset import CATPDataset, load_catp_dataset

project_root = Path(__file__).parent.parent

class Config(BaseModel):
    llm: dict
    dataset: str
    catp_seq_dataset: str
    catp_non_seq_dataset: str
    use_vanila_gepa: bool
    training_size: int
    test_size: int
    catp_alpha: float


def load_config() -> Config:
    config_path = Path(__file__).parent / "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    return Config(**config)


def get_llm(config: Config) -> OpenAI:
    provider = config.llm["provider"]
    model = config.llm["model"]
    if provider == "openai":
        # Please set API key in the environment variable OPENAI_API_KEY
        return OpenAI(model=model)
    else:
        raise ValueError(f"Unsupported provider: {provider}")
    

def get_catp_dataset(config: Config, seq: bool) -> PlanDataset:
    dataset_path = project_root / config.catp_seq_dataset if seq else project_root / config.catp_non_seq_dataset
    plan_pools = pickle.load(open(dataset_path, 'rb'))
    plan_dataset = PlanDataset(plan_pools, alpha=config.catp_alpha)
    return plan_dataset

def get_dataset(config: Config) -> CATPDataset:
    dataset_path = project_root / config.dataset
    return load_catp_dataset(dataset_path)