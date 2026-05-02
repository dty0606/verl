from __future__ import annotations

import importlib.util
import logging
import os
import sys

from omegaconf import OmegaConf

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def get_interaction_class(cls_name: str):
    """Dynamically import and return the configured interaction class."""

    module_name, class_name = cls_name.rsplit(".", 1)
    if module_name not in sys.modules:
        spec = importlib.util.find_spec(module_name)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import interaction module: {module_name}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    else:
        module = sys.modules[module_name]

    return getattr(module, class_name)


def initialize_interactions_from_config(interaction_config_file: str):
    """Initialize interactions from an OmegaConf YAML file."""

    interaction_config = OmegaConf.load(interaction_config_file)
    interaction_map = {}

    for interaction_item in interaction_config.interaction:
        cls_name = interaction_item.class_name
        interaction_cls = get_interaction_class(cls_name)
        config = OmegaConf.to_container(interaction_item.config, resolve=True)

        name = interaction_item.get("name", None)
        if name is None:
            class_simple_name = cls_name.split(".")[-1]
            name = class_simple_name[:-11].lower() if class_simple_name.endswith("Interaction") else class_simple_name.lower()

        if name in interaction_map:
            raise ValueError(f"Duplicate interaction name '{name}' found.")

        config["name"] = name
        interaction_map[name] = interaction_cls(config=config)
        logger.info("Initialized interaction '%s' with class '%s'", name, cls_name)

    return interaction_map
