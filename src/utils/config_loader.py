import yaml
from pathlib import Path

def get_project_root() -> Path:
    """Returns the project root directory."""
    return Path(__file__).parent.parent.parent

def load_config(config_name: str = "config.yaml") -> dict:
    """
    Loads a YAML configuration file from the configs directory.

    Args:
        config_name (str): The name of the configuration file to load.

    Returns:
        dict: The loaded configuration as a dictionary.
    """
    project_root = get_project_root()
    config_path = project_root / "src" / "configs" / config_name
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_prompts(prompts_name: str = "prompts.yaml") -> dict:
    """
    Loads the prompts YAML file from the configs directory.

    Args:
        prompts_name (str): The name of the prompts file to load.

    Returns:
        dict: The loaded prompts as a dictionary.
    """
    prompts_data = load_config(prompts_name)
    return prompts_data if prompts_data is not None else {}

if __name__ == '__main__':
    # Example usage for testing the loader
    try:
        main_config = load_config()
        print("Configuration (config.yaml) loaded successfully:")
        print(main_config)

        prompts_config = load_prompts()
        print("\nPrompts (prompts.yaml) loaded successfully:")
        print(prompts_config)

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred: {e}")