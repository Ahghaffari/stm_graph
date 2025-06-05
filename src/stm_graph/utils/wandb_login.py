import wandb


def authenticate_wandb(api_key: str) -> str:
    """
    Authenticate to Weights & Biases using the given API key.

    Args:
        api_key (str): Your W&B API key.

    Returns:
        str: "OK" if authentication succeeded, otherwise "ERROR: <message>".
    """
    try:
        wandb.login(key=api_key, relogin=True)
        return "OK"
    except Exception as e:
        return f"ERROR: {e}"


if __name__ == "__main__":
    key = input("Enter your W&B API key: ").strip()
    result = authenticate_wandb(key)
    if result == "OK":
        print("Authentication successful!")
    else:
        print(result)
