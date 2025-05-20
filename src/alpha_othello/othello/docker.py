import time
import subprocess
import tempfile
from pathlib import Path


def start_docker(name: str, image: str, memory_limit: int = 512, cpu_limit: int = 1):
    """Start a Docker container with the specified name and resource limits."""
    subprocess.run(
        [
            "docker",
            "run",
            "-it",
            "--rm",
            "--name",
            name,
            "--memory",
            f"{memory_limit}m",
            "--cpus",
            str(cpu_limit),
            "-d",
            image,
        ],
        check=True,
    )

    # Wait for the container to be ready
    while True:
        try:
            result = subprocess.run(
                ["docker", "inspect", "--format", "{{.State.Running}}", name],
                check=True,
                capture_output=True,
            )
            if result.stdout.strip() == b"true":
                break
        except subprocess.CalledProcessError:
            # Container not found or not started yet
            pass
        print("Waiting for Docker container to be ready...")
        time.sleep(1)


def stop_docker(name: str):
    """Stop the Docker container with the specified name."""
    subprocess.run(
        ["docker", "stop", name],
        check=True,
    )


def play_in_docker(
    ai1_code: str,
    ai2_code: str,
    n_games: int = 100,
    size: int = 6,
    time_control_millis: int = 20,
    memory_limit: int = 512,
    cpu_limit: int = 1,
) -> dict[tuple[str, str], int]:
    """Play Othello between two AIs in a Docker container."""
    eval_code_path = Path("src/alpha_othello/othello/eval.txt")
    with open(eval_code_path, "r") as f:
        eval_code = f.read()

    # Replace the AI code in the eval code
    eval_code = eval_code.replace("<AI_1>", ai1_code)
    eval_code = eval_code.replace("<AI_2>", ai2_code)

    docker_image = "python-othello"
    docker_container_name = "othello_eval_container"

    # Create a temporary directory to store the eval code
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_eval_code_path = Path(temp_dir) / "eval.py"
        with open(temp_eval_code_path, "w") as f:
            f.write(eval_code)

        # Start the Docker container
        start_docker(docker_container_name, docker_image, memory_limit, cpu_limit)
        # Copy the eval code to the Docker container
        subprocess.run(
            [
                "docker",
                "cp",
                str(temp_eval_code_path),
                f"{docker_container_name}:/app/eval.py",
            ],
            check=True,
        )

        # Run the eval code in the Docker container
        try:
            result = subprocess.run(
                [
                    "docker",
                    "exec",
                    docker_container_name,
                    "python",
                    "/app/eval.py",
                    str(n_games),
                    str(size),
                    str(time_control_millis),
                ],
                check=True,
                capture_output=True,
            )
        except Exception as e:
            stop_docker(docker_container_name)
            raise e

    # Stop the Docker container
    stop_docker(docker_container_name)

    # Check for errors
    if result.returncode != 0:
        raise RuntimeError(f"Docker run failed: {result.stderr}")

    # Parse the result
    output = result.stdout.strip().decode("utf-8")
    results = {}
    for line in output.splitlines():
        result, reason, count = line.split(",")
        results[(result, reason)] = int(count)

    return results
