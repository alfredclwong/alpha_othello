import subprocess
import tempfile
import time
from abc import ABC, abstractmethod
from pathlib import Path

from othello.types import T_PLAYER_FN, Player

from alpha_othello.llm import extract_tagged_text
from alpha_othello.othello.ai import get_function_source


class Evaluator(ABC):
    @abstractmethod
    def evaluate(self, *args, **kwargs) -> int:
        return 0


class OthelloDockerEvaluator(Evaluator):
    # TODO enforce strict and consistent memory and cpu limits
    def __init__(
        self,
        name: str,
        docker_image: str,
        memory_limit: str,
        cpu_limit: str,
        ais: list[T_PLAYER_FN],
        eval_script_path: Path,
        win_rate_threshold: float = 0.9,
        n_games: int = 100,
        size: int = 6,
        time_limit_ms: int = 20,
    ):
        # TODO multiple docker containers in parallel
        self.name = name
        self.docker_image = docker_image
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self.n_games = n_games
        self.size = size
        self.time_limit_ms = time_limit_ms
        self.win_rate_threshold = win_rate_threshold

        self._start_container()
        self._cp(eval_script_path, Path("/app/eval.py"))

        ai_completions = [get_function_source(ai) for ai in ais]
        self.opponents = [
            self.add_ai(i, completion)
            for i, completion in enumerate(ai_completions, start=1)
        ]

    def __del__(self):
        self._stop_container()

    def evaluate(self, completion: str) -> int:
        scores = []

        completion_path = self.add_ai(0, completion)

        for opponent_path in self.opponents:
            score = 0

            pairs = [
                (completion_path, opponent_path, 1),
                (opponent_path, completion_path, -1),
            ]

            for ai1_path, ai2_path, sign in pairs:
                docker_stdout = self._play(ai1_path, ai2_path)
                results = extract_tagged_text(docker_stdout, "RESULTS")
                if not results:
                    # Usually because the AI code is invalid
                    print(docker_stdout)
                    return -2 * self.n_games
                print(results)
                for line in results.strip().splitlines():
                    winner, reason, count = line.split(",")
                    # Black = 1, White = -1, Draw = 0
                    if winner == Player.BLACK.name:
                        score += int(count) * sign
                    elif winner == Player.WHITE.name:
                        score -= int(count) * sign

            score = max(0, score)
            scores.append(score)

            threshold = (2 * self.win_rate_threshold - 1) * self.n_games * 2
            if score < threshold:
                break

        print(f"Scores: {scores}")
        return sum(scores)

    def add_ai(self, id: int, completion: str) -> Path:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(completion.encode("utf-8"))
        temp_file_path = Path(temp_file.name)
        ai_path = Path(f"/app/ai/{id}")
        self._cp(temp_file_path, ai_path)
        temp_file_path.unlink(missing_ok=True)
        return ai_path

    def _play(self, black: Path, white: Path):
        try:
            result = subprocess.run(
            [
                "docker",
                "exec",
                self.name,
                "python3",
                "/app/eval.py",
                "-b",
                str(black),
                "-w",
                str(white),
                "-n",
                str(self.n_games),
                "-s",
                str(self.size),
                "-t",
                str(self.time_limit_ms),
            ],
            check=True,
            capture_output=True,
            )
            return result.stdout.decode("utf-8")
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while running Docker command: {e}")
            print(f"Command: {e.cmd}")
            print(f"Return code: {e.returncode}")
            print(f"Output: {e.output.decode('utf-8')}")
            print(f"Error output: {e.stderr.decode('utf-8')}")
        return ""

    def _start_container(self):
        """Start the Docker container and block until it's ready."""
        subprocess.run(
            [
                "docker",
                "run",
                "-it",
                "--rm",
                "--name",
                self.name,
                "--memory",
                self.memory_limit,
                "--cpus",
                self.cpu_limit,
                "-d",
                self.docker_image,
            ],
            check=True,
        )
        # Wait for the container to be ready
        while True:
            try:
                result = subprocess.run(
                    ["docker", "inspect", "--format", "{{.State.Running}}", self.name],
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
        print("Docker container is ready.")

    def _stop_container(self):
        """Stop the Docker container."""
        subprocess.run(
            ["docker", "stop", self.name],
            check=True,
        )
        print("Docker container stopped.")

    def _cp(self, src: Path, dest: Path):
        """Copy a file from the host to the Docker container."""
        subprocess.run(
            [
                "docker",
                "cp",
                str(src),
                f"{self.name}:{dest}",
            ],
            check=True,
        )
        print(f"Copied {src} to {dest} in Docker container.")
