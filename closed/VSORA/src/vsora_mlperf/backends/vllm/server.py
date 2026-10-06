"""vLLM as an OpenAI-compatible server: launches `vllm serve` with the yaml engine args, then drives it with the
generic OpenAI client (same path as an external endpoint, minus starting it by hand)."""
import os
import signal
import subprocess
import time

from ..openai_api.client import OpenAIBackend


class VllmServerBackend(OpenAIBackend):
    name = "vllm_server"

    def __init__(self, model: str, port: int = 8000, startup_timeout_s: float = 1800, max_concurrency: int = 1024, **engine_args):
        super().__init__(model, base_url=f"http://localhost:{port}", max_concurrency=max_concurrency)
        self.port = port
        self.startup_timeout_s = startup_timeout_s
        self.engine_args = engine_args
        self._proc = None

    def _command(self):
        cmd = ["vllm", "serve", self.model_name, "--port", str(self.port)]
        for key, value in self.engine_args.items():
            flag = "--" + key.replace("_", "-")
            if value is True:
                cmd.append(flag)
            elif value is not False and value is not None:
                cmd += [flag, str(value)]
        return cmd

    def load(self, scenario: str) -> None:
        import httpx

        cmd = self._command()
        print("launching:", " ".join(cmd), flush=True)
        self._proc = subprocess.Popen(cmd, start_new_session=True)
        deadline = time.monotonic() + self.startup_timeout_s
        while True:
            if self._proc.poll() is not None:
                raise RuntimeError(f"vllm serve exited with code {self._proc.returncode}")
            try:
                if httpx.get(f"{self.base_url}/health", timeout=5).status_code == 200:
                    break
            except httpx.TransportError:
                pass
            if time.monotonic() > deadline:
                self.close()
                raise TimeoutError(f"vllm serve not healthy after {self.startup_timeout_s}s")
            time.sleep(5)
        super().load(scenario)

    def close(self) -> None:
        super().close()
        if self._proc and self._proc.poll() is None:
            os.killpg(self._proc.pid, signal.SIGTERM)  # vllm serve spawns engine workers in its process group
            try:
                self._proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(self._proc.pid, signal.SIGKILL)
        self._proc = None
