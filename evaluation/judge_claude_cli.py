"""LLM juri untuk RAGAS yang memanggil Claude Code (`claude -p`) memakai langganan Claude, tanpa API key.

Tiap panggilan = satu proses `claude -p` tanpa tool, tanpa riwayat sesi, dijalankan dari folder netral
agar CLAUDE.md proyek tidak ikut terbaca. Konsekuensi: lambat dan memakai batas pesan langganan.
"""
import asyncio
import os
import shutil
import subprocess
import tempfile
from typing import Any, List, Optional

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM

SYSTEM_PROMPT = (
    "Kamu adalah juri penilaian yang mengikuti instruksi pengguna secara persis. "
    "Balas hanya dengan keluaran yang diminta (biasanya JSON valid), tanpa penjelasan tambahan, "
    "tanpa markdown code fence kecuali diminta."
)


class ClaudeCLI(LLM):
    model: str = "claude-sonnet-5-5"
    timeout: int = 300
    concurrency: int = 2

    @property
    def _llm_type(self) -> str:
        return "claude-cli"

    def _cmd(self) -> List[str]:
        exe = shutil.which("claude")
        if not exe:
            raise RuntimeError("Perintah `claude` tidak ditemukan di PATH. Pastikan Claude Code terpasang dan sudah login.")
        return [exe, "-p", "--model", self.model, "--tools", "", "--no-session-persistence",
                "--output-format", "text", "--system-prompt", SYSTEM_PROMPT]

    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> str:
        r = subprocess.run(self._cmd(), input=prompt, capture_output=True, text=True, encoding="utf-8",
                           timeout=self.timeout, cwd=tempfile.gettempdir())
        if r.returncode != 0:
            raise RuntimeError(f"claude -p gagal ({r.returncode}): {r.stderr.strip()[:300] or r.stdout.strip()[:300]}")
        return r.stdout.strip()

    async def _acall(self, prompt: str, stop: Optional[List[str]] = None,
                     run_manager: Optional[AsyncCallbackManagerForLLMRun] = None, **kwargs: Any) -> str:
        sem = _semaphore(self.concurrency)
        async with sem:
            proc = await asyncio.create_subprocess_exec(
                *self._cmd(), stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE, cwd=tempfile.gettempdir())
            try:
                out, err = await asyncio.wait_for(proc.communicate(prompt.encode("utf-8")), timeout=self.timeout)
            except asyncio.TimeoutError:
                proc.kill()
                raise
            if proc.returncode != 0:
                msg = (err.decode("utf-8", "replace") or out.decode("utf-8", "replace")).strip()[:300]
                raise RuntimeError(f"claude -p gagal ({proc.returncode}): {msg}")
            return out.decode("utf-8", "replace").strip()


_SEM: dict = {}


def _semaphore(n: int) -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    if loop not in _SEM:
        _SEM[loop] = asyncio.Semaphore(n)
    return _SEM[loop]


if __name__ == "__main__":
    print(ClaudeCLI(model=os.environ.get("JUDGE_CLI_MODEL", "claude-sonnet-5-5")).invoke('Balas JSON: {"ok": true}'))
