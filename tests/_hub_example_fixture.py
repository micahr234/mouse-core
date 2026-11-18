"""Capture notebook Hub uploads locally without any network writes."""
from pathlib import Path
import shutil
import huggingface_hub


def capture_upload(monkeypatch, destination):
    calls = []
    class Repository(str):
        @property
        def repo_id(self):
            return self.removeprefix('https://huggingface.co/')
    def create(_api, *, repo_id, **kwargs):
        resolved = repo_id if '/' in repo_id else 'test/' + repo_id
        return Repository('https://huggingface.co/' + resolved)
    def upload(_api, *, folder_path, repo_id, **kwargs):
        shutil.copytree(folder_path, destination, dirs_exist_ok=True)
        calls.append(dict(repo_id=repo_id, files=sorted(str(p.relative_to(destination))
            for p in Path(destination).rglob('*') if p.is_file())))
        return 'test-commit'
    monkeypatch.setattr(huggingface_hub.HfApi, 'create_repo', create)
    monkeypatch.setattr(huggingface_hub.HfApi, 'upload_folder', upload)
    return calls
