"""Google Drive channel for reading/writing control files.

Uses a GCP service account to authenticate and manages control JSON
files in a shared Google Drive folder. The Colab side reads these
files via its native Drive mount.
"""

from __future__ import annotations

import io
import json
import logging
from datetime import datetime, timezone

from .schemas import ControlFile

logger = logging.getLogger(__name__)

try:
    from google.oauth2.service_account import Credentials
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload

    _HAS_GDRIVE = True
except ImportError:
    _HAS_GDRIVE = False


_SCOPES = ["https://www.googleapis.com/auth/drive.file"]


class DriveChannel:
    """Reads and writes control files to Google Drive.

    Args:
        folder_id: Google Drive folder ID where control files are stored.
        service_account_path: Path to GCP service account JSON key file.
    """

    def __init__(self, folder_id: str, service_account_path: str) -> None:
        if not _HAS_GDRIVE:
            raise ImportError(
                "Google Drive dependencies required. Install with: "
                "pip install google-api-python-client google-auth"
            )
        self._folder_id = folder_id
        creds = Credentials.from_service_account_file(
            service_account_path, scopes=_SCOPES
        )
        self._service = build("drive", "v3", credentials=creds)
        # Cache: run_id -> Drive file ID
        self._file_ids: dict[str, str] = {}

    def write_control(self, control: ControlFile) -> None:
        """Write a control file to Google Drive.

        Creates a new file or updates the existing one for this run.
        """
        filename = f"control_{control.wandb_run_id}.json"
        content = control.model_dump_json(indent=2)
        media = MediaIoBaseUpload(
            io.BytesIO(content.encode("utf-8")),
            mimetype="application/json",
            resumable=False,
        )

        file_id = self._find_file(control.wandb_run_id)

        if file_id:
            # Update existing file
            self._service.files().update(
                fileId=file_id, media_body=media
            ).execute()
            logger.info("Updated control file %s (id=%s)", filename, file_id)
        else:
            # Create new file
            metadata = {
                "name": filename,
                "parents": [self._folder_id],
                "mimeType": "application/json",
            }
            result = (
                self._service.files()
                .create(body=metadata, media_body=media, fields="id")
                .execute()
            )
            file_id = result["id"]
            self._file_ids[control.wandb_run_id] = file_id
            logger.info("Created control file %s (id=%s)", filename, file_id)

    def read_control(self, run_id: str) -> ControlFile | None:
        """Read a control file from Google Drive.

        Returns:
            Parsed ``ControlFile`` or ``None`` if not found.
        """
        file_id = self._find_file(run_id)
        if not file_id:
            return None

        try:
            request = self._service.files().get_media(fileId=file_id)
            buffer = io.BytesIO()
            downloader = MediaIoBaseDownload(buffer, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
            buffer.seek(0)
            data = json.loads(buffer.read().decode("utf-8"))
            return ControlFile(**data)
        except Exception:
            logger.exception("Failed to read control file for run %s", run_id)
            return None

    def is_acknowledged(self, run_id: str) -> bool:
        """Check if the last decision for a run was acknowledged."""
        control = self.read_control(run_id)
        if control is None:
            return True  # No file = nothing pending
        return control.acknowledged

    def _find_file(self, run_id: str) -> str | None:
        """Find the Drive file ID for a run's control file."""
        if run_id in self._file_ids:
            return self._file_ids[run_id]

        filename = f"control_{run_id}.json"
        query = (
            f"name = '{filename}' and "
            f"'{self._folder_id}' in parents and "
            f"trashed = false"
        )
        try:
            results = (
                self._service.files()
                .list(q=query, fields="files(id)", pageSize=1)
                .execute()
            )
            files = results.get("files", [])
            if files:
                file_id = files[0]["id"]
                self._file_ids[run_id] = file_id
                return file_id
        except Exception:
            logger.exception("Failed to search Drive for %s", filename)

        return None


class LocalDriveChannel:
    """Local filesystem fallback for testing without Google Drive.

    Reads/writes control files to a local directory, mimicking the
    Drive channel interface.

    Args:
        control_dir: Local directory path for control files.
    """

    def __init__(self, control_dir: str) -> None:
        from pathlib import Path

        self._dir = Path(control_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def write_control(self, control: ControlFile) -> None:
        """Write control file to local directory."""
        path = self._dir / f"control_{control.wandb_run_id}.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(control.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(path)
        logger.info("Wrote local control file: %s", path)

    def read_control(self, run_id: str) -> ControlFile | None:
        """Read control file from local directory."""
        path = self._dir / f"control_{run_id}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return ControlFile(**data)
        except Exception:
            logger.exception("Failed to read local control file %s", path)
            return None

    def is_acknowledged(self, run_id: str) -> bool:
        """Check if the last decision was acknowledged."""
        control = self.read_control(run_id)
        if control is None:
            return True
        return control.acknowledged
