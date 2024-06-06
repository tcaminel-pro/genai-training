"""
API Test

Copyright (C) 2023 Eviden. All rights reserved
"""

import sys
from pathlib import Path

from fastapi.testclient import TestClient

# fmt: off
[sys.path.append(str(path)) for path in [Path.cwd(), Path.cwd().parent, Path.cwd().parent/"python"] if str(path) not in sys.path]  # type: ignore # fmt: on


from python.fastapi_app import app

# Define your FastAPI routes and functions here

client = TestClient(app)


def test_joke():
    response = client.get("/joke?topic=castor")
    assert response.status_code == 200
