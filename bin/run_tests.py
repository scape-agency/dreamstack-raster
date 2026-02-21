# -*- coding: utf-8 -*-


# =============================================================================
# Docstring
# =============================================================================

"""
Dreamstack Raster - Test Runner
===========


"""


# =============================================================================
# Imports
# =============================================================================

# Import | Future
from __future__ import annotations

# Import | Standard Library
import os
import sys
from pathlib import Path

# Import | Libraries


# Import | Local Modules


# =============================================================================
# Main
# =============================================================================


def main() -> int:
    """
    Run pytest with repo root on sys.path (equivalent to PYTHONPATH=.)
    and discover tests in tst/ by default.
    Pass through any extra CLI args: `poetry run test -q -k storage`.
    """
    # Ensure repo root & src/ are importable
    repo_root = Path.cwd()  # assumes you run from repo root
    sys.path.insert(0, str(repo_root))  # like PYTHONPATH=.
    sys.path.insert(
        0, str(repo_root / "src")
    )  # ensure `sturnus` package is importable

    # If you’re using tst.settings as DJANGO_SETTINGS_MODULE:
    os.environ.setdefault(
        "DJANGO_SETTINGS_MODULE",
        "tst.settings",
    )

    try:
        import pytest  # type: ignore
    except ImportError:
        print(
            "pytest is not installed in the current environment.",
            file=sys.stderr,
        )
        return 1

    # Default to running tests from tst/ if no paths were provided
    args = sys.argv[1:]
    if not any(
        a.startswith("-") or a.endswith(".py") or "/" in a for a in args
    ):
        args = ["tst", *args]

    return pytest.main(args)


if __name__ == "__main__":
    raise SystemExit(main())


# import sys

# import django
# from django.conf import settings

# settings.configure(
#     SILENCED_SYSTEM_CHECKS=["mysql.E001"],
#     # Application definition
#     INSTALLED_APPS=[
#         "django.contrib.admin",
#         "django.contrib.auth",
#         "django.contrib.contenttypes",
#         "django.contrib.sessions",
#         "django.contrib.messages",
#         "django.contrib.staticfiles",
#         "rest_framework",
#         "rest_framework.authtoken",
#         "on_demand",
#     ],
#     MIDDLEWARE=[
#         "django.middleware.security.SecurityMiddleware",
#         "django.contrib.sessions.middleware.SessionMiddleware",
#         "django.middleware.common.CommonMiddleware",
#         "django.middleware.csrf.CsrfViewMiddleware",
#         "django.contrib.auth.middleware.AuthenticationMiddleware",
#         "django.contrib.messages.middleware.MessageMiddleware",
#         "django.middleware.clickjacking.XFrameOptionsMiddleware",
#     ],
#     ROOT_URLCONF="on_demand.urls",
#     TEMPLATES=[
#         {
#             "BACKEND": "django.template.backends.django.DjangoTemplates",
#             "DIRS": [],
#             "APP_DIRS": True,
#             "OPTIONS": {
#                 "context_processors": [
#                     "django.template.context_processors.debug",
#                     "django.template.context_processors.request",
#                     "django.contrib.auth.context_processors.auth",
#                     "django.contrib.messages.context_processors.messages",
#                 ],
#             },
#         },
#     ],
#     DATABASES={
#         "default": {
#             "ENGINE": "django.db.backends.sqlite3",
#             "NAME": "on_demand",
#             "USER": "root",
#             "PASSWORD": "",
#             "HOST": "localhost",
#             "PORT": "3306",
#         }
#     },
# )

# django.setup()
# from django.test.runner import DiscoverRunner

# test_runner = DiscoverRunner(verbosity=2)

# failures = test_runner.run_tests(["."])
# if failures:
#     sys.exit(failures)
