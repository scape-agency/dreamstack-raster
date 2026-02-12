# =============================================================================
# Docstring
# =============================================================================

"""
Main Module for DreamStack


"""


# =============================================================================
# Import
# =============================================================================

# Import | Futures

# Import | Standard Library
import platform

try:
    import pkg_resources
except ImportError:
    pkg_resources = None

# Import | Libraries
import dreamstack

# Import | Local Modules


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print()
    print("palet is set!")
    print()
    print(f"tite: {dreamstack.__version__}")
    print(f"Python: {platform.python_version()} ({platform.python_implementation()})")

    if pkg_resources:
        working_set = pkg_resources.working_set
        packages = set([p.project_name for p in working_set]) - set(["palet"])
        palet_pkgs = [p for p in packages if p.lower().startswith("palet")]

        if palet_pkgs:
            print(f"Extensions: {[p for p in palet_pkgs]}")
