import pytest
import flood_tool.geo as geo
import flood_tool.tool as tool

def test_imports():
    assert geo is not None, "Failed to import geo module"
    assert tool is not None, "Failed to import tool module"
