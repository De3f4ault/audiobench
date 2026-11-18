from audiobench.daemon.intelligence.operator_registry import OperatorRegistry


def test_operator_registry_hot_register():
    registry = OperatorRegistry()
    assert len(registry.dynamic_operators) == 0

    # Stub test since actual hot_register is not fully implemented yet
    # We just ensure it doesn't crash
    registry.hot_register(expression_id=99)
