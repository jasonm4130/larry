class MockJawDriver:
    """JawDriver implementation for macOS dev — logs fractions to stdout."""

    def set_open_fraction(self, fraction: float) -> None:
        print(f"jaw: {fraction:.2f}")

    def close(self) -> None:
        print("jaw: closed")
