def sanitize(data):
    for item in data:
        if not isinstance(item, (int, float)):
            return None
    return data


class AxiomCoder:
    def execute(self, data):
        return sum(data)


if __name__ == "__main__":
    raw_data = [0.6, -0.2, 0.9, "error_string", 0.1]

    clean_data = sanitize(raw_data)

    if clean_data is not None:
        coder = AxiomCoder()
        print("System Status: Clean")
        print(f"Execution Result: {coder.execute(clean_data)}")
    else:
        print("System Status: Blocked")
        print("Healer Action: Invalid data detected and neutralized.")
