import sys

print_union = [str, int, list, tuple, bool]


def colorp(
        text: print_union,
        red: int = 255,
        blue: int = 255,
        green: int = 255,
) -> int:
    return sys.stdout.write(f'\033[38;2;{red};{blue};{green}m{text}[38;2;255;255;255m')
