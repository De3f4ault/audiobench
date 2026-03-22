"""Custom Click group with default command fallback and typo correction.

If the user types a command name that doesn't exist, we suggest the
closest match using difflib (e.g., 'trancribe' → 'Did you mean transcribe?').
"""

import difflib

import click


class DefaultGroup(click.Group):
    """Invokes a default subcommand if the subcommand is missing.

    Also provides typo correction for mistyped command names.
    """

    def __init__(self, *args, **kwargs):
        self.default_command = kwargs.pop("default_command", None)
        super().__init__(*args, **kwargs)

    def parse_args(self, ctx, args):
        if not args and self.default_command is not None:
            args.insert(0, self.default_command)
            return super().parse_args(ctx, args)

        # If we have arguments, check if the first one is a known command or an option.
        if (
            args
            and args[0] not in self.commands
            and not args[0].startswith("-")
            and self.default_command is not None
        ):
            args.insert(0, self.default_command)
        return super().parse_args(ctx, args)

    def resolve_command(self, ctx, args):
        """Override to add typo correction when a command is not found."""
        cmd_name = args[0] if args else None

        # Let Click resolve normally first
        try:
            return super().resolve_command(ctx, args)
        except click.UsageError:
            pass

        # Command not found — try fuzzy matching
        if cmd_name and cmd_name not in self.commands:
            close = difflib.get_close_matches(
                cmd_name,
                list(self.commands.keys()),
                n=1,
                cutoff=0.5,
            )
            if close:
                hint = f"Error: No such command '{cmd_name}'.\n\nDid you mean: {close[0]}?"
                raise click.UsageError(hint)

        raise click.UsageError(f"Error: No such command '{cmd_name}'.")
