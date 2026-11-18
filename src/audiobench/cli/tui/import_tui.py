import curses
from pathlib import Path

from audiobench.cli.tui.directory_navigator import DirectoryNavigator


class ImportFileManager:
    def __init__(self, start_path="~", start_state=None):
        self.navigator = DirectoryNavigator(start_path=Path(start_path).expanduser())
        self.confirmed_files = []
        self.cancelled = False
        self.next_mode = None
        self.ui_offset = 0

        if start_state:
            self.state_import(start_state)

    def state_export(self) -> dict:
        return {
            "current_path": str(self.navigator.current_path),
            "selected": self.navigator.selected,
            "show_hidden": self.navigator.show_hidden,
            "selected_files": [str(p) for p in self.navigator.selected_files]
        }

    def state_import(self, state: dict) -> None:
        if "current_path" in state:
            self.navigator.current_path = Path(state["current_path"])
        if "selected" in state:
            self.navigator.selected = state["selected"]
        if "show_hidden" in state:
            self.navigator.show_hidden = state["show_hidden"]
        if "selected_files" in state:
            self.navigator.selected_files = {Path(p) for p in state["selected_files"]}

    def draw(self, stdscr):
        curses.curs_set(0)
        stdscr.nodelay(0)
        stdscr.timeout(100)
        stdscr.clear()

        # Colors
        try:
            curses.start_color()
            curses.use_default_colors()
            curses.init_pair(1, curses.COLOR_CYAN, -1)  # Header/Footer
            curses.init_pair(2, curses.COLOR_WHITE, -1)  # Normal
            curses.init_pair(3, curses.COLOR_GREEN, -1)  # Selected folder
            curses.init_pair(4, curses.COLOR_YELLOW, -1)  # Hidden
            curses.init_pair(5, curses.COLOR_MAGENTA, -1)  # Status
            curses.init_pair(6, curses.COLOR_CYAN, -1)  # Separator/Dim Checkbox
        except:
            pass

        while True:
            stdscr.clear()
            height, width = stdscr.getmaxyx()

            header = " AudioBench Import Manager "
            path_str = f" Path: {self.navigator.current_path}"
            if len(path_str) > width - len(header) - 3:
                path_str = "..." + path_str[-(width - len(header) - 6) :]

            stdscr.addstr(0, 0, header, curses.color_pair(1) | curses.A_BOLD)
            stdscr.addstr(0, len(header), path_str.ljust(width - len(header)), curses.color_pair(1))

            info_text = f" Selected Files: {len(self.navigator.selected_files)}"
            if self.navigator.show_hidden:
                info_text += " | [HIDDEN VISIBLE]"

            items = self.navigator.list_items()
            if items:
                info_text += f" | {self.navigator.selected + 1}/{len(items)}"

            stdscr.addstr(1, 0, info_text.ljust(width), curses.color_pair(5))

            # Separator
            if width > 1:
                stdscr.addstr(2, 0, "─" * (width - 1), curses.color_pair(6))

            max_visible = height - 5

            if self.navigator.selected < self.ui_offset:
                self.ui_offset = self.navigator.selected
            elif self.navigator.selected >= self.ui_offset + max_visible:
                self.ui_offset = self.navigator.selected - max_visible + 1

            list_start_y = 3

            visible_items = items[self.ui_offset:self.ui_offset + max_visible]

            for idx_offset, item in enumerate(visible_items):
                idx = self.ui_offset + idx_offset
                y = idx_offset + list_start_y

                max_item_len = width - 10
                display_item = item[:max_item_len] + ".." if len(item) > max_item_len else item
                item_path = self.navigator.current_path / item

                attr_normal = curses.color_pair(2)
                attr_sel = curses.color_pair(3) | curses.A_BOLD

                if idx == self.navigator.selected:
                    row_attr = attr_sel
                elif item.startswith("."):
                    row_attr = curses.color_pair(4)
                elif item_path.is_dir():
                    row_attr = curses.color_pair(2) | curses.A_BOLD
                else:
                    row_attr = attr_normal

                # Icons
                if item_path.is_dir():
                    icon = "[DIR]"
                else:
                    icon = "[AUD]"

                # Checkbox for files
                if item_path.is_file():
                    checkbox = "[x]" if item_path in self.navigator.selected_files else "[ ]"
                    cb_attr = curses.color_pair(3) if item_path in self.navigator.selected_files else curses.color_pair(6)
                else:
                    checkbox = "   "
                    cb_attr = row_attr

                marker = "►" if idx == self.navigator.selected else " "

                try:
                    stdscr.addstr(y, 0, f"{marker} ", row_attr)
                    stdscr.addstr(y, 2, checkbox, cb_attr)
                    stdscr.addstr(y, 6, icon, row_attr)
                    stdscr.addstr(y, 12, display_item, row_attr)
                except curses.error:
                    pass

            # Footer
            footer_1 = " ↑↓/k j: Move | →/l/ENTER: Enter Dir | ←/h: Go Up | SPACE: Select File | s: Confirm "
            footer_2 = " Modes: [1]Audio [2]Auto-Detect [3]Single [4]Batch  |  q: Quit "

            if height > 2:
                try:
                    stdscr.addstr(height - 2, 0, footer_1.ljust(width - 1)[:width-1], curses.color_pair(1))
                    stdscr.addstr(height - 1, 0, footer_2.ljust(width - 1)[:width-1], curses.color_pair(1))
                except curses.error:
                    pass

            key = stdscr.getch()
            if key == ord("q"):
                self.cancelled = True
                break
            elif key == ord("1"):
                self.next_mode = "MODE_AUDIO"
                self.cancelled = True
                break
            elif key == ord("2"):
                self.next_mode = "MODE_AUTO"
                self.cancelled = True
                break
            elif key == ord("3"):
                self.next_mode = "MODE_SINGLE_TX"
                self.cancelled = True
                break
            elif key == ord("4"):
                self.next_mode = "MODE_BATCH_TX"
                self.cancelled = True
                break
            elif key == ord("s"):
                self.confirmed_files = list(self.navigator.selected_files)
                break
            elif key in (curses.KEY_RIGHT, 10, 13, ord("l")):
                self.navigator.enter()
            elif key in (curses.KEY_LEFT, ord("h")):
                self.navigator.go_up()
            elif key in (curses.KEY_UP, ord("k")):
                self.navigator.select_prev()
            elif key in (curses.KEY_DOWN, ord("j")):
                self.navigator.select_next()
            elif key == ord(" "):
                self.navigator.toggle_selection()
            elif key in (ord("t"), ord("H")):
                self.navigator.toggle_hidden_visibility()

    def run(self):
        curses.wrapper(self.draw)
        if self.next_mode:
            return self.next_mode
        return self.confirmed_files if not self.cancelled else None


def launch_file_manager(start_state=None):
    fm = ImportFileManager(start_state=start_state)
    return fm.run(), fm.state_export()
