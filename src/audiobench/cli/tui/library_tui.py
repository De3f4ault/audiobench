import curses

from audiobench.storage.repository import TranscriptionRepository


class LibraryTUI:
    def __init__(self, restore_state=None):
        self.repo = TranscriptionRepository()
        self.untranscribed = self.repo.get_untranscribed_files()
        self.idle_transcripts = self.repo.get_idle_transcripts()

        # Tabs: 0 = Untranscribed, 1 = Idle Transcripts
        self.current_tab = 0
        self.selected_index = 0
        self.scroll_offset = 0
        self.selected_files = set()  # Set of DB IDs
        self.action = None

        if restore_state:
            self.current_tab = restore_state.get("current_tab", 0)
            self.selected_index = restore_state.get("selected_index", 0)
            self.scroll_offset = restore_state.get("scroll_offset", 0)
            self.selected_files = set(restore_state.get("selected_files", []))

    def get_current_list(self):
        if self.current_tab == 0:
            return self.untranscribed
        return self.idle_transcripts

    def draw(self, stdscr):
        import json
        curses.curs_set(0)
        stdscr.nodelay(0)
        stdscr.timeout(100)
        stdscr.clear()

        try:
            curses.start_color()
            curses.use_default_colors()
            curses.init_pair(1, curses.COLOR_CYAN, -1)     # Header/Footer
            curses.init_pair(2, curses.COLOR_WHITE, -1)    # Normal
            curses.init_pair(3, curses.COLOR_GREEN, -1)    # Selected
            curses.init_pair(4, curses.COLOR_YELLOW, -1)   # Tab active
            curses.init_pair(5, curses.COLOR_MAGENTA, -1)  # Status Bar
            curses.init_pair(6, curses.COLOR_CYAN, -1)     # Badges/Metadata/Separator
        except:
            pass

        while True:
            stdscr.clear()
            height, width = stdscr.getmaxyx()

            header = " AudioBench Library Command Center "
            stdscr.addstr(0, 0, header.ljust(width), curses.color_pair(1) | curses.A_BOLD)

            # Tabs
            tab1 = " [1] Untranscribed Audio "
            tab2 = " [2] Idle Transcripts "

            stdscr.addstr(
                1,
                0,
                tab1,
                curses.color_pair(4) | curses.A_BOLD
                if self.current_tab == 0
                else curses.color_pair(2),
            )
            stdscr.addstr(
                1,
                len(tab1),
                tab2,
                curses.color_pair(4) | curses.A_BOLD
                if self.current_tab == 1
                else curses.color_pair(2),
            )

            # Status bar
            sel_count = len(self.selected_files)
            action_name = "Transcription" if self.current_tab == 0 else "Operations"
            status_text = f" Selected for {action_name}: {sel_count} files "
            stdscr.addstr(2, 0, status_text.ljust(width), curses.color_pair(5))

            # Separator
            if width > 1:
                stdscr.addstr(3, 0, "─" * (width - 1), curses.color_pair(6))

            items = self.get_current_list()
            list_start_y = 4

            max_visible = height - 7
            if max_visible < 1:
                max_visible = 1

            if self.selected_index < self.scroll_offset:
                self.scroll_offset = self.selected_index
            elif self.selected_index >= self.scroll_offset + max_visible:
                self.scroll_offset = self.selected_index - max_visible + 1

            if not items:
                stdscr.addstr(
                    list_start_y,
                    2,
                    "No items in this category! You are all caught up.",
                    curses.color_pair(3),
                )
            else:
                for i in range(max_visible):
                    idx = self.scroll_offset + i
                    if idx >= len(items):
                        break

                    item = items[idx]
                    y = i + list_start_y

                    name = item.get("file_name", "Unknown")
                    meta_info = []
                    badge = ""

                    if self.current_tab == 0:
                        duration = item.get("duration_seconds")
                        size_bytes = item.get("file_size_bytes")
                        if duration:
                            mins = int(duration // 60)
                            secs = int(duration % 60)
                            if mins >= 60:
                                hrs = mins // 60
                                mins = mins % 60
                                meta_info.append(f"{hrs}:{mins:02d}:{secs:02d}")
                            else:
                                meta_info.append(f"{mins}:{secs:02d}")
                        if size_bytes:
                            mb = size_bytes / (1024 * 1024)
                            meta_info.append(f"{mb:.1f}MB")

                        tags_str = item.get("tags", "[]")
                        if tags_str:
                            try:
                                tags = json.loads(tags_str)
                                for t in tags:
                                    if t.startswith("engine_preference:"):
                                        badge = f"[{t.split(':')[1].strip()}]"
                            except:
                                pass
                    else:
                        created = item.get("created_at", "")[:10]
                        if created:
                            meta_info.append(created)

                    meta_str = " | ".join(meta_info)
                    if badge:
                        meta_str = f"{badge} {meta_str}"
                    meta_str = f"[{meta_str}]" if meta_str else ""

                    marker = "►" if idx == self.selected_index else " "
                    checkbox = "[x]" if item["id"] in self.selected_files else "[ ]"
                    icon = "[AUD]" if self.current_tab == 0 else "[TXT]"

                    attr_normal = curses.color_pair(2)
                    attr_sel = curses.color_pair(3) | curses.A_BOLD
                    attr_meta = curses.color_pair(6)

                    row_attr = attr_sel if idx == self.selected_index else attr_normal

                    try:
                        stdscr.addstr(y, 0, f"{marker} ", row_attr)
                        cb_attr = curses.color_pair(3) if item["id"] in self.selected_files else curses.color_pair(6)
                        stdscr.addstr(y, 2, checkbox, cb_attr)
                        stdscr.addstr(y, 6, icon, row_attr)

                        max_name_len = width - 14 - len(meta_str)
                        if max_name_len < 5:
                            max_name_len = 5

                        display_name = name[:max_name_len - 2] + ".." if len(name) > max_name_len else name

                        stdscr.addstr(y, 12, display_name, row_attr)

                        if meta_str:
                            meta_x = width - len(meta_str) - 1
                            if meta_x > 12 + len(display_name):
                                stdscr.addstr(y, meta_x, meta_str, attr_meta)
                    except:
                        pass

            # Footer
            footer_1 = "←/→ or 1/2: Switch Tabs | ↑↓: Move | SPACE: Select"
            footer_2 = "t: Transcribe | i: Import OS Files | d: Delete | q: Quit"

            if height > 2:
                stdscr.addstr(height - 2, 0, footer_1.ljust(width - 1), curses.color_pair(1))
                stdscr.addstr(height - 1, 0, footer_2.ljust(width - 1), curses.color_pair(1))

            key = stdscr.getch()
            if key == ord("q"):
                break
            elif key in (curses.KEY_RIGHT, ord("2")):
                self.current_tab = 1
                self.selected_index = 0
                self.scroll_offset = 0
                self.selected_files.clear()
            elif key in (curses.KEY_LEFT, ord("1")):
                self.current_tab = 0
                self.selected_index = 0
                self.scroll_offset = 0
                self.selected_files.clear()
            elif key in (curses.KEY_UP, ord("k")):
                if items:
                    self.selected_index = (self.selected_index - 1) % len(items)
            elif key in (curses.KEY_DOWN, ord("j")):
                if items:
                    self.selected_index = (self.selected_index + 1) % len(items)
            elif key == ord(" "):
                if items:
                    item_id = items[self.selected_index]["id"]
                    if item_id in self.selected_files:
                        self.selected_files.remove(item_id)
                    else:
                        self.selected_files.add(item_id)
            elif key == ord("i"):
                self.action = "switch_to_import"
                break
            elif key == ord("t"):
                if items and not self.selected_files:
                    self.selected_files.add(items[self.selected_index]["id"])
                if self.selected_files:
                    self.action = "transcribe"
                    break
            elif key == ord("d"):
                if items and not self.selected_files:
                    self.selected_files.add(items[self.selected_index]["id"])
                if self.selected_files:
                    self.action = "delete"
                    break

    def state_export(self):
        return {
            "current_tab": self.current_tab,
            "selected_index": self.selected_index,
            "scroll_offset": self.scroll_offset,
            "selected_files": list(self.selected_files),
        }

    def run(self):
        curses.wrapper(self.draw)
        return {
            "action": self.action,
            "tab": "audio" if self.current_tab == 0 else "transcripts",
            "selected_ids": list(self.selected_files),
            "state": self.state_export()
        }


def launch_library_tui(restore_state=None):
    tui = LibraryTUI(restore_state=restore_state)
    return tui.run()
