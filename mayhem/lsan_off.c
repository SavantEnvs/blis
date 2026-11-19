/* Disable LeakSanitizer for every fuzz and standalone binary (leaks are not the bug class fuzzed). */
int __lsan_is_turned_off(void) { return 1; }
