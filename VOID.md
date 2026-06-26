
Rules:

In typescript, do NOT cast to types if not neccessary. NEVER lazily cast to 'any'. Find the correct type to apply and use it.
After you've made changes to the codebase, run the `read_lint_errors` tool on the files you edited.
You don't need to use `rewrite_file` to edit part of a file. This tool is designed to rewrite the entire file.
All tests you create must be added to  `npm run -s test-browser-void-quick`
