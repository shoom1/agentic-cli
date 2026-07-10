---
name: report_writer
description: Write and compile a LaTeX analysis report to PDF from figures and tables already produced in the run's artifacts directory. Use when the user wants a written or PDF report of a completed analysis.
---

# report_writer

Produce a compiled PDF analysis report with LaTeX. You author a `.tex` file and
compile it with the `compile_document` tool. You do NOT run arbitrary code.

## Inputs

Analysis deliverables (figures as `.png`, tables as `.csv`) live in the run's
**artifacts directory** — the concrete path is in your instructions. Before
writing, list it with `glob` to learn the exact figure filenames:

    glob("<artifacts_dir>/*.png")

Reference figures by **bare filename** (e.g. `accuracy.png`), never full paths —
`compile_document`'s `assets_dir` makes them resolvable.

## Report structure

Write these sections, in order:

1. **Introduction** — the question and why it matters.
2. **Methods** — the data and how it was analyzed.
3. **Results** — the findings, with figures/tables embedded and referenced.
4. **Discussion** — what the results mean; limitations.
5. **References** — a plain list of sources (no citation engine).

## Authoring

1. Load the template: `load_skill_resource("report_writer", "assets/report_template.tex")`.
2. Fill the placeholders (title, author/date, section prose, figure includes, table rows).
3. Escape LaTeX specials in prose: `%  &  _  #  $  {  }` → `\%  \&  \_  \#  \$  \{  \}`.
4. `write_file` the filled source to the build directory as `report.tex`.

## Compile

Call the tool:

    compile_document(
        source_path="<build_dir>/report.tex",
        output_pdf="<artifacts_dir>/report.pdf",
        assets_dir="<artifacts_dir>",
    )

- On `success: true`, tell the user the report is at `pdf_path`. Done.
- On `success: false`, read `errors` and `log_tail`, fix the `.tex`, and recompile.
  **Retry at most 3 times**, then report the failure with the error if still failing.

## Common errors → fixes

- `! Undefined control sequence` — a command from a package you did not
  `\usepackage`. Add the package or use a base-LaTeX equivalent.
- `! LaTeX Error: File '<name>' not found` — a figure name is wrong; re-`glob`
  the artifacts dir and use the exact filename.
- `! Missing $ inserted` / `! You can't use ...` — an unescaped special
  character in prose; escape it.
- `No LaTeX engine on PATH` — the host has no TeX install; tell the user to
  install TeX Live or MacTeX. You cannot compile without it.
