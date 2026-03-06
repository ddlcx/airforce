# Development Guidelines

## Philosophy

### Core Beliefs

- **Incremental progress over big bangs** - Small changes that compile and pass tests
- **Learning from existing code** - Study and plan before implementing
- **Pragmatic over dogmatic** - Adapt to project reality
- **Clear intent over clever code** - Be boring and obvious

### Simplicity

- **Single responsibility** per function/class
- **Avoid premature abstractions**
- **No clever tricks** - choose the boring solution
- If you need to explain it, it's too complex

## Technical Standards

### Architecture Principles

- **Composition over inheritance** - Use dependency injection
- **Interfaces over singletons** - Enable testing and flexibility
- **Explicit over implicit** - Clear data flow and dependencies
- **Test-driven when possible** - Never disable tests, fix them

### Error Handling

- **Fail fast** with descriptive messages
- **Include context** for debugging
- **Handle errors** at appropriate level
- **Never** silently swallow exceptions

## Project Integration

### Learn the Codebase

- Find similar features/components
- Identify common patterns and conventions
- Use same libraries/utilities when possible
- Follow existing test patterns

### Tooling

- Use project's existing build system
- Use project's existing test framework
- Use project's formatter/linter settings
- Don't introduce new tools without strong justification

### Code Style

- Follow existing conventions in the project
- Refer to linter configurations and .editorconfig, if present
- Text files should always end with an empty line

## MCP Tool Use

- Use Context7 to validate current documentation about software libraries
- Use searxng if your primary Web Search or Fetch tools fail
- Use Tavily ONLY when searxng doesn't give you enough information

## Documentation Standards

- **Clean and professional** - Documentation describes the current system state only
- **No deletion records** - Never document what was removed, why it was removed, or the history of removed features
- **No change logs in docs** - Technical documents are not changelogs; they describe what exists, not what changed
- **Present tense** - Write as if the current design was always the design

## Temporary Artifacts Management

- All verification outputs (test images, debug visualizations, pipeline previews) are **temporary artifacts**
- After a feature is implemented and user has reviewed/approved the results, **proactively remind the user** to delete temporary outputs
- Temporary artifacts include: `output/` subdirectories, `*_test_results/` directories, one-off test images, debug outputs
- Ensure all temporary output directories are listed in `.gitignore` so they never enter version control
- Keep the project tree clean — no leftover intermediate files after a feature is complete

## Important Reminders

**NEVER**:
- Use `--no-verify` to bypass commit hooks
- Disable tests instead of fixing them
- Commit code that doesn't compile
- Make assumptions - verify with existing code
- Record deletion history or removal reasons in documentation

**ALWAYS**:
- Commit working code incrementally
- Update plan documentation as you go
- Learn from existing implementations
- Stop after 3 failed attempts and reassess
- Keep documentation clean, describing only the current system
