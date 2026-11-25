# Cocoonet development guidance

FOR AGENTS: DO NOT MODIFY THIS FILE.

## 1 Project purpose

Cocoonet is a lightweight scientific Python module for evaluating machine learning models through independently configured local and online workers. The root owns coordination, while workers perform evaluations and return inspectable results.

## 2 Development philosophy

The project values scientific auditability, human interpretability, transparency and reproducibility.

## 3 Development practice

### Prefer functional programming style
- Prefer ordinary functions with explicit inputs and outputs. Straightforward loops and local mutation are appropriate when they keep scientific procedures understandable.
- Introduce classes for meaningful records, configuration, or owned resources. Keep state ownership explicit and separate calculation from operational effects.

### Unit-wise development
- Unit definition: A unit is normally one production function, one class method, or one declaration, together with its directly corresponding tests, typing, docstrings, and minimal supporting edits.
- A bounded set of tightly coupled helpers may form one unit only when the user explicitly approves the named bundle.
- Present the exact proposed unit and obtain explicit user approval before implementation.
- Review necessity, scope, and complexity before proposing a unit. Inspect the relevant source and tests, resolve current uncertainties through the conversation, and follow the exact agreed contracts. Broad phase approval does not authorize successive units or unrequested redesign.
- Complete and verify the approved unit, report the outcome, and stop before the next unit. An additional change revealed by validation needs its own approval.
- Keep changes within the requested scope. Avoid speculative abstractions, dependencies, wrappers, duplicated state, and unrelated restructuring.

### Docstrings
- Give every public function, class, and method a NumPy-style source docstring. Explain purpose, applicable parameters and returns, intentionally exposed exceptions, ownership, and relevant preconditions. Include useful examples for nontrivial usage; omit empty sections.
- Give every other private function, class, and method including helpers a one-line docstring describing its purpose.
- End each non-helper class or function's Notes section with a concise node-usage statement using `Node usage scope: Root -`, `Node usage scope: Worker -`, or `Node usage scope: Shared -`, followed by its responsibility-specific explanation.

### Type validation, code comments, over-engineering, tests and committing
- Prefer decorator-form type validation over direct-form shared validator over inline manual checks, using a later option only when it clearly improves performance or interpretability over the preceding option.
- Place an intention comment immediately above every complex code block. Explain its purpose, reasoning, assumptions, or invariant rather than restating syntax.
- Keep scientific procedures recognizable and source files cohesive. Decompose where it improves understanding or independent verification.
- Use the repository's configured typing, formatting, linting, and test checks. Test real behavior with representative inputs. Test doubles are limited to necessary network-boundary simulations. Report unavailable checks accurately.
- Do not commit automatically. Suggest a commit message with scope after implementation.

### Protected files
- Never modify `AGENTS.md` or `contracts.md`; report any necessary updates in your response instead.

## 4 Development documentation boundaries and ownership

- [contracts.md](contracts_reorg.md) contains only public API declarations with code-level typing and managed working-directory persistence layouts. It excludes internal interfaces, behavioral policies, algorithms, and technical routes.
- [roadmap.md](roadmap_reorg.md) contains one numbered phase map and, per phase, a short intention, a short technical route, a component-name/intention map, and concise tips for units containing multiple functions or methods. Each intention and route is one paragraph of at most 250 words.
- Update the roadmap alongside development to reflect design decisions agreed in conversations, keeping API signatures out of it. Express implementation details and internal interfaces solely in the audited source code and its docstrings.
