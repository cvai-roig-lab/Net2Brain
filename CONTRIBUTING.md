# Contributing to Net2Brain

Thank you for your interest in contributing! To keep the codebase clean and reviewable, please follow the guidelines below.


## Workflow

### 1. Fork and branch from `development`

Always branch off from the `development` branch — **not** `main`. This keeps the main branch stable and ensures your changes are integrated through proper review.

```bash
git checkout development
git pull origin development
git checkout -b feature/your-feature-name
```

Use a descriptive branch name that reflects what you're working on, e.g. `feature/add-my-model` or `fix/rdm-bug`.

### 2. Make your changes

Keep your changes focused. One feature or fix per PR makes reviewing much easier.

### 3. Write a clear PR description

When opening your pull request, **please explain**:

- **What** you changed (a concise summary of the code changes)
- **Why** you changed it (the motivation or problem you're solving)
- Any relevant context, links to issues, or caveats reviewers should know about
- Example code for quick triggering of the added functionality


### 4. Open a PR into `development`

Open your pull request against the `development` branch — not `main`.

### 5. Assign a reviewer

Always assign **@ToastyDom** as a reviewer.

---

## Questions?

Feel free to open an issue if you're unsure about anything before diving in.