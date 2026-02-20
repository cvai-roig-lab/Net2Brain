# Contributing to Net2Brain

Thank you for your interest in contributing to Net2Brain.  
To keep the project stable and maintainable, please follow the workflow described below.

---

## Branching Strategy (Required)

All contributions must follow this process:

1. Create a feature branch from `dev`.
2. Make your changes only in that feature branch.
3. Open a Pull Request (PR) targeting `dev`.
4. Assign **ToastyDom** as reviewer.

**Do not:**

- Push directly to `main`
- Push directly to `dev`

### Why?

- `main` is the stable release branch.
- `dev` is the integration branch.
- Feature branches keep changes isolated and reviewable.
- This prevents unintended side effects and keeps the toolbox stable for all users.

---

## Creating a Feature Branch

Example:

```bash
git checkout dev
git pull
git checkout -b feature/short-description