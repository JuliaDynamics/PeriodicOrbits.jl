---
name: Problem/bug report
about: Create a report for incorrect/unexpected/erroring behavior
title: ''
labels: ''
assignees: ''

---

**Describe the problem**

A clear and concise description of what the problem is, and what you were expecting to happen instead.

**Minimal Working Example**

Please provide a piece of code that leads to the bug you encounter.

This code should be **runnable**, **self-contained**, and minimal. Remove all irrelevant dependencies
and actively try to reduce the code to its smallest possible version.
This will help us identify and fix the problem faster.

**Package versions**

Please provide the versions you use. To do this, run the code:
```julia
import Pkg
Pkg.status(["Package1", "Package2"]; # etc.
    mode = PKGMODE_MANIFEST
)
```
