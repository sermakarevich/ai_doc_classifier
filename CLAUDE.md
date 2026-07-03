# Extractly

See [docs/index.md](docs/index.md) for project documentation.

# Instructions
- fleet is a python orchestrator of coding agents with centralized beads db
-- documentation: kb:fleet:add_task
-- tasks should always be created with --cwd /Users/sergii/git/extractly
-- complex and validation tasks should be created with --coder claude --model opus
-- execution / simple coding tasks: --coder opencode --model qwen3.6:latest
-- after adding tasks to fleet from sddw/design, or after a batch tasks insertion - automatically add additional task to do e2e test of what was done and review of evolvability, maintainability, simplicity, readability, review should end up adding more tasks to fleet
-- typical fleet call is:
```
fleet bd create --title --description --coder --model --deps --cwd
```

- when you asked to sddw feature, create tasks in fleet:
-- /sddw:requirements <feature-name> --auto instructions are in docs/feature/<feature-name> (claude:opus)
-- /sddw:design <feature-name> --auto take into account that executors are qwen3.6 small model, its better to make more smaller and simpler tasks with clear description, make a validation task after each task with instruction to add new task to fleet if adjustment is required (opencode:qwen3.6:latest), create finalizing task doing e2e + review as instructed above (claude:opus). add tasks to fleet, each next one with --deps on previo

- when asked to review the repo implementation:
-- conduct a through review with focus on evolvability, maintainability, simplicity, readability
-- add tasks to fleet (opencode:qwen3.6:latest)