# Non-interactive mode

For information about non-interactive mode, see [this documentation](https://developers.openai.com/codex/noninteractive).

## Task runner mode (Postgres + Redis)

When `TASK_ID` is set, `codex exec` runs in headless task mode. In this mode the agent:

- Pulls the task record from Postgres for the initial prompt, model, system prompt, and config.
- Streams messages and tool invocations into Postgres.
- Subscribes to Redis for corrections and publishes realtime events for UI updates.

Required environment variables:

- `TASK_ID`: UUID identifying the current task.
- `DATABASE_URL`: Postgres connection string (e.g., `postgresql://user:pass@host:5432/dbname`).
- `REDIS_URL`: Redis connection string (e.g., `redis://host:6379`).

Example:

```bash
TASK_ID=00000000-0000-0000-0000-000000000000 \
DATABASE_URL=postgresql://user:pass@host:5432/dbname \
REDIS_URL=redis://host:6379 \
codex exec
```
