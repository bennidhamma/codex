use crate::persistence::retry_with_backoff;
use anyhow::Context;
use chrono::Utc;
use deadpool_postgres::Manager;
use deadpool_postgres::Pool;
use serde_json::Value as JsonValue;
use tokio_postgres::NoTls;
use tokio_postgres::types::Json;
use uuid::Uuid;

#[derive(Debug, Clone)]
pub struct TaskRecord {
    pub id: Uuid,
    pub prompt: String,
    pub model: String,
    pub system_prompt: Option<String>,
    pub config: JsonValue,
}

#[derive(Debug, Clone)]
pub struct MessageRecord {
    pub id: Uuid,
    pub sequence_num: i32,
    pub role: String,
    pub content: Option<String>,
    pub content_blocks: Option<JsonValue>,
    pub tool_use_id: Option<String>,
    pub tokens: Option<i32>,
}

#[derive(Debug, Clone)]
pub struct ToolInvocationStart {
    pub tool_use_id: String,
    pub tool_name: String,
    pub input: JsonValue,
    pub working_dir: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ToolInvocationEnd {
    pub tool_use_id: String,
    pub output: Option<String>,
    pub error: Option<String>,
    pub exit_code: Option<i32>,
    pub duration_ms: Option<i64>,
}

#[derive(Clone)]
pub struct PostgresStore {
    pool: Pool,
}

impl PostgresStore {
    pub async fn connect(database_url: &str) -> anyhow::Result<Self> {
        let config: tokio_postgres::Config =
            database_url.parse().context("Invalid DATABASE_URL")?;
        let manager = Manager::new(config, NoTls);
        let pool = Pool::builder(manager)
            .max_size(16)
            .build()
            .context("Failed to build Postgres connection pool")?;
        Ok(Self { pool })
    }

    pub async fn connect_with_retry(database_url: &str) -> anyhow::Result<Self> {
        retry_with_backoff(
            || async {
                let store = Self::connect(database_url).await?;
                let _ = store
                    .pool
                    .get()
                    .await
                    .context("Failed to acquire Postgres connection")?;
                Ok(store)
            },
            "Postgres connection",
        )
        .await
    }

    pub async fn fetch_task(&self, task_id: Uuid) -> anyhow::Result<TaskRecord> {
        retry_with_backoff(
            || async {
                let client = self
                    .pool
                    .get()
                    .await
                    .context("Failed to get Postgres client")?;
                let row = client
                    .query_one(
                        "SELECT id, prompt, model, system_prompt, config FROM tasks WHERE id = $1",
                        &[&task_id],
                    )
                    .await
                    .context("Failed to fetch task record")?;
                Ok(TaskRecord {
                    id: row.get("id"),
                    prompt: row.get("prompt"),
                    model: row.get("model"),
                    system_prompt: row.get("system_prompt"),
                    config: row
                        .get::<_, Option<Json<JsonValue>>>("config")
                        .map(|json| json.0)
                        .unwrap_or(JsonValue::Object(serde_json::Map::new())),
                })
            },
            "Fetch task record",
        )
        .await
    }

    pub async fn mark_task_running(&self, task_id: Uuid) -> anyhow::Result<()> {
        retry_with_backoff(
            || async {
                let client = self
                    .pool
                    .get()
                    .await
                    .context("Failed to get Postgres client")?;
                client
                    .execute(
                        "UPDATE tasks SET status = 'running', started_at = now() WHERE id = $1",
                        &[&task_id],
                    )
                    .await
                    .context("Failed to update task status to running")?;
                Ok(())
            },
            "Update task running",
        )
        .await
    }

    pub async fn mark_task_completed(
        &self,
        task_id: Uuid,
        input_tokens: Option<i64>,
        output_tokens: Option<i64>,
    ) -> anyhow::Result<()> {
        retry_with_backoff(
            || async {
                let client = self
                    .pool
                    .get()
                    .await
                    .context("Failed to get Postgres client")?;
                client
                    .execute(
                        "UPDATE tasks SET status = 'completed', completed_at = now(), input_tokens = COALESCE($2, input_tokens), output_tokens = COALESCE($3, output_tokens) WHERE id = $1",
                        &[&task_id, &input_tokens, &output_tokens],
                    )
                    .await
                    .context("Failed to update task status to completed")?;
                Ok(())
            },
            "Update task completed",
        )
        .await
    }

    pub async fn mark_task_failed(
        &self,
        task_id: Uuid,
        input_tokens: Option<i64>,
        output_tokens: Option<i64>,
    ) -> anyhow::Result<()> {
        retry_with_backoff(
            || async {
                let client = self
                    .pool
                    .get()
                    .await
                    .context("Failed to get Postgres client")?;
                client
                    .execute(
                        "UPDATE tasks SET status = 'failed', completed_at = now(), input_tokens = COALESCE($2, input_tokens), output_tokens = COALESCE($3, output_tokens) WHERE id = $1",
                        &[&task_id, &input_tokens, &output_tokens],
                    )
                    .await
                    .context("Failed to update task status to failed")?;
                Ok(())
            },
            "Update task failed",
        )
        .await
    }

    pub async fn next_sequence_num(&self, task_id: Uuid) -> anyhow::Result<i32> {
        retry_with_backoff(
            || async {
                let client = self
                    .pool
                    .get()
                    .await
                    .context("Failed to get Postgres client")?;
                let row = client
                    .query_one(
                        "SELECT COALESCE(MAX(sequence_num), 0) FROM messages WHERE task_id = $1",
                        &[&task_id],
                    )
                    .await
                    .context("Failed to query message sequence")?;
                let max_seq: i64 = row.get(0);
                let next = max_seq.saturating_add(1).min(i64::from(i32::MAX));
                Ok(next as i32)
            },
            "Query next message sequence",
        )
        .await
    }

    pub async fn last_user_sequence(&self, task_id: Uuid) -> anyhow::Result<Option<i32>> {
        retry_with_backoff(
            || async {
                let client = self
                    .pool
                    .get()
                    .await
                    .context("Failed to get Postgres client")?;
                let row = client
                    .query_one(
                        "SELECT MAX(sequence_num) FROM messages WHERE task_id = $1 AND role = 'user' AND is_active = true",
                        &[&task_id],
                    )
                    .await
                    .context("Failed to query last user message sequence")?;
                let seq: Option<i64> = row.get(0);
                Ok(seq.map(|value| value.min(i64::from(i32::MAX)) as i32))
            },
            "Query last user message sequence",
        )
        .await
    }

    pub async fn insert_messages(
        &self,
        task_id: Uuid,
        messages: &[MessageRecord],
    ) -> anyhow::Result<()> {
        if messages.is_empty() {
            return Ok(());
        }
        retry_with_backoff(
            || async {
                let mut client = self
                    .pool
                    .get()
                    .await
                    .context("Failed to get Postgres client")?;
                let transaction = client
                    .transaction()
                    .await
                    .context("Failed to open message transaction")?;
                let stmt = transaction
                    .prepare(
                        "INSERT INTO messages (id, task_id, sequence_num, role, content, content_blocks, tool_use_id, tokens) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)",
                    )
                    .await
                    .context("Failed to prepare message insert")?;
                for message in messages {
                    let content_blocks = message.content_blocks.clone().map(Json);
                    transaction
                        .execute(
                            &stmt,
                            &[
                                &message.id,
                                &task_id,
                                &message.sequence_num,
                                &message.role,
                                &message.content,
                                &content_blocks,
                                &message.tool_use_id,
                                &message.tokens,
                            ],
                        )
                        .await
                        .with_context(|| {
                            let message_id = message.id;
                            format!("Failed to insert message {message_id} for task {task_id}")
                        })?;
                }
                transaction
                    .commit()
                    .await
                    .context("Failed to commit message insert transaction")?;
                Ok(())
            },
            "Insert messages",
        )
        .await
    }

    pub async fn supersede_messages_after(
        &self,
        task_id: Uuid,
        sequence_num: i32,
    ) -> anyhow::Result<()> {
        let superseded_at = Utc::now();
        retry_with_backoff(
            || async {
                let client = self
                    .pool
                    .get()
                    .await
                    .context("Failed to get Postgres client")?;
                client
                    .execute(
                        "UPDATE messages SET is_active = false, superseded_at = $3 WHERE task_id = $1 AND sequence_num > $2 AND is_active = true",
                        &[&task_id, &sequence_num, &superseded_at],
                    )
                    .await
                    .context("Failed to supersede messages")?;
                Ok(())
            },
            "Supersede messages",
        )
        .await
    }

    pub async fn insert_tool_start(
        &self,
        task_id: Uuid,
        message_id: Option<Uuid>,
        tool: ToolInvocationStart,
    ) -> anyhow::Result<()> {
        let tool_use_id = tool.tool_use_id.clone();
        let tool_name = tool.tool_name.clone();
        let tool_input = tool.input.clone();
        let working_dir = tool.working_dir.clone();
        retry_with_backoff(
            || async {
                let client = self
                    .pool
                    .get()
                    .await
                    .context("Failed to get Postgres client")?;
                client
                    .execute(
                        "INSERT INTO tool_invocations (task_id, message_id, tool_use_id, tool_name, input, working_dir) VALUES ($1, $2, $3, $4, $5, $6)",
                        &[
                            &task_id,
                            &message_id,
                            &tool_use_id,
                            &tool_name,
                            &Json(tool_input.clone()),
                            &working_dir,
                        ],
                    )
                    .await
                    .context("Failed to insert tool invocation")?;
                Ok(())
            },
            "Insert tool invocation",
        )
        .await
    }

    pub async fn update_tool_end(
        &self,
        task_id: Uuid,
        tool: ToolInvocationEnd,
    ) -> anyhow::Result<()> {
        let tool_use_id = tool.tool_use_id.clone();
        let output = tool.output.clone();
        let error = tool.error.clone();
        let exit_code = tool.exit_code;
        let duration_ms = tool.duration_ms;
        retry_with_backoff(
            || async {
                let client = self
                    .pool
                    .get()
                    .await
                    .context("Failed to get Postgres client")?;
                client
                    .execute(
                        "UPDATE tool_invocations SET output = $1, error = $2, exit_code = $3, completed_at = now(), duration_ms = $4 WHERE task_id = $5 AND tool_use_id = $6",
                        &[
                            &output,
                            &error,
                            &exit_code,
                            &duration_ms,
                            &task_id,
                            &tool_use_id,
                        ],
                    )
                    .await
                    .context("Failed to update tool invocation")?;
                Ok(())
            },
            "Update tool invocation",
        )
        .await
    }
}
