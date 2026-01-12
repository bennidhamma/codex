use crate::event_processor::CodexStatus;
use crate::event_processor::EventProcessor;
use crate::event_processor_with_human_output::EventProcessorWithHumanOutput;
use crate::event_processor_with_jsonl_output::EventProcessorWithJsonOutput;
use crate::persistence::MessageRecord;
use crate::persistence::PostgresStore;
use crate::persistence::RedisPublisher;
use crate::persistence::RedisSubscriber;
use crate::persistence::ToolInvocationEnd;
use crate::persistence::ToolInvocationStart;
use anyhow::Context;
use codex_common::oss::ensure_oss_provider_ready;
use codex_common::oss::get_default_model_for_oss_provider;
use codex_core::AuthManager;
use codex_core::ConversationManager;
use codex_core::LMSTUDIO_OSS_PROVIDER_ID;
use codex_core::NewConversation;
use codex_core::OLLAMA_OSS_PROVIDER_ID;
use codex_core::auth::enforce_login_restrictions;
use codex_core::config::Config;
use codex_core::config::ConfigOverrides;
use codex_core::config::find_codex_home;
use codex_core::config::load_config_as_toml_with_cli_overrides;
use codex_core::config::resolve_oss_provider;
use codex_core::git_info::get_git_repo_root;
use codex_core::protocol::AskForApproval;
use codex_core::protocol::Event;
use codex_core::protocol::EventMsg;
use codex_core::protocol::ExecCommandSource;
use codex_core::protocol::Op;
use codex_core::protocol::SessionSource;
use codex_protocol::config_types::ReasoningSummary;
use codex_protocol::config_types::SandboxMode;
use codex_protocol::protocol::TokenUsage;
use codex_protocol::user_input::UserInput;
use codex_utils_absolute_path::AbsolutePathBuf;
use serde::Serialize;
use serde_json::Value as JsonValue;
use serde_json::json;
use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::VecDeque;
use std::path::PathBuf;
use std::time::Duration;
use toml::Value as TomlValue;
use tracing::debug;
use tracing::error;
use tracing::info;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::prelude::*;
use uuid::Uuid;

use crate::cli::Cli;

#[derive(Clone, Debug)]
pub struct TaskEnv {
    pub task_id: Uuid,
    pub database_url: String,
    pub redis_url: String,
}

impl TaskEnv {
    pub fn from_env() -> anyhow::Result<Option<Self>> {
        let Ok(task_id_raw) = std::env::var("TASK_ID") else {
            return Ok(None);
        };
        let task_id = Uuid::parse_str(&task_id_raw)
            .map_err(|err| anyhow::anyhow!("Invalid TASK_ID {task_id_raw}: {err}"))?;
        let database_url = std::env::var("DATABASE_URL")
            .map_err(|_| anyhow::anyhow!("DATABASE_URL must be set when TASK_ID is present"))?;
        let redis_url = std::env::var("REDIS_URL")
            .map_err(|_| anyhow::anyhow!("REDIS_URL must be set when TASK_ID is present"))?;
        Ok(Some(Self {
            task_id,
            database_url,
            redis_url,
        }))
    }
}

#[derive(Clone)]
struct TaskPersistence {
    task_id: Uuid,
    postgres: PostgresStore,
    redis: RedisPublisher,
}

impl TaskPersistence {
    fn message_channel(&self) -> String {
        let task_id = self.task_id;
        format!("task:{task_id}:events")
    }

    async fn publish_event(&self, event_type: &str, payload: JsonValue) {
        let event = json!({
            "type": event_type,
            "task_id": self.task_id,
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "payload": payload,
        });
        if let Err(err) = self
            .redis
            .publish(&self.message_channel(), &event.to_string())
            .await
        {
            tracing::warn!("Failed to publish Redis event: {err:?}");
        }
    }

    async fn publish_status(&self, status: &str) {
        self.publish_event("status_change", json!({ "status": status }))
            .await;
    }
}

struct TurnDefaults {
    cwd: PathBuf,
    approval_policy: AskForApproval,
    sandbox_policy: codex_core::protocol::SandboxPolicy,
    model: String,
    effort: Option<codex_protocol::openai_models::ReasoningEffort>,
    summary: ReasoningSummary,
}

#[derive(Clone)]
struct PendingMessage {
    record: MessageRecord,
    event_payload: JsonValue,
}

struct TaskState {
    next_sequence: i32,
    last_user_sequence: Option<i32>,
    pending_user_inputs: VecDeque<String>,
    running_tools: HashSet<String>,
    tool_start_times: HashMap<String, std::time::Instant>,
    interrupt_pending: bool,
    turn_active: bool,
    last_token_usage: Option<TokenUsage>,
    pending_messages: Vec<PendingMessage>,
    error_seen: bool,
    shutdown_requested: bool,
}

impl TaskState {
    fn new(next_sequence: i32) -> Self {
        Self {
            next_sequence,
            last_user_sequence: None,
            pending_user_inputs: VecDeque::new(),
            running_tools: HashSet::new(),
            tool_start_times: HashMap::new(),
            interrupt_pending: false,
            turn_active: false,
            last_token_usage: None,
            pending_messages: Vec::new(),
            error_seen: false,
            shutdown_requested: false,
        }
    }

    fn next_sequence(&mut self) -> i32 {
        let current = self.next_sequence;
        self.next_sequence = self.next_sequence.saturating_add(1);
        current
    }

    fn queue_message(&mut self, message: PendingMessage) {
        self.pending_messages.push(message);
    }
}

#[derive(Serialize)]
struct ContentBlock {
    #[serde(rename = "type")]
    block_type: &'static str,
    text: String,
}

pub async fn run_task_mode(
    cli: Cli,
    codex_linux_sandbox_exe: Option<PathBuf>,
    task_env: TaskEnv,
) -> anyhow::Result<()> {
    if cli.command.is_some() {
        anyhow::bail!("Task mode does not support subcommands.");
    }
    if !cli.images.is_empty() {
        tracing::warn!("Task mode ignores --image inputs.");
    }
    if cli.prompt.is_some() {
        tracing::warn!("Task mode ignores CLI prompt input.");
    }

    let (stdout_with_ansi, stderr_with_ansi) = match cli.color {
        crate::cli::Color::Always => (true, true),
        crate::cli::Color::Never => (false, false),
        crate::cli::Color::Auto => (
            supports_color::on_cached(supports_color::Stream::Stdout).is_some(),
            supports_color::on_cached(supports_color::Stream::Stderr).is_some(),
        ),
    };

    let default_level = "error";
    let env_filter = EnvFilter::try_from_default_env()
        .or_else(|_| EnvFilter::try_new(default_level))
        .unwrap_or_else(|_| EnvFilter::new(default_level));
    let fmt_layer = tracing_subscriber::fmt::layer()
        .with_ansi(stderr_with_ansi)
        .with_writer(std::io::stderr)
        .with_filter(env_filter);

    let cli_kv_overrides = match cli.config_overrides.parse_overrides() {
        Ok(v) => v,
        #[allow(clippy::print_stderr)]
        Err(e) => {
            eprintln!("Error parsing -c overrides: {e}");
            std::process::exit(1);
        }
    };

    let resolved_cwd = cli.cwd.clone();
    let config_cwd = match resolved_cwd.as_deref() {
        Some(path) => AbsolutePathBuf::from_absolute_path(path.canonicalize()?)?,
        None => AbsolutePathBuf::current_dir()?,
    };

    let postgres = PostgresStore::connect_with_retry(&task_env.database_url).await?;
    let redis = RedisPublisher::connect_with_retry(&task_env.redis_url).await?;
    let persistence = TaskPersistence {
        task_id: task_env.task_id,
        postgres: postgres.clone(),
        redis,
    };
    let task = postgres.fetch_task(task_env.task_id).await?;

    persistence.postgres.mark_task_running(task.id).await?;
    persistence.publish_status("running").await;

    let mut combined_overrides = cli_kv_overrides;
    combined_overrides.extend(task_config_overrides(&task.config)?);

    #[allow(clippy::print_stderr)]
    let config_toml = {
        let codex_home = match find_codex_home() {
            Ok(codex_home) => codex_home,
            Err(err) => {
                eprintln!("Error finding codex home: {err}");
                std::process::exit(1);
            }
        };

        match load_config_as_toml_with_cli_overrides(
            &codex_home,
            &config_cwd,
            combined_overrides.clone(),
        )
        .await
        {
            Ok(config_toml) => config_toml,
            Err(err) => {
                eprintln!("Error loading config.toml: {err}");
                std::process::exit(1);
            }
        }
    };

    let model_provider = if cli.oss {
        let resolved = resolve_oss_provider(
            cli.oss_provider.as_deref(),
            &config_toml,
            cli.config_profile.clone(),
        );

        if let Some(provider) = resolved {
            Some(provider)
        } else {
            return Err(anyhow::anyhow!(
                "No default OSS provider configured. Use --local-provider=provider or set oss_provider to either {LMSTUDIO_OSS_PROVIDER_ID} or {OLLAMA_OSS_PROVIDER_ID} in config.toml"
            ));
        }
    } else {
        None
    };

    let model = Some(task.model.clone()).or_else(|| {
        if cli.oss {
            model_provider
                .as_ref()
                .and_then(|provider_id| get_default_model_for_oss_provider(provider_id))
                .map(std::borrow::ToOwned::to_owned)
        } else {
            None
        }
    });

    let sandbox_mode = if cli.full_auto {
        Some(SandboxMode::WorkspaceWrite)
    } else if cli.dangerously_bypass_approvals_and_sandbox {
        Some(SandboxMode::DangerFullAccess)
    } else {
        cli.sandbox_mode.map(Into::<SandboxMode>::into)
    };

    let overrides = ConfigOverrides {
        model,
        review_model: None,
        config_profile: cli.config_profile.clone(),
        approval_policy: Some(AskForApproval::Never),
        sandbox_mode,
        cwd: resolved_cwd,
        model_provider: model_provider.clone(),
        codex_linux_sandbox_exe,
        base_instructions: None,
        developer_instructions: task.system_prompt.clone(),
        compact_prompt: None,
        include_apply_patch_tool: None,
        show_raw_agent_reasoning: cli.oss.then_some(true),
        tools_web_search_request: None,
        additional_writable_roots: cli.add_dir.clone(),
    };

    let config =
        Config::load_with_cli_overrides_and_harness_overrides(combined_overrides, overrides)
            .await?;

    if let Err(err) = enforce_login_restrictions(&config).await {
        eprintln!("{err}");
        std::process::exit(1);
    }

    let otel = codex_core::otel_init::build_provider(&config, env!("CARGO_PKG_VERSION"));

    #[allow(clippy::print_stderr)]
    let otel = match otel {
        Ok(otel) => otel,
        Err(e) => {
            eprintln!("Could not create otel exporter: {e}");
            std::process::exit(1);
        }
    };

    let otel_logger_layer = otel.as_ref().and_then(|o| o.logger_layer());
    let otel_tracing_layer = otel.as_ref().and_then(|o| o.tracing_layer());

    let _ = tracing_subscriber::registry()
        .with(fmt_layer)
        .with(otel_tracing_layer)
        .with(otel_logger_layer)
        .try_init();

    let mut event_processor: Box<dyn EventProcessor> = match cli.json {
        true => Box::new(EventProcessorWithJsonOutput::new(
            cli.last_message_file.clone(),
        )),
        _ => Box::new(EventProcessorWithHumanOutput::create_with_ansi(
            stdout_with_ansi,
            &config,
            cli.last_message_file.clone(),
        )),
    };

    if cli.oss {
        let provider_id = match model_provider.as_ref() {
            Some(id) => id,
            None => {
                error!("OSS provider unexpectedly not set when oss flag is used");
                return Err(anyhow::anyhow!(
                    "OSS provider not set but oss flag was used"
                ));
            }
        };
        ensure_oss_provider_ready(provider_id, &config)
            .await
            .map_err(|e| anyhow::anyhow!("OSS setup failed: {e}"))?;
    }

    let default_cwd = config.cwd.to_path_buf();
    let default_approval_policy = config.approval_policy.value();
    let default_sandbox_policy = config.sandbox_policy.get();
    let default_effort = config.model_reasoning_effort;
    let default_summary = config.model_reasoning_summary;

    if !cli.skip_git_repo_check && get_git_repo_root(&default_cwd).is_none() {
        eprintln!("Not inside a trusted directory and --skip-git-repo-check was not specified.");
        std::process::exit(1);
    }

    let auth_manager = AuthManager::shared(
        config.codex_home.clone(),
        true,
        config.cli_auth_credentials_store_mode,
    );
    let conversation_manager = ConversationManager::new(auth_manager.clone(), SessionSource::Exec);
    let default_model = conversation_manager
        .get_models_manager()
        .get_model(&config.model, &config)
        .await;

    let NewConversation {
        conversation_id: _,
        conversation,
        session_configured,
    } = conversation_manager
        .new_conversation(config.clone())
        .await?;

    event_processor.print_config_summary(&config, &task.prompt, &session_configured);

    info!("Codex initialized with event: {session_configured:?}");

    let mut task_state = TaskState::new(persistence.postgres.next_sequence_num(task.id).await?);
    task_state.last_user_sequence = persistence.postgres.last_user_sequence(task.id).await?;

    if let Some(system_prompt) = task.system_prompt.as_ref() {
        enqueue_message(&mut task_state, "system", system_prompt.to_string(), None);
    }

    let turn_defaults = TurnDefaults {
        cwd: default_cwd,
        approval_policy: default_approval_policy,
        sandbox_policy: default_sandbox_policy.clone(),
        model: default_model,
        effort: default_effort,
        summary: default_summary,
    };

    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Event>();
    {
        let conversation = conversation.clone();
        tokio::spawn(async move {
            loop {
                match conversation.next_event().await {
                    Ok(event) => {
                        debug!("Received event: {event:?}");
                        let is_shutdown_complete = matches!(event.msg, EventMsg::ShutdownComplete);
                        if tx.send(event).is_err() {
                            break;
                        }
                        if is_shutdown_complete {
                            info!("Received shutdown event, exiting event loop.");
                            break;
                        }
                    }
                    Err(err) => {
                        error!("Error receiving event: {err:?}");
                        break;
                    }
                }
            }
        });
    }

    let mut redis_inputs = RedisSubscriber::connect_with_retry(&task_env.redis_url, task.id)
        .await?
        .spawn();

    submit_user_turn(
        &conversation,
        &persistence,
        &mut task_state,
        &turn_defaults,
        task.prompt.clone(),
    )
    .await?;

    let mut flush_interval = tokio::time::interval(Duration::from_millis(500));

    let mut sigterm = SigtermWatcher::new();

    loop {
        tokio::select! {
            event = rx.recv() => {
                let Some(event) = event else {
                    break;
                };
                if let EventMsg::ElicitationRequest(ev) = &event.msg {
                    conversation
                        .submit(Op::ResolveElicitation {
                            server_name: ev.server_name.clone(),
                            request_id: ev.id.clone(),
                            decision: codex_protocol::approvals::ElicitationAction::Cancel,
                        })
                        .await
                        .ok();
                }
                let shutdown = event_processor.process_event(event.clone());
                match shutdown {
                    CodexStatus::Running => {}
                    CodexStatus::InitiateShutdown => {
                        conversation.submit(Op::Shutdown).await.ok();
                    }
                    CodexStatus::Shutdown => {
                        task_state.shutdown_requested = true;
                    }
                }

                handle_event(
                    &conversation,
                    &persistence,
                    &mut task_state,
                    &turn_defaults,
                    event,
                )
                .await?;
            }
            input = redis_inputs.recv() => {
                let Some(input) = input else {
                    continue;
                };
                handle_user_input(
                    &conversation,
                    &persistence,
                    &mut task_state,
                    &turn_defaults,
                    input,
                )
                .await?;
            }
            _ = flush_interval.tick() => {
                flush_messages(&persistence, &mut task_state).await?;
            }
            _ = sigterm.wait(), if !task_state.shutdown_requested => {
                task_state.shutdown_requested = true;
                persistence.publish_status("failed").await;
                conversation.submit(Op::Shutdown).await.ok();
            }
        }
        if should_exit(&task_state) {
            break;
        }
    }

    flush_messages(&persistence, &mut task_state).await?;

    let token_usage = task_state.last_token_usage.clone();
    if task_state.error_seen || task_state.shutdown_requested {
        persistence
            .postgres
            .mark_task_failed(
                task.id,
                token_usage.as_ref().map(|usage| usage.input_tokens),
                token_usage.as_ref().map(|usage| usage.output_tokens),
            )
            .await?;
        persistence.publish_status("failed").await;
    } else {
        persistence
            .postgres
            .mark_task_completed(
                task.id,
                token_usage.as_ref().map(|usage| usage.input_tokens),
                token_usage.as_ref().map(|usage| usage.output_tokens),
            )
            .await?;
        persistence.publish_status("completed").await;
    }

    event_processor.print_final_output();

    if task_state.error_seen {
        std::process::exit(1);
    }

    Ok(())
}

enum SigtermWatcher {
    #[cfg(unix)]
    Unix(tokio::signal::unix::Signal),
    Disabled,
}

impl SigtermWatcher {
    fn new() -> Self {
        #[cfg(unix)]
        {
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                .map(Self::Unix)
                .unwrap_or(Self::Disabled)
        }
        #[cfg(not(unix))]
        {
            Self::Disabled
        }
    }

    async fn wait(&mut self) {
        match self {
            #[cfg(unix)]
            Self::Unix(signal) => {
                signal.recv().await;
            }
            Self::Disabled => {
                std::future::pending::<()>().await;
            }
        }
    }
}

fn should_exit(state: &TaskState) -> bool {
    state.shutdown_requested && !state.turn_active && state.running_tools.is_empty()
}

async fn handle_user_input(
    conversation: &codex_core::CodexConversation,
    persistence: &TaskPersistence,
    state: &mut TaskState,
    defaults: &TurnDefaults,
    input: String,
) -> anyhow::Result<()> {
    if let Some(last_sequence) = state.last_user_sequence {
        persistence
            .postgres
            .supersede_messages_after(persistence.task_id, last_sequence)
            .await?;
    }
    persistence.publish_status("running").await;
    state.pending_user_inputs.push_back(input);
    state.interrupt_pending = true;
    if state.running_tools.is_empty() {
        if state.turn_active {
            state.interrupt_pending = false;
            conversation.submit(Op::Interrupt).await.ok();
        } else {
            submit_next_user_turn(conversation, persistence, state, defaults).await?;
        }
    }
    Ok(())
}

async fn handle_event(
    conversation: &codex_core::CodexConversation,
    persistence: &TaskPersistence,
    state: &mut TaskState,
    defaults: &TurnDefaults,
    event: Event,
) -> anyhow::Result<()> {
    if matches!(event.msg, EventMsg::Error(_)) {
        state.error_seen = true;
    }

    match &event.msg {
        EventMsg::TaskStarted(_) => {
            state.turn_active = true;
        }
        EventMsg::TaskComplete(_) | EventMsg::TurnAborted(_) => {
            state.turn_active = false;
            if state.pending_user_inputs.is_empty() {
                state.shutdown_requested = true;
            } else if let Err(err) =
                submit_next_user_turn(conversation, persistence, state, defaults).await
            {
                tracing::warn!("Failed to submit queued user input: {err:?}");
            }
        }
        EventMsg::TokenCount(ev) => {
            if let Some(info) = &ev.info {
                state.last_token_usage = Some(info.total_token_usage.clone());
            }
        }
        EventMsg::AgentMessage(ev) => {
            enqueue_message(state, "assistant", ev.message.clone(), None);
        }
        EventMsg::ExecCommandBegin(ev) => {
            state.running_tools.insert(ev.call_id.clone());
            state
                .tool_start_times
                .insert(ev.call_id.clone(), std::time::Instant::now());
            let tool_name = match ev.source {
                ExecCommandSource::UserShell => "user_shell",
                _ => "bash",
            };
            let parsed_cmd = serde_json::to_value(&ev.parsed_cmd).unwrap_or(JsonValue::Null);
            let input = json!({
                "command": ev.command,
                "cwd": ev.cwd,
                "source": ev.source,
                "parsed_cmd": parsed_cmd,
                "interaction_input": ev.interaction_input,
            });
            persistence
                .postgres
                .insert_tool_start(
                    persistence.task_id,
                    None,
                    ToolInvocationStart {
                        tool_use_id: ev.call_id.clone(),
                        tool_name: tool_name.to_string(),
                        input: input.clone(),
                        working_dir: Some(ev.cwd.to_string_lossy().into_owned()),
                    },
                )
                .await?;
            persistence
                .publish_event(
                    "tool_start",
                    json!({
                        "tool_use_id": ev.call_id,
                        "tool_name": tool_name,
                        "input": input,
                        "working_dir": ev.cwd,
                    }),
                )
                .await;
        }
        EventMsg::ExecCommandEnd(ev) => {
            state.running_tools.remove(&ev.call_id);
            state.tool_start_times.remove(&ev.call_id);
            let tool_name = match ev.source {
                ExecCommandSource::UserShell => "user_shell",
                _ => "bash",
            };
            let error = if ev.stderr.trim().is_empty() {
                None
            } else {
                Some(ev.stderr.clone())
            };
            let duration_ms = Some(ev.duration.as_millis() as i64);
            persistence
                .postgres
                .update_tool_end(
                    persistence.task_id,
                    ToolInvocationEnd {
                        tool_use_id: ev.call_id.clone(),
                        output: Some(ev.formatted_output.clone()),
                        error: error.clone(),
                        exit_code: Some(ev.exit_code),
                        duration_ms,
                    },
                )
                .await?;
            persistence
                .publish_event(
                    "tool_complete",
                    json!({
                        "tool_use_id": ev.call_id,
                        "tool_name": tool_name,
                        "output": ev.formatted_output,
                        "error": error,
                        "exit_code": ev.exit_code,
                        "duration_ms": duration_ms,
                    }),
                )
                .await;
        }
        EventMsg::McpToolCallBegin(ev) => {
            state.running_tools.insert(ev.call_id.clone());
            state
                .tool_start_times
                .insert(ev.call_id.clone(), std::time::Instant::now());
            let input = json!({
                "server": ev.invocation.server,
                "tool": ev.invocation.tool,
                "arguments": ev.invocation.arguments,
            });
            persistence
                .postgres
                .insert_tool_start(
                    persistence.task_id,
                    None,
                    ToolInvocationStart {
                        tool_use_id: ev.call_id.clone(),
                        tool_name: ev.invocation.tool.clone(),
                        input: input.clone(),
                        working_dir: None,
                    },
                )
                .await?;
            persistence
                .publish_event(
                    "tool_start",
                    json!({
                        "tool_use_id": ev.call_id,
                        "tool_name": ev.invocation.tool,
                        "input": input,
                    }),
                )
                .await;
        }
        EventMsg::McpToolCallEnd(ev) => {
            state.running_tools.remove(&ev.call_id);
            state.tool_start_times.remove(&ev.call_id);
            let (output, error) = match &ev.result {
                Ok(result) => (serde_json::to_string(result).ok(), None),
                Err(err) => (None, Some(err.clone())),
            };
            let duration_ms = Some(ev.duration.as_millis() as i64);
            persistence
                .postgres
                .update_tool_end(
                    persistence.task_id,
                    ToolInvocationEnd {
                        tool_use_id: ev.call_id.clone(),
                        output: output.clone(),
                        error: error.clone(),
                        exit_code: None,
                        duration_ms,
                    },
                )
                .await?;
            persistence
                .publish_event(
                    "tool_complete",
                    json!({
                        "tool_use_id": ev.call_id,
                        "tool_name": ev.invocation.tool,
                        "output": output,
                        "error": error,
                        "duration_ms": duration_ms,
                    }),
                )
                .await;
        }
        EventMsg::PatchApplyBegin(ev) => {
            state.running_tools.insert(ev.call_id.clone());
            state
                .tool_start_times
                .insert(ev.call_id.clone(), std::time::Instant::now());
            let input = json!({
                "auto_approved": ev.auto_approved,
                "changes": ev.changes,
            });
            persistence
                .postgres
                .insert_tool_start(
                    persistence.task_id,
                    None,
                    ToolInvocationStart {
                        tool_use_id: ev.call_id.clone(),
                        tool_name: "apply_patch".to_string(),
                        input: input.clone(),
                        working_dir: None,
                    },
                )
                .await?;
            persistence
                .publish_event(
                    "tool_start",
                    json!({
                        "tool_use_id": ev.call_id,
                        "tool_name": "apply_patch",
                        "input": input,
                    }),
                )
                .await;
        }
        EventMsg::PatchApplyEnd(ev) => {
            state.running_tools.remove(&ev.call_id);
            let duration_ms = state
                .tool_start_times
                .remove(&ev.call_id)
                .map(|start| start.elapsed().as_millis() as i64);
            let error = if ev.success {
                None
            } else if ev.stderr.trim().is_empty() {
                Some("apply_patch failed".to_string())
            } else {
                Some(ev.stderr.clone())
            };
            persistence
                .postgres
                .update_tool_end(
                    persistence.task_id,
                    ToolInvocationEnd {
                        tool_use_id: ev.call_id.clone(),
                        output: Some(ev.stdout.clone()),
                        error: error.clone(),
                        exit_code: None,
                        duration_ms,
                    },
                )
                .await?;
            persistence
                .publish_event(
                    "tool_complete",
                    json!({
                        "tool_use_id": ev.call_id,
                        "tool_name": "apply_patch",
                        "output": ev.stdout,
                        "error": error,
                        "duration_ms": duration_ms,
                    }),
                )
                .await;
        }
        EventMsg::ViewImageToolCall(ev) => {
            let input = json!({ "path": ev.path });
            persistence
                .postgres
                .insert_tool_start(
                    persistence.task_id,
                    None,
                    ToolInvocationStart {
                        tool_use_id: ev.call_id.clone(),
                        tool_name: "view".to_string(),
                        input: input.clone(),
                        working_dir: None,
                    },
                )
                .await?;
            persistence
                .postgres
                .update_tool_end(
                    persistence.task_id,
                    ToolInvocationEnd {
                        tool_use_id: ev.call_id.clone(),
                        output: None,
                        error: None,
                        exit_code: None,
                        duration_ms: Some(0),
                    },
                )
                .await?;
            persistence
                .publish_event(
                    "tool_complete",
                    json!({
                        "tool_use_id": ev.call_id,
                        "tool_name": "view",
                        "input": input,
                        "duration_ms": 0,
                    }),
                )
                .await;
        }
        _ => {}
    }

    maybe_flush_messages(persistence, state).await?;

    if state.running_tools.is_empty() && state.interrupt_pending && state.turn_active {
        state.interrupt_pending = false;
        conversation.submit(Op::Interrupt).await.ok();
    }

    if matches!(event.msg, EventMsg::ShutdownComplete) {
        state.shutdown_requested = true;
    }

    if !state.turn_active && !state.shutdown_requested {
        submit_next_user_turn(conversation, persistence, state, defaults).await?;
    }

    Ok(())
}

fn enqueue_message(
    state: &mut TaskState,
    role: &str,
    content: String,
    tool_use_id: Option<String>,
) {
    let sequence_num = state.next_sequence();
    let id = Uuid::new_v4();
    if role == "user" {
        state.last_user_sequence = Some(sequence_num);
    }
    let content_blocks = vec![ContentBlock {
        block_type: "text",
        text: content.clone(),
    }];
    let record = MessageRecord {
        id,
        sequence_num,
        role: role.to_string(),
        content: Some(content.clone()),
        content_blocks: Some(json!(content_blocks)),
        tool_use_id,
        tokens: None,
    };
    let event_payload = json!({
        "message_id": id,
        "sequence_num": sequence_num,
        "role": role,
        "content": content,
        "content_blocks": content_blocks,
    });
    state.queue_message(PendingMessage {
        record,
        event_payload,
    });
}

async fn flush_messages(
    persistence: &TaskPersistence,
    state: &mut TaskState,
) -> anyhow::Result<()> {
    if state.pending_messages.is_empty() {
        return Ok(());
    }
    let pending = std::mem::take(&mut state.pending_messages);
    let records: Vec<MessageRecord> = pending.iter().map(|msg| msg.record.clone()).collect();
    persistence
        .postgres
        .insert_messages(persistence.task_id, &records)
        .await?;
    for message in pending {
        persistence
            .publish_event("message", message.event_payload)
            .await;
    }
    Ok(())
}

async fn maybe_flush_messages(
    persistence: &TaskPersistence,
    state: &mut TaskState,
) -> anyhow::Result<()> {
    if state.pending_messages.len() >= 10 {
        flush_messages(persistence, state).await?;
    }
    Ok(())
}

async fn submit_user_turn(
    conversation: &codex_core::CodexConversation,
    persistence: &TaskPersistence,
    state: &mut TaskState,
    defaults: &TurnDefaults,
    prompt: String,
) -> anyhow::Result<()> {
    let items = vec![UserInput::Text {
        text: prompt.clone(),
    }];
    conversation
        .submit(Op::UserTurn {
            items,
            cwd: defaults.cwd.clone(),
            approval_policy: defaults.approval_policy,
            sandbox_policy: defaults.sandbox_policy.clone(),
            model: defaults.model.clone(),
            effort: defaults.effort,
            summary: defaults.summary,
            final_output_json_schema: None,
        })
        .await?;
    state.turn_active = true;
    enqueue_message(state, "user", prompt, None);
    maybe_flush_messages(persistence, state).await?;
    Ok(())
}

async fn submit_next_user_turn(
    conversation: &codex_core::CodexConversation,
    persistence: &TaskPersistence,
    state: &mut TaskState,
    defaults: &TurnDefaults,
) -> anyhow::Result<()> {
    if state.turn_active {
        return Ok(());
    }
    if let Some(prompt) = state.pending_user_inputs.pop_front() {
        submit_user_turn(conversation, persistence, state, defaults, prompt).await?;
    }
    Ok(())
}

fn task_config_overrides(config: &JsonValue) -> anyhow::Result<Vec<(String, TomlValue)>> {
    if config.is_null() {
        return Ok(Vec::new());
    }
    let toml_value: TomlValue = serde_json::from_value(config.clone())
        .context("Task config JSON is not compatible with TOML")?;
    let TomlValue::Table(table) = toml_value else {
        tracing::warn!("Task config should be a JSON object to apply overrides.");
        return Ok(Vec::new());
    };
    let mut overrides = Vec::new();
    for (key, value) in table {
        flatten_overrides(&key, value, &mut overrides);
    }
    Ok(overrides)
}

fn flatten_overrides(prefix: &str, value: TomlValue, overrides: &mut Vec<(String, TomlValue)>) {
    match value {
        TomlValue::Table(table) => {
            for (key, value) in table {
                let path = format!("{prefix}.{key}");
                flatten_overrides(&path, value, overrides);
            }
        }
        _ => {
            overrides.push((prefix.to_string(), value));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn task_config_overrides_ignores_non_table_configs() {
        let overrides = task_config_overrides(&json!("not an object")).unwrap();

        assert_eq!(overrides, Vec::new());
    }

    #[test]
    fn task_config_overrides_flattens_nested_tables() {
        let overrides = task_config_overrides(&json!({
            "model": {
                "temperature": 0.2,
                "max_tokens": 1200,
                "stop": ["final", "done"],
            },
            "logging": {
                "level": "info",
            },
            "enable": true,
        }))
        .unwrap();

        let mut flattened = overrides;
        flattened.sort_by(|(left, _), (right, _)| left.cmp(right));

        assert_eq!(
            flattened,
            vec![
                ("enable".to_string(), TomlValue::Boolean(true)),
                (
                    "logging.level".to_string(),
                    TomlValue::String("info".to_string()),
                ),
                ("model.max_tokens".to_string(), TomlValue::Integer(1200),),
                (
                    "model.stop".to_string(),
                    TomlValue::Array(vec![
                        TomlValue::String("final".to_string()),
                        TomlValue::String("done".to_string()),
                    ]),
                ),
                ("model.temperature".to_string(), TomlValue::Float(0.2),),
            ]
        );
    }

    #[test]
    fn task_config_overrides_handles_null_config() {
        let overrides = task_config_overrides(&JsonValue::Null).unwrap();

        assert_eq!(overrides, Vec::new());
    }
}
