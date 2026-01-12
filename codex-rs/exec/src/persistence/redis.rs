use crate::persistence::retry_with_backoff;
use anyhow::Context;
use futures::StreamExt;
use redis::AsyncCommands;
use redis::aio::ConnectionManager;
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio::sync::mpsc;
use uuid::Uuid;

#[derive(Clone)]
pub struct RedisPublisher {
    client: redis::Client,
    manager: Arc<Mutex<ConnectionManager>>,
}

impl RedisPublisher {
    pub async fn connect(redis_url: &str) -> anyhow::Result<Self> {
        let client = redis::Client::open(redis_url).context("Invalid REDIS_URL")?;
        let manager = ConnectionManager::new(client.clone())
            .await
            .context("Failed to connect to Redis")?;
        Ok(Self {
            client,
            manager: Arc::new(Mutex::new(manager)),
        })
    }

    pub async fn connect_with_retry(redis_url: &str) -> anyhow::Result<Self> {
        retry_with_backoff(
            || async { Self::connect(redis_url).await },
            "Redis publisher connection",
        )
        .await
    }

    pub async fn publish(&self, channel: &str, payload: &str) -> anyhow::Result<()> {
        let mut manager = self.manager.lock().await;
        let publish_result: redis::RedisResult<i64> = manager.publish(channel, payload).await;
        if publish_result.is_ok() {
            return Ok(());
        }
        drop(manager);
        let mut manager = ConnectionManager::new(self.client.clone())
            .await
            .context("Failed to reconnect to Redis")?;
        let _: i64 = manager
            .publish(channel, payload)
            .await
            .context("Failed to publish Redis message")?;
        let mut guard = self.manager.lock().await;
        *guard = manager;
        Ok(())
    }
}

pub struct RedisSubscriber {
    client: redis::Client,
    task_id: Uuid,
}

impl RedisSubscriber {
    pub fn new(client: redis::Client, task_id: Uuid) -> Self {
        Self { client, task_id }
    }

    pub async fn connect_with_retry(redis_url: &str, task_id: Uuid) -> anyhow::Result<Self> {
        retry_with_backoff(
            || async {
                let client = redis::Client::open(redis_url).context("Invalid REDIS_URL")?;
                client
                    .get_multiplexed_async_connection()
                    .await
                    .context("Failed to connect to Redis")?;
                Ok(Self::new(client, task_id))
            },
            "Redis subscriber connection",
        )
        .await
    }

    pub fn spawn(self) -> mpsc::UnboundedReceiver<String> {
        let (tx, rx) = mpsc::unbounded_channel();
        let task_id = self.task_id;
        let client = self.client;
        tokio::spawn(async move {
            let channel = format!("task:{task_id}:input");
            let mut delay_ms = 200u64;
            loop {
                match client.get_async_pubsub().await {
                    Ok(mut pubsub) => {
                        if let Err(err) = pubsub.subscribe(&channel).await {
                            tracing::warn!(
                                "Failed to subscribe to Redis channel {channel}: {err:?}"
                            );
                        } else {
                            delay_ms = 200;
                            let mut stream = pubsub.on_message();
                            while let Some(message) = stream.next().await {
                                let payload: redis::RedisResult<String> = message.get_payload();
                                match payload {
                                    Ok(text) => {
                                        if tx.send(text).is_err() {
                                            return;
                                        }
                                    }
                                    Err(err) => {
                                        tracing::warn!("Failed to decode Redis payload: {err:?}");
                                    }
                                }
                            }
                        }
                    }
                    Err(err) => {
                        tracing::warn!("Redis connection failed: {err:?}");
                    }
                }
                tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
                delay_ms = (delay_ms * 2).min(10_000);
            }
        });
        rx
    }
}
